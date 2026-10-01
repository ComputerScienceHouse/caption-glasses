import asyncio
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyaudio 
import websockets

from config import DEVICE_CAPTURE_RATE
from client.settings import apply_settings_from_server, update_telemetry
from client.state import (
    invalidate_caption_cache,
    invalidate_partial_cache,
    mark_caption_received,
    state,
    state_lock,
    trim_finals_history,
)
from client.tuning_specs import WS_AUDIO_QUEUE_MAX, WS_CMD_OUTBOUND_MAX, WS_CMD_QUEUE_MAX
from client.ui.captions import parse_sound_labels

FORMAT = pyaudio.paFloat32
CHANNELS = 1
TARGET_RATE = 16000
SERVER_CHUNK_SAMPLES = 2048
FRAMES_PER_BUFFER = 4096

_ui_cmd_queue: queue.Queue | None = None

ws_audio_queue: asyncio.Queue | None = None
ws_cmd_outbound_queue: asyncio.Queue | None = None

_audio_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="caption-audio")
_mic = {"p": None, "stream": None, "rate": TARGET_RATE, "out_buf": np.zeros(0, dtype=np.float32)}

_net = {
    "thread": None,
    "stop": None,
    "connected": None,
    "error": None,
    "loop": None,
}

def detect_default_input_rate(fallback: int = 44100) -> int:
    p = pyaudio.PyAudio()
    try:
        info = p.get_default_input_device_info()
        return int(float(info.get("defaultSampleRate", fallback)))
    except Exception:
        return fallback
    finally:
        p.terminate()

def _device_supports_rate(p: pyaudio.PyAudio, device_index: int, rate: int) -> bool:
    try:
        return bool(
            p.is_format_supported(
                rate,
                input_device=device_index,
                input_channels=CHANNELS,
                input_format=FORMAT,
            )
        )
    except ValueError:
        return False

def _choose_capture_rate(p: pyaudio.PyAudio, device_index: int) -> tuple[int, int]:
    info = p.get_device_info_by_index(device_index)
    device_default = int(float(info.get("defaultSampleRate", 44100)))
    forced = DEVICE_CAPTURE_RATE

    if forced is not None:
        if _device_supports_rate(p, device_index, forced):
            return forced, device_default
        print(
            f"DEVICE_CAPTURE_RATE={forced} not supported by input device; "
            f"falling back to device default {device_default} Hz"
        )
        return device_default, device_default

    return device_default, device_default

def _resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Lightweight linear resample — enough for speech, cheap on the GIL."""
    if src_rate == TARGET_RATE:
        return audio.astype(np.float32, copy=False)
    n = int(audio.size)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    target_len = max(1, int(round(n * TARGET_RATE / src_rate)))
    if target_len == n:
        return audio.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, n, endpoint=False)
    x_new = np.linspace(0.0, 1.0, target_len, endpoint=False)
    return np.interp(x_new, x_old, audio.astype(np.float64, copy=False)).astype(
        np.float32
    )

def _open_mic():
    p = pyaudio.PyAudio()
    device_index = int(p.get_default_input_device_info()["index"])
    rate, device_default = _choose_capture_rate(p, device_index)
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )
    name = p.get_device_info_by_index(device_index).get("name", "default")
    print(
        f"Mic: {name!r} capture={rate} Hz "
        f"(device default={device_default} Hz, env="
        f"{'auto' if DEVICE_CAPTURE_RATE is None else DEVICE_CAPTURE_RATE})"
    )
    return p, stream, rate

def _read_mic_frames() -> list[bytes]:
    stream = _mic["stream"]
    rate = int(_mic["rate"])
    data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
    audio = np.frombuffer(data, dtype=np.float32)
    resampled = _resample_to_16k(audio, rate)

    buf = _mic["out_buf"]
    if buf.size:
        buf = np.concatenate([buf, resampled])
    else:
        buf = resampled

    frames: list[bytes] = []
    while buf.size >= SERVER_CHUNK_SAMPLES:
        chunk = buf[:SERVER_CHUNK_SAMPLES]
        buf = buf[SERVER_CHUNK_SAMPLES:]
        frames.append(chunk.astype(np.float32, copy=False).tobytes())
    _mic["out_buf"] = buf
    return frames

def init_queues() -> None:
    global _ui_cmd_queue
    _ui_cmd_queue = queue.Queue(maxsize=WS_CMD_QUEUE_MAX)


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return 0.0
    return obj


def ws_send(obj):
    if _ui_cmd_queue is None:
        return
    payload = json.dumps(_sanitize_for_json(obj), allow_nan=False)
    try:
        _ui_cmd_queue.put_nowait(payload)
    except queue.Full:
        try:
            _ui_cmd_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _ui_cmd_queue.put_nowait(payload)
        except queue.Full:
            pass


def network_connected() -> bool:
    ev = _net["connected"]
    return bool(ev and ev.is_set() and _net["error"] is None)


def network_error() -> BaseException | None:
    return _net["error"]

def _put_outbound_nowait(payload) -> None:
    try:
        ws_cmd_outbound_queue.put_nowait(payload)
    except asyncio.QueueFull:
        try:
            ws_cmd_outbound_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            ws_cmd_outbound_queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass


def _put_audio_nowait(payload: bytes) -> None:
    try:
        ws_audio_queue.put_nowait(payload)
    except asyncio.QueueFull:
        try:
            ws_audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            ws_audio_queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass


async def ws_outbound(websocket):
    while True:
        payload = None
        try:
            payload = ws_audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        if payload is None:
            audio_task = asyncio.create_task(ws_audio_queue.get())
            cmd_task = asyncio.create_task(ws_cmd_outbound_queue.get())
            done, pending = await asyncio.wait(
                {audio_task, cmd_task}, return_when=asyncio.FIRST_COMPLETED
            )
            payload = next(iter(done)).result()
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        try:
            await websocket.send(payload)
        except websockets.ConnectionClosed:
            break


async def ws_sender():
    """Bridge thread-safe UI commands into the asyncio outbound queue."""
    loop = asyncio.get_running_loop()
    while True:
        msg = await loop.run_in_executor(None, _ui_cmd_queue.get)
        if msg is None:
            break
        _put_outbound_nowait(msg)


async def send_audio():
    loop = asyncio.get_running_loop()
    try:
        p, stream, rate = await loop.run_in_executor(_audio_executor, _open_mic)
        _mic["p"] = p
        _mic["stream"] = stream
        _mic["rate"] = rate
        _mic["out_buf"] = np.zeros(0, dtype=np.float32)
        while True:
            frames = await loop.run_in_executor(_audio_executor, _read_mic_frames)
            for payload in frames:
                _put_audio_nowait(payload)
            await asyncio.sleep(0)
    finally:
        stream = _mic["stream"]
        p = _mic["p"]
        _mic["stream"] = None
        _mic["p"] = None
        _mic["out_buf"] = np.zeros(0, dtype=np.float32)
        if stream is not None:
            stream.close()
        if p is not None:
            p.terminate()


async def receive_text(websocket):
    while True:
        try:
            raw = await websocket.recv()
        except websockets.ConnectionClosed:
            break
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        try:
            t = msg.get("type")
            if t == "partial":
                entry = {
                    "text": msg["text"],
                    "speaker": msg.get("speaker", "SPEAKER_01"),
                    "language": msg.get("language"),
                }
                with state_lock:
                    state["partial"] = entry
                mark_caption_received()
                update_telemetry(msg)
            elif t == "final":
                entry = {
                    "text": msg["text"],
                    "speaker": msg.get("speaker", "SPEAKER_01"),
                    "language": msg.get("language"),
                }
                with state_lock:
                    if msg.get("revise") and state["finals"]:
                        state["finals"][-1] = entry
                    else:
                        state["finals"].append(entry)
                    state["partial"] = {"text": "", "speaker": ""}
                if msg.get("revise"):
                    invalidate_caption_cache()
                else:
                    invalidate_partial_cache()
                mark_caption_received()
                trim_finals_history()
                update_telemetry(msg)
            elif t == "telemetry":
                update_telemetry(msg)
            elif t == "settings":
                apply_settings_from_server(msg.get("settings", {}))
            elif t == "sound":
                labels = parse_sound_labels(msg)
                with state_lock:
                    state["sound_labels"] = labels
                    state["sound"] = msg.get("text") or ", ".join(
                        d["label"] for d in labels
                    )
                    state["sound_timestamp"] = int(time.monotonic() * 1000)
        except Exception:
            continue


async def _watch_stop(stop: threading.Event, websocket):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, stop.wait)
    try:
        await websocket.close()
    except Exception:
        pass


async def _network_main(uri: str, stop: threading.Event) -> None:
    global ws_audio_queue, ws_cmd_outbound_queue

    _net["loop"] = asyncio.get_running_loop()
    ws_audio_queue = asyncio.Queue(maxsize=WS_AUDIO_QUEUE_MAX)
    ws_cmd_outbound_queue = asyncio.Queue(maxsize=WS_CMD_OUTBOUND_MAX)

    try:
        async with websockets.connect(uri) as websocket:
            _net["error"] = None
            _net["connected"].set()
            print("Connected to WebSocket.")
            tasks = [
                asyncio.create_task(receive_text(websocket), name="receive_text"),
                asyncio.create_task(ws_sender(), name="ws_sender"),
                asyncio.create_task(ws_outbound(websocket), name="ws_outbound"),
                asyncio.create_task(_watch_stop(stop, websocket), name="watch_stop"),
            ]
            if state.get("enable_mic", True):
                tasks.append(asyncio.create_task(send_audio(), name="send_audio"))
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            results = await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, asyncio.CancelledError):
                    print(f"Network task {task.get_name()} failed: {type(exc).__name__}: {exc}")
            for result in results:
                if isinstance(result, Exception) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    print(f"Network task failed: {type(result).__name__}: {result}")
    except Exception as e:
        _net["error"] = e
        _net["connected"].set()
        print(f"Connection Error: {e}")
    finally:
        _net["loop"] = None


def _network_thread_main(uri: str, stop: threading.Event) -> None:
    try:
        asyncio.run(_network_main(uri, stop))
    except Exception as e:
        _net["error"] = e
        if _net["connected"] is not None:
            _net["connected"].set()
        print(f"Connection Error: {e}")


def start_network(uri: str) -> None:
    if _net["thread"] is not None and _net["thread"].is_alive():
        return
    _net["error"] = None
    _net["stop"] = threading.Event()
    _net["connected"] = threading.Event()
    _net["thread"] = threading.Thread(
        target=_network_thread_main,
        args=(uri, _net["stop"]),
        name="caption-network",
        daemon=True,
    )
    _net["thread"].start()


def stop_network(timeout: float = 5.0) -> None:
    stop = _net["stop"]
    if stop is not None:
        stop.set()
    if _ui_cmd_queue is not None:
        try:
            _ui_cmd_queue.put_nowait(None)
        except queue.Full:
            pass
    thread = _net["thread"]
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
    _net["thread"] = None
    _net["stop"] = None
    _net["connected"] = None
    _net["loop"] = None
