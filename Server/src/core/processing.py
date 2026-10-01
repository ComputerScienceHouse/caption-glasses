import asyncio
import collections
import functools
import gc
import json
import threading
import time
import uuid
import noisereduce as nr
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import WebSocket
from logging import Logger, getLogger
from starlette.websockets import WebSocketDisconnect

from config import (
    CAPTION_ENV_TELEMETRY,
    CHUNK_SIZE,
    PARTIAL_AUDIO_MAX_SEC,
    PHRASE_TIMEOUT,
    PROFANITY_FILTER_DEFAULT,
    SAMPLE_RATE,
    SESSION_MAINTENANCE_CHUNKS,
)
from core.yamnet_categories import (
    YAMNET_CATEGORY_IDS,
    YAMNET_HOP_CHUNKS,
    YAMNET_WINDOW_SAMPLES,
    normalize_category,
)
from core.acoustics import (
    ConnectionAcoustics,
    YamnetAdaptiveState,
    chunk_rms,
    effective_silence_limit,
)
from core.session_settings import MODE_NAMES, SessionSettings
from core.sfx import SfxTracker
from modules import diarization, profanity_filter, transcription

logger: Logger = getLogger(__name__)
logger.info("Initiating Threads")

CHUNKS_PER_SEC: float = SAMPLE_RATE / CHUNK_SIZE
UTTERANCE_OVERLAP_SEC: float = 0.5
UTTERANCE_OVERLAP_CHUNKS: int = max(1, int(UTTERANCE_OVERLAP_SEC * CHUNKS_PER_SEC))

gpu_lock: asyncio.Lock = asyncio.Lock()
whisper_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
vad_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
sound_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
yamnet_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
diart_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

INT16_FRAME_BYTES: int = CHUNK_SIZE * 2
FLOAT_FRAME_BYTES: int = CHUNK_SIZE * 4
GPU_LOCK_SLOW_WAIT_SEC: float = 0.5
DENOISE_MIN_NOISE_SCORE: float = 0.28
PARTIAL_UNSTABLE_TAIL_TOKENS: int = 2

DIART_MAX_PENDING_CHUNKS: int = 8
DIART_DROP_LOG_EVERY: int = 200

_diart_pending: int = 0
_diart_dropped: int = 0
_diart_lock: threading.Lock = threading.Lock()

def _diart_push(chunk: np.ndarray) -> None:
    global _diart_pending
    try:
        diarization.audio_source.push_audio(chunk)
    except Exception as e:
        logger.error("Diart push error: %s", e)
    finally:
        with _diart_lock:
            _diart_pending -= 1

def _submit_diart_chunk(loop: asyncio.AbstractEventLoop, chunk: np.ndarray) -> None:
    global _diart_pending, _diart_dropped
    with _diart_lock:
        if _diart_pending >= DIART_MAX_PENDING_CHUNKS:
            _diart_dropped += 1
            dropped = _diart_dropped
            pending = _diart_pending
            submit = False
        else:
            _diart_pending += 1
            submit = True
    if submit:
        loop.run_in_executor(diart_executor, _diart_push, chunk.copy())
    elif dropped % DIART_DROP_LOG_EVERY == 1:
        logger.warning(
            "Diarization behind real time; dropped %d chunks so far (pending=%d)",
            dropped,
            pending,
        )

def _sanitize_audio(audio: np.ndarray) -> np.ndarray:
    if np.isfinite(audio).all():
        return audio
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def _normalize_chunk(audio: np.ndarray) -> np.ndarray | None:
    if audio.size == CHUNK_SIZE:
        return _sanitize_audio(audio)
    if audio.size > CHUNK_SIZE:
        return _sanitize_audio(audio[:CHUNK_SIZE].copy())
    if audio.size > 0:
        padded = np.pad(audio.astype(np.float32), (0, CHUNK_SIZE - int(audio.size)))
        return _sanitize_audio(padded)
    return None

def _decode_audio_frame(raw_bytes: bytes) -> np.ndarray | None:
    if not raw_bytes:
        return None
    if len(raw_bytes) == INT16_FRAME_BYTES:
        audio = np.frombuffer(raw_bytes, np.int16).astype(np.float32) / 32768.0
    elif len(raw_bytes) == FLOAT_FRAME_BYTES:
        audio = np.frombuffer(raw_bytes, np.float32).copy()
    elif len(raw_bytes) % 4 == 0 and len(raw_bytes) >= 4:
        audio = np.frombuffer(raw_bytes, np.float32).copy()
        logger.debug("Nonstandard float frame length %s bytes", len(raw_bytes))
    elif len(raw_bytes) % 2 == 0:
        audio = np.frombuffer(raw_bytes, np.int16).astype(np.float32) / 32768.0
        logger.debug("Nonstandard int16 frame length %s bytes", len(raw_bytes))
    else:
        logger.warning("Unrecognized audio frame length %s", len(raw_bytes))
        return None
    return _normalize_chunk(audio)

def _parse_json_commands(raw: str | bytes) -> list[dict] | None:
    if isinstance(raw, bytes):
        s = raw.lstrip()
        if not s or s[:1] != b"{":
            return None
        try:
            text = s.decode("utf-8")
        except UnicodeDecodeError:
            return None
    else:
        text = raw.lstrip()
        if not text or text[0] != "{":
            return None

    commands: list[dict] = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            data, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            return commands if commands else None
        if isinstance(data, dict) and data.get("type") is not None:
            commands.append(data)
        idx = end
    return commands if commands else None

def _prepare_audio(
    audio: np.ndarray,
    ac: ConnectionAcoustics,
    settings: SessionSettings,
) -> np.ndarray:
    if settings.skip_denoise:
        gain = ac.post_gain()
        return _sanitize_audio(np.clip(audio * gain, -1.0, 1.0).astype(np.float32))
    reduced = nr.reduce_noise(
        y=audio,
        sr=SAMPLE_RATE,
        prop_decrease=ac.nr_prop_decrease(),
        stationary=ac.nr_stationary(),
        n_fft=512,
        hop_length=128,
    )
    return _sanitize_audio(
        np.clip(reduced * ac.post_gain(), -1.0, 1.0).astype(np.float32)
    )

def _current_speaker(settings: SessionSettings) -> str:
    return diarization.get_speaker_now(settings.speaker_lookback_sec)

def _format_caption_speaker(
    speaker: str,
    *,
    task: str,
    language: str | None,
) -> str:
    normalized = diarization.speaker_client_id(speaker)
    lang = (language or "").strip().lower()[:2]
    if task == "translate" and lang and lang != "en":
        return f"{normalized} ({lang})"
    return normalized

class WebSocketData:
    __slots__ = (
        "connection",
        "uuid",
        "voiced_buffer",
        "is_speaking",
        "is_transcribing",
        "silence_counter",
        "utterance_start_time",
        "pre_roll",
        "yamnet_buffer",
        "chunk_counter",
        "task",
        "yamnet_profile",
        "sfx_enabled_categories",
        "caption_env_telemetry",
        "acoustics",
        "yamnet_adaptive",
        "last_partial_monotonic",
        "last_partial_chunk_count",
        "settings",
        "last_speech_monotonic",
        "speaker_timeline_reset",
        "lyrics_vad_hangover",
        "partial_transcription_inflight",
        "yamnet_inflight",
        "pending_yamnet",
        "sfx_tracker",
        "pending_partial",
        "last_partial_text",
        "last_partial_speaker",
        "last_partial_language",
        "translate_source_lang",
        "rollover_skip_chunks",
        "prefer_final",
    )

    def __init__(self, websocket: WebSocket, uuid_str: str):
        self.connection: WebSocket = websocket
        self.uuid: str = uuid_str
        self.voiced_buffer: list = []
        self.is_speaking: bool = False
        self.is_transcribing = False
        self.silence_counter: float = 0
        self.utterance_start_time: float = time.monotonic()
        self.pre_roll: collections.deque = collections.deque(maxlen=10)
        self.yamnet_buffer: collections.deque = collections.deque(
            maxlen=YAMNET_WINDOW_SAMPLES
        )
        self.chunk_counter: int = 0
        self.task: str = "transcribe"
        self.yamnet_profile: str = "default"
        self.sfx_enabled_categories: set[str] = set(YAMNET_CATEGORY_IDS)
        self.caption_env_telemetry: bool = False
        self.acoustics: ConnectionAcoustics = ConnectionAcoustics()
        self.yamnet_adaptive: YamnetAdaptiveState = YamnetAdaptiveState()
        self.last_partial_monotonic: float = 0.0
        self.last_partial_chunk_count: int = 0
        self.settings: SessionSettings = SessionSettings(
            phrase_timeout_sec=PHRASE_TIMEOUT
        )
        self.last_speech_monotonic: float = 0.0
        self.speaker_timeline_reset: bool = False
        self.lyrics_vad_hangover: int = 0
        self.partial_transcription_inflight: bool = False
        self.yamnet_inflight: bool = False
        self.pending_yamnet: np.ndarray | None = None
        self.sfx_tracker: SfxTracker = SfxTracker()
        self.pending_partial: tuple[np.ndarray, str, bool] | None = None
        self.last_partial_text: str = ""
        self.last_partial_speaker: str = ""
        self.last_partial_language: str | None = None
        self.translate_source_lang: str | None = None
        self.rollover_skip_chunks: int = 0
        self.prefer_final: bool = False

_connections: set[WebSocketData] = set()
_profanity_filter_enabled: bool = PROFANITY_FILTER_DEFAULT


def _telemetry_enabled(websocket: WebSocketData) -> bool:
    return (
        CAPTION_ENV_TELEMETRY
        or websocket.caption_env_telemetry
        or any(c.caption_env_telemetry for c in _connections)
    )


async def _broadcast_json(payload: dict) -> None:
    for client in tuple(_connections):
        try:
            await client.connection.send_json(payload)
        except Exception:
            _connections.discard(client)


def _merge_stable_partial(
    prev: str,
    new: str,
    *,
    unstable_tail: int = PARTIAL_UNSTABLE_TAIL_TOKENS,
) -> str:
    prev = (prev or "").strip()
    new = (new or "").strip()
    if not new:
        return prev
    if not prev:
        return new

    prev_toks = prev.split()
    new_toks = new.split()

    common = 0
    for a, b in zip(prev_toks, new_toks):
        if a == b:
            common += 1
        else:
            break

    if common == len(prev_toks):
        return " ".join(new_toks)
    if common == len(new_toks) and len(new_toks) <= len(prev_toks):
        return prev

    lock_len = max(0, len(prev_toks) - max(1, int(unstable_tail)))
    if common >= lock_len and len(new_toks) >= lock_len:
        return " ".join(prev_toks[:lock_len] + new_toks[lock_len:])
    if len(new_toks) > lock_len:
        return " ".join(prev_toks[:lock_len] + new_toks[lock_len:])
    return prev


_TOKEN_STRIP_CHARS = ".,!?;:…\"'()[]"


def _norm_token(tok: str) -> str:
    return tok.strip(_TOKEN_STRIP_CHARS).lower()


def _stitch_sliding_partial(
    prev: str,
    new: str,
    *,
    unstable_tail: int = PARTIAL_UNSTABLE_TAIL_TOKENS,
) -> str:
    prev = (prev or "").strip()
    new = (new or "").strip()
    if not new:
        return prev
    if not prev:
        return new

    prev_toks = prev.split()
    new_toks = new.split()
    tail = max(1, int(unstable_tail))
    if len(prev_toks) <= tail:
        return new if len(new_toks) >= len(prev_toks) else prev

    norm_new = [_norm_token(t) for t in new_toks]
    for tail_drop in range(0, tail + 1):
        end = len(prev_toks) - tail_drop
        for anchor_len in (4, 3, 2):
            if end < anchor_len:
                continue
            anchor = [_norm_token(t) for t in prev_toks[end - anchor_len : end]]
            if not all(anchor):
                continue
            for j in range(len(new_toks) - anchor_len, -1, -1):
                if norm_new[j : j + anchor_len] == anchor:
                    return " ".join(prev_toks[:end] + new_toks[j + anchor_len :])
    return prev


def _clear_partial_state(websocket: WebSocketData) -> None:
    websocket.last_partial_text = ""
    websocket.last_partial_speaker = ""
    websocket.last_partial_language = None
    websocket.pending_partial = None


async def handle_client_command(data: dict, client: WebSocketData) -> None:
    if not isinstance(data, dict):
        return
    cmd = data.get("type")
    if cmd == "set_task":
        client.task = str(data.get("value", "transcribe"))
        client.translate_source_lang = None
        logger.info("Task updated to: %s", client.task)
    elif cmd == "set_profanity_filter":
        global _profanity_filter_enabled
        _profanity_filter_enabled = bool(data.get("value", False))
        logger.info("Profanity filter: %s", _profanity_filter_enabled)
    elif cmd == "set_yamnet_profile":
        p = data.get("value", "default")
        if isinstance(p, str) and p.strip():
            client.yamnet_profile = p.strip()
        else:
            client.yamnet_profile = "default"
        logger.info("YAMNet profile: %s", client.yamnet_profile)
    elif cmd == "set_sfx_categories":
        raw = data.get("value")
        if isinstance(raw, list):
            enabled = {
                normalize_category(str(item))
                for item in raw
                if str(item).strip()
            }
            client.sfx_enabled_categories = {
                cat for cat in enabled if cat in YAMNET_CATEGORY_IDS
            }
        elif raw is None:
            client.sfx_enabled_categories = set(YAMNET_CATEGORY_IDS)
        logger.info(
            "SFX categories enabled: %s",
            sorted(client.sfx_enabled_categories),
        )
    elif cmd == "set_caption_telemetry":
        client.caption_env_telemetry = bool(data.get("value", False))
    elif cmd == "set_mode":
        mode = str(data.get("value", "balanced"))
        if mode in MODE_NAMES:
            client.settings.apply_mode(mode)
            logger.info("Session mode: %s", mode)
            await client.connection.send_json(
                {"type": "settings", "mode": mode, "settings": client.settings.as_dict()}
            )
        else:
            logger.warning("Unknown mode %s (valid: %s)", mode, MODE_NAMES)
    elif cmd == "set_settings":
        updates = data.get("value")
        if isinstance(updates, dict):
            applied = client.settings.patch(updates)
            logger.info("Patched settings: %s", applied)
            if any(
                k in applied
                for k in (
                    "partial_min_interval_sec",
                    "partial_every_n_chunks",
                    "partial_beam",
                )
            ):
                client.last_partial_chunk_count = 0
                client.last_partial_monotonic = 0.0
            await client.connection.send_json(
                {"type": "settings", "patched": applied, "settings": client.settings.as_dict()}
            )
    elif cmd == "get_settings":
        await client.connection.send_json(
            {"type": "settings", "settings": client.settings.as_dict()}
        )

async def _dispatch_client_commands(raw: str | bytes, client: WebSocketData) -> bool:
    commands = _parse_json_commands(raw)
    if not commands:
        return False
    preview = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    if len(commands) == 1:
        logger.info("WS command (client %s): %s", client.uuid, commands[0])
    else:
        logger.info(
            "WS commands (client %s, count=%d): %s",
            client.uuid,
            len(commands),
            preview[:500],
        )
    for cmd in commands:
        await handle_client_command(cmd, client)
    return True


def _gpu_cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

async def _session_maintenance(websocket: WebSocketData) -> None:
    logger.info(
        "Session maintenance at chunk %s (client %s)",
        websocket.chunk_counter,
        websocket.uuid,
    )
    diarization.reset_streaming_state()
    websocket.yamnet_adaptive = YamnetAdaptiveState()
    websocket.sfx_tracker = SfxTracker()
    websocket.pending_yamnet = None
    websocket.acoustics.recenter_for_long_session()
    if len(websocket.voiced_buffer) > websocket.rollover_skip_chunks + 4:
        await _flush_utterance(websocket, rolling=False, await_completion=True)
    websocket.voiced_buffer.clear()
    websocket.rollover_skip_chunks = 0
    websocket.is_speaking = False
    websocket.silence_counter = 0
    websocket.lyrics_vad_hangover = 0
    _clear_partial_state(websocket)
    websocket.translate_source_lang = None
    websocket.prefer_final = False
    websocket.last_partial_chunk_count = 0
    websocket.last_partial_monotonic = 0.0
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(sound_executor, _gpu_cleanup)

async def _send_telemetry(websocket: WebSocketData, speech_prob: float) -> None:
    if not _telemetry_enabled(websocket):
        return
    ac = websocket.acoustics
    await _broadcast_json(
        {
            "type": "telemetry",
            "env": ac.env_bucket,
            "noise_score": round(float(ac.noise_score), 3),
            "content_mode": websocket.settings.content_mode,
            "speech_prob": round(float(speech_prob), 3),
            "input_rms": round(float(ac.last_input_rms), 4),
            "capturing": ac.last_gate_open,
            "vad_threshold": round(float(ac.effective_vad_threshold), 3),
        }
    )

def _speech_gate(
    websocket: WebSocketData,
    audio_chunk: np.ndarray,
    speech_prob: float,
) -> bool:
    settings = websocket.settings
    ac = websocket.acoustics
    rms = chunk_rms(audio_chunk)
    is_speech = speech_prob > ac.effective_vad_threshold

    if settings.content_mode == "lyrics":
        energy_floor = max(settings.transcribe_min_rms * 0.85, ac.noise_rms_ema * 0.9)
        has_energy = rms >= energy_floor
        if websocket.is_speaking or websocket.lyrics_vad_hangover > 0:
            if has_energy or speech_prob > 0.08:
                is_speech = True
        elif has_energy and speech_prob > 0.14:
            is_speech = True
        elif speech_prob > ac.effective_vad_threshold:
            is_speech = True
        if is_speech:
            websocket.lyrics_vad_hangover = max(websocket.lyrics_vad_hangover, 10)
        elif websocket.lyrics_vad_hangover > 0:
            is_speech = True
            websocket.lyrics_vad_hangover -= 1
    elif is_speech:
        websocket.lyrics_vad_hangover = 0

    ac.last_gate_open = is_speech
    return is_speech

async def _send_caption_message(
    websocket: WebSocketData,
    msg_type: str,
    text: str,
    speaker: str,
    *,
    language: str | None = None,
    revise: bool = False,
) -> None:
    display_speaker = _format_caption_speaker(
        speaker,
        task=websocket.task,
        language=language,
    )
    payload: dict = {"type": msg_type, "text": text, "speaker": display_speaker}
    lang = (language or "").strip().lower()[:2]
    if lang and websocket.task == "translate" and lang != "en":
        payload["language"] = lang
    if revise and msg_type == "final":
        payload["revise"] = True
    if _telemetry_enabled(websocket):
        payload["env"] = websocket.acoustics.env_bucket
        payload["noise_score"] = round(float(websocket.acoustics.noise_score), 3)
        payload["content_mode"] = websocket.settings.content_mode
    await _broadcast_json(payload)

async def create_connection(websocket: WebSocket) -> None:
    generated_uuid: str = str(uuid.uuid4())
    logger.info("Established connection with Client %s", generated_uuid)

    client_connection = WebSocketData(websocket, generated_uuid)
    _connections.add(client_connection)
    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes") is not None:
                raw = message["bytes"]
                if raw.lstrip()[:1] == b"{" and await _dispatch_client_commands(
                    raw, client_connection
                ):
                    continue
                await process_websocket_bytes(raw, client_connection)

            elif message.get("text") is not None:
                text = message["text"]
                if text.lstrip().startswith("{"):
                    if await _dispatch_client_commands(text, client_connection):
                        continue
                    logger.error("Error parsing command frame (raw=%r)", text[:500])
                else:
                    logger.warning("Ignored non-command text frame: %r", text[:200])
    except WebSocketDisconnect:
        logger.info("Disconnected with Client %s", generated_uuid)
    except Exception as e:
        logger.error("Error: %s: %s", type(e).__name__, e)
    finally:
        _connections.discard(client_connection)

async def process_audio_task(
    websocket: WebSocketData,
    audio_data: np.ndarray,
    speaker_at_capture: str,
    is_final: bool = True,
    *,
    fresh_context: bool = False,
    revise: bool = False,
    provisional_text: str = "",
    window_slid: bool = False,
) -> None:
    if audio_data is None or len(audio_data) == 0:
        return
    if not is_final and websocket.prefer_final:
        return

    loop = asyncio.get_running_loop()
    ac = websocket.acoustics
    settings = websocket.settings
    use_denoise = (
        is_final
        and not settings.skip_denoise
        and float(ac.noise_score) >= DENOISE_MIN_NOISE_SCORE
    )
    if use_denoise:
        denoise_fn = functools.partial(_prepare_audio, audio_data, ac, settings)
        clean_audio = await loop.run_in_executor(sound_executor, denoise_fn)
    else:
        gain = ac.post_gain()
        clean_audio = _sanitize_audio(
            np.clip(audio_data * gain, -1.0, 1.0).astype(np.float32)
        )

    if not is_final and websocket.prefer_final:
        return

    wait_start = time.monotonic()
    await gpu_lock.acquire()
    waited = time.monotonic() - wait_start
    if waited >= GPU_LOCK_SLOW_WAIT_SEC:
        logger.info(
            "GPU lock wait %.2fs (final=%s, client=%s)",
            waited,
            is_final,
            websocket.uuid,
        )
    websocket.is_transcribing = True
    try:
        if not is_final and websocket.prefer_final:
            return
        result = await loop.run_in_executor(
            whisper_executor,
            transcription.get_speech,
            clean_audio,
            is_final,
            websocket.task,
            ac.noise_score,
            settings,
            fresh_context,
            websocket.translate_source_lang,
        )
        if not is_final and websocket.prefer_final:
            return
        text = result["text"].strip()
        lang = result.get("language") or None
        if (
            websocket.task == "translate"
            and lang
            and lang != "en"
        ):
            websocket.translate_source_lang = lang
        if is_final and not text and websocket.last_partial_text.strip():
            text = websocket.last_partial_text.strip()
            speaker_at_capture = websocket.last_partial_speaker or speaker_at_capture
            lang = websocket.last_partial_language or lang
            logger.debug("Sending last partial as final (%d chars)", len(text))
        if not is_final:
            if window_slid:
                text = _stitch_sliding_partial(websocket.last_partial_text, text)
            else:
                text = _merge_stable_partial(websocket.last_partial_text, text)
            if not text or text == websocket.last_partial_text.strip():
                if text:
                    websocket.last_partial_speaker = speaker_at_capture
                    websocket.last_partial_language = lang
                return
            text = profanity_filter.mask_caption_text(
                text, _profanity_filter_enabled
            )
            if not text:
                return
            websocket.last_partial_text = text
            websocket.last_partial_speaker = speaker_at_capture
            websocket.last_partial_language = lang
            await _send_caption_message(
                websocket, "partial", text, speaker_at_capture, language=lang
            )
            return

        provisional = (provisional_text or "").strip()
        if not text:
            if revise and provisional:
                _clear_partial_state(websocket)
                return
            _clear_partial_state(websocket)
            return

        text = profanity_filter.mask_caption_text(
            text, _profanity_filter_enabled
        )
        if not text:
            if revise and provisional:
                _clear_partial_state(websocket)
                return
            _clear_partial_state(websocket)
            return

        if revise and provisional and text == provisional:
            _clear_partial_state(websocket)
            return

        await _send_caption_message(
            websocket,
            "final",
            text,
            speaker_at_capture,
            language=lang,
            revise=bool(revise and provisional),
        )
        _clear_partial_state(websocket)
    except Exception as e:
        logger.exception("Transcription Error: %s", e)
    finally:
        websocket.is_transcribing = False
        if is_final:
            websocket.prefer_final = False
        gpu_lock.release()


def _max_utterance_chunks(settings: SessionSettings) -> int:
    return max(2, int(settings.max_utterance_sec * CHUNKS_PER_SEC))

def _slice_voiced_audio(
    websocket: WebSocketData,
    *,
    end: int | None = None,
    max_chunks: int | None = None,
) -> np.ndarray | None:
    buf = websocket.voiced_buffer
    start = websocket.rollover_skip_chunks
    stop = len(buf) if end is None else end
    if max_chunks is not None and stop - start > max_chunks:
        start = stop - max_chunks
    if stop <= start + 1:
        return None
    return np.concatenate(buf[start:stop])


def _partial_cadence(settings: SessionSettings, task: str = "transcribe") -> tuple[int, float]:
    _ = task
    return (
        max(1, int(settings.partial_every_n_chunks)),
        max(0.0, float(settings.partial_min_interval_sec)),
    )

def _partial_max_chunks() -> int:
    return max(8, int(PARTIAL_AUDIO_MAX_SEC * CHUNKS_PER_SEC))

async def _flush_pending_partial(websocket: WebSocketData) -> None:
    try:
        while (
            websocket.pending_partial is not None and not websocket.prefer_final
        ):
            audio_data, speaker, window_slid = websocket.pending_partial
            websocket.pending_partial = None
            await process_audio_task(
                websocket,
                audio_data,
                speaker,
                is_final=False,
                window_slid=window_slid,
            )
    finally:
        websocket.partial_transcription_inflight = False
        if websocket.pending_partial is not None and not websocket.prefer_final:
            websocket.partial_transcription_inflight = True
            asyncio.create_task(_flush_pending_partial(websocket))

async def _schedule_partial_transcription(
    websocket: WebSocketData,
    audio_data: np.ndarray,
    speaker: str,
    *,
    is_final: bool,
    fresh_context: bool = False,
    window_slid: bool = False,
) -> None:
    if not is_final:
        if websocket.prefer_final:
            return
        websocket.pending_partial = (audio_data.copy(), speaker, window_slid)
        if websocket.partial_transcription_inflight:
            return
        websocket.partial_transcription_inflight = True
        asyncio.create_task(_flush_pending_partial(websocket))
        return

    websocket.pending_partial = None
    websocket.prefer_final = True
    asyncio.create_task(
        process_audio_task(
            websocket,
            audio_data,
            speaker,
            is_final=True,
            fresh_context=fresh_context,
        )
    )

async def _flush_utterance(
    websocket: WebSocketData,
    *,
    rolling: bool = False,
    overlap_chunks: int | None = None,
    await_completion: bool = False,
) -> None:
    overlap = overlap_chunks if overlap_chunks is not None else UTTERANCE_OVERLAP_CHUNKS
    if rolling:
        commit_end = len(websocket.voiced_buffer) - overlap
        audio = _slice_voiced_audio(websocket, end=commit_end)
    else:
        audio = _slice_voiced_audio(websocket)
    if audio is None:
        return

    speaker = _current_speaker(websocket.settings)
    websocket.pending_partial = None
    websocket.prefer_final = True

    provisional = ""
    sent_provisional = False
    if not rolling:
        provisional = websocket.last_partial_text.strip()
        provisional_speaker = websocket.last_partial_speaker or speaker
        provisional_lang = websocket.last_partial_language
        if provisional:
            masked = profanity_filter.mask_caption_text(
                provisional, _profanity_filter_enabled
            )
            if masked:
                await _send_caption_message(
                    websocket,
                    "final",
                    masked,
                    provisional_speaker,
                    language=provisional_lang,
                )
                sent_provisional = True
                provisional = masked

    task = process_audio_task(
        websocket,
        audio,
        speaker,
        is_final=True,
        fresh_context=rolling,
        revise=sent_provisional,
        provisional_text=provisional if sent_provisional else "",
    )
    if await_completion:
        await task
    else:
        asyncio.create_task(task)
    if rolling:
        keep = max(1, overlap)
        websocket.voiced_buffer = list(websocket.voiced_buffer[-keep:])
        websocket.rollover_skip_chunks = len(websocket.voiced_buffer)
        websocket.last_partial_chunk_count = 0
    else:
        websocket.voiced_buffer = []
        websocket.rollover_skip_chunks = 0
        websocket.last_partial_chunk_count = 0
        websocket.is_speaking = False
        websocket.pre_roll.clear()

async def process_speaking(websocket: WebSocketData, audio_chunk: np.ndarray) -> None:
    now = time.monotonic()
    websocket.last_speech_monotonic = now
    websocket.speaker_timeline_reset = False

    if not websocket.is_speaking:
        websocket.is_speaking = True
        websocket.utterance_start_time = now
        websocket.rollover_skip_chunks = 0
        websocket.prefer_final = False
        _clear_partial_state(websocket)
        websocket.last_partial_chunk_count = 0
        websocket.voiced_buffer.extend(list(websocket.pre_roll))
    websocket.voiced_buffer.append(audio_chunk)
    websocket.silence_counter = 0

    n, partial_interval = _partial_cadence(websocket.settings, websocket.task)
    new_audio_chunks = len(websocket.voiced_buffer) - websocket.rollover_skip_chunks
    chunks_since_partial = new_audio_chunks - websocket.last_partial_chunk_count
    if new_audio_chunks >= n and chunks_since_partial >= n:
        if (
            partial_interval <= 0.0
            or (now - websocket.last_partial_monotonic) >= partial_interval
        ):
            max_chunks = _partial_max_chunks()
            audio = _slice_voiced_audio(websocket, max_chunks=max_chunks)
            if audio is not None:
                websocket.last_partial_monotonic = now
                websocket.last_partial_chunk_count = new_audio_chunks
                await _schedule_partial_transcription(
                    websocket,
                    audio,
                    _current_speaker(websocket.settings),
                    is_final=False,
                    window_slid=new_audio_chunks > max_chunks,
                )

    if len(websocket.voiced_buffer) >= _max_utterance_chunks(websocket.settings):
        overlap = UTTERANCE_OVERLAP_CHUNKS
        if websocket.settings.content_mode == "lyrics":
            overlap = max(overlap, int(1.2 * CHUNKS_PER_SEC))
        await _flush_utterance(websocket, rolling=True, overlap_chunks=overlap)

async def process_silence(websocket: WebSocketData, audio_chunk: np.ndarray) -> None:
    now = time.monotonic()

    if websocket.is_speaking:
        websocket.voiced_buffer.append(audio_chunk)
        websocket.silence_counter += 1
        limit = effective_silence_limit(
            websocket.acoustics,
            websocket.settings.phrase_timeout_sec,
            CHUNKS_PER_SEC,
            content_mode=websocket.settings.content_mode,
        )
        if websocket.silence_counter > limit:
            websocket.is_speaking = False
            await _flush_utterance(websocket, rolling=False)
    else:
        websocket.pre_roll.append(audio_chunk)
        if websocket.last_speech_monotonic > 0 and not websocket.speaker_timeline_reset:
            silent_for = now - websocket.last_speech_monotonic
            if silent_for >= websocket.settings.speaker_reset_sec:
                diarization.reset_speaker_timeline()
                websocket.speaker_timeline_reset = True
                if websocket.translate_source_lang:
                    logger.info(
                        "Clearing translate source language %s after %.1fs silence",
                        websocket.translate_source_lang,
                        silent_for,
                    )
                websocket.translate_source_lang = None
                logger.debug("Speaker timeline reset after %.1fs silence", silent_for)

async def process_websocket_bytes(raw_bytes: bytes, websocket: WebSocketData) -> None:
    loop = asyncio.get_running_loop()

    audio_chunk = _decode_audio_frame(raw_bytes)
    if audio_chunk is None:
        return

    _submit_diart_chunk(loop, audio_chunk)

    websocket.yamnet_buffer.extend(audio_chunk)
    websocket.chunk_counter += 1

    hop = max(1, YAMNET_HOP_CHUNKS)
    if (
        websocket.chunk_counter % hop == 0
        and len(websocket.yamnet_buffer) == YAMNET_WINDOW_SAMPLES
    ):
        if websocket.yamnet_inflight:
            websocket.pending_yamnet = np.array(websocket.yamnet_buffer, copy=True)
        else:

            async def ps(buf: np.ndarray, ws: WebSocketData) -> None:
                ws.yamnet_inflight = True
                try:
                    current: np.ndarray | None = buf
                    while current is not None:
                        exclude = set(YAMNET_CATEGORY_IDS) - set(
                            ws.sfx_enabled_categories
                        )
                        fn = functools.partial(
                            diarization.get_sounds,
                            current,
                            profile=ws.yamnet_profile,
                            adaptive=ws.yamnet_adaptive,
                            exclude_categories=exclude or None,
                            settings=ws.settings,
                            noise_score=float(ws.acoustics.noise_score),
                        )
                        detected = await loop.run_in_executor(yamnet_executor, fn)
                        to_send = ws.sfx_tracker.update(
                            detected,
                            max_labels=int(ws.settings.yamnet_max_labels),
                        )
                        if to_send:
                            combined_text = ", ".join(d["label"] for d in to_send)
                            await _broadcast_json(
                                {
                                    "type": "sound",
                                    "text": combined_text,
                                    "labels": to_send,
                                    "speaker": "",
                                }
                            )
                        current, ws.pending_yamnet = ws.pending_yamnet, None
                except Exception as e:
                    logger.error("YAMNet Error: %s", e)
                finally:
                    ws.yamnet_inflight = False

            asyncio.create_task(ps(np.array(websocket.yamnet_buffer, copy=True), websocket))

    speech_prob: float = await loop.run_in_executor(
        vad_executor, transcription.check_vad, audio_chunk
    )
    websocket.acoustics.update(audio_chunk, speech_prob, websocket.settings)

    is_speech = _speech_gate(websocket, audio_chunk, speech_prob)

    if (
        SESSION_MAINTENANCE_CHUNKS > 0
        and websocket.chunk_counter % SESSION_MAINTENANCE_CHUNKS == 0
    ):
        asyncio.create_task(_session_maintenance(websocket))

    if websocket.chunk_counter % 4 == 0 and _telemetry_enabled(websocket):
        await _send_telemetry(websocket, speech_prob)

    if is_speech:
        await process_speaking(websocket, audio_chunk)
    else:
        await process_silence(websocket, audio_chunk)