import asyncio
import collections
import time
import numpy as np
import uuid

from fastapi import WebSocket
from concurrent.futures import ThreadPoolExecutor
from logging import Logger, getLogger

from modules import diarization, transcription
from config import SAMPLE_RATE, CHUNK_SIZE, PHRASE_TIMEOUT, VAD_THRESHOLD, MAX_DURATION

from starlette.websockets import WebSocketDisconnect

CHUNKS_PER_SEC: float = SAMPLE_RATE / CHUNK_SIZE
SILENCE_LIMIT: int = int(PHRASE_TIMEOUT * CHUNKS_PER_SEC)

logger: Logger = getLogger(__name__)
logger.info("Initiating Threads")

gpu_lock: asyncio.Lock = asyncio.Lock()
whisper_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
sound_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
diart_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

class WebSocketData:
    __slots__ = ("connection","uuid","voiced_buffer","is_speaking","is_transcribing","silence_counter","utterance_start_time","pre_roll","yamnet_buffer","chunk_counter")

    def __init__(self, websocket: WebSocket, uuid: str):
        self.connection: WebSocket = websocket
        self.uuid: str = uuid
        self.voiced_buffer: list = []
        self.is_speaking: bool = False
        self.is_transcribing = False
        self.silence_counter: float = 0
        self.utterance_start_time: float = time.monotonic()
        self.pre_roll: collections.deque = collections.deque(
            maxlen=10
        )
        self.yamnet_buffer: collections.deque = collections.deque(maxlen=16384)
        self.chunk_counter: int = 0

async def create_connection(websocket: WebSocket):
    generated_uuid: str = uuid.uuid4()

    logger.info(
        f"Established connection with Client {generated_uuid} at %s", time.monotonic()
    )

    client_connection: WebSocketData = WebSocketData(websocket, generated_uuid)

    try:
        while True:
            raw_bytes: bytes = await websocket.receive_bytes()
            await process_websocket_bytes(
                raw_bytes, client_connection
            )
    except WebSocketDisconnect as e:
        logger.info(
            f"Disconnected with Client {generated_uuid} at %s", time.monotonic()
        )
    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}")
 


async def process_audio_task(
    websocket: WebSocketData,
    audio_data: np.ndarray,
    speaker_at_capture: str,
    is_final: bool = True,
) -> None:
    """
    Processes the audio chunk for text, making sure to lock required threads as necessary.
    Sends the transcibed text back through the inputted websocket

    Arguments:
        audio_chunk (np.ndarray): The audio chunk to be processed as a portion of silence
    """

    global is_transcribing

    if audio_data is None or len(audio_data) == 0:
        return

    if gpu_lock.locked() and not is_final:
        return

    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

    async with gpu_lock:
        websocket.is_transcribing = True
        try:
            result = await loop.run_in_executor(
                whisper_executor, transcription.get_speech, audio_data, is_final
            )
            text = result["text"].strip()
            if text:
                msg_type = "final" if is_final else "partial"
                await websocket.connection.send_json(
                    {"type": msg_type, "text": text, "speaker": speaker_at_capture}
                )
        except Exception as e:
            print(f"Transcription Error: {e}")
        finally:
            websocket.is_transcribing = False


async def process_speaking(websocket: WebSocketData, audio_chunk: np.ndarray) -> None:
    """
    Processes the audio chunk as a portion of dialogue

    Arguments:
        websocket (WebSocketData): The websocket connection data
        audio_chunk (np.ndarray): The audio chunk to be processed as a portion of silence
    """
    if not websocket.is_speaking:
        websocket.is_speaking = True
        websocket.utterance_start_time = time.monotonic()
        websocket.voiced_buffer.extend(list(websocket.pre_roll))
    websocket.voiced_buffer.append(audio_chunk)
    websocket.silence_counter = 0

    # Sends partial transcription every 4 chunks while speaking
    if len(websocket.voiced_buffer) % 4 == 0:
        speaker_snapshot: str = diarization.get_speaker_at(websocket.utterance_start_time)
        asyncio.create_task(
            process_audio_task(
                websocket,
                np.concatenate(websocket.voiced_buffer),
                speaker_snapshot,
                is_final=False,
            )
        )

    # Force final transcriptions if buffer gets too long
    if (len(websocket.voiced_buffer) * CHUNK_SIZE) / SAMPLE_RATE >= MAX_DURATION:
        speaker_snapshot: str = diarization.get_speaker_at(websocket.utterance_start_time)
        asyncio.create_task(
            process_audio_task(
                websocket,
                np.concatenate(websocket.voiced_buffer),
                speaker_snapshot,
                is_final=True,
            )
        )
        websocket.voiced_buffer = []
        websocket.is_speaking = False
        websocket.pre_roll.clear()


async def process_silence(websocket: WebSocketData, audio_chunk: np.ndarray) -> None:
    """
    Processes the audio chunk as a silence chunk

    Arguments:
        audio_chunk (np.ndarray): The audio chunk to be processed as a portion of silence
    """
    if websocket.is_speaking:
        websocket.voiced_buffer.append(audio_chunk)
        websocket.silence_counter += 1
        if websocket.silence_counter > SILENCE_LIMIT:
            websocket.is_speaking = False
            if len(websocket.voiced_buffer) > 4:
                speaker_snapshot: str = diarization.get_speaker_at(websocket.utterance_start_time)
                asyncio.create_task(
                    process_audio_task(
                        websocket,
                        np.concatenate(websocket.voiced_buffer),
                        speaker_snapshot,
                        is_final=True,
                    )
                )
            websocket.voiced_buffer = []
            websocket.pre_roll.clear()
    else:
        websocket.pre_roll.append(audio_chunk)


async def process_websocket_bytes(raw_bytes: bytes, websocket: WebSocketData) -> None:
    """
    Processes the raw bytes recieved from a websocket

    Arguments:
        raw_bytes (bytes): The raw_bytes recieved through the websocket
        websocket (WebSocketData): The websocket connection object
    """
    if not raw_bytes:
        return
    
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

    #If incoming bytes are int16, convert to float32
    #some clients might send int16 and thats how the app is set up right now
    if len(raw_bytes) == 4096:
        audio_chunk = np.frombuffer(raw_bytes, np.int16).astype(np.float32) / 32768.0
    else:
        audio_chunk = np.frombuffer(raw_bytes, np.float32).copy()
    loop.run_in_executor(
        diart_executor, diarization.audio_source.push_audio, audio_chunk.copy()
    )

    websocket.yamnet_buffer.extend(audio_chunk)
    websocket.chunk_counter += 1

    if websocket.chunk_counter % 8 == 0 and len(websocket.yamnet_buffer) == 16384:
        async def ps(buf):
            try:
                s, sc = await loop.run_in_executor(
                    sound_executor, diarization.get_sounds, buf.flatten()
                )
                if sc > 0.45 and s not in ["Silence", "Speech"]:    
                    await websocket.connection.send_json({"type": "sound", "text": s, "speaker": ""})
            except:
                pass

        asyncio.create_task(ps(np.array(websocket.yamnet_buffer)))

    speech_prob: asyncio.AbstractEventLoop = await loop.run_in_executor(None, transcription.check_vad, audio_chunk)

    if speech_prob > VAD_THRESHOLD:
        await process_speaking(websocket, audio_chunk)
    else:
        await process_silence(websocket, audio_chunk)
