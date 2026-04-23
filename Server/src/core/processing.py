import asyncio
import collections
import time
import numpy as np

from fastapi import WebSocket
from concurrent.futures import ThreadPoolExecutor
from logging import Logger, getLogger

from modules import diarization, transcription
from config import SAMPLE_RATE, CHUNK_SIZE, PHRASE_TIMEOUT, VAD_THRESHOLD, MAX_DURATION

logger: Logger = getLogger(__name__)
logger.info("Initiating Threads")

gpu_lock: asyncio.Lock = asyncio.Lock()
whisper_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
sound_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
diart_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

voiced_buffer: list = []  # audio chunks for current speech
is_speaking: bool = False
is_transcribing: bool = False
silence_counter: float = 0
utterance_start_time: float = time.monotonic()

CHUNKS_PER_SEC: float = SAMPLE_RATE / CHUNK_SIZE
SILENCE_LIMIT: int = int(PHRASE_TIMEOUT * CHUNKS_PER_SEC)

pre_roll: collections.deque = collections.deque(
    maxlen=10
)  # keeps audio just before speech starts to avoid cutting off beginning of phrases
yamnet_buffer: collections.deque = collections.deque(maxlen=16384)
chunk_counter: int = 0

loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()


async def process_audio_task(websocket: WebSocket, audio_data: np.ndarray, speaker_at_capture: str, is_final: bool = True) -> None:
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

    async with gpu_lock:
        is_transcribing = True
        try:
            result = await loop.run_in_executor(
                whisper_executor, transcription.get_speech, audio_data, is_final
            )
            text = result["text"].strip()
            if text:
                msg_type = "final" if is_final else "partial"
                await websocket.send_json(
                    {"type": msg_type, "text": text, "speaker": speaker_at_capture}
                )
        except Exception as e:
            print(f"Transcription Error: {e}")
        finally:
            is_transcribing = False


async def procees_speaking(audio_chunk: np.ndarray) -> None:
    """
    Processes the audio chunk as a portion of dialogue

    Arguments:
        audio_chunk (np.ndarray): The audio chunk to be processed as a portion of silence
    """
    global is_speaking, utterance_start_time, voiced_buffer, silence_counter, pre_roll

    if not is_speaking:
        is_speaking = True
        utterance_start_time = time.monotonic()
        voiced_buffer.extend(list(pre_roll))
    voiced_buffer.append(audio_chunk)
    silence_counter = 0

    # Sends partial transcription every 4 chunks while speaking
    if len(voiced_buffer) % 4 == 0:
        speaker_snapshot: str = diarization.get_speaker_at(utterance_start_time)
        asyncio.create_task(
            process_audio_task(
                np.concatenate(voiced_buffer),
                speaker_snapshot,
                is_final=False,
            )
        )

    # Force final transcriptions if buffer gets too long
    if (len(voiced_buffer) * CHUNK_SIZE) / SAMPLE_RATE >= MAX_DURATION:
        speaker_snapshot: str = diarization.get_speaker_at(utterance_start_time)
        asyncio.create_task(
            process_audio_task(
                np.concatenate(voiced_buffer),
                speaker_snapshot,
                is_final=True,
            )
        )
        voiced_buffer = []
        is_speaking = False
        pre_roll.clear()


async def process_silence(audio_chunk: np.ndarray) -> None:
    """
    Processes the audio chunk as a silence chunk

    Arguments:
        audio_chunk (np.ndarray): The audio chunk to be processed as a portion of silence
    """
    global is_speaking, utterance_start_time, voiced_buffer, silence_counter, pre_roll

    if is_speaking:
        voiced_buffer.append(audio_chunk)
        silence_counter += 1
        if silence_counter > SILENCE_LIMIT:
            is_speaking = False
            if len(voiced_buffer) > 4:
                speaker_snapshot: str = diarization.get_speaker_at(utterance_start_time)
                asyncio.create_task(
                    process_audio_task(
                        np.concatenate(voiced_buffer),
                        speaker_snapshot,
                        is_final=True,
                    )
                )
            voiced_buffer = []
            pre_roll.clear()
    else:
        pre_roll.append(audio_chunk)


async def process_websocket_bytes(raw_bytes: bytes, websocket: WebSocket) -> None:
    """
    Processes the raw bytes recieved from a websocket

    Arguments:
        raw_bytes (bytes): The raw_bytes recieved through the websocket
        websocket (Websocket): The websocket connection that was done
    """

    audio_chunk: np.ndarray = np.frombuffer(raw_bytes, np.float32).copy()

    loop.run_in_executor(
        diart_executor, diarization.audio_source.push_audio, audio_chunk.copy()
    )

    yamnet_buffer.extend(audio_chunk)
    chunk_counter += 1

    if chunk_counter % 8 == 0 and len(yamnet_buffer) == 16384:
        async def ps(buf):
            try:
                s, sc = await loop.run_in_executor(
                    sound_executor, transcription.get_sounds, buf
                )
                if sc > 0.45 and s not in ["Silence", "Speech"]:
                    await websocket.send_json({"type": "sound", "sound": s})
            except:
                pass

        asyncio.create_task(ps(np.array(yamnet_buffer)))

    speech_prob: asyncio.AbstractEventLoop = await loop.run_in_executor(None, transcription.check_vad)

    if speech_prob > VAD_THRESHOLD:
        await procees_speaking(audio_chunk=audio_chunk)
    else:
        await process_silence(audio_chunk=audio_chunk)
