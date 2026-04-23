# import asyncio
# import numpy as np
# import collections
# import tensorflow as tf

# tf.config.set_visible_devices([], "GPU")  # Forces TF to use CPU
# import time


# from fastapi import WebSWebSocketocket

# from concurrent.futures import ThreadPoolExecutor
# import pyannote.audio.core.model as _pyannote_model

# import logging

# from config import (
#     SAMPLE_RATE,
#     CHUNK_SIZE,
#     VAD_THRESHOLD,
#     MAX_DURATION,
#     PHRASE_TIMEOUT,
# )

# logging.basicConfig(level=logging.INFO)
# logger: logging.Logger = logging.getLogger(__name__)

# logger.info("Initiating Threads")

# gpu_lock = asyncio.Lock()

# whisper_executor = ThreadPoolExecutor(max_workers=1)
# sound_executor = ThreadPoolExecutor(max_workers=1)
# diart_executor = ThreadPoolExecutor(max_workers=1)


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("Client connected.")

#     voiced_buffer = []  # audio chunks for current speech
#     is_speaking = False
#     silence_counter = 0
#     is_transcribing = False
#     utterance_start_time = time.monotonic()

#     chunks_per_sec = SAMPLE_RATE / CHUNK_SIZE
#     silence_limit = int(PHRASE_TIMEOUT * chunks_per_sec)

#     pre_roll = collections.deque(
#         maxlen=10
#     )  # keeps audio just before speech starts to avoid cutting off beginning of phrases
#     yamnet_buffer = collections.deque(maxlen=16384)
#     chunk_counter = 0

#     loop = asyncio.get_running_loop()

#     async def process_audio_task(audio_data, speaker_at_capture, is_final=True):
#         nonlocal is_transcribing

#         if audio_data is None or len(audio_data) == 0:
#             return

#         if gpu_lock.locked() and not is_final:
#             return

#         async with gpu_lock:
#             is_transcribing = True
#             try:
#                 result = await loop.run_in_executor(
#                     whisper_executor, get_speech, audio_data, is_final
#                 )
#                 text = result["text"].strip()
#                 if text:
#                     msg_type = "final" if is_final else "partial"
#                     await websocket.send_json(
#                         {"type": msg_type, "text": text, "speaker": speaker_at_capture}
#                     )
#             except Exception as e:
#                 print(f"Transcription Error: {e}")
#             finally:
#                 is_transcribing = False

#     try:
#         while True:
#             raw_bytes = await websocket.receive_bytes()
#             audio_chunk = np.frombuffer(raw_bytes, np.float32).copy()

#             loop.run_in_executor(
#                 diart_executor, audio_source.push_audio, audio_chunk.copy()
#             )

#             yamnet_buffer.extend(audio_chunk)
#             chunk_counter += 1

#             if chunk_counter % 8 == 0 and len(yamnet_buffer) == 16384:

#                 async def ps(buf):
#                     try:
#                         s, sc = await loop.run_in_executor(
#                             sound_executor, get_sounds, buf
#                         )
#                         if sc > 0.45 and s not in ["Silence", "Speech"]:
#                             await websocket.send_json({"type": "sound", "sound": s})
#                     except:
#                         pass

#                 asyncio.create_task(ps(np.array(yamnet_buffer)))

#             def check_vad():
#                 with torch.no_grad():
#                     sub_chunks = torch.from_numpy(audio_chunk).to(device).split(512)
#                     max_prob = 0.0
#                     for sub in sub_chunks:
#                         prob = vad_model(sub, SAMPLE_RATE).item()
#                         max_prob = max(max_prob, prob)
#                     return max_prob

#             speech_prob = await loop.run_in_executor(None, check_vad)

#             if speech_prob > VAD_THRESHOLD:
#                 if not is_speaking:
#                     is_speaking = True
#                     utterance_start_time = time.monotonic()
#                     voiced_buffer.extend(list(pre_roll))
#                 voiced_buffer.append(audio_chunk)
#                 silence_counter = 0

#                 # Sends partial transcription every 4 chunks while speaking
#                 if len(voiced_buffer) % 4 == 0:
#                     speaker_snapshot = get_speaker_at(utterance_start_time)
#                     asyncio.create_task(
#                         process_audio_task(
#                             np.concatenate(voiced_buffer),
#                             speaker_snapshot,
#                             is_final=False,
#                         )
#                     )

#                 # Force final transcriptions if buffer gets too long
#                 if (len(voiced_buffer) * CHUNK_SIZE) / SAMPLE_RATE >= MAX_DURATION:
#                     speaker_snapshot = get_speaker_at(utterance_start_time)
#                     asyncio.create_task(
#                         process_audio_task(
#                             np.concatenate(voiced_buffer),
#                             speaker_snapshot,
#                             is_final=True,
#                         )
#                     )
#                     voiced_buffer = []
#                     is_speaking = False
#                     pre_roll.clear()
#             else:
#                 # Silence handling
#                 if is_speaking:
#                     voiced_buffer.append(audio_chunk)
#                     silence_counter += 1
#                     if silence_counter > silence_limit:
#                         is_speaking = False
#                         if len(voiced_buffer) > 4:
#                             speaker_snapshot = get_speaker_at(utterance_start_time)
#                             asyncio.create_task(
#                                 process_audio_task(
#                                     np.concatenate(voiced_buffer),
#                                     speaker_snapshot,
#                                     is_final=True,
#                                 )
#                             )
#                         voiced_buffer = []
#                         pre_roll.clear()
#                 else:
#                     pre_roll.append(audio_chunk)

#     except Exception as e:
#         print(f"WS Disconnected: {e}")
