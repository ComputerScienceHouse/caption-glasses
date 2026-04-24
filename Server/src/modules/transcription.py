"""
Contains all of the functions f
"""

import tensorflow as tf
import torch

from numpy import ndarray

from faster_whisper import WhisperModel
from logging import Logger, getLogger

from config import SAMPLE_RATE

logger: Logger = getLogger(__name__)

device: str = "cuda" if torch.cuda.is_available() else "cpu"
compute_type: str = "float16" if device == "cuda" else "int8"

logger.info(f"Loading Whisper using {device.upper()} for transcription.")
speech_model: WhisperModel = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device=device,
    compute_type=compute_type,
)

logger.info(f"Loading VAD via torch hub!")
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
vad_model: torch.nn.Module = vad_model.to(device)


logger.info("Loading YAMNet...")

def get_speech(audio: ndarray, is_final: bool = True) -> dict[str, str]:
    """
    Processed audio array and returns the resulting audio

    Arguments:
        audio (ndarray): The audio byte array to be processed
        is_final (bool): Whether or not the audio is finalized, defaults to true

    Returns:
        dict[str, str]: The dictionary "text" with the returned speech
    """

    beam: int = 5 if is_final else 1  # more accurate for final, faster for partial
    segments, _ = speech_model.transcribe(
        audio,  # The audio to be processed
        beam_size=beam,  # Search width for the audio. Higher = More accurate but slower
        language="en",  # English
        condition_on_previous_text=True,  # Feeds previous inputs for better result
        temperature=[0.0, 0.2, 0.4],  # Fallback chain for tempature readings
        vad_filter=True,  # Skip silent regions
        vad_parameters=dict(
            min_silence_duration_ms=500, speech_pad_ms=200, threshold=0.2
        ),
        no_speech_threshold=0.65,
        log_prob_threshold=-2.0,
        compression_ratio_threshold=2.0,
        repetition_penalty=1.2,
    )
    text = " ".join([segment.text for segment in segments])
    return {"text": text}

def check_vad(audio: ndarray) -> float:
    """
    Processes audio to validate and return the VAD level.

    Arguments:
        audio (ndarray): The audio byte array to be processed

    Returns:
        float: The float level of the VAD
    """
    with torch.no_grad():
        sub_chunks: tf.Tensor = torch.from_numpy(audio).to(device).split(512)
        max_prob: float = 0.0
        for sub in sub_chunks:
            prob = vad_model(sub, SAMPLE_RATE).item()
            max_prob = max(max_prob, prob)
        return max_prob
