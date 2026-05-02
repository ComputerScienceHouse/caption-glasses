"""
Contains all of the functions f
"""

import tensorflow as tf
import torch
import numpy as np

from numpy import ndarray

from faster_whisper import WhisperModel
from logging import Logger, getLogger

from config import SAMPLE_RATE

logger: Logger = getLogger(__name__)

device: str = "cuda" if torch.cuda.is_available() else "cpu"
compute_type: str = "float16" if device == "cuda" else "int8"

logger.info(f"Loading Whisper using {device.upper()} for transcription.")
speech_model: WhisperModel = WhisperModel(
    "Systran/faster-distil-whisper-large-v3",
    device=device,
    compute_type=compute_type,
)

logger.info(f"Loading VAD via torch hub!")
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
vad_model: torch.nn.Module = vad_model.to("cpu")


logger.info("Loading YAMNet...")

def get_speech(audio: ndarray, is_final: bool = True, task: str = "transcribe") -> dict[str, str]:
    """
    Processed audio array and returns the resulting audio

    Arguments:
        audio (ndarray): The audio byte array to be processed
        is_final (bool): Whether or not the audio is finalized, defaults to true
        task (str): The task to be performed, either "transcribe" or "translate

    Returns:
        dict[str, str]: The dictionary "text" with the returned speech
    """
    if len(audio) < 1600: 
        return {"text": ""}
    
    if task == "translate":
        beam = 1
    else:
        beam = 3  # more accurate for final, faster for partial

    use_previous = True if task == "transcribe" else False
    try:
        segments, _ = speech_model.transcribe(beam = 5 if is_final else 1
            audio,  # The audio to be processed
            beam_size=beam,  # Search width for the audio. Higher = More accurate but slower
            language="en" if task == "transcribe" else None,
            task=task,
            condition_on_previous_text=use_previous,  # Might feed previous inputs for better result
            temperature=0.0,
            vad_filter=True,  # Skip silent regions
            vad_parameters=dict(
                min_silence_duration_ms=500, speech_pad_ms=200, threshold=0.3
            ),
            no_speech_threshold=0.95,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=1.5,
            repetition_penalty=1.5,
        )
        text = " ".join([segment.text for segment in segments])
        return {"text": text}
    except (ValueError, RuntimeError) as e:
        return {"text": ""}

def check_vad(audio: ndarray) -> float:
    """
    Processes audio to validate and return the VAD level.

    Arguments:
        audio (ndarray): The audio byte array to be processed

    Returns:
        float: The float level of the VAD
    """
    with torch.no_grad():
        sub_chunks: tf.Tensor = torch.from_numpy(audio).to("cpu").split(512)
        max_prob: float = 0.0
        for sub in sub_chunks:
            if sub.shape[0] < 512:
                sub = torch.nn.functional.pad(sub, (0, 512 - sub.shape[0]))
            prob = vad_model(sub, SAMPLE_RATE).item()
            max_prob = max(max_prob, prob)
        return max_prob
