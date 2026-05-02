import time
import numpy as np
import collections
from logging import Logger, getLogger
from pyannote.core import Annotation, Segment
import noisereduce as nr

import tensorflow as tf
#this makes tensorflow use CPU only
tf.config.set_visible_devices([], 'GPU')
tf.config.optimizer.set_jit(False)

import tensorflow_hub as hub

from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.sources import AudioSource
from diart.inference import StreamingInference

from config import HF_TOKEN, SAMPLE_RATE


yamnet_model: hub.KerasLayer = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path: bytes = yamnet_model.class_map_path().numpy()
class_names: list[str] = []

with tf.io.gfile.GFile(class_map_path) as f:
    class_names = [
        line.split(",")[2].strip().strip('"') for line in f.read().splitlines()[1:]
    ]


# Audio source to feed websocket audio into Diart
class WebSocketAudioSource(AudioSource):
    def __init__(self, sample_rate):
        super().__init__(uri="websocket_stream", sample_rate=sample_rate)

    def read(self):
        pass

    def close(self):
        self.stream.on_completed()

    def push_audio(self, chunk: np.ndarray):
        self.stream.on_next(chunk.reshape(1, -1))

logger: Logger = getLogger(__name__)
logger.info("Loading Diart (Pyannote)...")

diart_config: SpeakerDiarizationConfig = SpeakerDiarizationConfig(
    duration=3.0, step=1.0, latency="max", sample_rate=SAMPLE_RATE, hf_token=HF_TOKEN, tau_active=0.6
)
diarization: SpeakerDiarization = SpeakerDiarization(diart_config)

audio_source: WebSocketAudioSource = WebSocketAudioSource(SAMPLE_RATE)
pipeline: StreamingInference = StreamingInference(diarization, audio_source)


speaker_timeline: collections.deque[tuple[float, str]] = collections.deque(
    maxlen=200
)  # store recent speaker labels with timestamps


def on_diarization_update(result: tuple[Annotation] | Annotation) -> None:
    """
    Adds result processed from the Audio Pipeline to be added to the speaker queue

    Arguments:
        result (tuple[Annotation] | Annotation): Either a tuple or just an annotation, depending on current buffer
    """

    annotation: Annotation = result[0] if isinstance(result, tuple) else result

    if not hasattr(annotation, "labels") or not annotation.labels():
        return

    try:
        tracks: list[tuple[Segment, str, str]] = list(
            annotation.itertracks(yield_label=True)
        )
        if not tracks:
            return
        # Get most recent speaker segment
        latest_track: tuple[Segment, str, str] = max(tracks, key=lambda x: x[0].end)
        speaker_timeline.append((time.monotonic(), latest_track[2]))
    except Exception:
        return


def get_sounds(audio: np.ndarray) -> list[str]:
    """
    Processes audio for sounds using category-based deduplication.
    """
    clean_audio = nr.reduce_noise(
        y=audio, 
        sr=SAMPLE_RATE, 
        stationary=True, 
        prop_decrease=0.7
    )
    scores, _, _ = yamnet_model(clean_audio)
    class_scores: tf.Tensor = tf.reduce_max(scores, axis=0)

    top_indices = tf.argsort(class_scores, direction='DESCENDING')[:20].numpy()

    category_ranges = {
        "human": range(0, 67),
        "animal": range(67, 132),
        "music": range(132, 277),
        "natural": range(277, 294),
        "vehicle": range(294, 348),
        "domestic": range(348, 412),
        "tools": range(412, 420),
        "explosive": range(420, 456),
        "misc": range(456, 521),
    }

    # Per-class floor: below this, ignore. Slightly raised so faint activations do not stack.
    base_threshold = 0.52
    vehicle_threshold = 0.68
    natural_threshold = 0.64

    # After picking the best label per category, only keep a category if it is nearly as
    # strong as the global winner. That way one real-world sound (often firing several
    # YAMNet classes in different buckets) usually yields one caption, while two distinct
    # loud events in the same window can still both appear.
    secondary_min_ratio = 0.90

    candidates = []
    for idx in top_indices:
        label = class_names[idx]
        score = float(class_scores[idx].numpy())

        th = base_threshold
        if idx in category_ranges["vehicle"]:
            th = vehicle_threshold
        if idx in category_ranges["natural"]:
            th = natural_threshold

        if label in ["Silence", "Speech"] or score < th:
            continue

        assigned_cat = "none"
        for cat_name, cat_range in category_ranges.items():
            if idx in cat_range:
                assigned_cat = cat_name
                break

        candidates.append(
            {"label": label, "score": score, "category": assigned_cat}
        )

    best_per_category: dict[str, dict] = {}
    for cand in candidates:
        cat = cand["category"]
        if cat not in best_per_category or cand["score"] > best_per_category[cat]["score"]:
            best_per_category[cat] = cand

    if not best_per_category:
        return []

    global_top = max(c["score"] for c in best_per_category.values())
    floor = global_top * secondary_min_ratio

    kept = [c for c in best_per_category.values() if c["score"] >= floor]
    kept.sort(key=lambda x: x["score"], reverse=True)
    return [c["label"] for c in kept[:3]]

def get_speaker_at(timestamp: float, max_age: float = 1.0) -> str:
    """
    Finds the most recent speaker at or before the given timestamp, or 00 if none is found

    Arguments:
        timestamp (float): The timestamp of the speaker to look for
        max_age (float): The max age to look back, defaults to 1.5

    Returns:
        str: The label of the speaker, or SPEAKER_00 if none is found
    """

    best: str | None = None

    for ts, spk in reversed(speaker_timeline):
        if timestamp - 1.0 <= ts <= timestamp + max_age:
            return spk
        if ts <= timestamp + max_age:
            best = spk
            break
    return best or "SPEAKER_00"

pipeline.stream.subscribe(on_diarization_update)
