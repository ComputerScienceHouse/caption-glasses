import re
import time
import warnings
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

from config import (
    HF_TOKEN,
    SAMPLE_RATE,
    SFX_DENOISE_MIN_NOISE_SCORE,
    YAMNET_BASE_THRESHOLD,
    YAMNET_DENOISE_STRENGTH,
    YAMNET_MAX_LABELS,
    YAMNET_MUSIC_THRESHOLD,
    YAMNET_NATURAL_THRESHOLD,
    YAMNET_PRESETS_MERGED,
    YAMNET_SECONDARY_MIN_RATIO,
    YAMNET_TOP_K,
    YAMNET_VEHICLE_THRESHOLD,
)
from core.acoustics import YamnetAdaptiveState
from core.session_settings import SessionSettings
from core.yamnet_categories import (
    TRANSIENT_THRESHOLD_DELTA,
    VAGUE_SCORE_MARGIN,
    YAMNET_SPEECH_LABELS,
    YAMNET_TRANSIENT_INDICES,
    YAMNET_VAGUE_LABELS,
    category_for_index,
    normalize_category,
)

yamnet_model: hub.KerasLayer = hub.load("https://tfhub.dev/google/yamnet/1")

class_map_path: bytes = yamnet_model.class_map_path().numpy()
class_names: list[str] = []

with tf.io.gfile.GFile(class_map_path) as f:
    class_names = [
        line.split(",")[2].strip().strip('"') for line in f.read().splitlines()[1:]
    ]

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
warnings.filterwarnings(
    "ignore",
    message="Mismatch between frames .* and weights .* numbers.",
    category=UserWarning,
    module=r"pyannote\.audio\.models\.blocks\.pooling",
)

diart_config: SpeakerDiarizationConfig = SpeakerDiarizationConfig(
    duration=5.0,
    step=0.5,
    latency="min",
    sample_rate=SAMPLE_RATE,
    hf_token=HF_TOKEN,
    tau_active=0.5,
)
diarization: SpeakerDiarization = SpeakerDiarization(diart_config)

audio_source: WebSocketAudioSource = WebSocketAudioSource(SAMPLE_RATE)
pipeline: StreamingInference = StreamingInference(
    diarization,
    audio_source,
    do_profile=False,
    show_progress=False,
)

speaker_timeline: collections.deque[tuple[float, str]] = collections.deque(
    maxlen=200
)

_SPEAKER_RE = re.compile(r"speaker[_\s-]*(\d+)", re.IGNORECASE)

def normalize_speaker_id(speaker: str | None) -> str:
    if not speaker or not str(speaker).strip():
        return "SPEAKER_00"
    base = str(speaker).strip().split(" (", 1)[0].strip()
    match = _SPEAKER_RE.search(base)
    if match:
        return f"SPEAKER_{int(match.group(1)):02d}"
    return base.upper().replace(" ", "_")

def speaker_client_id(speaker: str | None) -> str:
    normalized = normalize_speaker_id(speaker)
    match = _SPEAKER_RE.search(normalized)
    if match:
        return f"SPEAKER_{int(match.group(1)) + 1:02d}"
    return normalized

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
        latest_track: tuple[Segment, str, str] = max(tracks, key=lambda x: x[0].end)
        speaker_timeline.append(
            (time.monotonic(), normalize_speaker_id(latest_track[2]))
        )
    except Exception:
        return

def _yamnet_thresholds_for_profile(profile: str) -> dict[str, float]:
    base = {
        "base_threshold": YAMNET_BASE_THRESHOLD,
        "vehicle_threshold": YAMNET_VEHICLE_THRESHOLD,
        "natural_threshold": YAMNET_NATURAL_THRESHOLD,
        "music_threshold": YAMNET_MUSIC_THRESHOLD,
        "secondary_min_ratio": YAMNET_SECONDARY_MIN_RATIO,
        "top_k": float(YAMNET_TOP_K),
        "max_labels": float(YAMNET_MAX_LABELS),
    }
    preset = YAMNET_PRESETS_MERGED.get(profile) or YAMNET_PRESETS_MERGED.get("default") or {}
    for key, delta in preset.items():
        if key in base:
            base[key] = float(base[key]) + float(delta)
        else:
            base[key] = float(delta)
    base["secondary_min_ratio"] = float(np.clip(base["secondary_min_ratio"], 0.5, 0.99))
    base["base_threshold"] = float(np.clip(base["base_threshold"], 0.2, 0.95))
    base["vehicle_threshold"] = float(np.clip(base["vehicle_threshold"], 0.2, 0.95))
    base["natural_threshold"] = float(np.clip(base["natural_threshold"], 0.2, 0.95))
    base["music_threshold"] = float(np.clip(base["music_threshold"], 0.2, 0.95))
    return base

def _resolve_yamnet_thresholds(
    profile: str,
    settings: SessionSettings | None,
) -> dict[str, float]:
    thr = _yamnet_thresholds_for_profile(profile)
    if settings is None:
        return thr
    thr["base_threshold"] = float(settings.yamnet_base_threshold)
    thr["music_threshold"] = float(settings.yamnet_music_threshold)
    thr["secondary_min_ratio"] = float(settings.yamnet_secondary_min_ratio)
    thr["max_labels"] = float(settings.yamnet_max_labels)
    thr["secondary_min_ratio"] = float(np.clip(thr["secondary_min_ratio"], 0.5, 0.99))
    thr["base_threshold"] = float(np.clip(thr["base_threshold"], 0.2, 0.95))
    thr["music_threshold"] = float(np.clip(thr["music_threshold"], 0.2, 0.95))
    return thr

def get_sounds(
    audio: np.ndarray,
    profile: str = "default",
    adaptive: YamnetAdaptiveState | None = None,
    exclude_categories: set[str] | None = None,
    settings: SessionSettings | None = None,
    noise_score: float | None = None,
) -> list[dict[str, str | float | bool]]:
    """
    Processes audio for sounds using category-based deduplication.
    Returns dicts: label, category, score, transient.
    Filtering is by enabled categories only — no hard-coded label skip list,
    except speech-only classes, which captions already cover.
    """
    thr = _resolve_yamnet_thresholds(profile, settings)
    base_threshold = thr["base_threshold"]
    vehicle_threshold = thr["vehicle_threshold"]
    natural_threshold = thr["natural_threshold"]
    music_threshold = thr["music_threshold"]
    secondary_min_ratio = thr["secondary_min_ratio"]
    top_k = int(thr.get("top_k", YAMNET_TOP_K))
    max_labels = int(thr.get("max_labels", YAMNET_MAX_LABELS))
    skip_cats = exclude_categories or set()
    denoise_strength = YAMNET_DENOISE_STRENGTH
    if settings is not None:
        denoise_strength = float(np.clip(settings.yamnet_denoise_strength, 0.0, 0.95))

    floor_delta = 0.0
    if adaptive is not None:
        floor_delta = adaptive.take_floor_delta()

    base_threshold += floor_delta
    vehicle_threshold += floor_delta
    natural_threshold += floor_delta

    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if noise_score is not None and noise_score < SFX_DENOISE_MIN_NOISE_SCORE:
        denoise_strength = 0.0
    if denoise_strength > 0.05:
        clean_audio = nr.reduce_noise(
            y=audio,
            sr=SAMPLE_RATE,
            stationary=True,
            prop_decrease=denoise_strength,
            n_fft=512,
            hop_length=128,
        )
    else:
        clean_audio = audio
    if not np.isfinite(clean_audio).all():
        clean_audio = np.nan_to_num(
            clean_audio, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
    scores, _, _ = yamnet_model(clean_audio)
    class_scores: tf.Tensor = 0.55 * tf.reduce_max(scores, axis=0) + 0.45 * tf.reduce_mean(
        scores, axis=0
    )

    top_indices = tf.argsort(class_scores, direction="DESCENDING")[:top_k].numpy()

    candidates = []
    considered_top: float | None = None
    for idx in top_indices:
        idx_i = int(idx)
        label = class_names[idx_i]
        score = float(class_scores[idx_i].numpy())
        assigned_cat = normalize_category(category_for_index(idx_i))
        if assigned_cat in skip_cats:
            continue
        if label in YAMNET_SPEECH_LABELS:
            continue

        if considered_top is None or score > considered_top:
            considered_top = score

        th = base_threshold
        if assigned_cat == "vehicle":
            th = vehicle_threshold
        elif assigned_cat == "natural":
            th = natural_threshold
        elif assigned_cat == "music":
            th = music_threshold

        is_transient = idx_i in YAMNET_TRANSIENT_INDICES
        if is_transient:
            th = max(0.18, th - TRANSIENT_THRESHOLD_DELTA)
        elif label in YAMNET_VAGUE_LABELS:
            th = min(0.95, th + VAGUE_SCORE_MARGIN)

        if score < th:
            continue

        candidates.append(
            {
                "label": label,
                "score": score,
                "category": assigned_cat,
                "transient": is_transient,
            }
        )

    if adaptive is not None and considered_top is not None:
        adaptive.after_window(considered_top)

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
    return [
        {
            "label": c["label"],
            "category": c["category"],
            "score": round(float(c["score"]), 3),
            "transient": bool(c["transient"]),
        }
        for c in kept[:max_labels]
    ]

def reset_speaker_timeline() -> None:
    speaker_timeline.clear()

def reset_streaming_state() -> None:
    speaker_timeline.clear()
    try:
        pipeline.accumulator._prediction = None
    except Exception:
        pass
    try:
        diarization.reset()
    except Exception:
        pass

def get_speaker_now(lookback_sec: float = 0.35) -> str:
    now = time.monotonic()
    for ts, spk in reversed(speaker_timeline):
        if now - ts <= lookback_sec:
            return normalize_speaker_id(spk)
    return "SPEAKER_00"

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
            return normalize_speaker_id(spk)
        if ts <= timestamp + max_age:
            best = spk
            break
    return normalize_speaker_id(best) if best else "SPEAKER_00"

pipeline.stream.subscribe(on_diarization_update)