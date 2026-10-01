from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from config import (
    ACOUSTIC_NOISE_RMS_ALPHA,
    ACOUSTIC_NOISE_FLOOR_STATIONARY_MAX,
    ACOUSTIC_PHRASE_TIMEOUT_NOISE_SCALE,
    ACOUSTIC_NOISE_SCORE_CAP_RMS,
    ACOUSTIC_SILENCE_VAD_CUTOFF,
    ACOUSTIC_VAD_OFFSET_PER_RMS,
    NR_PROP_DECREASE_LOUD,
    NR_PROP_DECREASE_QUIET,
    POST_GAIN_LOUD,
    POST_GAIN_NOISE_RMS_END,
    POST_GAIN_NOISE_RMS_START,
    POST_GAIN_QUIET,
    VAD_THRESHOLD_MAX,
    VAD_THRESHOLD_MIN,
    YAMNET_ADAPT_ALPHA,
    YAMNET_ADAPT_FLOOR_DELTA_MAX,
)
from core.session_settings import SessionSettings


def chunk_rms(audio: ndarray) -> float:
    if audio is None or len(audio) == 0:
        return 0.0
    samples = audio.astype(np.float64)
    if not np.isfinite(samples).all():
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(np.mean(np.square(samples))))

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))

@dataclass
class YamnetAdaptiveState:
    global_top_ema: float = 0.55
    pending_floor_delta: float = 0.0

    def take_floor_delta(self) -> float:
        d = self.pending_floor_delta
        self.pending_floor_delta = 0.0
        return d

    def after_window(self, global_top: float) -> None:
        a = YAMNET_ADAPT_ALPHA
        self.global_top_ema = (1.0 - a) * self.global_top_ema + a * float(global_top)
        if self.global_top_ema < 0.42:
            self.pending_floor_delta = max(
                -YAMNET_ADAPT_FLOOR_DELTA_MAX, (self.global_top_ema - 0.5) * 0.5
            )
        elif self.global_top_ema > 0.68:
            self.pending_floor_delta = min(
                YAMNET_ADAPT_FLOOR_DELTA_MAX, (self.global_top_ema - 0.62) * 0.35
            )
        else:
            self.pending_floor_delta = 0.0

@dataclass
class ConnectionAcoustics:
    noise_rms_ema: float = 0.012
    effective_vad_threshold: float = 0.4
    noise_score: float = 0.0
    env_bucket: str = "moderate"
    last_input_rms: float = 0.0
    last_gate_open: bool = False

    def update(
        self,
        audio_chunk: ndarray,
        speech_prob: float,
        settings: SessionSettings,
    ) -> None:
        rms = chunk_rms(audio_chunk)
        self.last_input_rms = rms
        a = ACOUSTIC_NOISE_RMS_ALPHA
        if settings.content_mode == "lyrics":
            if rms < 0.01 and speech_prob < ACOUSTIC_SILENCE_VAD_CUTOFF:
                self.noise_rms_ema = (1.0 - a) * self.noise_rms_ema + a * rms
        elif speech_prob < ACOUSTIC_SILENCE_VAD_CUTOFF:
            self.noise_rms_ema = (1.0 - a) * self.noise_rms_ema + a * rms

        base = settings.vad_threshold_base
        excess = max(0.0, self.noise_rms_ema - 0.012)
        offset = ACOUSTIC_VAD_OFFSET_PER_RMS * excess
        if settings.content_mode == "lyrics":
            base -= 0.08
            offset *= 0.25
            vad_max = 0.32
        else:
            vad_max = VAD_THRESHOLD_MAX
        clipped = float(np.clip(base + offset, VAD_THRESHOLD_MIN, vad_max))
        self.effective_vad_threshold = max(
            VAD_THRESHOLD_MIN, clipped - settings.vad_sensitivity_boost
        )

        cap = max(ACOUSTIC_NOISE_SCORE_CAP_RMS, 1e-6)
        self.noise_score = float(np.clip(self.noise_rms_ema / cap, 0.0, 1.0))

        if self.noise_rms_ema < 0.022:
            self.env_bucket = "quiet"
        elif self.noise_rms_ema < 0.045:
            self.env_bucket = "moderate"
        else:
            self.env_bucket = "loud"

    def recenter_for_long_session(self) -> None:
        self.noise_rms_ema = 0.7 * self.noise_rms_ema + 0.3 * 0.014
        cap = max(ACOUSTIC_NOISE_SCORE_CAP_RMS, 1e-6)
        self.noise_score = float(np.clip(self.noise_rms_ema / cap, 0.0, 1.0))
        if self.noise_rms_ema < 0.022:
            self.env_bucket = "quiet"
        elif self.noise_rms_ema < 0.045:
            self.env_bucket = "moderate"
        else:
            self.env_bucket = "loud"

    def nr_stationary(self) -> bool:
        return self.noise_rms_ema < ACOUSTIC_NOISE_FLOOR_STATIONARY_MAX

    def nr_prop_decrease(self) -> float:
        t = (self.noise_rms_ema - 0.015) / max(1e-6, 0.05 - 0.015)
        return _lerp(NR_PROP_DECREASE_QUIET, NR_PROP_DECREASE_LOUD, t)

    def post_gain(self) -> float:
        t = (self.noise_rms_ema - POST_GAIN_NOISE_RMS_START) / max(
            1e-6, POST_GAIN_NOISE_RMS_END - POST_GAIN_NOISE_RMS_START
        )
        return _lerp(POST_GAIN_QUIET, POST_GAIN_LOUD, t)

    def silence_chunk_limit(
        self,
        phrase_timeout_sec: float,
        chunks_per_sec: float,
        *,
        content_mode: str = "speech",
    ) -> int:
        base_chunks = max(1, int(phrase_timeout_sec * chunks_per_sec))
        scale = 1.0 + ACOUSTIC_PHRASE_TIMEOUT_NOISE_SCALE * self.noise_score
        if content_mode == "lyrics":
            scale *= 1.5
        return max(1, int(math.ceil(base_chunks * scale)))

def effective_silence_limit(
    acoustics: ConnectionAcoustics,
    phrase_timeout_sec: float,
    chunks_per_sec: float,
    *,
    content_mode: str = "speech",
) -> int:
    return acoustics.silence_chunk_limit(
        phrase_timeout_sec, chunks_per_sec, content_mode=content_mode
    )