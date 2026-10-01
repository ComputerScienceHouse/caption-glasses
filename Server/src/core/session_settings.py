from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from config import (
    LYRICS_PARTIAL_EVERY_N_CHUNKS,
    LYRICS_PARTIAL_MIN_INTERVAL_SEC,
    MAX_DURATION,
    PARTIAL_EVERY_N_CHUNKS,
    PARTIAL_MIN_INTERVAL_SEC,
    PHRASE_TIMEOUT,
    SEGMENT_MAX_NO_SPEECH_PROB_FINAL,
    SEGMENT_MAX_NO_SPEECH_PROB_PARTIAL,
    SEGMENT_MIN_AVG_LOGPROB_FINAL,
    SEGMENT_MIN_AVG_LOGPROB_PARTIAL,
    TRANSCRIBE_MIN_RMS,
    VAD_THRESHOLD,
    WHISPER_FINAL_BEAM,
    WHISPER_PARTIAL_BEAM,
    YAMNET_BASE_THRESHOLD,
    YAMNET_DENOISE_STRENGTH,
    YAMNET_MAX_LABELS,
    YAMNET_MUSIC_THRESHOLD,
    YAMNET_SECONDARY_MIN_RATIO,
)

_TYPE_BY_NAME = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
}


@dataclass
class SessionSettings:
    content_mode: str = "speech"

    phrase_timeout_sec: float = PHRASE_TIMEOUT
    max_utterance_sec: float = MAX_DURATION
    speaker_reset_sec: float = 45.0

    partial_every_n_chunks: int = PARTIAL_EVERY_N_CHUNKS
    partial_min_interval_sec: float = PARTIAL_MIN_INTERVAL_SEC

    vad_threshold_base: float = VAD_THRESHOLD
    vad_sensitivity_boost: float = 0.0
    transcribe_min_rms: float = TRANSCRIBE_MIN_RMS
    skip_denoise: bool = False

    partial_beam: int = WHISPER_PARTIAL_BEAM
    final_beam: int = WHISPER_FINAL_BEAM
    whisper_vad_filter: bool = True
    segment_min_logprob_partial: float = SEGMENT_MIN_AVG_LOGPROB_PARTIAL
    segment_min_logprob_final: float = SEGMENT_MIN_AVG_LOGPROB_FINAL
    segment_max_no_speech_partial: float = SEGMENT_MAX_NO_SPEECH_PROB_PARTIAL
    segment_max_no_speech_final: float = SEGMENT_MAX_NO_SPEECH_PROB_FINAL

    speaker_lookback_sec: float = 0.35

    yamnet_base_threshold: float = YAMNET_BASE_THRESHOLD
    yamnet_music_threshold: float = YAMNET_MUSIC_THRESHOLD
    yamnet_secondary_min_ratio: float = YAMNET_SECONDARY_MIN_RATIO
    yamnet_max_labels: int = YAMNET_MAX_LABELS
    yamnet_denoise_strength: float = YAMNET_DENOISE_STRENGTH

    def apply_mode(self, mode: str) -> None:
        if mode not in _MODE_PRESETS:
            return
        fresh = SessionSettings()
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, getattr(fresh, field_name))
        presets = _MODE_PRESETS[mode]
        for k, v in presets.items():
            if hasattr(self, k):
                setattr(self, k, copy.deepcopy(v))

    @staticmethod
    def _resolve_field_type(ft: Any) -> Any:
        if isinstance(ft, str):
            return _TYPE_BY_NAME.get(ft, ft)
        origin = get_origin(ft)
        if origin is not None:
            args = [a for a in get_args(ft) if a is not type(None)]
            if len(args) == 1:
                return SessionSettings._resolve_field_type(args[0])
        return ft

    def _coerce_field_value(self, key: str, value: Any) -> Any:
        ft = self._resolve_field_type(self.__dataclass_fields__[key].type)
        if ft is int:
            return int(round(float(value)))
        if ft is float:
            return float(value)
        if ft is bool:
            if isinstance(value, str):
                return value.lower() in ("1", "true", "yes")
            return bool(value)
        if ft is str:
            return str(value)
        return value

    def patch(self, updates: dict[str, Any]) -> list[str]:
        applied: list[str] = []
        for key, value in updates.items():
            if key not in self.__dataclass_fields__:
                continue
            setattr(self, key, self._coerce_field_value(key, value))
            applied.append(key)
        return applied

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}

#presets just in case, but settings can be changed manually during runtime
_MODE_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {},
    "soft_speech": {
        "vad_sensitivity_boost": 0.08,
        "transcribe_min_rms": 0.002,
        "phrase_timeout_sec": 0.7,
        "max_utterance_sec": 6.0,
        "final_beam": 3,
        "segment_min_logprob_final": -1.15,
        "segment_max_no_speech_final": 0.85,
    },
    "noisy_hall": {
        "vad_sensitivity_boost": 0.05,
        "phrase_timeout_sec": 0.7,
        "max_utterance_sec": 6.0,
        "skip_denoise": False,
        "final_beam": 3,
        "segment_min_logprob_final": -1.1,
    },
    "accuracy": {
        "partial_beam": 2,
        "final_beam": 4,
        "max_utterance_sec": 8.0,
        "phrase_timeout_sec": 0.8,
        "segment_min_logprob_final": -1.2,
        "segment_max_no_speech_final": 0.88,
        "speaker_lookback_sec": 0.25,
    },
    "lyrics": {
        "content_mode": "lyrics",
        "vad_sensitivity_boost": 0.14,
        "transcribe_min_rms": 0.0015,
        "phrase_timeout_sec": 1.5,
        "max_utterance_sec": 14.0,
        "partial_every_n_chunks": LYRICS_PARTIAL_EVERY_N_CHUNKS,
        "partial_min_interval_sec": LYRICS_PARTIAL_MIN_INTERVAL_SEC,
        "skip_denoise": True,
        "whisper_vad_filter": False,
        "final_beam": 3,
        "segment_min_logprob_partial": -0.82,
        "segment_min_logprob_final": -1.0,
        "segment_max_no_speech_partial": 0.55,
        "segment_max_no_speech_final": 0.74,
        "yamnet_denoise_strength": 0.1,
        "yamnet_music_threshold": 0.42,
        "yamnet_base_threshold": 0.34,
    },
}

MODE_NAMES: tuple[str, ...] = tuple(_MODE_PRESETS.keys())