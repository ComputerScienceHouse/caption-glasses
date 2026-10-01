import json

from config.env import env_float, env_int, get_env_variable, logger

YAMNET_BASE_THRESHOLD: float = env_float("YAMNET_BASE_THRESHOLD", "0.38")
YAMNET_VEHICLE_THRESHOLD: float = env_float("YAMNET_VEHICLE_THRESHOLD", "0.48")
YAMNET_NATURAL_THRESHOLD: float = env_float("YAMNET_NATURAL_THRESHOLD", "0.46")
YAMNET_MUSIC_THRESHOLD: float = env_float("YAMNET_MUSIC_THRESHOLD", "0.48")
YAMNET_SECONDARY_MIN_RATIO: float = env_float("YAMNET_SECONDARY_MIN_RATIO", "0.78")
YAMNET_TOP_K: int = env_int("YAMNET_TOP_K", "20")
YAMNET_MAX_LABELS: int = env_int("YAMNET_MAX_LABELS", "4")
YAMNET_ADAPT_ALPHA: float = env_float("YAMNET_ADAPT_ALPHA", "0.12")
YAMNET_ADAPT_FLOOR_DELTA_MAX: float = env_float("YAMNET_ADAPT_FLOOR_DELTA_MAX", "0.04")
YAMNET_DENOISE_STRENGTH: float = env_float("YAMNET_DENOISE_STRENGTH", "0.15")

SFX_LABEL_TTL_SEC: float = env_float("SFX_LABEL_TTL_SEC", "3.0")
SFX_KEEPALIVE_SEC: float = env_float("SFX_KEEPALIVE_SEC", "2.0")
SFX_HYSTERESIS_WINDOWS: int = env_int("SFX_HYSTERESIS_WINDOWS", "2")
SFX_DENOISE_MIN_NOISE_SCORE: float = env_float("SFX_DENOISE_MIN_NOISE_SCORE", "0.28")

_yamnet_presets_raw = get_env_variable("YAMNET_PROFILE_PRESETS", "")
if _yamnet_presets_raw:
    try:
        YAMNET_PROFILE_PRESETS: dict = json.loads(_yamnet_presets_raw)
    except json.JSONDecodeError:
        logger.warning("YAMNET_PROFILE_PRESETS invalid JSON; using built-in presets")
        YAMNET_PROFILE_PRESETS = {}
else:
    YAMNET_PROFILE_PRESETS = {}

_BUILTIN_YAMNET_PRESETS: dict[str, dict[str, float]] = {
    "default": {},
    "quiet": {
        "base_threshold": -0.04,
        "vehicle_threshold": -0.04,
        "natural_threshold": -0.04,
        "music_threshold": -0.04,
        "secondary_min_ratio": -0.04,
    },
    "noisy": {
        "base_threshold": 0.06,
        "vehicle_threshold": 0.05,
        "natural_threshold": 0.05,
        "music_threshold": 0.04,
        "secondary_min_ratio": 0.04,
    },
    "media": {
        "base_threshold": -0.08,
        "vehicle_threshold": -0.06,
        "natural_threshold": -0.06,
        "music_threshold": -0.10,
        "secondary_min_ratio": -0.10,
    },
}


def merged_yamnet_profile_presets() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k, v in _BUILTIN_YAMNET_PRESETS.items():
        out[k] = dict(v)
    for k, v in YAMNET_PROFILE_PRESETS.items():
        if isinstance(v, dict):
            out.setdefault(k, {})
            out[k].update({str(kk): float(vv) for kk, vv in v.items()})
    return out


YAMNET_PRESETS_MERGED: dict[str, dict[str, float]] = merged_yamnet_profile_presets()