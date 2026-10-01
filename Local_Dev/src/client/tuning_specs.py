"""
Values, labels, and UI layout for settings sliders.
"""

SLIDER_SPECS = [
    {
        "key": "vad_sensitivity_boost",
        "label": "Quiet speech",
        "min": 0.0, "max": 0.12, "step": 0.01, "default": 0.0, "fmt": ".2f",
    },
    {
        "key": "phrase_timeout_sec",
        "label": "Pause ends line",
        "min": 0.35, "max": 1.5, "step": 0.05, "default": 0.55, "fmt": ".2f", "suffix": "s",
    },
    {
        "key": "max_utterance_sec",
        "label": "Max line length",
        "min": 3.0, "max": 14.0, "step": 0.5, "default": 6.0, "fmt": ".1f", "suffix": "s",
    },
    {
        "key": "partial_min_interval_sec",
        "label": "Partial delay",
        "min": 0.05, "max": 0.60, "step": 0.05, "default": 0.25, "fmt": ".2f", "suffix": "s",
    },
    {
        "key": "partial_every_n_chunks",
        "label": "Partial stride",
        "min": 1, "max": 12, "step": 1, "default": 3, "fmt": "d",
    },
    {
        "key": "partial_beam",
        "label": "Partial accuracy",
        "min": 1, "max": 5, "step": 1, "default": 1, "fmt": "d",
    },
    {
        "key": "final_beam",
        "label": "Word accuracy",
        "min": 1, "max": 5, "step": 1, "default": 2, "fmt": "d",
    },
    {
        "key": "transcribe_min_rms",
        "label": "Volume floor",
        "min": 0.001, "max": 0.008, "step": 0.0005, "default": 0.004, "fmt": ".3f",
    },
    {
        "key": "segment_min_logprob_final",
        "label": "Uncertain words",
        "min": -1.3, "max": -0.85, "step": 0.05, "default": -1.12, "fmt": ".2f",
    },
    {
        "key": "speaker_lookback_sec",
        "label": "Speaker speed",
        "min": 0.15, "max": 0.75, "step": 0.05, "default": 0.35, "fmt": ".2f", "suffix": "s",
    },
    {
        "key": "speaker_reset_sec",
        "label": "Reset speakers",
        "min": 20, "max": 120, "step": 5, "default": 45, "fmt": "d", "suffix": "s",
    },
]

SFX_SLIDER_SPECS = [
    {
        "key": "yamnet_base_threshold",
        "label": "SFX sensitivity",
        "min": 0.22, "max": 0.72, "step": 0.02, "default": 0.38, "fmt": ".2f",
    },
    {
        "key": "yamnet_music_threshold",
        "label": "Music / choir",
        "min": 0.28, "max": 0.75, "step": 0.02, "default": 0.48, "fmt": ".2f",
    },
    {
        "key": "yamnet_secondary_min_ratio",
        "label": "Extra SFX labels",
        "min": 0.60, "max": 0.98, "step": 0.02, "default": 0.78, "fmt": ".2f",
    },
    {
        "key": "yamnet_max_labels",
        "label": "Max SFX chips",
        "min": 1, "max": 6, "step": 1, "default": 4, "fmt": "d",
    },
    {
        "key": "yamnet_denoise_strength",
        "label": "SFX denoise",
        "min": 0.0, "max": 0.85, "step": 0.05, "default": 0.15, "fmt": ".2f",
    },
]

ALL_SLIDER_SPECS = SLIDER_SPECS + SFX_SLIDER_SPECS

YAMNET_PROFILES = ["default", "media", "quiet", "noisy"]

MEDIA_SFX_DEFAULTS = {
    "yamnet_base_threshold": 0.32,
    "yamnet_music_threshold": 0.40,
    "yamnet_secondary_min_ratio": 0.70,
    "yamnet_max_labels": 5,
    "yamnet_denoise_strength": 0.10,
}

SIDEBAR_W = 272
SIDEBAR_PAD = 10
HEADER_H = 26
SLIDER_H = 38
BTN_H = 30
BTN_GAP = 5
SFX_CAT_COLS = 2
SFX_CAT_ROW_H = 28
SFX_CAT_GAP = 4
FOOTER_H = 96
LINE_H = 34
MAX_SENTENCE_HISTORY = 200
SOUND_DISPLAY_DURATION = 3500
FRAME_SEC = 1 / 60
BTN_DEBOUNCE_SEC = 0.3
WS_CMD_QUEUE_MAX = 48
WS_AUDIO_QUEUE_MAX = 256
WS_CMD_OUTBOUND_MAX = 16
CLIENT_GC_INTERVAL_SEC = 1800