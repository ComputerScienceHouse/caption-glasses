YAMNET_CATEGORY_RANGES: dict[str, range] = {
    "human": range(0, 67),
    "animal": range(67, 132),
    "music": range(132, 277),
    "natural": range(277, 294),
    "vehicle": range(294, 348),
    "domestic": range(348, 412),
    "tools": range(412, 420),
    "explosive": range(420, 456),
    "tones": range(456, 499),
    "ambient": range(499, 521),
}

YAMNET_CATEGORY_IDS: tuple[str, ...] = tuple(YAMNET_CATEGORY_RANGES.keys())

YAMNET_VAGUE_LABELS: frozenset[str] = frozenset(
    {
        "Clicking",
        "Clickety-clack",
        "Hum",
        "Rumble",
        "Rustle",
        "Whir",
        "Static",
        "White noise",
        "Pink noise",
        "Mains hum",
        "Reverberation",
        "Echo",
        "Noise",
        "Environmental noise",
        "Silence",
    }
)

YAMNET_SPEECH_LABELS: frozenset[str] = frozenset(
    {
        "Speech",
        "Child speech, kid speaking",
        "Conversation",
        "Narration, monologue",
        "Babbling",
        "Speech synthesizer",
        "Whispering",
    }
)

YAMNET_TRANSIENT_INDICES: frozenset[int] = frozenset(
    {
        35,   # Whistling
        57,   # Finger snapping
        58,   # Clapping
        61,   # Cheering
        62,   # Applause
        353,  # Knock
        355,  # Squeak
        420,  # Explosion
        421,  # Gunshot, gunfire
        427,  # Firecracker
        429,  # Eruption
        430,  # Boom
        437,  # Shatter
        460,  # Bang
        475,  # Beep, bleep
        476,  # Ping
        477,  # Ding
        478,  # Clang
    }
)
YAMNET_WINDOW_SAMPLES: int = 15600
YAMNET_HOP_CHUNKS: int = 4
TRANSIENT_THRESHOLD_DELTA: float = 0.12
VAGUE_SCORE_MARGIN: float = 0.04


def category_for_index(idx: int) -> str:
    for cat_name, cat_range in YAMNET_CATEGORY_RANGES.items():
        if idx in cat_range:
            return cat_name
    return "ambient"


def normalize_category(category: str | None) -> str:
    if not category:
        return "ambient"
    if category == "misc":
        return "ambient"
    if category in YAMNET_CATEGORY_RANGES:
        return category
    return "ambient"
