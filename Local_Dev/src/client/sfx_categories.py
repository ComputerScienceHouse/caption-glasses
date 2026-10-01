from pathlib import Path

SFX_CATEGORY_DEFS: list[dict[str, str]] = [
    {"id": "human", "label": "People"},
    {"id": "animal", "label": "Animals"},
    {"id": "music", "label": "Music"},
    {"id": "natural", "label": "Nature"},
    {"id": "vehicle", "label": "Vehicles"},
    {"id": "domestic", "label": "Home"},
    {"id": "tools", "label": "Tools"},
    {"id": "explosive", "label": "Impacts"},
    {"id": "tones", "label": "Beeps"},
    {"id": "ambient", "label": "Other"},
]

ALL_SFX_CATEGORY_IDS: tuple[str, ...] = tuple(c["id"] for c in SFX_CATEGORY_DEFS)

#From Font Awesome 5 Free Solid
SFX_CATEGORY_ICONS: dict[str, str] = {
    "human": "\uf007",
    "animal": "\uf1b0",
    "music": "\uf001",
    "natural": "\uf06c",
    "vehicle": "\uf1b9",
    "domestic": "\uf015",
    "tools": "\uf0ad",
    "explosive": "\uf0e7",
    "tones": "\uf028",
    "ambient": "\uf0c2",
}

def fa_solid_font_path() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "fonts" / "fa-solid-900.ttf"

def normalize_sfx_category(category: str | None) -> str:
    if not category:
        return "ambient"
    if category == "misc":
        return "ambient"
    if category in ALL_SFX_CATEGORY_IDS:
        return category
    return "ambient"

def category_icon_char(category: str | None) -> str:
    return SFX_CATEGORY_ICONS.get(
        normalize_sfx_category(category), SFX_CATEGORY_ICONS["ambient"]
    )

def default_enabled_categories() -> dict[str, bool]:
    return {cid: True for cid in ALL_SFX_CATEGORY_IDS}

def enabled_category_ids(enabled: dict[str, bool] | None) -> list[str]:
    src = enabled if enabled is not None else default_enabled_categories()
    return [cid for cid in ALL_SFX_CATEGORY_IDS if src.get(cid, True)]