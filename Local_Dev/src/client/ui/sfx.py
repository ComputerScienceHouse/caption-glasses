import pygame
from client import theme
from client.sfx_categories import category_icon_char, normalize_sfx_category
from client.state import state

_chip_cache: dict[tuple, pygame.Surface] = {}

CHIP_PAD_X = 8
CHIP_PAD_Y = 4
CHIP_GAP = 8
CHIP_ICON_GAP = 6

SCREEN_WIDTH_BASE = 1100
SCREEN_SCALE_THRESHOLDS = (
    (1600, 1.4),
    (1400, 1.3),
    (1200, 1.2),
    (1100, 1.1),
)


def _style_for(category: str | None) -> dict:
    cat = normalize_sfx_category(category)
    return theme.SFX_CATEGORY_STYLES.get(cat, theme.SFX_DEFAULT_STYLE)

def _screen_scale(screen: pygame.Surface) -> float:
    width = screen.get_width() if screen else SCREEN_WIDTH_BASE
    for threshold, scale in SCREEN_SCALE_THRESHOLDS:
        if width >= threshold:
            return scale
    return 1.0


def _render_chip(label: str, category: str | None, *args) -> pygame.Surface:
    scale = float(args[0]) if args else 1.0
    cat = normalize_sfx_category(category)
    key = (label, cat, id(theme.sfx_font), id(theme.small_font), round(scale, 2))
    cached = _chip_cache.get(key)
    if cached is not None:
        return cached

    style = _style_for(cat)
    icon = category_icon_char(cat)
    icon_surf = theme.sfx_font.render(icon, True, style["fg"])
    text_surf = theme.small_font.render(label, True, style["fg"])

    pad_x = int(CHIP_PAD_X * scale)
    pad_y = int(CHIP_PAD_Y * scale)
    icon_gap = int(CHIP_ICON_GAP * scale)
    radius = max(6, int(6 * scale))

    width = pad_x * 2 + icon_surf.get_width() + icon_gap + text_surf.get_width()
    height = max(icon_surf.get_height(), text_surf.get_height()) + pad_y * 2
    chip = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(chip, style["bg"], chip.get_rect(), border_radius=radius)
    chip.blit(icon_surf, (pad_x, (height - icon_surf.get_height()) // 2))
    chip.blit(
        text_surf,
        (
            pad_x + icon_surf.get_width() + icon_gap,
            (height - text_surf.get_height()) // 2,
        ),
    )
    _chip_cache[key] = chip
    if len(_chip_cache) > 64:
        _chip_cache.clear()
        _chip_cache[key] = chip
    return chip

def visible_sound_labels() -> list[dict]:
    enabled = state.get("sfx_enabled_categories") or {}
    out = []
    for item in state.get("sound_labels") or []:
        cat = normalize_sfx_category(item.get("category"))
        if enabled.get(cat, True):
            out.append(item)
    return out

def draw_sound_labels(screen: pygame.Surface, x: int, y: int) -> None:
    labels = visible_sound_labels()
    if not labels:
        return
    scale = _screen_scale(screen)
    gap = max(4, int(CHIP_GAP * scale))
    cursor_x = x
    for item in labels:
        chip = _render_chip(str(item.get("label", "")), item.get("category"), scale)
        screen.blit(chip, (cursor_x, y - chip.get_height()))
        cursor_x += chip.get_width() + gap
