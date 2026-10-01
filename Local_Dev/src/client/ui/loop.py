import gc
import time
import pygame

from client import theme
from client import state as app_state
from client.settings import (
    reset_lyrics_settings,
    reset_speech_settings,
    settings_payload,
    slider_stored_value,
)
from client.state import invalidate_caption_cache, state, state_lock
from client.tuning_specs import (
    ALL_SLIDER_SPECS,
    BTN_GAP,
    BTN_H,
    CLIENT_GC_INTERVAL_SEC,
    FOOTER_H,
    HEADER_H,
    LINE_H,
    MEDIA_SFX_DEFAULTS,
    SIDEBAR_PAD,
    SIDEBAR_W,
    SLIDER_H,
    SLIDER_SPECS,
    SFX_SLIDER_SPECS,
    SOUND_DISPLAY_DURATION,
    YAMNET_PROFILES,
)
from client.sfx_categories import enabled_category_ids
from client.ui.captions import get_caption_height, get_caption_lines, rebuild_captions
from client.ui.noise_meter import draw_noise_meter
from client.ui.sfx import draw_sound_labels
from client.ui.sidebar import (
    btn_label,
    btn_ready,
    button_color,
    buttons_y,
    clamp,
    draw_sfx_category_filters,
    draw_slider_row,
    fit_hint,
    handle_sfx_category_hit,
    handle_slider_hit,
    sidebar_height,
    sidebar_x,
    slider_track_for_key,
    slider_value,
    slider_y,
    sfx_section_y,
    sfx_slider_y,
)
from client.ws_client import ws_send

BTN_LABELS = [
    "caption_translate",
    "singing",
    "speakers",
    "profanity",
    "yamnet",
    "telemetry",
    "reset_speech",
    "reset_lyrics",
]

SPEECH_SLIDER_COLORS = ((55, 110, 170), (90, 160, 230))
SFX_SLIDER_COLORS = ((100, 75, 140), (160, 120, 190))

_frame = {
    "scroll_y": 0,
    "auto_scroll": True,
    "bootstrapped": False,
    "caption_state_key": None,
}

def paint_startup_frame(message: str = "Connecting…") -> None:
    pygame.event.pump()
    theme.screen.fill((18, 18, 18))
    surf = theme.ui_font.render(message, True, (150, 150, 150))
    theme.screen.blit(surf, (18, 14))
    pygame.display.flip()

def _push_sfx_categories() -> None:
    ws_send(
        {
            "type": "set_sfx_categories",
            "value": enabled_category_ids(state.get("sfx_enabled_categories")),
        }
    )


def _bootstrap_ws() -> None:
    if _frame["bootstrapped"]:
        return
    _frame["bootstrapped"] = True
    # Monitor mode is receive-only: do not push settings or category filters.
    if not state.get("show_sidebar", True):
        if state["caption_telemetry"]:
            ws_send({"type": "set_caption_telemetry", "value": True})
        return
    if state["caption_telemetry"]:
        ws_send({"type": "set_caption_telemetry", "value": True})
    ws_send({"type": "get_settings"})
    _push_sfx_categories()

def tick_frame() -> bool:
    _bootstrap_ws()
    scroll_y = _frame["scroll_y"]
    auto_scroll = _frame["auto_scroll"]
    show_sidebar = state.get("show_sidebar", True)

    pygame.event.pump()
    w, h = theme.screen.get_size()
    sb_x = sidebar_x(w) if show_sidebar else w
    scroll = clamp(state["sidebar_scroll"], 0, max(0, sidebar_height() - h)) if show_sidebar else 0
    state["sidebar_scroll"] = scroll
    mouse = pygame.mouse.get_pos()
    track_w = SIDEBAR_W - SIDEBAR_PAD * 2

    with state_lock:
        caption_state_key = (
            len(state["finals"]),
            state["partial"].get("text"),
            state["partial"].get("speaker"),
            state["partial"].get("language"),
        )
    if caption_state_key != _frame["caption_state_key"]:
        auto_scroll = True
        _frame["caption_state_key"] = caption_state_key

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.VIDEORESIZE:
            theme.resize_window(event.size[0], event.size[1])
            invalidate_caption_cache()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not show_sidebar or event.pos[0] < sb_x:
                auto_scroll = True
                continue
            hit_slider = handle_slider_hit(event.pos, sb_x, scroll, track_w)
            if hit_slider:
                state["dragging_slider"] = hit_slider
            elif handle_sfx_category_hit(event.pos, sb_x, scroll, track_w):
                _push_sfx_categories()
            else:
                btn_y = buttons_y() - scroll
                for i, name in enumerate(BTN_LABELS):
                    rect = pygame.Rect(
                        sb_x + SIDEBAR_PAD, btn_y + i * (BTN_H + BTN_GAP), track_w, BTN_H
                    )
                    if not rect.collidepoint(event.pos):
                        continue
                    if not btn_ready(name):
                        break
                    if name == "caption_translate":
                        state["translate_mode"] = not state["translate_mode"]
                        ws_send(
                            {
                                "type": "set_task",
                                "value": "translate" if state["translate_mode"] else "transcribe",
                            }
                        )
                    elif name == "singing":
                        state["singing_mode"] = not state["singing_mode"]
                        ws_send(
                            {
                                "type": "set_mode",
                                "value": "lyrics" if state["singing_mode"] else "balanced",
                            }
                        )
                    elif name == "speakers":
                        state["show_speakers"] = not state["show_speakers"]
                        invalidate_caption_cache()
                    elif name == "profanity":
                        state["profanity_filter"] = not state["profanity_filter"]
                        ws_send({"type": "set_profanity_filter", "value": state["profanity_filter"]})
                    elif name == "yamnet":
                        state["yamnet_profile_index"] = (
                            state["yamnet_profile_index"] + 1
                        ) % len(YAMNET_PROFILES)
                        profile = YAMNET_PROFILES[state["yamnet_profile_index"]]
                        ws_send({"type": "set_yamnet_profile", "value": profile})
                        if profile == "media":
                            for key, val in MEDIA_SFX_DEFAULTS.items():
                                state["slider_values"][key] = val
                            ws_send({"type": "set_settings", "value": settings_payload()})
                    elif name == "telemetry":
                        state["caption_telemetry"] = not state["caption_telemetry"]
                        ws_send({"type": "set_caption_telemetry", "value": state["caption_telemetry"]})
                    elif name == "reset_speech":
                        reset_speech_settings()
                        invalidate_caption_cache()
                    elif name == "reset_lyrics":
                        reset_lyrics_settings()
                        invalidate_caption_cache()
                    break
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and state["dragging_slider"]:
            key = state["dragging_slider"]
            state["dragging_slider"] = None
            if show_sidebar:
                spec = next(s for s in ALL_SLIDER_SPECS if s["key"] == key)
                ws_send(
                    {
                        "type": "set_settings",
                        "value": {
                            key: slider_stored_value(spec, state["slider_values"][key])
                        },
                    }
                )
        elif event.type == pygame.MOUSEMOTION and state["dragging_slider"] and show_sidebar:
            key = state["dragging_slider"]
            track, spec = slider_track_for_key(key, sb_x, scroll, track_w)
            ratio = (event.pos[0] - track.x) / max(1, track.width)
            state["slider_values"][key] = slider_value(spec, ratio)
        elif event.type == pygame.MOUSEWHEEL:
            if show_sidebar and mouse[0] >= sb_x:
                state["sidebar_scroll"] -= event.y * 36
            else:
                auto_scroll = False
                scroll_y -= event.y * LINE_H

    theme.screen.fill((18, 18, 18))

    if show_sidebar:
        pygame.draw.rect(theme.screen, (32, 32, 32), (sb_x, 0, SIDEBAR_W, h))
        pygame.draw.line(theme.screen, (55, 55, 55), (sb_x, 0), (sb_x, h))

        header = theme.ui_font.render("Live tuning", True, (130, 130, 130))
        theme.screen.blit(header, (sb_x + SIDEBAR_PAD, 6))

        for i, spec in enumerate(SLIDER_SPECS):
            row_y = slider_y(i) - scroll
            if row_y + SLIDER_H < HEADER_H or row_y > h:
                continue
            draw_slider_row(
                theme.screen,
                spec,
                row_y,
                track_w,
                sb_x + SIDEBAR_PAD,
                state["dragging_slider"] == spec["key"],
                SPEECH_SLIDER_COLORS,
            )

        sfx_hdr_y = sfx_section_y() - scroll
        if sfx_hdr_y + 12 > HEADER_H and sfx_hdr_y < h:
            theme.screen.blit(theme.ui_font.render("SFX detection", True, (130, 130, 130)), (sb_x + SIDEBAR_PAD, sfx_hdr_y))

        for i, spec in enumerate(SFX_SLIDER_SPECS):
            row_y = sfx_slider_y(i) - scroll
            if row_y + SLIDER_H < HEADER_H or row_y > h:
                continue
            draw_slider_row(
                theme.screen,
                spec,
                row_y,
                track_w,
                sb_x + SIDEBAR_PAD,
                state["dragging_slider"] == spec["key"],
                SFX_SLIDER_COLORS,
            )

        draw_sfx_category_filters(theme.screen, sb_x, scroll, track_w, mouse)

        btn_y = buttons_y() - scroll
        for i, name in enumerate(BTN_LABELS):
            rect = pygame.Rect(sb_x + SIDEBAR_PAD, btn_y + i * (BTN_H + BTN_GAP), track_w, BTN_H)
            if rect.bottom < 0 or rect.top > h:
                continue
            on = (
                (name == "caption_translate" and state["translate_mode"])
                or (name == "singing" and state["singing_mode"])
                or (name == "speakers" and state["show_speakers"])
                or (name == "profanity" and state["profanity_filter"])
                or (name == "telemetry" and state["caption_telemetry"])
            )
            col = button_color(name, on, rect.collidepoint(mouse))
            pygame.draw.rect(theme.screen, col, rect, border_radius=4)
            label = fit_hint(btn_label(name), track_w - 8)
            txt = theme.ui_font.render(label, True, (240, 240, 240))
            theme.screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

    text_w = (w - SIDEBAR_W - 36) if show_sidebar else (w - 36)
    cap_top = 12
    cap_bottom = h - FOOTER_H
    rebuild_captions(text_w)
    caption_height = get_caption_height()
    if auto_scroll:
        scroll_y = max(0, caption_height - (cap_bottom - cap_top))

    clip_w = (w - SIDEBAR_W) if show_sidebar else w
    clip = pygame.Rect(0, cap_top, clip_w, cap_bottom - cap_top)
    theme.screen.set_clip(clip)
    for surf, y in get_caption_lines():
        dy = cap_top + y - scroll_y
        if cap_top - LINE_H < dy < cap_bottom:
            theme.screen.blit(surf, (18, dy))
    theme.screen.set_clip(None)

    now_ms = int(time.monotonic() * 1000)
    with state_lock:
        has_sfx = bool(state.get("sound_labels"))
        sound_timestamp = int(state.get("sound_timestamp") or 0)
    if has_sfx and now_ms - sound_timestamp < SOUND_DISPLAY_DURATION:
        sfx_x = 110 if state["caption_telemetry"] else 18
        draw_sound_labels(theme.screen, sfx_x, h - 28)
    if state["caption_telemetry"]:
        draw_noise_meter(theme.screen, 34, h - 34)

    pygame.display.flip()

    _frame["scroll_y"] = scroll_y
    _frame["auto_scroll"] = auto_scroll
    return True

def run_pygame_loop() -> None:
    """Blocking UI loop for the main thread. Network I/O runs elsewhere."""
    clock = pygame.time.Clock()
    app_state._last_client_gc = time.monotonic()
    _frame["bootstrapped"] = False
    while True:
        try:
            if not tick_frame():
                return
        except Exception as exc:
            print(f"UI frame error: {type(exc).__name__}: {exc}")
        now = time.monotonic()
        if now - app_state._last_client_gc >= CLIENT_GC_INTERVAL_SEC:
            gc.collect()
            app_state._last_client_gc = now
        clock.tick(60)
