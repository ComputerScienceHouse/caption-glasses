from client import theme
from client.state import (
    _caption_height,
    _caption_lines,
    _finals_block_cache,
    _finals_height,
    _finals_layout_key,
    _partial_cache_key,
    _partial_height,
    _partial_lines,
    state,
    state_lock,
)
from client.theme import SPEAKER_COLORS
from client.tuning_specs import LINE_H

import re

def wrap_text(text, font_obj, max_width):
    words = text.split(" ")
    lines, cur = [], []
    for word in words:
        trial = " ".join(cur + [word])
        if font_obj.size(trial)[0] <= max_width:
            cur.append(word)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [word]
    if cur:
        lines.append(" ".join(cur))
    return lines

def _caption_body(item, is_partial: bool) -> str:
    body = item.get("text", "") + ("..." if is_partial else "")
    lang = (item.get("language") or "").strip().lower()[:2]
    speaker = item.get("speaker", "")
    speaker_has_lang = " (" in speaker
    if (
        state["translate_mode"]
        and lang
        and lang != "en"
        and not speaker_has_lang
    ):
        body = f"[{lang}] {body}"
    return body

def _base_speaker_id(speaker_id: str) -> str:
    base = speaker_id.split(" (", 1)[0].strip()
    match = re.match(r"speaker[_\s-]*(\d+)", base, re.IGNORECASE)
    if match:
        zero_based = max(0, int(match.group(1)) - 1)
        return f"SPEAKER_{zero_based:02d}"
    return "SPEAKER_00"

def _speaker_label(speaker_id: str) -> str:
    lang = ""
    if " (" in speaker_id:
        speaker_id, lang_part = speaker_id.split(" (", 1)
        lang = lang_part.rstrip(")")
    match = re.match(r"speaker[_\s-]*(\d+)", speaker_id, re.IGNORECASE)
    if match:
        label = f"Speaker {int(match.group(1))}"
    else:
        label = speaker_id
    if lang:
        label = f"{label} ({lang})"
    return label

def _render_caption_items(items, text_w, *, mark_partial: bool = False):
    lines, y, last_spk = [], 0, None
    for item in items:
        if not item.get("text"):
            continue
        body = _caption_body(item, mark_partial)
        speaker = item.get("speaker", "SPEAKER_01")
        if not state.get("show_speakers", True):
            speaker = "SPEAKER_01"
        color = SPEAKER_COLORS.get(_base_speaker_id(speaker), (240, 240, 240))
        for i, row in enumerate(wrap_text(body, theme.font, text_w)):
            if (
                state.get("show_speakers", True)
                and state["display_mode"] == "label"
                and i == 0
                and speaker != last_spk
            ):
                row = f"[{_speaker_label(speaker)}] {row}"
            surf = theme.font.render(row, True, color)
            if mark_partial:
                surf.set_alpha(190)
            lines.append((surf, y))
            y += LINE_H
        last_spk = speaker
    return lines, y

def _caption_item_key(item) -> tuple:
    return (item.get("text"), item.get("speaker"), item.get("language"))

def rebuild_captions(text_w):
    global _finals_block_cache, _finals_layout_key, _finals_height
    global _partial_cache_key, _partial_lines, _partial_height
    global _caption_lines, _caption_height

    with state_lock:
        finals = list(state["finals"])
        partial = dict(state["partial"])
        display_mode = state["display_mode"]
        translate_mode = state["translate_mode"]
        show_speakers = state.get("show_speakers", True)

    layout_key = (text_w, display_mode, translate_mode, show_speakers)
    if layout_key != _finals_layout_key:
        _finals_layout_key = layout_key
        _finals_block_cache.clear()

    while len(_finals_block_cache) > len(finals):
        _finals_block_cache.pop(0)

    mismatch_at = None
    for i in range(min(len(_finals_block_cache), len(finals))):
        if _finals_block_cache[i][0] != _caption_item_key(finals[i]):
            mismatch_at = i
            break
    if mismatch_at is not None:
        del _finals_block_cache[mismatch_at:]

    while len(_finals_block_cache) < len(finals):
        idx = len(_finals_block_cache)
        item = finals[idx]
        block_lines, block_h = _render_caption_items([item], text_w)
        _finals_block_cache.append((_caption_item_key(item), block_lines, block_h))

    finals_lines: list[tuple] = []
    y = 0
    for _, block_lines, block_h in _finals_block_cache:
        for surf, by in block_lines:
            finals_lines.append((surf, y + by))
        y += block_h
    _finals_height = y

    partial_key = (
        partial.get("text"),
        partial.get("speaker"),
        partial.get("language"),
        text_w,
        display_mode,
        translate_mode,
        show_speakers,
    )
    if partial_key != _partial_cache_key:
        _partial_cache_key = partial_key
        if partial.get("text"):
            _partial_lines, _partial_height = _render_caption_items(
                [partial], text_w, mark_partial=True
            )
        else:
            _partial_lines, _partial_height = [], 0

    _caption_lines = list(finals_lines)
    _caption_height = _finals_height
    if _partial_lines:
        _caption_lines.extend((surf, y + _finals_height) for surf, y in _partial_lines)
        _caption_height = _finals_height + _partial_height

def parse_sound_labels(msg) -> list[dict]:
    from client.sfx_categories import normalize_sfx_category

    raw = msg.get("labels")
    if isinstance(raw, list) and raw:
        out = []
        for entry in raw:
            if isinstance(entry, dict) and entry.get("label"):
                out.append(
                    {
                        "label": str(entry["label"]),
                        "category": normalize_sfx_category(entry.get("category")),
                    }
                )
            elif isinstance(entry, str) and entry.strip():
                out.append({"label": entry.strip(), "category": "ambient"})
        return out
    text = (msg.get("text") or "").strip()
    if not text:
        return []
    return [
        {"label": part.strip(), "category": "ambient"}
        for part in text.split(",")
        if part.strip()
    ]

def get_caption_lines():
    return _caption_lines

def get_caption_height():
    return _caption_height
