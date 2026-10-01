import pygame

from client import theme
from client.state import _meter_cache, state
from client.theme import ENV_COLORS

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def draw_noise_meter(screen, x, y):
    tel = state["telemetry"]
    noise_target = _clamp(tel["noise_score"], 0.0, 1.0)
    input_target = _clamp(tel["input_rms"] / 0.06, 0.0, 1.0)
    tel["display_noise"] += (noise_target - tel["display_noise"]) * 0.2
    tel["display_input"] += (input_target - tel["display_input"]) * 0.28
    cap_target = 1.0 if tel["capturing"] else 0.0
    tel["display_capture"] += (cap_target - tel["display_capture"]) * 0.35

    level = max(tel["display_noise"], tel["display_input"])
    color = ENV_COLORS.get(tel.get("env", "moderate"), (140, 180, 140))
    if tel["display_capture"] > 0.35:
        color = (
            int(color[0] + 40 * tel["display_capture"]),
            int(color[1] + 30 * tel["display_capture"]),
            int(color[2] + 10 * tel["display_capture"]),
        )
    inner_r = 8
    outer_r = inner_r + 5 + level * 30
    size = int(outer_r * 2 + 6)
    center = int(outer_r + 3)
    if _meter_cache["size"] != size or _meter_cache["surf"] is None:
        _meter_cache["surf"] = pygame.Surface((size, size), pygame.SRCALPHA)
        _meter_cache["size"] = size
    pulse = _meter_cache["surf"]
    pulse.fill((0, 0, 0, 0))
    alpha = int(40 + 50 * tel["display_input"])
    pygame.draw.circle(pulse, (*color, alpha), (center, center), int(outer_r))
    if tel["display_capture"] > 0.05:
        ring_a = int(70 * tel["display_capture"])
        pygame.draw.circle(
            pulse, (180, 230, 180, ring_a), (center, center), int(outer_r + 5), 2
        )
    screen.blit(pulse, (x - center, y - center))
    inner_color = (
        int(120 + 100 * tel["display_capture"]),
        int(170 + 60 * tel["display_capture"]),
        int(120 + 40 * tel["display_capture"]),
    )
    pygame.draw.circle(screen, inner_color, (x, y), inner_r)
    pygame.draw.circle(screen, (235, 235, 235), (x, y), inner_r, 1)
    status = "Capturing" if tel["display_capture"] > 0.5 else "Listening"
    screen.blit(theme.hint_font.render(status, True, inner_color), (x + inner_r + 10, y - 8))
    screen.blit(
        theme.hint_font.render(
            f"in {tel['display_input'] * 100:.0f}%  vad {tel.get('speech_prob', 0):.2f}",
            True,
            (150, 150, 150),
        ),
        (x + inner_r + 10, y + 4),
    )
