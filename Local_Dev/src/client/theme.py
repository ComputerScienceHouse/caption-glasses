import pygame
from client.sfx_categories import fa_solid_font_path

WIDTH, HEIGHT = 1100, 620

SPEAKER_COLORS = {
    "SPEAKER_00": (240, 240, 240),
    "SPEAKER_01": (255, 223, 130),
    "SPEAKER_02": (163, 255, 177),
    "SPEAKER_03": (177, 163, 255),
    "SPEAKER_04": (255, 163, 177),
}

ENV_COLORS = {
    "quiet": (72, 195, 118),
    "moderate": (218, 186, 64),
    "loud": (228, 98, 78),
}

SFX_CATEGORY_STYLES = {
    "human": {"bg": (78, 58, 108), "fg": (238, 220, 255)},
    "animal": {"bg": (38, 92, 58), "fg": (180, 245, 190)},
    "music": {"bg": (110, 42, 88), "fg": (255, 190, 235)},
    "natural": {"bg": (32, 88, 72), "fg": (170, 240, 210)},
    "vehicle": {"bg": (48, 68, 98), "fg": (200, 220, 255)},
    "domestic": {"bg": (98, 72, 38), "fg": (255, 225, 170)},
    "tools": {"bg": (70, 70, 78), "fg": (220, 220, 230)},
    "explosive": {"bg": (120, 48, 38), "fg": (255, 190, 170)},
    "tones": {"bg": (88, 58, 48), "fg": (255, 210, 180)},
    "ambient": {"bg": (58, 58, 68), "fg": (210, 210, 220)},
    "misc": {"bg": (58, 58, 68), "fg": (210, 210, 220)},
}
SFX_DEFAULT_STYLE = SFX_CATEGORY_STYLES["ambient"]

screen: pygame.Surface
font: pygame.font.Font
small_font: pygame.font.Font
ui_font: pygame.font.Font
hint_font: pygame.font.Font
sfx_font: pygame.font.Font

def init_pygame() -> None:
    global screen, font, small_font, ui_font, hint_font, sfx_font
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Transcription Display")
    font = pygame.font.SysFont("arial", 28)
    small_font = pygame.font.SysFont("arial", 18)
    ui_font = pygame.font.SysFont("arial", 15)
    hint_font = pygame.font.SysFont("arial", 13)
    fa_path = fa_solid_font_path()
    if fa_path.is_file():
        sfx_font = pygame.font.Font(str(fa_path), 18)
    else:
        sfx_font = pygame.font.SysFont("arial", 18, bold=True)

def resize_window(w: int, h: int) -> None:
    global WIDTH, HEIGHT, screen
    WIDTH, HEIGHT = w, h
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
