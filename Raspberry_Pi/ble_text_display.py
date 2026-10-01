# ------------------------- PI SETUP INSTRUCTIONS ------------------------- 
# 1. Install dependencies:
#	sudo apt update
#	sudo apt install python3-pygame python3-gi python3-dbus
# 2. Set up a Python virtual environment and install Bluezero:
#	python3 -m venv .venv
#	source .venv/bin/activate
#	pip install --upgrade pip
#	pip3 install bluezero
# 3. Run this script on your Raspberry Pi:
#	source .venv/bin/activate
#	python3 main.py
# ------------------------- PI SETUP INSTRUCTIONS ------------------------- 


import json
import sys
import threading
import time
from pathlib import Path

import pygame

from bluezero import adapter
from bluezero import peripheral
from bluezero import device

# ---------------- BLE UUIDs ----------------
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
CAPTION_RX_CHARACTERISTIC_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
SOUND_RX_CHARACTERISTIC_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
MAX_DISPLAY_CHARS = 150
SOUND_DISPLAY_DURATION_MS = 3500

# ---------------- SFX categories ----------------
SFX_CATEGORY_IDS = (
	"human", "animal", "music", "natural", "vehicle",
	"domestic", "tools", "explosive", "tones", "ambient",
)

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
}
SFX_DEFAULT_STYLE = SFX_CATEGORY_STYLES["ambient"]

IGNORED_SOUND_LABELS = {"silence", "speech"}


def normalize_sfx_category(category):
	if not category or category == "misc":
		return "ambient"
	if category in SFX_CATEGORY_IDS:
		return category
	return "ambient"


def parse_sound_labels(raw_text: str):
	"""
	Parse a BLE sound payload into [{"label": ..., "category": ...}, ...].
	"""
	raw_text = raw_text.strip()[:4096]
	if not raw_text:
		return []

	entries = None
	if raw_text[0] in "[{":
		try:
			decoded = json.loads(raw_text)
			if isinstance(decoded, dict):
				entries = decoded.get("labels")
				if entries is None and decoded.get("text"):
					entries = [p.strip() for p in str(decoded["text"]).split(",")]
			elif isinstance(decoded, list):
				entries = decoded
		except (ValueError, TypeError):
			entries = None

	if entries is None:
		entries = [p.strip() for p in raw_text.split(",")]

	labels = []
	for entry in entries:
		if isinstance(entry, dict) and entry.get("label"):
			label = str(entry["label"]).strip()
			category = normalize_sfx_category(entry.get("category"))
		elif isinstance(entry, str) and entry.strip():
			label = entry.strip()
			category = "ambient"
		else:
			continue
		if label.lower() in IGNORED_SOUND_LABELS:
			continue
		labels.append({"label": label[:MAX_DISPLAY_CHARS], "category": category})
	return labels


# ---------------- Shared text state ----------------
caption_lock = threading.Lock()
current_caption = "Waiting for BLE text..."
current_sound_labels = []
current_sound_timestamp_ms = 0


def set_caption(new_text: str):
	global current_caption
	new_text = new_text.strip()[:MAX_DISPLAY_CHARS]
	if not new_text:
		return
	with caption_lock:
		current_caption = new_text


def get_caption() -> str:
	with caption_lock:
		return current_caption


def set_sound_effect(new_sound: str):
	global current_sound_labels, current_sound_timestamp_ms
	labels = parse_sound_labels(new_sound)
	with caption_lock:
		current_sound_labels = labels
		if labels:
			current_sound_timestamp_ms = int(time.monotonic() * 1000)


def get_sound_labels():
	with caption_lock:
		if not current_sound_labels:
			return []
		now_ms = int(time.monotonic() * 1000)
		if now_ms - current_sound_timestamp_ms >= SOUND_DISPLAY_DURATION_MS:
			return []
		return list(current_sound_labels)


# ---------------- Pygame setup ----------------
pygame.init()
screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)
pygame.display.set_caption("Display Text in Pygame")
font = pygame.font.SysFont("Sans", 32)
sfx_label_font = pygame.font.SysFont("Sans", 26)

_fa_font_path = Path(__file__).resolve().parent / "fa-solid-900.ttf"
if _fa_font_path.is_file():
	sfx_icon_font = pygame.font.Font(str(_fa_font_path), 26)
else:
	print(f"Font Awesome font not found at {_fa_font_path}, using fallback")
	sfx_icon_font = pygame.font.SysFont("Sans", 26, bold=True)


def render_text(text):
	return font.render(text, True, (255, 255, 255))


# ---------------- SFX chips ---------------
CHIP_PAD_X = 12
CHIP_PAD_Y = 6
CHIP_GAP = 12
CHIP_ICON_GAP = 8

_chip_cache = {}


def _render_chip(label: str, category: str) -> pygame.Surface:
	key = (label, category)
	cached = _chip_cache.get(key)
	if cached is not None:
		return cached

	style = SFX_CATEGORY_STYLES.get(category, SFX_DEFAULT_STYLE)
	icon = SFX_CATEGORY_ICONS.get(category, SFX_CATEGORY_ICONS["ambient"])
	icon_surf = sfx_icon_font.render(icon, True, style["fg"])
	text_surf = sfx_label_font.render(label, True, style["fg"])
	width = CHIP_PAD_X * 2 + icon_surf.get_width() + CHIP_ICON_GAP + text_surf.get_width()
	height = max(icon_surf.get_height(), text_surf.get_height()) + CHIP_PAD_Y * 2
	chip = pygame.Surface((width, height), pygame.SRCALPHA)
	pygame.draw.rect(chip, style["bg"], chip.get_rect(), border_radius=8)
	chip.blit(icon_surf, (CHIP_PAD_X, (height - icon_surf.get_height()) // 2))
	chip.blit(
		text_surf,
		(
			CHIP_PAD_X + icon_surf.get_width() + CHIP_ICON_GAP,
			(height - text_surf.get_height()) // 2,
		),
	)
	_chip_cache[key] = chip
	if len(_chip_cache) > 64:
		_chip_cache.clear()
		_chip_cache[key] = chip
	return chip


def draw_sound_chips(surface: pygame.Surface, labels, center_x: int, bottom_y: int, max_width: int) -> int:
	"""
	Draw SFX chips in a row centered on center_x with bottoms at bottom_y.
	Chips that would push the row past max_width are dropped (labels arrive
	sorted by score, so the strongest detections survive).
	Returns the total height used (0 if nothing drawn).
	"""
	if not labels:
		return 0
	chips = []
	total_width = 0
	for item in labels:
		chip = _render_chip(item["label"], item["category"])
		added = chip.get_width() + (CHIP_GAP if chips else 0)
		if chips and total_width + added > max_width:
			break
		chips.append(chip)
		total_width += added
	row_height = max(c.get_height() for c in chips)
	cursor_x = center_x - total_width // 2
	for chip in chips:
		surface.blit(chip, (cursor_x, bottom_y - chip.get_height()))
		cursor_x += chip.get_width() + CHIP_GAP
	return row_height


def wrap_text(text, font, max_width):
	words = text.split()
	if not words:
		return [""]

	lines = []
	current_line = words[0]

	for word in words[1:]:
		test_line = current_line + " " + word
		if font.size(test_line)[0] <= max_width:
			current_line = test_line
		else:
			lines.append(current_line)
			current_line = word

	lines.append(current_line)
	return lines


# ---------------- BLE ----------------
class BLETextReceiver:
	@classmethod
	def on_connect(cls, ble_device: device.Device):
		print(f"Phone connected: {ble_device.address}")

	@classmethod
	def on_disconnect(cls, adapter_address, device_address):
		print(f"Phone disconnected: {device_address}")

	@classmethod
	def caption_rx_write(cls, value, options):
		try:
			text = bytes(value).decode("utf-8")
		except Exception:
			text = str(bytes(value))

		# print("Caption received:", text)
		set_caption(text)

	@classmethod
	def sound_rx_write(cls, value, options):
		try:
			text = bytes(value).decode("utf-8")
		except Exception:
			text = str(bytes(value))

		# print("Sound effect received:", text)
		set_sound_effect(text)


def start_ble():
	"""
	Start BLE peripheral in a background thread to receive text from the phone and update the caption.
	"""
	adapters = list(adapter.Adapter.available())
	if not adapters:
		raise RuntimeError("No Bluetooth adapter found")

	ble_uart = peripheral.Peripheral(
		adapters[0].address,
		local_name="CGPI"
	)

	ble_uart.add_service(
		srv_id=1,
		uuid=UART_SERVICE_UUID,
		primary=True
	)

	ble_uart.add_characteristic(
		srv_id=1,
		chr_id=1,
		uuid=CAPTION_RX_CHARACTERISTIC_UUID,
		value=[],
		notifying=False,
		flags=["write", "write-without-response"],
		write_callback=BLETextReceiver.caption_rx_write,
		read_callback=None,
		notify_callback=None
	)

	ble_uart.add_characteristic(
		srv_id=1,
		chr_id=2,
		uuid=SOUND_RX_CHARACTERISTIC_UUID,
		value=[],
		notifying=False,
		flags=["write", "write-without-response"],
		write_callback=BLETextReceiver.sound_rx_write,
		read_callback=None,
		notify_callback=None
	)

	ble_uart.on_connect = BLETextReceiver.on_connect
	ble_uart.on_disconnect = BLETextReceiver.on_disconnect

	print("Advertising as CGPI")
	ble_uart.publish()


# ---------------- Main loop ----------------
def main():
	ble_thread = threading.Thread(target=start_ble, daemon=True)
	ble_thread.start()

	clock = pygame.time.Clock()

	while True:
		win_w, win_h = pygame.display.get_window_size()
		text_y = win_h - (win_h / 6)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()

		caption = get_caption()
		lines = wrap_text(caption, font, win_w - 80)
		sound_labels = get_sound_labels()

		screen.fill((0, 0, 0))

		line_height = font.get_linesize()
		total_height = len(lines) * line_height
		start_y = text_y - total_height // 2

		if sound_labels:
			chips_bottom_y = int(start_y - line_height // 2)
			draw_sound_chips(screen, sound_labels, win_w // 2, chips_bottom_y, win_w - 80)

		for i, line in enumerate(lines):
			text = render_text(line)
			text_rect = text.get_rect(center=(win_w // 2, int(start_y + i * line_height)))
			screen.blit(text, text_rect)

		pygame.display.flip()
		clock.tick(60)


if __name__ == "__main__":
	main()
