# ------------------------- PI SETUP INSTRUCTIONS ------------------------- 
# 1. Install dependencies:
#	sudo apt update
#	sudo apt install python3-pygame python3-gi python3-dbus
# 2. Set up a Python virtual environment and install Bluezero:
#	python3 -m venv .venv
#	source .venv/bin/activate
#	pip install --upgrade pip
#	pip3 install bluezero
# 2. Run this script on your Raspberry Pi:
#	source .venv/bin/activate
#	python3 main.py
# ------------------------- PI SETUP INSTRUCTIONS ------------------------- 


import sys
import threading
import pygame

from bluezero import adapter
from bluezero import peripheral
from bluezero import device

# ---------------- BLE UUIDs ----------------
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
RX_CHARACTERISTIC_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
MAX_DISPLAY_CHARS = 150

# ---------------- Shared text state ----------------
caption_lock = threading.Lock()
current_caption = "Waiting for BLE text..."


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


# ---------------- Pygame setup ----------------
pygame.init()
screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)
pygame.display.set_caption("Display Text in Pygame")
font = pygame.font.SysFont("Sans", 32)


def render_text(text):
	return font.render(text, True, (255, 255, 255))


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
	def rx_write(cls, value, options):
		try:
			text = bytes(value).decode("utf-8")
		except Exception:
			text = str(bytes(value))

		print("Received:", text)
		set_caption(text)


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
		uuid=RX_CHARACTERISTIC_UUID,
		value=[],
		notifying=False,
		flags=["write", "write-without-response"],
		write_callback=BLETextReceiver.rx_write,
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

		screen.fill((0, 0, 0))

		line_height = font.get_linesize()
		total_height = len(lines) * line_height
		start_y = text_y - total_height // 2

		for i, line in enumerate(lines):
			text = render_text(line)
			text_rect = text.get_rect(center=(win_w // 2, int(start_y + i * line_height)))
			screen.blit(text, text_rect)

		pygame.display.flip()
		clock.tick(60)


if __name__ == "__main__":
	main()
