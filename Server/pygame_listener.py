import asyncio
import websockets
import json
import pyaudio
import pygame
import sys
import argparse

parser = argparse.ArgumentParser(description="Transcription Display")
parser.add_argument(
    "--mode",
    choices=["label", "color"],
    default="label",
    help="Display mode: 'label' shows [Speaker], 'color' relies on text color.",
)
args = parser.parse_args()

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
CHUNK = 2048

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE | pygame.SCALED)
pygame.display.set_caption("Transcription Display")
font = pygame.font.SysFont("arial", 32)

SPEAKER_COLORS = {
    "[Speaker 0]": (240, 240, 240),
    "[Speaker 1]": (255, 223, 130),
    "[Speaker 2]": (163, 255, 177),
    "[Speaker 3]": (177, 163, 255),
    "[Speaker 4]": (255, 163, 177),
}

state = {
    "finals": [],
    "partial": {"text": "", "speaker": ""},
    "sound": "",
    "sound_timestamp": 0,
}

MAX_SENTENCE_HISTORY = 100
SOUND_DISPLAY_DURATION = 2000


def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        width, _ = font.size(test_line)

        if width <= max_width:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines


async def send_audio(websocket):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )
    try:
        while True:
            available = stream.get_read_available()
            if available > CHUNK:
                stream.read(available - CHUNK, exception_on_overflow=False)

            data = stream.read(CHUNK, exception_on_overflow=False)
            await websocket.send(data)
            await asyncio.sleep(0.001)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


async def receive_text(websocket):
    while True:
        data = await websocket.recv()
        msg = json.loads(data)

        if msg["type"] == "partial":
            state["partial"] = {
                "text": msg["text"],
                "speaker": msg.get("speaker", "SPEAKER_00"),
            }

        elif msg["type"] == "final":
            state["finals"].append(
                {"text": msg["text"], "speaker": msg.get("speaker", "SPEAKER_00")}
            )
            state["partial"] = {"text": "", "speaker": ""}

            if len(state["finals"]) > MAX_SENTENCE_HISTORY:
                state["finals"].pop(0)

        elif msg["type"] == "sound":
            state["sound"] = msg["sound"]
            state["sound_timestamp"] = pygame.time.get_ticks()


async def pygame_loop():
    global WIDTH, HEIGHT, screen
    line_height = 40
    scroll_y = 0
    auto_scroll = True

    while True:
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEWHEEL:
                auto_scroll = False
                scroll_y -= event.y * 30

        screen.fill((15, 15, 15))
        max_text_width = WIDTH - 40

        layout = []
        current_y = 20
        last_speaker = None

        all_items = list(state["finals"])
        if state["partial"]["text"]:
            all_items.append(state["partial"])

        for item in all_items:
            if not item["text"]:
                continue

            text_to_wrap = item["text"] + ("..." if item == state["partial"] else "")
            wrapped = wrap_text(text_to_wrap, font, max_text_width)

            current_speaker = item["speaker"]
            display_name = current_speaker.replace("SPEAKER_", "Speaker ").replace(
                "speaker", "Speaker "
            )
            color = SPEAKER_COLORS.get(current_speaker, (240, 240, 240))

            for i, line in enumerate(wrapped):
                prefix = ""
                if args.mode == "label":
                    if i == 0:
                        if current_speaker != last_speaker and last_speaker is not None:
                            current_y += 15
                        if current_speaker != last_speaker:
                            prefix = f"[{display_name}] "

                layout.append(
                    {"text": prefix + line, "color": color, "absolute_y": current_y}
                )
                current_y += line_height

            last_speaker = current_speaker

        total_content_height = current_y
        usable_height = HEIGHT - 80

        max_scroll = max(0, total_content_height - usable_height)

        if auto_scroll:
            scroll_y = max_scroll
        else:
            scroll_y = max(0, min(scroll_y, max_scroll))
            if scroll_y == max_scroll:
                auto_scroll = True

        text_rect = pygame.Rect(0, 0, WIDTH, usable_height)
        screen.set_clip(text_rect)

        for item in layout:
            draw_y = item["absolute_y"] - scroll_y
            if draw_y > -line_height and draw_y < usable_height:
                text_surface = font.render(item["text"], True, item["color"])
                screen.blit(text_surface, (20, draw_y))

        screen.set_clip(None)

        if state["sound"] and (
            current_time - state["sound_timestamp"] > SOUND_DISPLAY_DURATION
        ):
            state["sound"] = ""

        if state["sound"]:
            sound_surface = font.render(
                f"[{state['sound'].upper()}]", True, (100, 150, 255)
            )
            screen.blit(sound_surface, (20, HEIGHT - 50))

        pygame.display.flip()
        await asyncio.sleep(0.01)


async def main():
    uri = "wss://{website.com}/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket. Launching Pygame UI...")
            await asyncio.gather(
                send_audio(websocket), receive_text(websocket), pygame_loop()
            )
    except Exception as e:
        print(f"Connection Error: {e}")
        pygame.quit()


if __name__ == "__main__":
    asyncio.run(main())
