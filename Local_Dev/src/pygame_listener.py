import asyncio
import websockets
import json
import pyaudio
import pygame
import sys
import argparse
import numpy as np
from scipy.signal import resample_poly
from config import DEVICE_CAPTURE_RATE, WEBSOCKET_URI

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
WIDTH, HEIGHT = 900, 500  
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Transcription Display")
font = pygame.font.SysFont("arial", 28)
small_font = pygame.font.SysFont("arial", 20)

SPEAKER_COLORS = {
    "SPEAKER_00": (240, 240, 240),
    "SPEAKER_01": (255, 223, 130),
    "SPEAKER_02": (163, 255, 177),
    "SPEAKER_03": (177, 163, 255),
    "SPEAKER_04": (255, 163, 177),
}

state = {
    "finals": [],
    "partial": {"text": "", "speaker": ""},
    "sound": "",
    "sound_timestamp": 0,
    "translate_mode": False,
    "speaker_names": {},
    
}

SIDEBAR_WIDTH = 180
MAX_SENTENCE_HISTORY = 50
SOUND_DISPLAY_DURATION = 2000


def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        if font.size(test_line)[0] <= max_width:
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
        format=FORMAT,
        channels=CHANNELS,
        rate=DEVICE_CAPTURE_RATE,
        input=True,
        frames_per_buffer=4096,
    )
    try:
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.float32)
            resampled = resample_poly(audio, 160, DEVICE_CAPTURE_RATE/100).astype(np.float32)
            await websocket.send(resampled.tobytes())
            await asyncio.sleep(0.001)
    finally:
        stream.close()
        p.terminate()


async def receive_text(websocket):
    while True:
        try:
            data = await websocket.recv()
            msg = json.loads(data)
            if msg["type"] == "partial":
                state["partial"] = {"text": msg["text"], "speaker": msg.get("speaker", "SPEAKER_00")}
            elif msg["type"] == "final":
                state["finals"].append({"text": msg["text"], "speaker": msg.get("speaker", "SPEAKER_00")})
                state["partial"] = {"text": "", "speaker": ""}
                if len(state["finals"]) > MAX_SENTENCE_HISTORY:
                    state["finals"].pop(0)
            elif msg["type"] == "sound":
                state["sound"] = msg["text"]
                state["sound_timestamp"] = pygame.time.get_ticks()
        except Exception:
            break


async def pygame_loop(websocket):
    global WIDTH, HEIGHT, screen
    line_height = 35
    scroll_y = 0
    auto_scroll = True

    while True:
        curr_width, curr_height = screen.get_size()
        mouse_pos = pygame.mouse.get_pos()
        btn_rect = pygame.Rect(curr_width - SIDEBAR_WIDTH + 10, 20, SIDEBAR_WIDTH - 20, 50)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = event.size
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if btn_rect.collidepoint(event.pos):
                    state["translate_mode"] = not state["translate_mode"]
                    task = "translate" if state["translate_mode"] else "transcribe"
                    await websocket.send(json.dumps({"type": "set_task", "value": task}))
            elif event.type == pygame.MOUSEWHEEL:
                auto_scroll = False
                scroll_y -= event.y * 30

        screen.fill((18, 18, 18))

        pygame.draw.rect(screen, (35, 35, 35), (curr_width - SIDEBAR_WIDTH, 0, SIDEBAR_WIDTH, curr_height))
        
        btn_color = (40, 160, 40) if state["translate_mode"] else (70, 70, 70)
        if btn_rect.collidepoint(mouse_pos):
            btn_color = tuple(min(255, c + 30) for c in btn_color)
        
        pygame.draw.rect(screen, btn_color, btn_rect, border_radius=5)
        
        label = "TRANSLATING" if state["translate_mode"] else "CAPTIONING"
        label_surf = small_font.render(label, True, (255, 255, 255))
        screen.blit(label_surf, (btn_rect.centerx - label_surf.get_width()//2, 
                                 btn_rect.centery - label_surf.get_height()//2))

        max_text_width = curr_width - SIDEBAR_WIDTH - 40
        layout = []
        current_y = 20
        last_speaker = None

        all_items = state["finals"] + ([state["partial"]] if state["partial"]["text"] else [])

        for item in all_items:
            text_to_wrap = item["text"] + ("..." if item == state["partial"] else "")
            wrapped = wrap_text(text_to_wrap, font, max_text_width)
            color = SPEAKER_COLORS.get(item["speaker"], (240, 240, 240))

            for i, line in enumerate(wrapped):
                prefix = ""
                if args.mode == "label" and i == 0 and item["speaker"] != last_speaker:
                    prefix = f"[{item['speaker'].replace('SPEAKER_', 'Speaker ')}] "
                
                layout.append({"text": prefix + line, "color": color, "absolute_y": current_y})
                current_y += line_height
            last_speaker = item["speaker"]

        if auto_scroll:
            scroll_y = max(0, current_y - (curr_height - 100))

        text_area_rect = pygame.Rect(0, 0, curr_width - SIDEBAR_WIDTH, curr_height - 80)
        screen.set_clip(text_area_rect)
        for item in layout:
            draw_y = item["absolute_y"] - scroll_y
            if -40 < draw_y < curr_height:
                screen.blit(font.render(item["text"], True, item["color"]), (20, draw_y))
        screen.set_clip(None)

        if state["sound"] and pygame.time.get_ticks() - state["sound_timestamp"] < SOUND_DISPLAY_DURATION:
            s_surf = font.render(f"({state['sound'].upper()})", True, (120, 180, 255))
            screen.blit(s_surf, (20, curr_height - 50))

        pygame.display.flip()
        await asyncio.sleep(0.01)


async def main():
    uri = WEBSOCKET_URI
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket.")
            await asyncio.gather(
                send_audio(websocket),
                receive_text(websocket),
                pygame_loop(websocket),
                return_exceptions=True,
            )
    except Exception as e:
        print(f"Connection Error: {e}")
        pygame.quit()


if __name__ == "__main__":
    asyncio.run(main())