import asyncio
import websockets
import json
import pyaudio

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
CHUNK = 512


async def send_audio(websocket):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )
    try:
        while True:
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
            # overwrites the same line with partial updates until final is received
            print(f"\r\033[90mPartial: {msg['text']}\033[0m", end="", flush=True)
        elif msg["type"] == "final":
            print(f"\r\033[92mFinal: {msg['text']}\033[0m" + " " * 20)
        elif msg["type"] == "sound":
            print(f"\n[{msg['sound'].upper()}]")


async def main():
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket. Start speaking!")
        await asyncio.gather(send_audio(websocket), receive_text(websocket))


asyncio.run(main())
