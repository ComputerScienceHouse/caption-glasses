import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import sys

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform


def main():
    parser = argparse.ArgumentParser(
        description="Real-time mic transcription with SpeechRecognition + whisper, now word-by-word & faster."
    )
    parser.add_argument(
        "--model",
        default="tiny",  # default to "tiny" for speed
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model to use.",
    )
    parser.add_argument(
        "--non_english",
        action="store_true",
        help="Don't force the English model if it's smaller than 'large'.",
    )
    parser.add_argument(
        "--energy_threshold",
        default=1000,
        type=int,
        help="Mic detection threshold (SpeechRecognition).",
    )
    parser.add_argument(
        "--record_timeout",
        default=1.0,
        type=float,
        help="Seconds of audio after which we process a chunk.",
    )
    parser.add_argument(
        "--phrase_timeout",
        default=1.0,
        type=float,
        help="Silence gap (sec) between chunks => new line in transcription.",
    )
    if "linux" in platform:
        parser.add_argument(
            "--default_microphone",
            default="pulse",
            type=str,
            help="Mic name on Linux. Use 'list' to list devices.",
        )
    args = parser.parse_args()

    # SpeechRecognition setup
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Microphone selection
    if "linux" in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == "list":
            print("Available microphones:")
            for idx, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"  {idx}: {name}")
            return
        else:
            found_idx = None
            for idx, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    found_idx = idx
                    break
            if found_idx is None:
                print(f"ERROR: Microphone '{mic_name}' not found.")
                return
            source = sr.Microphone(sample_rate=16000, device_index=found_idx)
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load whisper model
    model_str = args.model
    if args.model != "large" and args.model != "turbo" and not args.non_english:
        model_str += ".en"
    print(f"Loading whisper model: {model_str} ...")
    audio_model = whisper.load_model(model_str)
    print("Whisper model loaded.\n")

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    # We'll store each chunk's recognized words in separate lines
    # if there's a big pause
    lines = []  # each item is a list of words
    lines.append([])  # start with empty list of words

    phrase_time = None
    data_queue = Queue()

    # Callback function when a chunk is done
    def record_callback(_, audio: sr.AudioData):
        data = audio.get_raw_data()
        data_queue.put(data)

    with source:
        print("Calibrating mic for ambient noise...")
        recorder.adjust_for_ambient_noise(source, duration=1)
        print("Calibration complete. Starting background listening...")

    # Start background thread
    recorder.listen_in_background(
        source,
        record_callback,
        # phrase_time_limit=record_timeout
    )

    print("Recording... Speak!\n")
    print(recorder.pause_threshold)
    while True:
        try:
            # now = datetime.now()
            if not data_queue.empty():
                # chunk_gap = False
                # if phrase_time and (now - phrase_time) > timedelta(seconds=phrase_timeout):
                #     chunk_gap = True
                # phrase_time = now

                # Gather raw audio data
                audio_data = b"".join(list(data_queue.queue))
                data_queue.queue.clear()

                audio_np = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                # Try GPU if available
                use_fp16 = torch.cuda.is_available()
                # Transcribe with word timestamps
                result = audio_model.transcribe(
                    audio_np,
                    fp16=use_fp16,
                    beam_size=1,  # faster
                    word_timestamps=True,  # key for word-by-word
                )

                # If there's a big pause => new line of words
                # if chunk_gap:
                #     lines.append([])

                # Extract each recognized word from segments
                # and append to the last line in 'lines'
                segments = result["segments"]
                for seg in segments:
                    for w in seg["words"]:
                        word = w["word"]  # Adjust to the actual key in your data
                        print(word, end="", flush=True)

                        lines[-1].append(word)

                # Clear screen & re-print everything word-by-word
                os.system("cls" if os.name == "nt" else "clear")
                for word_list in lines:
                    # Join words with a space
                    # sys.stdout.write("\033[F")
                    # sys.stdout.write("\033[K")
                    print("".join(word_list))

                print("", end="", flush=True)

            else:
                # Avoid burning CPU in an infinite loop
                sleep(0.25)

        except KeyboardInterrupt:
            break

    print("\n\nFinal Transcript:")
    for i, word_list in enumerate(lines):
        print(f"Line {i + 1}: {''.join(word_list)}")


if __name__ == "__main__":
    main()
