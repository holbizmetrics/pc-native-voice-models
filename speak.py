#!/usr/bin/env python3
"""speak.py — pc-native-voice-models v1: type text, hear it.

Talk-only TTS over Kokoro (ONNX, CPU). Streams sentence-by-sentence so the
first words start ~1s after you hit enter, regardless of total length (the
generator runs ~2x faster than playback, so it stays ahead — gapless).

Usage:
    python speak.py "Hello, this is my own voice model."
    python speak.py "..." --voice af_bella --speed 1.1
    python speak.py --file story.txt
    echo "piped text works too" | python speak.py
    python speak.py "save instead of play" --save out.wav
    python speak.py --list-voices
    python speak.py -h        # full help with examples

Voice library: 54 voices, 8 languages (Kokoro). --list-voices to see them.
Model files expected at models/kokoro-v1.0.onnx + models/voices-v1.0.bin.
"""
from __future__ import annotations

import argparse
import queue
import re
import sys
import threading
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
MODEL = REPO / "models" / "kokoro-v1.0.onnx"
VOICES = REPO / "models" / "voices-v1.0.bin"
DEFAULT_VOICE = "af_sarah"
DEFAULT_LANG = "en-us"

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def load_kokoro():
    if not MODEL.is_file() or not VOICES.is_file():
        sys.exit(f"Model files missing:\n  {MODEL}\n  {VOICES}\n"
                 f"(download per README: kokoro-v1.0.onnx + voices-v1.0.bin)")
    from kokoro_onnx import Kokoro
    return Kokoro(str(MODEL), str(VOICES))


def list_voices(kokoro) -> list[str]:
    # kokoro-onnx exposes voices differently across versions; try the common paths.
    for attr in ("get_voices", "voices"):
        obj = getattr(kokoro, attr, None)
        if callable(obj):
            try:
                return sorted(obj())
            except Exception:
                pass
        elif obj is not None:
            try:
                return sorted(obj.keys()) if hasattr(obj, "keys") else sorted(list(obj))
            except Exception:
                pass
    return []


def speak_streaming(kokoro, text: str, voice: str, speed: float, lang: str) -> None:
    """Generate sentence-by-sentence in a producer thread, play gaplessly via a
    single OutputStream whose blocking write() paces playback."""
    import sounddevice as sd

    sentences = split_sentences(text)
    chunk_q: "queue.Queue" = queue.Queue()

    def producer():
        for sent in sentences:
            try:
                samples, sr = kokoro.create(sent, voice=voice, speed=speed, lang=lang)
                chunk_q.put(("chunk", samples.astype(np.float32), sr))
            except Exception as e:
                chunk_q.put(("error", str(e)))
                break
        chunk_q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    stream = None
    try:
        while True:
            item = chunk_q.get()
            if item is None:
                break
            if item[0] == "error":
                sys.exit(f"generation error: {item[1]}")
            _, samples, sr = item
            if stream is None:
                stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
                stream.start()
            stream.write(samples)
    finally:
        if stream is not None:
            stream.stop()
            stream.close()


def speak_to_file(kokoro, text: str, voice: str, speed: float, lang: str, out_path: Path) -> None:
    import soundfile as sf
    sentences = split_sentences(text)
    all_samples = []
    sr = 24000
    for sent in sentences:
        samples, sr = kokoro.create(sent, voice=voice, speed=speed, lang=lang)
        all_samples.append(samples.astype(np.float32))
    full = np.concatenate(all_samples) if all_samples else np.zeros(0, dtype=np.float32)
    sf.write(str(out_path), full, sr)
    dur = len(full) / sr if sr else 0
    print(f"wrote {out_path} ({dur:.1f}s audio)")


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="speak.py",
        description="pc-native-voice-models v1 — talk-only TTS (Kokoro, CPU, streaming). "
                    "Type or pipe text, hear it spoken. No GPU, no network at runtime.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  speak.py "Hello, this runs entirely on my CPU."     speak inline text
  speak.py "..." --voice af_bella --speed 1.1         pick a voice, adjust speed
  speak.py --file story.txt                            speak a text file
  echo "piped text" | speak.py                         speak from stdin
  speak.py "..." --save out.wav                        write a WAV instead of playing
  speak.py --list-voices                               show all 54 voices

text source precedence: --file > inline arg > stdin.
voices: 54 across 8 languages (af_=US female, am_=US male, bf_/bm_=British, etc.).
streaming: first words start ~1s in, gapless after (generator runs ahead of playback).
""",
    )
    p.add_argument("text", nargs="?", help="text to speak (or pipe via stdin, or use --file)")
    p.add_argument("-f", "--file", metavar="PATH", help="read text to speak from a file")
    p.add_argument("--voice", default=DEFAULT_VOICE, help=f"voice name (default {DEFAULT_VOICE}; --list-voices to see all)")
    p.add_argument("--speed", type=float, default=1.0, help="speech speed multiplier (default 1.0)")
    p.add_argument("--lang", default=DEFAULT_LANG, help=f"language code (default {DEFAULT_LANG})")
    p.add_argument("--save", metavar="PATH", help="write a WAV instead of playing")
    p.add_argument("--list-voices", action="store_true", help="print available voices and exit")
    args = p.parse_args(argv)

    kokoro = load_kokoro()

    if args.list_voices:
        voices = list_voices(kokoro)
        if voices:
            print(f"{len(voices)} voices:")
            for v in voices:
                print(f"  {v}")
        else:
            print("could not enumerate voices from this kokoro-onnx version; "
                  "common ones: af_sarah, af_bella, af_heart, am_adam, am_michael, bf_emma, bm_george")
        return

    # text source precedence: --file > explicit "-" stdin > positional arg > piped stdin.
    # IMPORTANT: only read stdin when it's actually piped (not a TTY) or explicitly
    # requested with "-". Otherwise `speak.py` with no args would block forever on
    # stdin.read() in an interactive terminal (no EOF coming).
    if args.file:
        fpath = Path(args.file)
        if not fpath.is_file():
            sys.exit(f"file not found: {fpath}")
        text = fpath.read_text(encoding="utf-8")
    elif args.text == "-":
        text = sys.stdin.read()
    elif args.text is not None:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        p.error("no text to speak. Pass text as an argument, use --file PATH, "
                "or pipe via stdin. Run with -h for examples.")
    if not text.strip():
        sys.exit("no text given (pass as arg, --file PATH, or pipe via stdin)")

    if args.save:
        speak_to_file(kokoro, text, args.voice, args.speed, args.lang, Path(args.save))
    else:
        speak_streaming(kokoro, text, args.voice, args.speed, args.lang)


if __name__ == "__main__":
    main(sys.argv[1:])
