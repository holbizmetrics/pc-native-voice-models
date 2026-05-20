#!/usr/bin/env python3
"""Streaming Kokoro — prove time-to-first-audio stays ~1s for long text via chunked generation.

The bench_kokoro.py result showed: short utterances generate in ~1s (PASS), but a long
paragraph's TOTAL generation is 6.67s (over the 1.5s target). Since RTF is ~0.48 (generation
is ~2x faster than playback), streaming fixes this: split text into sentences, generate
sentence 1 and start playing it (~1s), generate the rest while sentence 1 plays. The generator
stays ahead of playback, so the user hears audio ~1s after request regardless of total length.

This script measures time-to-first-audio (TTFA) for a long paragraph and confirms it ~= the
short-utterance latency, validating the streaming approach.

Usage:
    .venv/Scripts/python.exe benchmarks/stream_kokoro.py
    .venv/Scripts/python.exe benchmarks/stream_kokoro.py --play   # play gaplessly through speakers
"""
from __future__ import annotations

import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "kokoro-v1.0.onnx"
VOICES = REPO / "models" / "voices-v1.0.bin"
OUT_DIR = REPO / "benchmarks" / "out"
VOICE = "af_sarah"

LONG_TEXT = (
    "Voice synthesis on consumer hardware has come a long way. "
    "The question is no longer whether it can run locally, but whether it can match the naturalness. "
    "It must also match the timing, and the emotional range that people expect from a real conversation. "
    "If the first words arrive quickly, the rest can stream in behind them without anyone noticing the seams. "
    "That is the whole trick: start fast, then keep up with the speed of speech."
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def main(argv: list[str]) -> None:
    play = "--play" in argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL.is_file() or not VOICES.is_file():
        sys.exit(f"Model files missing:\n  {MODEL}\n  {VOICES}")

    print("Loading Kokoro (ONNX, CPU)...")
    from kokoro_onnx import Kokoro
    kokoro = Kokoro(str(MODEL), str(VOICES))
    kokoro.create("Warming up.", voice=VOICE, speed=1.0, lang="en-us")  # warm-up, excluded

    sentences = split_sentences(LONG_TEXT)
    print(f"\nLong text split into {len(sentences)} sentences.")
    print(f"Total text length: {len(LONG_TEXT)} chars\n")

    # Producer/consumer: producer generates each sentence and timestamps it; consumer
    # collects in order. We measure TTFA = time from request-start to first chunk ready.
    chunk_q: "queue.Queue" = queue.Queue()
    request_start = time.perf_counter()
    timings = []

    def producer():
        for i, sent in enumerate(sentences):
            t0 = time.perf_counter()
            samples, sr = kokoro.create(sent, voice=VOICE, speed=1.0, lang="en-us")
            ready_at = time.perf_counter() - request_start
            chunk_q.put((i, samples, sr, ready_at))
            timings.append((i, len(samples) / sr, time.perf_counter() - t0, ready_at))
        chunk_q.put(None)  # sentinel

    prod = threading.Thread(target=producer, daemon=True)
    prod.start()

    all_samples = []
    sample_rate = 24000
    ttfa = None
    play_cursor = 0.0  # cumulative audio seconds queued for playback

    while True:
        item = chunk_q.get()
        if item is None:
            break
        i, samples, sr, ready_at = item
        sample_rate = sr
        if ttfa is None:
            ttfa = ready_at
            print(f"*** TIME TO FIRST AUDIO: {ttfa:.2f}s ***\n")
        all_samples.append(samples)
        audio_len = len(samples) / sr
        # "Behind by" = does the chunk arrive before the playhead would need it?
        behind = ready_at - play_cursor
        status = "ahead-of-playback" if ready_at <= play_cursor or i == 0 else f"+{behind:.2f}s late"
        print(f"sentence {i}: ready@{ready_at:.2f}s  audio={audio_len:.2f}s  ({status})")
        play_cursor += audio_len

    prod.join()
    full = np.concatenate(all_samples)
    total_audio = len(full) / sample_rate
    total_wall = time.perf_counter() - request_start

    out_path = OUT_DIR / "kokoro_stream_long.wav"
    sf.write(str(out_path), full, sample_rate)

    print("\n" + "=" * 60)
    print(f"Sentences:            {len(sentences)}")
    print(f"Time to first audio:  {ttfa:.2f}s   (target <= 1.5s CPU)  -> {'PASS' if ttfa <= 1.5 else 'OVER'}")
    print(f"Total audio duration: {total_audio:.2f}s")
    print(f"Total wall time:      {total_wall:.2f}s")
    print(f"Streaming advantage:  user hears audio at {ttfa:.2f}s instead of {total_wall:.2f}s "
          f"({total_wall - ttfa:.2f}s saved before first sound)")
    # Did the generator keep up? After sentence 0, each chunk should be ready before its play slot.
    kept_up = all(ready <= max(0.01, sum(t[1] for t in timings[:idx])) + ttfa
                  for idx, (_, _, _, ready) in enumerate(timings) if idx > 0)
    print(f"Generator kept ahead of playback: {'YES (gapless possible)' if kept_up else 'NO (would stutter)'}")
    print(f"WAV: {out_path.relative_to(REPO)}")

    if play:
        import sounddevice as sd
        sd.play(full, sample_rate)
        sd.wait()


if __name__ == "__main__":
    main(sys.argv[1:])
