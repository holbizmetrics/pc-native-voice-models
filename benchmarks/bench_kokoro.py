#!/usr/bin/env python3
"""Benchmark Kokoro 82M (ONNX, CPU) — latency + RTF against the SCOPE-DECISION targets.

Measures, for each test sentence:
- generation wall-time (= time-to-first-audio for non-streaming create())
- audio duration
- RTF (real-time factor = gen_time / audio_duration; <1.0 means faster than realtime)

Target (per docs/SCOPE-DECISION.md): TTS time-to-first-audio <= 1.5s in CPU-only mode.

Usage:
    .venv/Scripts/python.exe benchmarks/bench_kokoro.py
    .venv/Scripts/python.exe benchmarks/bench_kokoro.py --play   # also play through speakers
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "models" / "kokoro-v1.0.onnx"
VOICES = REPO / "models" / "voices-v1.0.bin"
OUT_DIR = REPO / "benchmarks" / "out"
VOICE = "af_sarah"

# Test sentences: short (latency-critical), medium, long, and one with implied non-verbal
# (Kokoro won't laugh — included to confirm it just reads the bracket literally or skips).
SENTENCES = {
    "short":  "Hello, how can I help you today?",
    "medium": "The quick brown fox jumps over the lazy dog while the sun sets behind the mountains.",
    "long":   "Voice synthesis on consumer hardware has come a long way. The question is no longer "
              "whether it can run locally, but whether it can match the naturalness, the timing, and "
              "the emotional range that people expect from a real conversation partner.",
    "nonverbal_probe": "That's hilarious [laughs] I can't believe it actually worked.",
}


def main(argv: list[str]) -> None:
    play = "--play" in argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL.is_file() or not VOICES.is_file():
        sys.exit(f"Model files missing. Expected:\n  {MODEL}\n  {VOICES}")

    print("Loading Kokoro (ONNX, CPU)...")
    t0 = time.perf_counter()
    from kokoro_onnx import Kokoro
    kokoro = Kokoro(str(MODEL), str(VOICES))
    load_time = time.perf_counter() - t0
    print(f"Model load time: {load_time:.2f}s  (one-time, not per-utterance)\n")

    # Warm-up run (first inference includes graph optimization; not representative)
    print("Warm-up run (excluded from results)...")
    kokoro.create("Warming up.", voice=VOICE, speed=1.0, lang="en-us")
    print("Warm-up done.\n")

    print(f"{'case':<18} {'gen_s':>8} {'audio_s':>8} {'RTF':>6} {'target':>8} {'verdict':>8}")
    print("-" * 64)

    results = {}
    for name, text in SENTENCES.items():
        t0 = time.perf_counter()
        samples, sample_rate = kokoro.create(text, voice=VOICE, speed=1.0, lang="en-us")
        gen_time = time.perf_counter() - t0
        audio_dur = len(samples) / sample_rate
        rtf = gen_time / audio_dur if audio_dur > 0 else float("nan")
        # Target relevance: 'short' is the latency-critical case for time-to-first-audio
        target = 1.5
        verdict = "PASS" if gen_time <= target else "OVER"
        results[name] = {"gen": gen_time, "audio": audio_dur, "rtf": rtf}
        print(f"{name:<18} {gen_time:>7.2f}s {audio_dur:>7.2f}s {rtf:>6.2f} {target:>7.1f}s {verdict:>8}")

        out_path = OUT_DIR / f"kokoro_{name}.wav"
        sf.write(str(out_path), samples, sample_rate)
        if play:
            import sounddevice as sd
            sd.play(samples, sample_rate)
            sd.wait()

    print("-" * 64)
    short = results["short"]
    print(f"\nLatency-critical case (short utterance): {short['gen']:.2f}s "
          f"vs {1.5:.1f}s CPU target -> {'PASS' if short['gen'] <= 1.5 else 'OVER'}")
    print(f"Throughput (RTF) on long utterance: {results['long']['rtf']:.2f} "
          f"({'faster' if results['long']['rtf'] < 1 else 'slower'} than realtime)")
    print(f"\nWAVs written to {OUT_DIR.relative_to(REPO)}/")
    print("Listen to kokoro_nonverbal_probe.wav to hear how Kokoro handles a [laughs] marker "
          "(expected: reads it literally or skips — confirms we need a separate non-verbal engine).")


if __name__ == "__main__":
    main(sys.argv[1:])
