#!/usr/bin/env python3
"""Sesame CSM-1B spike test — does it laugh? + CPU latency.

CSM generates non-verbal cues (laughs, breath) from conversational CONTEXT and
the acoustic-token stream, NOT from [laugh] bracket markup (that's Bark). So we
try a few elicitation strategies and save each as a WAV for the operator to judge.

Full-precision (float32) on CPU — the gold-standard quality test. Slow on CPU
(1B model); that's expected. We're testing "does it laugh well?" not production
latency here.

Usage:
    HF_HOME=D:/FromGitHubEtc/pc-native-voice-models/models/hf-cache \
      .venv/Scripts/python.exe benchmarks/spike_csm.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "benchmarks" / "out"
MODEL_ID = "sesame/csm-1b"
SPEAKER = "0"

# Elicitation strategies for non-verbal / laughter:
VARIANTS = {
    "plain":            "Hello, how can I help you today?",
    "laughter_text":    "That is so funny, hahaha! I can't believe it actually worked.",
    "explicit_marker":  "That's hilarious [laugh] I can't believe it actually worked.",
    "amused_context":   "Oh no... hah... okay that genuinely made me laugh out loud.",
}


def main() -> None:
    import torch
    from transformers import CsmForConditionalGeneration, AutoProcessor

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} (float32, CPU) — first run downloads ~2-4GB to HF_HOME...", flush=True)
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CsmForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model = model.to("cpu")
    load_s = time.perf_counter() - t0
    print(f"Loaded in {load_s:.1f}s (includes download on first run)\n", flush=True)

    print(f"{'variant':<18} {'gen_s':>8} {'audio_s':>8} {'RTF':>7}", flush=True)
    print("-" * 48, flush=True)

    for name, text in VARIANTS.items():
        conversation = [{"role": SPEAKER, "content": [{"type": "text", "text": text}]}]
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True
        ).to("cpu")
        t0 = time.perf_counter()
        audio = model.generate(**inputs, output_audio=True)
        gen_s = time.perf_counter() - t0

        out_path = OUT_DIR / f"csm_{name}.wav"
        processor.save_audio(audio, str(out_path))

        # Derive audio duration from the saved file
        try:
            import soundfile as sf
            data, sr = sf.read(str(out_path))
            audio_s = len(data) / sr
            rtf = gen_s / audio_s if audio_s > 0 else float("nan")
        except Exception:
            audio_s, rtf = float("nan"), float("nan")
        print(f"{name:<18} {gen_s:>7.1f}s {audio_s:>7.2f}s {rtf:>7.2f}", flush=True)

    print("-" * 48, flush=True)
    print(f"\nWAVs in {OUT_DIR.relative_to(REPO)}/ — listen to judge laugh quality:", flush=True)
    print("  csm_plain.wav           (baseline naturalness)", flush=True)
    print("  csm_laughter_text.wav   ('hahaha' in text)", flush=True)
    print("  csm_explicit_marker.wav ('[laugh]' marker — does CSM read it or vocalize?)", flush=True)
    print("  csm_amused_context.wav  ('hah' interjection in context)", flush=True)
    print("\nThe question: does ANY variant produce a convincing, natural laugh?", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nSPIKE FAILED: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
