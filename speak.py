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
import os
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

# Voice-prefix -> Kokoro/espeak language code. Lets --lang auto-derive from the
# chosen voice so you never have to match them by hand (the prefix already
# encodes the language: af/am=US, bf/bm=GB, ef/em=ES, ff/fm=FR, hf/hm=HI,
# if/im=IT, pf/pm=PT-BR, jf/jm=JA, zf/zm=ZH).
VOICE_LANG = {
    "af": "en-us", "am": "en-us",
    "bf": "en-gb", "bm": "en-gb",
    "ef": "es",    "em": "es",
    "ff": "fr-fr", "fm": "fr-fr",
    "hf": "hi",    "hm": "hi",
    "if": "it",    "im": "it",
    "pf": "pt-br", "pm": "pt-br",
    "jf": "ja",    "jm": "ja",
    "zf": "zh",    "zm": "zh",
}


def lang_for(voice: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    return VOICE_LANG.get(voice[:2], DEFAULT_LANG)


# Chinese needs a different phonemizer than the rest. kokoro-onnx phonemizes via
# espeak only, but Kokoro's Chinese (zf_/zm_) voices were trained on misaki[zh]
# pinyin phonemes — espeak's Mandarin produces the wrong phoneme set. So for
# Chinese we run misaki[zh] G2P ourselves and feed phonemes with is_phonemes=True.
_ZH_G2P = None


def _is_chinese(voice: str, lang: str) -> bool:
    return voice[:2] in ("zf", "zm") or (lang or "").lower() in ("zh", "cmn", "zh-cn")


def _zh_phonemes(text: str) -> str:
    global _ZH_G2P
    if _ZH_G2P is None:
        try:
            from misaki import zh
        except ImportError:
            sys.exit("Mandarin needs the Chinese phonemizer: pip install \"misaki[zh]\"")
        _ZH_G2P = zh.ZHG2P()
    ph, _ = _ZH_G2P(text)
    return ph


def generate(kokoro, text: str, voice: str, speed: float, lang: str):
    """One generation call. Routes Chinese through misaki[zh]; everything else
    through kokoro's normal (espeak) path. Returns (samples, sample_rate)."""
    if _is_chinese(voice, lang):
        return kokoro.create(_zh_phonemes(text), voice=voice, speed=speed, is_phonemes=True)
    return kokoro.create(text, voice=voice, speed=speed, lang=lang)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")
# If the opening sentence exceeds this, split it at its first clause boundary so
# the first audio chunk is small (faster time-to-first-audio). No-op if the
# opening has no early clause boundary (avoids unnatural mid-clause word-breaks).
FIRST_CHUNK_MAX_CHARS = 45
FIRST_CLAUSE_MIN_CHARS = 8


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_for_streaming(text: str) -> list[str]:
    """Sentence list, but with a first-chunk-small optimization: if the opening
    sentence is long AND has an early clause boundary, peel off the opening clause
    as its own chunk so first audio fires sooner. Subsequent chunks are full
    sentences. No-op (returns plain sentences) when the opening can't be cleanly
    split early."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    first = sentences[0]
    if len(first) > FIRST_CHUNK_MAX_CHARS:
        parts = CLAUSE_SPLIT.split(first, maxsplit=1)
        if len(parts) == 2 and len(parts[0]) >= FIRST_CLAUSE_MIN_CHARS:
            return [parts[0], parts[1]] + sentences[1:]
    return sentences


def _select_provider() -> str:
    """Pick the ONNX Runtime execution provider.

    Order: explicit ONNX_PROVIDER env wins > KOKORO_CPU=1 forces CPU >
    auto-prefer CUDA when onnxruntime-gpu exposes it > CPU. kokoro-onnx reads
    ONNX_PROVIDER at session construction (kokoro_onnx/__init__.py), so we just
    set it. DirectML (DmlExecutionProvider) is intentionally NOT auto-selected:
    it fails on Kokoro's F0 ConvTranspose op (tested 2026-05-24).
    """
    env = os.getenv("ONNX_PROVIDER")
    if env:
        return env
    if os.getenv("KOKORO_CPU"):
        return "CPUExecutionProvider"
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "CUDAExecutionProvider"
    except Exception:
        pass
    return "CPUExecutionProvider"


def _register_cuda_dlls() -> None:
    """Make the CUDA 12 / cuDNN 9 DLLs from the nvidia-*-cu12 pip wheels findable.

    ort.preload_dlls() loads the top-level DLLs but NOT cuDNN 9's lazily-loaded
    sub-libraries (e.g. cudnn_engines_tensor_ir64_9.dll); without the wheel bin
    dirs on the DLL search path the first Conv silently falls back to CPU
    (tested 2026-05-24). So register each nvidia/*/bin dir explicitly, then
    preload. Windows-only; on Linux the wheels use RPATH/LD_LIBRARY_PATH.
    """
    if hasattr(os, "add_dll_directory"):
        try:
            import glob
            import importlib.util
            spec = importlib.util.find_spec("nvidia")
            for root in (spec.submodule_search_locations or []) if spec else []:
                for bindir in glob.glob(os.path.join(root, "*", "bin")):
                    os.add_dll_directory(bindir)
        except Exception:
            pass
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:
        pass


def load_kokoro():
    if not MODEL.is_file() or not VOICES.is_file():
        sys.exit(f"Model files missing:\n  {MODEL}\n  {VOICES}\n"
                 f"(download per README: kokoro-v1.0.onnx + voices-v1.0.bin)")

    provider = _select_provider()
    if provider == "CUDAExecutionProvider":
        _register_cuda_dlls()
    os.environ["ONNX_PROVIDER"] = provider
    print(f"[speak] ORT provider: {provider}", file=sys.stderr)

    from kokoro_onnx import Kokoro
    try:
        k = Kokoro(str(MODEL), str(VOICES))
    except Exception as e:
        if provider == "CPUExecutionProvider":
            raise
        print(f"[speak] {provider} init failed ({type(e).__name__}: {e}); "
              f"falling back to CPU", file=sys.stderr)
        os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"
        k = Kokoro(str(MODEL), str(VOICES))

    # CUDA can silently fall back to CPU (ORT always appends CPU) if a DLL is
    # missing — report the provider that actually bound.
    try:
        bound = k.sess.get_providers()[0]
        if bound != os.environ["ONNX_PROVIDER"]:
            print(f"[speak] NOTE: requested {os.environ['ONNX_PROVIDER']} "
                  f"but bound {bound}", file=sys.stderr)
    except Exception:
        pass
    return k


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

    sentences = chunk_for_streaming(text)
    chunk_q: "queue.Queue" = queue.Queue()

    def producer():
        for sent in sentences:
            try:
                samples, sr = generate(kokoro, sent, voice, speed, lang)
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
        samples, sr = generate(kokoro, sent, voice, speed, lang)
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
    p.add_argument("--lang", default=None, help="language code (default: auto-derived from voice prefix, e.g. ff_->fr-fr)")
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

    lang = lang_for(args.voice, args.lang)
    if args.save:
        speak_to_file(kokoro, text, args.voice, args.speed, lang, Path(args.save))
    else:
        speak_streaming(kokoro, text, args.voice, args.speed, lang)


if __name__ == "__main__":
    main(sys.argv[1:])
