#!/usr/bin/env python3
"""speak.py — pc-native-voice-models v1: type text, hear it.

Talk-only TTS over Kokoro (ONNX, CPU by default; GPU opt-in via KOKORO_GPU).
Streams sentence-by-sentence so, once the model is loaded, the first words start
~1s after you hit enter, regardless of total length (the generator runs ~2x
faster than playback, so it stays ahead — gapless). A cold CLI launch adds the
one-time model load (~3-4s) before that first ~1s.

Usage:
    python speak.py "Hello, this is my own voice model."
    python speak.py "..." --voice af_bella --speed 1.1
    python speak.py --file story.txt
    echo "piped text works too" | python speak.py
    python speak.py "save instead of play" --save out.wav
    python speak.py "hear it AND keep it" --record out.mp3
    python speak.py --file story.txt --read       # read-along: words appear as spoken
    python speak.py --list-voices
    python speak.py -h        # full help with examples

Voice library: 54 voices, 9 languages (Kokoro). --list-voices to see them.
Model files expected at models/kokoro-v1.0.onnx + models/voices-v1.0.bin.
"""
from __future__ import annotations

import argparse
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

_T0 = time.time()  # process start, for SPEAK_TIMING time-to-first-audio

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

    Order: KOKORO_CPU=1 forces CPU (escape hatch) > explicit ONNX_PROVIDER env >
    KOKORO_GPU=1 opts into CUDA > CPU (default).

    CPU is the default *even when CUDA is available*. A fresh process pays the
    full CUDA cold-start every launch (context init + cuDNN first-conv autotune):
    measured ~5.6s warm-disk / ~18.8s cold-disk to first audio vs ~3.2s on CPU
    (RTX 3060, 2026-05-24). The ~5x warm-gen win only amortizes where the model
    loads ONCE and is reused — the resident bus monitor (per-message gen ~2.5s →
    ~0.5s) or a long streamed text — so those opt in explicitly via KOKORO_GPU=1
    / ONNX_PROVIDER=CUDAExecutionProvider. DirectML is never selected (fails on
    Kokoro's F0 ConvTranspose op).
    """
    if os.getenv("KOKORO_CPU"):
        return "CPUExecutionProvider"
    env = os.getenv("ONNX_PROVIDER")
    if env:
        return env
    if os.getenv("KOKORO_GPU"):
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


def _play_read_along(stream, writer, samples, sr: int, sent: str) -> None:
    """Reading mode, one chunk: write `samples` to the stream in small slices and
    print the words of `sent` to stdout as playback reaches them — each word shows
    up as it's spoken. Kokoro gives no per-word timestamps, so word timing is
    proportional to word length, paced off this chunk's own audio duration; that
    keeps the text roughly in sync with the voice. Records each slice too if a
    writer is given. Ends the sentence with a newline."""
    words = sent.split()
    n = len(samples)
    dur = n / sr if sr else 0.0
    # Start time of each word (proportional to its length in characters).
    starts = []
    if words:
        weights = [max(1, len(w)) for w in words]
        tot = sum(weights)
        acc = 0
        for w in weights:
            starts.append(dur * acc / tot)
            acc += w
    frame = max(1, int(sr * 0.04))  # 40 ms slices pace the word reveal
    played = next_word = 0
    while played < n:
        sl = samples[played:played + frame]
        stream.write(sl)
        if writer is not None:
            writer.write(sl)
        played += len(sl)
        t = played / sr if sr else dur
        while next_word < len(words) and t >= starts[next_word]:
            sys.stdout.write(words[next_word] + " ")
            sys.stdout.flush()
            next_word += 1
    # Flush any words not yet shown (e.g. the final word) and end the line.
    if next_word < len(words):
        sys.stdout.write(" ".join(words[next_word:]) + " ")
    sys.stdout.write("\n")
    sys.stdout.flush()


def speak_streaming(kokoro, text: str, voice: str, speed: float, lang: str,
                    record_path: Path | None = None, read: bool = False) -> None:
    """Generate sentence-by-sentence in a producer thread, play gaplessly via a
    single OutputStream whose blocking write() paces playback.

    If record_path is given, each chunk is written to disk as it plays (streamed
    to the file, not buffered at the end) — so you hear it AND keep a copy in one
    pass, and a partial file stays valid if interrupted. Format is inferred from
    the extension; .wav/.flac/.ogg/.mp3 all work via libsndfile (no ffmpeg/LAME
    install needed).

    If read is True (reading mode), the text is printed to the console word by
    word IN SYNC with the speech — each word appears as it's spoken — one sentence
    per line. Combine with record_path to read along AND keep the file."""
    import sounddevice as sd

    sentences = chunk_for_streaming(text)
    # Bounded queue = backpressure. The generator runs ~2-3x faster than playback
    # (RTF ~0.37), so an UNbounded queue would race ahead and buffer the ENTIRE
    # audio in RAM for a long input (a ~1hr text ≈ hundreds of MB). maxsize caps how
    # far ahead generation runs; the producer's put() blocks until the consumer
    # drains. 8 sentences ahead is far more than enough to stay gapless.
    chunk_q: "queue.Queue" = queue.Queue(maxsize=8)

    def producer():
        # Catch BaseException, not just Exception: _zh_phonemes (missing misaki[zh])
        # calls sys.exit() -> SystemExit, which is a BaseException. Uncaught in this
        # worker thread it would kill the thread WITHOUT queuing the sentinel, leaving
        # the consumer blocked forever on chunk_q.get(). The finally guarantees the
        # sentinel is always sent, so a failure surfaces as an error, never a hang.
        try:
            for sent in sentences:
                samples, sr = generate(kokoro, sent, voice, speed, lang)
                chunk_q.put(("chunk", samples.astype(np.float32), sr, sent))
        except BaseException as e:
            chunk_q.put(("error", str(e) or type(e).__name__))
        finally:
            chunk_q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    stream = None
    writer = None
    try:
        while True:
            item = chunk_q.get()
            if item is None:
                break
            if item[0] == "error":
                sys.exit(f"generation error: {item[1]}")
            _, samples, sr, sent = item
            if stream is None:
                stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
                stream.start()
                # Open the recording file lazily on the first chunk — sr is only
                # known once generation has produced audio.
                if record_path is not None:
                    import soundfile as sf
                    try:
                        writer = sf.SoundFile(str(record_path), mode="w",
                                              samplerate=sr, channels=1)
                    except Exception as e:
                        sys.exit(f"cannot record to {record_path}: {e}\n"
                                 f"(use a .wav / .flac / .ogg / .mp3 extension)")
                if os.getenv("SPEAK_TIMING"):
                    print(f"[speak] time-to-first-audio: {time.time() - _T0:.2f}s",
                          file=sys.stderr)
            if read:
                _play_read_along(stream, writer, samples, sr, sent)
            else:
                stream.write(samples)
                if writer is not None:
                    writer.write(samples)
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        if writer is not None:
            writer.close()
            print(f"[speak] recorded {record_path}", file=sys.stderr)


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
        description="pc-native-voice-models v1 — talk-only TTS (Kokoro, streaming; CPU by "
                    "default, GPU opt-in via KOKORO_GPU). Type or pipe text, hear it spoken. "
                    "No network at runtime.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  speak.py "Hello, this runs entirely on my CPU."     speak inline text
  speak.py "..." --voice af_bella --speed 1.1         pick a voice, adjust speed
  speak.py --file story.txt                            speak a text file
  echo "piped text" | speak.py                         speak from stdin
  speak.py "..." --save out.wav                        write a file instead of playing
  speak.py "..." --record out.mp3                       play AND save (wav/flac/ogg/mp3)
  speak.py --file story.txt --read                      read-along: words appear as spoken
  speak.py --list-voices                               show all 54 voices

text source precedence: --file > inline arg > stdin.
output: default plays live; --record plays AND streams to a file; --save writes only (silent).
reading mode: add --read to print the text word-by-word in sync with the speech.
voices: 54 across 9 languages (af_=US female, am_=US male, bf_/bm_=British, etc.).
streaming: once loaded, first words ~1s in, gapless after (generator runs ahead);
           a cold launch adds the one-time model load (~3-4s) first.
""",
    )
    p.add_argument("text", nargs="?", help="text to speak (or pipe via stdin, or use --file)")
    p.add_argument("--file", "-f", metavar="PATH", help="read text to speak from a file")
    p.add_argument("--voice", default=DEFAULT_VOICE, help=f"voice name (default {DEFAULT_VOICE}; --list-voices to see all)")
    p.add_argument("--speed", type=float, default=1.0, help="speech speed multiplier (default 1.0)")
    p.add_argument("--lang", default=None, help="language code (default: auto-derived from voice prefix, e.g. ff_->fr-fr)")
    out = p.add_mutually_exclusive_group()
    out.add_argument("--save", metavar="PATH", help="write to a file instead of playing (no audio out)")
    out.add_argument("--record", "-r", metavar="PATH",
                     help="play AND save in one pass — stream the spoken audio to a file "
                          "(format from extension: .wav/.flac/.ogg/.mp3)")
    p.add_argument("--read", action="store_true",
                   help="reading mode: print the text word-by-word in sync with the "
                        "speech (one sentence per line). Combines with --record.")
    p.add_argument("--list-voices", action="store_true", help="print available voices and exit")
    args = p.parse_args(argv)

    if args.save and args.read:
        p.error("--read shows the text during playback and can't be combined with --save "
                "(silent / writes only). Use --record to also keep a file.")

    # Resolve + validate the text source BEFORE loading the model (~4s), so a bare
    # `speak.py` or any missing-text invocation errors instantly instead of after
    # the load. --list-voices needs the model but no text, so it skips this.
    #
    # text source precedence: --file > explicit "-" stdin > positional arg > piped stdin.
    # IMPORTANT: only read stdin when it's actually piped (not a TTY) or explicitly
    # requested with "-". Otherwise `speak.py` with no args would block forever on
    # stdin.read() in an interactive terminal (no EOF coming).
    text = None
    if not args.list_voices:
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

    lang = lang_for(args.voice, args.lang)
    if args.save:
        speak_to_file(kokoro, text, args.voice, args.speed, lang, Path(args.save))
    else:
        speak_streaming(kokoro, text, args.voice, args.speed, lang,
                        record_path=Path(args.record) if args.record else None,
                        read=args.read)


if __name__ == "__main__":
    main(sys.argv[1:])
