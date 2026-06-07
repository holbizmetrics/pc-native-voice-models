#!/usr/bin/env python3
"""dictate.py — pc-native-voice-models: talk, and it types. The inverse of speak.py.

speak.py turns text into speech. dictate.py turns speech into text and drops it
into whatever window has focus — a local, own-the-stack Wispr Flow. No network at
runtime; the model runs on your machine.

Hold the push-to-talk key (default Right-Ctrl), talk, release. faster-whisper
transcribes locally; the text lands in the focused app via clipboard-paste
(most reliable across apps) or direct typing (--type).

The thing a raw dictation tool does NOT do — and the reason to own the stack —
is step 2: --clean runs a local pass that shapes the spoken prose before it
lands ("type what you meant, not what you said"). Stubbed here; wired next.

Usage:
    python dictate.py                      # hold Right-Ctrl, talk, release -> it types
    python dictate.py --model base.en      # smaller/faster model (default small.en)
    python dictate.py --key f9             # push-to-talk on F9 instead
    python dictate.py --type               # inject by typing (default: clipboard paste)
    python dictate.py --gpu                # CUDA via CTranslate2 (default: CPU int8)
    python dictate.py --once               # one capture -> print to stdout, exit (good for testing)
    python dictate.py --list-mics          # show input devices and exit
    python dictate.py -h

Models download on first use to the faster-whisper cache (~/.cache/huggingface).
Suggested sizes: tiny.en (~75MB) base.en (~150MB) small.en (~500MB) large-v3 (~3GB).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parent

DEFAULT_MODEL = os.environ.get("DICTATE_MODEL", "small.en")
DEFAULT_KEY = os.environ.get("DICTATE_KEY", "ctrl_r")
DEFAULT_LANG = os.environ.get("DICTATE_LANG", "en")
SAMPLE_RATE = 16000  # whisper expects 16 kHz mono float32
MIN_UTTERANCE_S = 0.2  # ignore key-taps with no real audio
PREROLL_S = float(os.environ.get("DICTATE_PREROLL", "0.4"))  # audio kept from *before*
#   the keypress, so the OS mic-activation lag can't clip your first word (the
#   "first word missing" bug Wispr/Aqua/etc. punt to "wait a second after pressing").
_BLOCK_S = 0.05  # mic callback chunk size (50 ms)


# ── clean-up layer: "type what you meant, not what you said" ────────────────────
# The stage a cloud dictation tool can't give you, because you don't own its
# middle. This is the deterministic, fully-local v1 (zero deps, instant). A
# local-LLM backend (rewrite-grade "what you meant") is the natural next upgrade
# and plugs in right here, behind its own model download.

# Vocalized fillers that are never meaningful words -> safe to drop by default.
# Override with DICTATE_FILLERS="um,uh,like,you know" (empty disables).
_DEFAULT_FILLERS = ["um", "uh", "uhm", "erm", "er", "hmm", "mm", "mhm"]
FILLERS = [f.strip() for f in
           os.environ.get("DICTATE_FILLERS", ",".join(_DEFAULT_FILLERS)).split(",")
           if f.strip()]

# Spoken formatting commands -> what they become in the text.
VOICE_COMMANDS = {
    "new paragraph": "\n\n",
    "new line": "\n",
    "next line": "\n",
}


def clean_text(text: str, fillers: list[str] | None = None, commands: dict | None = None) -> str:
    """Tidy a raw dictation transcript into what you meant to write. Deterministic
    and local: applies spoken formatting commands, drops vocalized fillers, fixes
    spacing/punctuation/capitalization. Conservative by design -- it never rewrites
    word choice (that's the future LLM backend), only removes cruft Whisper keeps."""
    if not text:
        return text
    fillers = FILLERS if fillers is None else fillers
    commands = VOICE_COMMANDS if commands is None else commands
    s = text
    # 1) spoken formatting commands -> real formatting (eat an adjacent comma/period)
    for phrase, repl in commands.items():
        s = re.sub(rf"\s*\b{re.escape(phrase)}\b[.,!?]?\s*", repl, s, flags=re.IGNORECASE)
    # 2) drop standalone vocalized fillers (and a trailing comma they often carry)
    if fillers:
        s = re.sub(rf"\b(?:{'|'.join(map(re.escape, fillers))})\b,?", "", s, flags=re.IGNORECASE)
    # 3) capitalize the standalone pronoun 'i'
    s = re.sub(r"\bi\b", "I", s)
    # 4) tidy whitespace + orphan punctuation that filler-removal leaves behind
    s = re.sub(r"[ \t]+", " ", s)                # collapse runs of spaces
    s = re.sub(r"(^|\n)\s*,\s*", r"\1", s)       # drop a comma stranded at a start
    s = re.sub(r"\s*,(?:\s*,)+", ",", s)         # collapse comma runs -> one
    s = re.sub(r"\s*,\s*([.!?])", r"\1", s)      # drop a comma stranded before . ! ?
    s = re.sub(r" +([,.!?;:])", r"\1", s)        # no space before punctuation
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)       # trim around newlines
    s = re.sub(r"\n{3,}", "\n\n", s)             # cap blank-line runs
    s = s.strip()
    # 5) capitalize sentence starts (string start, after . ! ?, or after a newline)
    s = re.sub(r"(^|[.!?]\s+|\n+)([a-z])",
               lambda m: m.group(1) + m.group(2).upper(), s)
    return s


# ── audio capture ─────────────────────────────────────────────────────────────
class Recorder:
    """Always-on mic capture with a pre-roll ring buffer. The stream stays open for
    the whole session and the last PREROLL_S seconds are always retained in memory,
    so when you press the key the first word is already captured -- sidestepping the
    OS mic-activation lag that clips the first word in most dictation tools. Nothing
    leaves the machine: the ring is in-memory and continuously discarded."""

    def __init__(self, device: int | None = None, samplerate: int = SAMPLE_RATE,
                 preroll_s: float = PREROLL_S):
        import numpy as np  # lazy
        self._np = np
        self.device = device
        self.samplerate = samplerate
        self._blocksize = max(1, int(_BLOCK_S * samplerate))
        self._ring: deque = deque(maxlen=max(1, round(preroll_s / _BLOCK_S)))
        self._frames: list = []
        self._lock = threading.Lock()
        self._stream = None
        self.recording = False

    def _callback(self, indata, frames, time_info, status):  # sounddevice thread
        chunk = indata.copy().reshape(-1)
        with self._lock:
            self._ring.append(chunk)        # always feed the pre-roll ring
            if self.recording:
                self._frames.append(chunk)

    def open(self) -> None:
        """Open the persistent mic stream (begins pre-roll buffering immediately)."""
        import sounddevice as sd  # lazy
        self._stream = sd.InputStream(
            samplerate=self.samplerate, channels=1, dtype="float32",
            device=self.device, blocksize=self._blocksize, callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def start(self) -> None:
        """Begin an utterance, seeded with the pre-roll already in the buffer.
        Fast (no device open) -- so push-to-talk feels instant."""
        with self._lock:
            self._frames = list(self._ring)  # the ~PREROLL_S captured before the press
            self.recording = True

    def stop(self):
        with self._lock:
            self.recording = False
            frames, self._frames = self._frames, []
        if not frames:
            return self._np.zeros(0, dtype="float32")
        return self._np.concatenate(frames, axis=0)


# ── ASR ───────────────────────────────────────────────────────────────────────
class Transcriber:
    """faster-whisper wrapper. Loads once; transcribes a numpy waveform.
    CPU int8 by default (fine for dictation-length bursts); --gpu uses CUDA."""

    def __init__(self, model_name: str = DEFAULT_MODEL, gpu: bool = False, lang: str = DEFAULT_LANG):
        self.model_name = model_name
        self.gpu = gpu
        self.lang = lang
        self._model = None

    def load(self) -> float:
        from faster_whisper import WhisperModel  # lazy (slow import + first-use download)
        device = "cuda" if self.gpu else "cpu"
        compute = "float16" if self.gpu else "int8"
        t0 = time.time()
        self._model = WhisperModel(self.model_name, device=device, compute_type=compute)
        return time.time() - t0

    def transcribe(self, audio) -> str:
        if self._model is None:
            self.load()
        lang = None if self.lang in ("", "auto") else self.lang
        # vad_filter drops non-speech: keeps the always-on pre-roll from feeding
        # silence into Whisper (which otherwise hallucinates "you"/"thank you").
        segments, _info = self._model.transcribe(audio, language=lang, beam_size=5,
                                                 vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()


# ── text injection ─────────────────────────────────────────────────────────────
def _set_clipboard_windows(text: str) -> bool:
    """Set the Windows clipboard to text via CF_UNICODETEXT (ctypes, no dep).
    Pointer-correct on 64-bit (restypes set, else handles truncate -> crash)."""
    import ctypes
    CF_UNICODETEXT = 13
    GMEM_MOVEABLE = 0x0002
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    if not user32.OpenClipboard(None):
        return False
    try:
        user32.EmptyClipboard()
        data = text.encode("utf-16-le") + b"\x00\x00"
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not handle:
            return False
        ptr = kernel32.GlobalLock(handle)
        ctypes.memmove(ptr, data, len(data))
        kernel32.GlobalUnlock(handle)
        user32.SetClipboardData(CF_UNICODETEXT, handle)  # clipboard owns handle now
        return True
    finally:
        user32.CloseClipboard()


def inject(text: str, mode: str) -> None:
    """Drop text into the focused app. mode='paste' (clipboard + Ctrl-V) or 'type'."""
    from pynput.keyboard import Controller, Key
    kb = Controller()
    if mode == "paste" and sys.platform == "win32" and _set_clipboard_windows(text):
        time.sleep(0.05)  # let the OS register the clipboard + focus settle
        with kb.pressed(Key.ctrl):
            kb.press("v")
            kb.release("v")
    else:
        kb.type(text)  # fallback / --type: synthetic keystrokes, no clipboard clobber


def _parse_key(name: str):
    from pynput.keyboard import Key, KeyCode
    name = name.lower()
    if hasattr(Key, name):           # ctrl_r, alt_r, cmd_r, f1..f12, space, ...
        return getattr(Key, name)
    if len(name) == 1:
        return KeyCode.from_char(name)
    raise SystemExit(f"unknown --key {name!r}; try ctrl_r, alt_r, f9, or a single char")


# ── run loop ───────────────────────────────────────────────────────────────────
def run(args) -> None:
    rec = Recorder(device=args.device)
    asr = Transcriber(model_name=args.model, gpu=args.gpu, lang=args.lang)
    print(f"Loading whisper '{args.model}' ({'GPU' if args.gpu else 'CPU int8'})...", file=sys.stderr)
    dt = asr.load()
    print(f"  ready in {dt:.1f}s", file=sys.stderr)

    from pynput import keyboard
    hotkey = _parse_key(args.key)
    state = {"busy": False}

    def handle_utterance() -> None:
        audio = rec.stop()
        dur = len(audio) / SAMPLE_RATE
        if dur < MIN_UTTERANCE_S:
            print("  (too short - ignored)", file=sys.stderr)
            state["busy"] = False
            return
        print(f"  transcribing {dur:.1f}s...", file=sys.stderr)
        t0 = time.time()
        text = asr.transcribe(audio)
        print(f"  [{time.time() - t0:.1f}s] {text!r}", file=sys.stderr)
        if args.clean and text:
            text = clean_text(text)
            print(f"  cleaned: {text!r}", file=sys.stderr)
        if text:
            if args.once:
                print(text)
            else:
                inject(text, "type" if args.type_mode else "paste")
        state["busy"] = False
        if args.once:
            os._exit(0)

    def on_press(key) -> None:
        if key == hotkey and not rec.recording and not state["busy"]:
            rec.start()
            print(">> recording... (release to transcribe)", file=sys.stderr)

    def on_release(key) -> None:
        if key == hotkey and rec.recording:
            state["busy"] = True
            print("<< stopped", file=sys.stderr)
            threading.Thread(target=handle_utterance, daemon=True).start()

    rec.open()  # mic on now, pre-roll buffering, so the first word is never clipped
    print(f"Hold [{args.key}] to dictate. Ctrl-C to quit.", file=sys.stderr)
    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\nbye", file=sys.stderr)
    finally:
        rec.close()


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="dictate.py",
        description="Talk, and it types. Local speech-to-text dictation (the inverse of speak.py).",
        epilog=__doc__[__doc__.index("Usage:"):],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"whisper model: tiny.en/base.en/small.en/medium.en/large-v3 (default {DEFAULT_MODEL})")
    p.add_argument("--key", default=DEFAULT_KEY,
                   help=f"push-to-talk key (default {DEFAULT_KEY}; e.g. ctrl_r, alt_r, f9)")
    p.add_argument("--lang", default=DEFAULT_LANG, help=f"language code or 'auto' (default {DEFAULT_LANG})")
    p.add_argument("--gpu", action="store_true", help="use CUDA (CTranslate2) instead of CPU int8")
    inj = p.add_mutually_exclusive_group()
    inj.add_argument("--type", dest="type_mode", action="store_true",
                     help="inject by typing synthetic keystrokes")
    inj.add_argument("--paste", dest="type_mode", action="store_false",
                     help="inject via clipboard + Ctrl-V (default; more reliable)")
    p.set_defaults(type_mode=False)
    p.add_argument("--device", type=int, default=None, help="input device index (see --list-mics)")
    p.add_argument("--list-mics", action="store_true", help="list input devices and exit")
    p.add_argument("--once", action="store_true",
                   help="capture one utterance, print to stdout, exit (no typing - good for testing)")
    p.add_argument("--clean", action="store_true",
                   help='clean the dictation before it lands: drop fillers (um/uh), apply '
                        'voice commands ("new paragraph"/"new line"), tidy spacing + caps')
    args = p.parse_args(argv)

    if args.list_mics:
        import sounddevice as sd
        print(sd.query_devices())
        return
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
