#!/usr/bin/env python3
"""Bus-speech bridge — speak inbound SecuredChat messages aloud (in-process).

Reads `chat.py watch --json` (merged with stderr) on stdin. For each inbound
message: (1) print a text notification to stdout (so an upstream Monitor still
surfaces it), and (2) speak a short summary — "<sender> says: <first N words>".

Loads Kokoro ONCE at startup and generates in-process (this is a long-running
process, so loading once is natural). Earlier this spawned a fresh `speak.py`
per message, paying the ~2.5s model-load reload every time; in-process drops
per-message lag from ~3.7s to ~1.2s.

Config (env, set before launch):
  BUS_VOICE      Kokoro voice                          (default af_sarah)
  BUS_LANG       lang code                             (default: auto from voice prefix)
  BUS_MAX_WORDS  max words spoken per message summary  (default 18)

Wire it as:
  SECUREDCHAT_BUS=... python <SecuredChat>/cli/chat.py --room R --identity ME \
    watch --addressed-to-me --exclude-self --since <id> --poll 30 --json 2>&1 \
    | BUS_VOICE=af_nicole python integrations/bus_speak.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # import speak.py from the repo root

import speak  # noqa: E402  — reuse load_kokoro + generate

# Config via env (set before launch; defaults match the original constants).
#   BUS_VOICE      Kokoro voice                          (default af_sarah)
#   BUS_LANG       lang code                             (default: auto from voice prefix)
#   BUS_MAX_WORDS  max words spoken per message summary  (default 18)
VOICE = os.getenv("BUS_VOICE", "af_sarah")
LANG = os.getenv("BUS_LANG") or speak.VOICE_LANG.get(VOICE[:2], "en-us")
MAX_WORDS = int(os.getenv("BUS_MAX_WORDS", "18"))

_KOKORO = None
_SD = None


def _ensure_loaded():
    """Load Kokoro + sounddevice once. Returns True if speech is available."""
    global _KOKORO, _SD
    if _KOKORO is None:
        try:
            _KOKORO = speak.load_kokoro()
            import sounddevice as sd
            _SD = sd
            print("MONITOR_READY: bus_speak loaded Kokoro (in-process speech)", flush=True)
        except Exception as e:
            print(f"MONITOR_WARN: speech unavailable ({type(e).__name__}: {e}); text-only", flush=True)
            _KOKORO = False  # sentinel: tried + failed, don't retry
    return bool(_KOKORO)


def speak_text(text: str) -> None:
    if not _ensure_loaded():
        return
    try:
        samples, sr = speak.generate(_KOKORO, text, VOICE, 1.0, LANG)
        _SD.play(samples, sr)
        _SD.wait()
    except Exception as e:
        # Surface but don't propagate — the bridge must never die on a speech
        # glitch, but a silent swallow hid a real diagnostic for an hour during
        # the 2026-05-26 mango-investigation. stderr keeps it visible without
        # corrupting the [bus] event stream on stdout.
        print(f"[bus] speech failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


def main() -> None:
    _ensure_loaded()  # load up front so the first message isn't delayed by load
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                m = json.loads(line)
            except Exception:
                continue
            sender = str(m.get("from", "someone"))
            body = str(m.get("body", "")).replace("\n", " ").replace("\r", " ")
            mid = str(m.get("id", ""))[:8]
            print(f"[bus] from={sender} id={mid}: {body[:180]}", flush=True)
            words = body.split()
            summary = " ".join(words[:MAX_WORDS])
            if len(words) > MAX_WORDS:
                summary += ", and more"
            speak_text(f"{sender} says: {summary}")
        elif "securedchat:" in line and ("not found" in line or "ambiguous" in line):
            print(f"[bus-ALERT] cursor issue -> {line[:200]}", flush=True)


if __name__ == "__main__":
    main()
