#!/usr/bin/env python3
"""One-off: speak each newly-rendered chapter as render_binding writes it.

Watches the audiobook output dir; when a new NNN_*.mp3 appears, speaks its title
("Chapter Thirty-Eight, Stephany Sees"). Seeds 'seen' with whatever already
exists so it only announces chapters finished from now on. Loads Kokoro once
(CPU, so it doesn't fight the GPU render). Stops at TOTAL files or after an idle
gap (render finished / stalled).
"""
from __future__ import annotations
import re
import sys
import time
from pathlib import Path

import speak

OUTDIR = Path(r"C:\Users\Holger\Downloads\The_Binding_audio")
VOICE = "af_nicole"
LANG = speak.lang_for(VOICE, None)
TOTAL = 144
POLL = 3.0
IDLE_STOP = 150.0   # seconds with no new file -> assume render done/stalled


def title_of(fname: str) -> str:
    stem = re.sub(r"\.mp3$", "", fname)
    stem = re.sub(r"^\d+_", "", stem)          # drop NNN_ index
    return stem.replace("_", " ").strip()


def main() -> None:
    import sounddevice as sd
    kokoro = speak.load_kokoro()
    print("MONITOR_READY: announce_chapters (CPU)", flush=True)
    seen = {p.name for p in OUTDIR.glob("*.mp3")} if OUTDIR.is_dir() else set()
    print(f"[announce] seeded {len(seen)} existing; announcing new ones", flush=True)
    last_new = time.time()
    while True:
        cur = sorted(p.name for p in OUTDIR.glob("*.mp3")) if OUTDIR.is_dir() else []
        new = [f for f in cur if f not in seen]
        for f in new:
            seen.add(f)
            last_new = time.time()
            title = title_of(f)
            print(f"[announce] {f} -> {title!r}", flush=True)
            try:
                samples, sr = speak.generate(kokoro, title + ".", VOICE, 1.0, LANG)
                sd.play(samples, sr)
                sd.wait()
            except Exception as e:
                print(f"[announce] speech failed: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
        if len(seen) >= TOTAL:
            print("[announce] all chapters announced; done", flush=True)
            return
        if time.time() - last_new > IDLE_STOP:
            print(f"[announce] idle {IDLE_STOP:.0f}s, stopping "
                  f"({len(seen)}/{TOTAL} seen)", flush=True)
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
