#!/usr/bin/env python3
"""Stitch a folder of numbered mp3s into one file (the whole audiobook).

libsndfile-native (no ffmpeg): decodes each input in blocks and writes them to a
single output, so memory stays bounded even for a 14h book. Inputs are taken in
sorted filename order (NNN_ prefix == play order). Output format from extension
(.mp3/.flac/.wav/.ogg). All inputs must share a sample rate (Kokoro = 24000).

  python stitch_audio.py <indir> <out.mp3> [--limit N]
"""
from __future__ import annotations
import sys
from pathlib import Path

import soundfile as sf

BLOCK = 1 << 16  # 65536 frames per read


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        sys.exit("usage: stitch_audio.py <indir> <out.mp3> [--limit N]")
    indir = Path(argv[0])
    out = Path(argv[1])
    limit = None
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])

    files = sorted(indir.glob("*.mp3"))
    if limit:
        files = files[:limit]
    if not files:
        sys.exit(f"no mp3s in {indir}")

    sr0 = sf.info(str(files[0])).samplerate
    ch0 = sf.info(str(files[0])).channels
    print(f"[stitch] {len(files)} files -> {out}  ({sr0} Hz, {ch0}ch)", flush=True)

    written = 0
    skipped = []
    with sf.SoundFile(str(out), mode="w", samplerate=sr0, channels=ch0) as w:
        for i, f in enumerate(files, 1):
            info = sf.info(str(f))
            if info.samplerate != sr0 or info.channels != ch0:
                skipped.append((f.name, f"{info.samplerate}Hz/{info.channels}ch"))
                continue
            with sf.SoundFile(str(f)) as r:
                while True:
                    data = r.read(BLOCK, dtype="float32")
                    if len(data) == 0:
                        break
                    w.write(data)
                    written += len(data)
            if i % 20 == 0 or i == len(files):
                print(f"[stitch] {i}/{len(files)}  ({written/sr0/3600:.2f}h)", flush=True)

    if skipped:
        print(f"[stitch] WARNING skipped {len(skipped)} mismatched: {skipped[:5]}",
              flush=True)
    print(f"[stitch] DONE -> {out}  {written/sr0/3600:.2f}h "
          f"({out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
