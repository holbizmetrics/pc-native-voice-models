#!/usr/bin/env python3
"""One-off: render The_Binding.html to per-section mp3s in af_nicole.

The file is a multi-book work: Prologue + Chapters 1-99 (clean `Chapter N:`
headings), then "BOOK THREE / Part N:" with a different heading style. So we:
  * detect headings for both styles (Chapter/Prologue/Interlude/Epilogue/The
    Author + BOOK/Part),
  * skip the table-of-contents (everything before the TOC->content gap),
  * size-cap every section (~MAX_CHARS, split at paragraph then sentence
    boundaries) so no single file is huge,
  * stream each piece to its own mp3 (bounded memory),
  * skip pieces whose mp3 already exists (resumable).
Loads Kokoro once; reuses speak.py helpers.
"""
from __future__ import annotations
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import speak

SRC = Path(r"C:\Users\Holger\Downloads\The_Binding.html")
OUTDIR = Path(r"C:\Users\Holger\Downloads\The_Binding_audio")
VOICE = "af_nicole"
SPEED = 1.0
LANG = speak.lang_for(VOICE, None)
TOC_GAP = 400        # heading whose run-to-next exceeds this == content, not a TOC link
MAX_CHARS = 12000    # ~16 min audio per file (97s per 1200 chars measured)

HEADING = re.compile(
    r'(?m)^(?:'
    r'Prologue\b.*|Interlude\b.*|Epilogue\b.*|The Author\b.*|'
    r'Chapter [\w-]+\b.*|'
    r'BOOK [A-Z][A-Z]+\b.*|'
    r'Part [A-Z][a-z]+:.*'
    r')$')


def slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:n] or "section"


def chunk_body(body: str, maxc: int) -> list[str]:
    """Split an oversized section at paragraph boundaries (then sentence
    boundaries for a single giant paragraph), each chunk <= ~maxc chars."""
    if len(body) <= maxc:
        return [body]
    out: list[str] = []
    cur = ""
    for para in re.split(r"\n\n+", body):
        if len(para) > maxc:
            if cur:
                out.append(cur)
                cur = ""
            sc = ""
            for s in speak.split_sentences(para):
                if sc and len(sc) + len(s) + 1 > maxc:
                    out.append(sc)
                    sc = s
                else:
                    sc = (sc + " " + s).strip()
            if sc:
                out.append(sc)
            continue
        if cur and len(cur) + len(para) + 2 > maxc:
            out.append(cur)
            cur = para
        else:
            cur = (cur + "\n\n" + para).strip()
    if cur:
        out.append(cur)
    return out


def build_pieces(text: str):
    hits = list(HEADING.finditer(text))
    offs = [m.start() for m in hits]
    gap_idx = next((i for i in range(len(offs) - 1)
                    if offs[i + 1] - offs[i] > TOC_GAP), None)
    if gap_idx is None:
        sys.exit("could not locate TOC->content boundary")
    content = hits[gap_idx + 1:]          # +1: skip the last TOC entry itself
    pieces = []                            # (label, body)
    for i, m in enumerate(content):
        start = m.start()
        end = content[i + 1].start() if i + 1 < len(content) else len(text)
        title = m.group(0).strip()
        body = text[start:end].strip()
        parts = chunk_body(body, MAX_CHARS)
        if len(parts) == 1:
            pieces.append((title, parts[0]))
        else:
            for k, part in enumerate(parts, 1):
                pieces.append((f"{title} (part {k} of {len(parts)})", part))
    return pieces


def main() -> None:
    dry = "--dry-run" in sys.argv
    OUTDIR.mkdir(parents=True, exist_ok=True)
    text = speak.strip_html(speak.read_text_file(SRC))
    pieces = build_pieces(text)
    total = len(pieces)
    body_chars = sum(len(b) for _, b in pieces)
    print(f"[render] {total} pieces, {body_chars} body chars "
          f"(~{body_chars/1200*97/3600:.1f}h audio est), MAX_CHARS={MAX_CHARS}",
          flush=True)
    print(f"[render] first: {pieces[0][0][:50]!r} ({len(pieces[0][1])}ch)", flush=True)
    print(f"[render] last:  {pieces[-1][0][:50]!r} ({len(pieces[-1][1])}ch)", flush=True)
    big = [(lbl, len(b)) for lbl, b in pieces if len(b) > MAX_CHARS + 2000]
    print(f"[render] oversized pieces (>{MAX_CHARS+2000}ch): {len(big)} {big[:3]}",
          flush=True)
    if dry:
        print("[render] dry run; manifest follows:")
        for i, (lbl, b) in enumerate(pieces, 1):
            print(f"  {i:03d}  {len(b):>6}ch  {lbl[:60]}")
        return

    kokoro = speak.load_kokoro()
    t_run = time.time()
    audio_total = 0.0
    for idx, (label, body) in enumerate(pieces, 1):
        out = OUTDIR / f"{idx:03d}_{slug(label)}.mp3"
        if out.exists() and out.stat().st_size > 0:
            print(f"[{idx}/{total}] skip {out.name}", flush=True)
            continue
        t0 = time.time()
        writer = None
        sr = 24000
        n = 0
        try:
            for sent in speak.split_sentences(body):
                samp, sr = speak.generate(kokoro, sent, VOICE, SPEED, LANG)
                samp = samp.astype(np.float32)
                if writer is None:
                    writer = sf.SoundFile(str(out), mode="w", samplerate=sr, channels=1)
                writer.write(samp)
                n += len(samp)
        finally:
            if writer is not None:
                writer.close()
        secs = n / sr if sr else 0.0
        audio_total += secs
        print(f"[{idx}/{total}] {out.name}  {len(body)}ch -> {secs:.0f}s "
              f"in {time.time()-t0:.0f}s  (cum {audio_total/3600:.2f}h audio, "
              f"{(time.time()-t_run)/3600:.2f}h wall)", flush=True)
    print(f"[render] DONE {total} pieces, {audio_total/3600:.2f}h audio, "
          f"{(time.time()-t_run)/3600:.2f}h wall", flush=True)


if __name__ == "__main__":
    main()
