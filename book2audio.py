#!/usr/bin/env python3
"""book2audio.py — turn a long document into a chaptered audiobook.

Any .html / .txt / .md -> a folder of ordered, size-capped mp3s, generated on
Kokoro (load-once, GPU by default since this is a batch). Built on speak.py's
text helpers (strip_html / strip_markdown / encoding-robust read / generate).

What's reliable vs best-effort (read this before trusting the output):
  * RELIABLE: numbered size-chunks. The text is split at paragraph (then
    sentence) boundaries into ~--max-min pieces. Works on ANY document.
  * BEST-EFFORT: --by-heading names pieces after detected headings
    (Chapter/Prologue/Part/# Markdown/... by default; --heading-regex to
    override). Heading conventions vary wildly between books, so this is a
    convenience layer — it ALWAYS falls back to size-chunks for anything it
    can't split, and a book with an idiom the pattern doesn't know just
    renders as plain numbered chunks. Don't read "chapter-accurate" into it
    unless you've eyeballed --dry-run on your file.

Resumable: pieces whose mp3 already exists are skipped, so an interrupted run
re-runs from where it stopped. Bounded memory: each piece streams to disk.

Examples:
  python book2audio.py book.html                      # numbered ~16-min mp3s
  python book2audio.py book.html --by-heading --voice af_nicole
  python book2audio.py notes.md --out audio --max-min 10
  python book2audio.py book.html --by-heading --dry-run   # preview the split
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

import speak

# Measured on Kokoro: ~1200 chars -> ~97s audio (RTX 3060 + CPU agree on duration).
CHARS_PER_MIN = int(1200 / (97 / 60))  # ~742

# Default best-effort heading pattern: common English book conventions + Markdown
# ATX headers. Anchored at line start (strip_* puts blocks on their own lines).
# Deliberately STRICT to avoid matching prose: bare "Part of her..." must NOT
# count, so Part requires a colon; Chapter/BOOK/keywords are specific enough.
DEFAULT_HEADING = (
    r'(?m)^(?:'
    r'#{1,6}\s+\S.*|'                                       # markdown headers (md inputs)
    r'BOOK\s+[A-Z][A-Za-z]*\b.*|'                           # BOOK THREE The Wrong Sky
    r'(?:Prologue|Epilogue|Interlude|Foreword|Preface|'
    r'Introduction|Afterword|Appendix)\b.*|'
    r'Chapter\s+[\w-]+\b.*|'                                # Chapter Twelve: Title
    r'Part\s+[A-Z][\w-]*:.*'                                # Part Seven: Title (colon required)
    r')$'
)

# A real front table-of-contents shows up as a tight run of heading-only lines
# (tiny gaps) followed by a big jump to the first real section. Skip it ONLY when
# such a run actually precedes the content — so a book with no TOC keeps its
# first chapter instead of having it eaten.
TOC_GAP = 400        # chars; a heading whose run-to-next exceeds this is content
MIN_TOC_RUN = 8      # need at least this many tight heading links to call it a TOC


def detect_format(path: Path, override: str | None) -> str:
    if override and override != "auto":
        return override
    ext = path.suffix.lower()
    if ext in (".html", ".htm"):
        return "html"
    if ext in (".md", ".markdown"):
        return "md"
    return "txt"


def extract_text(path: Path, fmt: str, encoding: str) -> str:
    raw = speak.read_text_file(path, encoding)
    if fmt == "html":
        return speak.strip_html(raw)
    if fmt == "md":
        return speak.strip_markdown(raw)
    return raw.strip()


def slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:n] or "section"


def size_chunks(body: str, maxc: int) -> list[str]:
    """Split body into <=~maxc-char chunks at paragraph then sentence boundaries."""
    if len(body) <= maxc:
        return [body] if body.strip() else []
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


def split_by_heading(text: str, pattern: str, maxc: int) -> list[tuple[str, str]]:
    """Best-effort heading split. Skips a leading table-of-contents when one is
    detected (tight heading run + big jump to content); keeps a real title-page
    preamble otherwise. Oversized sections are further size-chunked. Falls back
    to pure size-chunks if no heading matches at all."""
    rx = re.compile(pattern)
    hits = list(rx.finditer(text))
    if not hits:
        return [(f"part {i+1}", c) for i, c in enumerate(size_chunks(text, maxc))]
    # TOC detection: first heading whose gap to the next exceeds TOC_GAP. If that
    # boundary sits after a run of >= MIN_TOC_RUN tight links, treat everything
    # up to and including it as a TOC and start content after it.
    offs = [m.start() for m in hits]
    gap_idx = next((i for i in range(len(offs) - 1)
                    if offs[i + 1] - offs[i] > TOC_GAP), None)
    toc_skipped = gap_idx is not None and gap_idx >= MIN_TOC_RUN
    content_offset0 = hits[gap_idx + 1].start() if toc_skipped else 0
    if toc_skipped:
        hits = hits[gap_idx + 1:]
    pieces: list[tuple[str, str]] = []
    # Preamble before the first content heading (a title page) -> its own piece(s),
    # but only when we did NOT skip a TOC (the pre-TOC text is the TOC itself).
    pre = "" if toc_skipped else text[:hits[0].start()].strip()
    if pre:
        for k, c in enumerate(size_chunks(pre, maxc), 1):
            pieces.append((f"Intro (part {k})" if len(pre) > maxc else "Intro", c))
    for i, m in enumerate(hits):
        start = m.start()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        title = m.group(0).strip().lstrip("#").strip()
        body = text[start:end].strip()
        parts = size_chunks(body, maxc)
        if len(parts) <= 1:
            pieces.append((title, parts[0] if parts else body))
        else:
            for k, part in enumerate(parts, 1):
                pieces.append((f"{title} (part {k} of {len(parts)})", part))
    return pieces


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="book2audio.py",
        description="Turn a long .html/.txt/.md document into a folder of ordered, "
                    "size-capped mp3s (Kokoro, load-once, GPU by default). "
                    "Numbered size-chunks are reliable; --by-heading naming is best-effort.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  book2audio.py book.html                       numbered ~16-min mp3s\n"
               "  book2audio.py book.html --by-heading --voice af_nicole\n"
               "  book2audio.py book.html --by-heading --dry-run    preview the split\n",
    )
    p.add_argument("input", help="path to the .html / .txt / .md document")
    p.add_argument("--out", metavar="DIR",
                   help="output folder (default: <input>_audio next to the input)")
    p.add_argument("--voice", default="af_nicole", help="Kokoro voice (default af_nicole)")
    p.add_argument("--speed", type=float, default=1.0, help="speech speed (default 1.0)")
    p.add_argument("--lang", default=None, help="language code (default: from voice prefix)")
    p.add_argument("--format", choices=["auto", "html", "md", "txt"], default="auto",
                   help="input format (default auto from extension)")
    p.add_argument("--encoding", default="utf-8-sig",
                   help="input encoding (default utf-8-sig; auto-falls-back on error)")
    p.add_argument("--max-min", type=float, default=16.0,
                   help="max minutes of audio per file (default 16)")
    p.add_argument("--by-heading", action="store_true",
                   help="best-effort: name pieces after detected headings, falling "
                        "back to size-chunks (see --heading-regex)")
    p.add_argument("--heading-regex", default=DEFAULT_HEADING,
                   help="override the heading pattern used by --by-heading")
    p.add_argument("--cpu", action="store_true",
                   help="force CPU (default: GPU if available — this is a load-once batch)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the piece manifest and exit (no audio)")
    args = p.parse_args(argv)

    src = Path(args.input)
    if not src.is_file():
        sys.exit(f"input not found: {src}")
    outdir = Path(args.out) if args.out else src.with_name(src.stem + "_audio")

    fmt = detect_format(src, args.format)
    text = extract_text(src, fmt, args.encoding)
    if not text.strip():
        sys.exit("no speakable text extracted from the input")
    maxc = max(1000, int(args.max_min * CHARS_PER_MIN))

    if args.by_heading:
        pieces = split_by_heading(text, args.heading_regex, maxc)
    else:
        pieces = [(f"part {i+1}", c) for i, c in enumerate(size_chunks(text, maxc))]
    if not pieces:
        sys.exit("nothing to render")

    total = len(pieces)
    chars = sum(len(b) for _, b in pieces)
    est_h = chars / CHARS_PER_MIN / 60
    print(f"[book2audio] {src.name} [{fmt}] -> {total} pieces, {chars} chars "
          f"(~{est_h:.1f}h audio), <= {args.max_min:.0f} min each", flush=True)
    print(f"[book2audio] out: {outdir}", flush=True)

    if args.dry_run:
        for i, (lbl, b) in enumerate(pieces, 1):
            print(f"  {i:03d}  {len(b):>6}ch  {lbl[:64]}")
        return

    if not args.cpu:
        os.environ["KOKORO_GPU"] = "1"   # load_kokoro auto-falls-back to CPU if it fails
    outdir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    lang = speak.lang_for(args.voice, args.lang)
    kokoro = speak.load_kokoro()

    t_run = time.time()
    audio_total = 0.0
    for idx, (label, body) in enumerate(pieces, 1):
        out = outdir / f"{idx:03d}_{slug(label)}.mp3"
        if out.exists() and out.stat().st_size > 0:
            print(f"[{idx}/{total}] skip {out.name}", flush=True)
            continue
        t0 = time.time()
        writer = None
        sr = 24000
        n = 0
        try:
            for sent in speak.split_sentences(body):
                samp, sr = speak.generate(kokoro, sent, args.voice, args.speed, lang)
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
    print(f"[book2audio] DONE {total} pieces, {audio_total/3600:.2f}h audio, "
          f"{(time.time()-t_run)/3600:.2f}h wall -> {outdir}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
