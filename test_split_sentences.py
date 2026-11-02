#!/usr/bin/env python3
"""test_split_sentences.py — no piece handed to Kokoro may reach its 510-phoneme limit.

Locks the 2026-11-02 fix: kokoro_onnx crashed with `IndexError: index 510 is
out of bounds` (off-by-one at its own truncation limit) on piece #140 of the
operator's 2.3M-char story — 568 chars. split_sentences now caps every piece at
PIECE_MAX_CHARS, splitting at clause boundaries first, then words, then
hard-cutting.

WHAT THESE TESTS DO NOT SHOW: the crash itself. Kokoro batches long input at
punctuation internally, so a synthetic comma-rich run-on (as below) did NOT crash
the old code end-to-end — only a stretch it cannot batch does. The crash was
reproduced on the real piece (old: IndexError, new: 3 pieces OK), and the real
file was phonemized whole: old longest 20 pieces all >= 510 phonemes (max 1360),
new 30,577 pieces max 326, none >= 510. These tests lock the CAP, which is what
makes that true for any text.

Run:  python test_split_sentences.py   (plain asserts, no pytest needed)
  or: python -m pytest test_split_sentences.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speak  # noqa: E402
from speak import split_sentences, PIECE_MAX_CHARS  # noqa: E402

MAX = PIECE_MAX_CHARS
words = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731


def test_normal_sentences_unchanged():
    text = "Hello there. How are you? I am fine, thanks!"
    assert split_sentences(text) == ["Hello there.", "How are you?", "I am fine, thanks!"]


def test_runon_sentence_is_capped_and_loses_no_words():
    # ~1,200 chars, commas but no sentence ending — the shape that crashed.
    clause = "she walked along the river and kept thinking about what he had said"
    text = ", ".join([clause] * 18)
    assert len(text) > 1000
    pieces = split_sentences(text)
    assert len(pieces) > 1
    assert all(len(p) <= MAX for p in pieces), [len(p) for p in pieces]
    # Nothing dropped or duplicated: rejoined text has the same words in order.
    assert words(" ".join(pieces)).replace(",", "") == words(text).replace(",", "")


def test_prefers_clause_boundaries():
    clause = "x" * 100
    text = f"{clause}, {clause}, {clause}"   # 304 chars, two commas
    pieces = split_sentences(text)
    assert all(len(p) <= MAX for p in pieces)
    assert pieces[0].endswith(",")          # broke AT a comma, not mid-clause


def test_no_punctuation_at_all_falls_back_to_words():
    text = " ".join(["word"] * 200)           # 999 chars, no , ; : . ! ?
    pieces = split_sentences(text)
    assert all(len(p) <= MAX for p in pieces)
    assert words(" ".join(pieces)) == words(text)


def test_giant_single_token_is_hard_cut():
    url = "https://example.com/" + "a" * 600
    pieces = split_sentences(url)
    assert all(len(p) <= MAX for p in pieces)
    assert "".join(pieces) == url


def test_CONTROL_old_splitter_would_have_passed_the_long_piece_through():
    # Proves the cases above are the dangerous shape: the pre-fix behaviour
    # (sentence split only) yields a piece far over the cap.
    clause = "she walked along the river and kept thinking about what he had said"
    text = ", ".join([clause] * 18)
    old = [s.strip() for s in speak.SENTENCE_SPLIT.split(text) if s.strip()]
    assert max(len(p) for p in old) > 2 * MAX


def test_empty_input():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed, 0 failed")
