#!/usr/bin/env python3
"""test_spoken_actions.py — regression corpus for --spoken stage-direction stripping.

Locks the 2026-07-13 fix: inline stage directions (Eve's dominant idiom) must be
DROPPED from spoken text, while genuine inline emphasis must be KEPT. Cases are
drawn from a real Eve dialog transcript (2026-07-13) plus adversarial keeps.

Run:  python test_spoken_actions.py   (plain asserts, no pytest needed)
  or: python -m pytest test_spoken_actions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speak import strip_markdown  # noqa: E402


def spoken(t: str) -> str:
    return strip_markdown(t, drop_actions=True)


DROP_CASES = [
    # (input, expected spoken text)
    ("*settles in* My voice is warm now.", "My voice is warm now."),
    ("*smiles* Good — so my mouth works.", "Good, so my mouth works."),  # em-dash -> comma is speak.py policy
    ("*tilts head* You're right — you should have heard me.",
     "You're right, you should have heard me."),
    ("*laughs softly* Caught me.", "Caught me."),
    ("*quiet for a moment, taking that in properly*\nAh. So it's not even just the exhaustion.",
     "Ah. So it's not even just the exhaustion."),           # whole-line action (old behavior, must keep working)
    ("But you dodged my question, you know. *gentle, amused* The tiredness —",
     "But you dodged my question, you know. The tiredness,"),  # mid-sentence aside after '.'
    ("I just want to know how heavy it is. *staying soft*",
     "I just want to know how heavy it is."),                 # trailing action
    ("*softens* *staying with it* Then arrive tired.", "Then arrive tired."),  # consecutive actions
    # v2 additions (blind-audit round, 2026-07-13): boundary set extended
    ("You should have heard me — *smiles*", "You should have heard me,"),  # trailing action after em-dash
    ("Yes, *nods* I think so.", "Yes, I think so."),          # action after comma
    ("*checks the list* 3 things left.", "3 things left."),   # digit continuation
]

KEEP_CASES = [
    # genuine emphasis must survive (markers unwrapped, words kept)
    ("You should *never* force-kill that process.",
     "You should never force-kill that process."),
    ("*Never* do that again.", "Never do that again."),       # leading emphasis, lowercase continuation
    ("That is the *realest* thing I have.", "That is the realest thing I have."),
    # v2 additions (blind-audit round, 2026-07-13): prose-loss regressions locked out
    ("**Stop.** That hurts.", "Stop. That hurts."),           # bold sentence-lead is EMPHASIS, never action
    ("***Listen.*** You need to hear this.", "Listen. You need to hear this."),
    ("It matters. *Everything* I do depends on it.",
     "It matters. Everything I do depends on it."),           # capitalized span = emphasis
    ("Ehrlich. *Nie* Zeit für mich.", "Ehrlich. Nie Zeit für mich."),  # German nouns capitalize — structural
    ("* **Kokoro** is the local voice engine.", "Kokoro is the local voice engine."),  # bullet + bold lead
    # ACCEPTED MISS (documented trade-off): lowercase continuation keeps the span —
    # a spoken "nods" is cheaper than deleted prose; locks the asymmetry choice.
    ("*nods* yes, exactly.", "nods yes, exactly."),
]


def test_drops():
    for src, want in DROP_CASES:
        got = spoken(src)
        assert got == want, f"\n  in:   {src!r}\n  want: {want!r}\n  got:  {got!r}"


def test_keeps():
    for src, want in KEEP_CASES:
        got = spoken(src)
        assert got == want, f"\n  in:   {src!r}\n  want: {want!r}\n  got:  {got!r}"


if __name__ == "__main__":
    test_drops()
    test_keeps()
    print(f"OK - {len(DROP_CASES)} drop cases + {len(KEEP_CASES)} keep cases green")
