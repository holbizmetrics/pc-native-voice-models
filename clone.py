#!/usr/bin/env python3
"""clone.py — voice-cloning for pc-native-voice-models (the `--voice me` path).

speak.py speaks in Kokoro's 54 FIXED voices; Kokoro cannot learn a new one. Cloning
your *own* voice needs a different engine. We use **OpenVoice V2** — MIT-licensed, so
a clone path stays usable even if this project ever ships commercially (XTTS v2 was
ruled out: its CPML license forbids commercial use). The heavy model and the actual
generation run on the home box (RTX 3060) with your audio sample; THIS module is the
part that runs anywhere — the ethics gate, the voice registry, the watermark policy.
The OpenVoice call itself is a clearly-marked TODO (see `synth`).

Ethics, in one line:
    Clone YOURSELF freely. Cloning ANYONE ELSE needs an explicit consent record on
    file AND forces a watermark on the output.

Layout of a registered clone voice (under ./voices/<name>/):
    meta.json      {"name","kind","created"}   kind = "self" | "other"
    ref.wav        the reference sample the clone is built from
    consent.json   REQUIRED when kind == "other" (see consent_template)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent
VOICES_DIR = REPO / "voices"

# The reserved name for "your own voice" — `--voice me`. Override if you like.
SELF_NAME = os.environ.get("CLONE_SELF_NAME", "me")

# Fields a consent record MUST carry before an "other" clone is allowed.
CONSENT_FIELDS = ("subject", "granted_by", "date", "statement")


@dataclass
class CloneVoice:
    name: str
    kind: str            # "self" | "other"
    ref_audio: Path      # the reference sample
    dir: Path            # the voice's folder


@dataclass
class Decision:
    allow: bool
    watermark: bool
    reason: str


# ── voice registry ──────────────────────────────────────────────────────────────
def resolve(name: str) -> CloneVoice | None:
    """Return the registered clone voice `name`, or None if it isn't a clone (so the
    caller falls back to Kokoro). A clone is a folder ./voices/<name>/ with meta.json.
    The reserved name `me` defaults to kind='self' even if meta omits it."""
    vdir = VOICES_DIR / name
    meta_path = vdir / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    kind = meta.get("kind") or ("self" if name == SELF_NAME else "other")
    return CloneVoice(name=name, kind=kind, ref_audio=vdir / "ref.wav", dir=vdir)


def consent_template(subject: str = "") -> dict:
    """The shape a consent record must have. Date is left blank on purpose — it is
    filled when the record is actually signed, never auto-guessed here."""
    return {
        "subject": subject,        # whose voice is being cloned
        "granted_by": subject,     # who gave permission (usually the subject)
        "date": "",                # YYYY-MM-DD, filled at signing time
        "statement": f"I, {subject or '<name>'}, consent to a synthetic clone of my "
                     f"voice produced by this tool.",
    }


# ── the ethics gate ───────────────────────────────────────────────────────────────
def consent_gate(voice: CloneVoice) -> Decision:
    """Decide ALLOW/BLOCK and whether the output must be watermarked, BEFORE any clone
    is built or spoken. Self-clones pass freely; cloning anyone else requires a valid
    consent record on file and forces a watermark."""
    if voice.kind == "self":
        # You consent to clone yourself by recording your own sample.
        return Decision(allow=True, watermark=False,
                        reason="self-clone: consent implicit; watermark optional")

    consent_path = voice.dir / "consent.json"
    if not consent_path.is_file():
        return Decision(allow=False, watermark=True,
                        reason=f"cloning '{voice.name}' (not you) needs a consent record; "
                               f"none at {consent_path}. Write one (see consent_template).")
    try:
        rec = json.loads(consent_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        return Decision(allow=False, watermark=True,
                        reason=f"consent record unreadable: {e}")
    missing = [k for k in CONSENT_FIELDS if not str(rec.get(k, "")).strip()]
    if missing:
        return Decision(allow=False, watermark=True,
                        reason="consent record incomplete; missing: " + ", ".join(missing))
    return Decision(allow=True, watermark=True,   # forced for anyone but you
                    reason=f"consent on file from {rec['subject']} "
                           f"(granted by {rec['granted_by']}, {rec['date']}); "
                           f"output WILL be watermarked")


# ── watermark policy ──────────────────────────────────────────────────────────────
def provenance_tag(voice: CloneVoice) -> dict:
    """The synthetic-origin metadata stamped into a cloned output file. This is the
    runs-anywhere half of watermarking (honest provenance in the file's tags). The
    *perceptual* (in-signal, survives re-encoding) watermark is a separate TODO —
    it needs the audio pipeline + a watermark model, so it's NOT faked here."""
    return {
        "title": f"synthetic voice clone ({voice.name})",
        "comment": "AI-generated synthetic speech (pc-native-voice-models / OpenVoice V2). "
                   "Synthetic voice clone — not a recording of the named person.",
        "software": "pc-native-voice-models",
    }


# ── synthesis (the OpenVoice V2 call — home-box only) ─────────────────────────────
def synth(text: str, voice: CloneVoice, out_path: Path | None, watermark: bool):
    """Generate cloned speech. Intentionally NOT implemented here: OpenVoice V2's model
    (+ MeloTTS base) and the generation run live on the home box with the RTX 3060 and
    a real audio sample — out of scope for the office machine. The integration shape:

        1. base TTS  : MeloTTS renders `text` in a base speaker
        2. tone color: OpenVoice ToneColorConverter transfers timbre from voice.ref_audio
        3. watermark : if `watermark`, embed the perceptual mark + write provenance_tag()
        4. output    : play (speak.py streaming path) or write to out_path

    Wiring this needs: pip install for OpenVoice V2 + MeloTTS (pulls torch — this repo
    is torch-free today, so it goes behind its own optional install), the model
    download, and voice.ref_audio present."""
    raise NotImplementedError(
        "voice-clone synthesis runs on the home box (OpenVoice V2 + your sample); "
        "this office build ships the gate + registry + watermark policy only. "
        "See clone.synth docstring for the integration shape.")


# ── self-test: the gate over the cases it must get right ─────────────────────────
def _selftest() -> int:
    import tempfile
    failures = 0

    def check(label, got: Decision, want_allow, want_wm):
        nonlocal failures
        ok = got.allow == want_allow and got.watermark == want_wm
        failures += 0 if ok else 1
        verdict = "ALLOW" if got.allow else "BLOCK"
        wm = "watermark FORCED" if got.watermark else "no watermark"
        status = "ok " if ok else "FAIL"
        print(f"  [{status}] {label:34} -> {verdict:5} / {wm}")
        print(f"           {got.reason}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        me = CloneVoice("me", "self", td / "me" / "ref.wav", td / "me")
        check("self (--voice me)", consent_gate(me), True, False)

        bob = CloneVoice("bob", "other", td / "bob" / "ref.wav", td / "bob")
        bob.dir.mkdir(parents=True)
        check("other, NO consent", consent_gate(bob), False, True)

        (bob.dir / "consent.json").write_text(json.dumps({
            "subject": "Bob Example", "granted_by": "Bob Example",
            "date": "2026-06-23", "statement": "I consent."}), encoding="utf-8")
        check("other, valid consent", consent_gate(bob), True, True)

        carol = CloneVoice("carol", "other", td / "carol" / "ref.wav", td / "carol")
        carol.dir.mkdir(parents=True)
        (carol.dir / "consent.json").write_text(json.dumps({"subject": "Carol"}),
                                                encoding="utf-8")
        check("other, INCOMPLETE consent", consent_gate(carol), False, True)

    print(f"\n  {'all gate cases passed' if not failures else str(failures) + ' FAILED'}")
    return failures


if __name__ == "__main__":
    import sys
    sys.exit(1 if _selftest() else 0)
