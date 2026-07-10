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


# ── synthesis (the OpenVoice V2 call, via the worker subprocess) ──────────────────
VENV_PY = REPO / ".venv-openvoice" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
WORKER = REPO / "clone_worker.py"


def synth(text: str, voice: CloneVoice, out_path: Path | None, watermark: bool,
          speed: float = 1.0, lang: str | None = None):
    """Generate cloned speech. The heavy stack (torch + OpenVoice V2 + MeloTTS) lives in
    its own venv (.venv-openvoice) and runs in a SUBPROCESS (clone_worker.py) — the
    caller's venv stays torch-free. Plays the result when out_path is None, else writes
    it. `lang` is a MeloTTS base-speaker language (default EN_NEWEST; env CLONE_LANG)."""
    if not VENV_PY.is_file():
        raise NotImplementedError(
            "OpenVoice venv not found (.venv-openvoice). Home-box setup: create it and "
            "install OpenVoice V2 + MeloTTS + checkpoints (see clone_worker.py docstring).")
    if not voice.ref_audio.is_file():
        raise RuntimeError(
            f"no reference sample at {voice.ref_audio} — record ~30s of the voice and "
            f"register it: python clone.py register {voice.name} <sample.wav>")

    import subprocess
    import tempfile
    lang = lang or os.environ.get("CLONE_LANG", "EN_NEWEST")
    tmp_name = None
    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        tmp_name = tmp.name
        target = Path(tmp_name)
    else:
        target = out_path
    cmd = [str(VENV_PY), str(WORKER), "--text", text, "--ref", str(voice.ref_audio),
           "--voice-dir", str(voice.dir), "--out", str(target), "--speed", str(speed),
           "--lang", lang]
    if watermark:
        cmd.append("--watermark")
    try:
        proc = subprocess.run(cmd, cwd=str(REPO))   # worker stderr = live progress
        if proc.returncode != 0:
            raise RuntimeError(f"clone worker failed (exit {proc.returncode}) — see above")
        if out_path is None:
            _play(target)
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _play(path: Path) -> None:
    """Whole-file playback (MeloTTS renders complete files; no streaming to chase)."""
    import sounddevice as sd
    import soundfile as sf
    data, sr = sf.read(str(path))
    sd.play(data, sr)
    sd.wait()


# ── registration (enroll a reference sample as a named clone voice) ───────────────
def register(name: str, sample: Path, kind: str | None = None) -> CloneVoice:
    """Enroll `sample` as ./voices/<name>/ref.wav + meta.json. Non-wav inputs are
    converted via libsndfile (mp3/flac/ogg fine; m4a is not — convert those first).
    A stale cached embedding (se.pth) is dropped so the next synth re-extracts."""
    import shutil
    from datetime import date
    kind = kind or ("self" if name == SELF_NAME else "other")
    vdir = VOICES_DIR / name
    vdir.mkdir(parents=True, exist_ok=True)
    ref = vdir / "ref.wav"
    if sample.suffix.lower() == ".wav":
        shutil.copyfile(sample, ref)
    else:
        import soundfile as sf
        data, sr = sf.read(str(sample))
        sf.write(str(ref), data, sr)
    (vdir / "meta.json").write_text(json.dumps(
        {"name": name, "kind": kind, "created": date.today().isoformat()},
        indent=2), encoding="utf-8")
    se_cache = vdir / "se.pth"
    if se_cache.is_file():
        se_cache.unlink()
    return CloneVoice(name=name, kind=kind, ref_audio=ref, dir=vdir)


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
    if len(sys.argv) >= 4 and sys.argv[1] == "register":
        name, sample = sys.argv[2], Path(sys.argv[3])
        kind = sys.argv[4] if len(sys.argv) > 4 else None
        if not sample.is_file():
            sys.exit(f"sample not found: {sample}")
        cv = register(name, sample, kind)
        print(f"registered clone voice '{cv.name}' (kind={cv.kind}) at {cv.dir}")
        if cv.kind == "other":
            print("NOTE: kind=other requires voices/%s/consent.json before it will speak "
                  "(see consent_template)." % cv.name)
        sys.exit(0)
    if len(sys.argv) > 1:
        sys.exit("usage: clone.py                          # run the gate self-test\n"
                 "       clone.py register <name> <sample> [self|other]")
    sys.exit(1 if _selftest() else 0)
