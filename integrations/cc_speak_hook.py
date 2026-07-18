#!/usr/bin/env python3
"""cc_speak_hook.py — Claude Code Stop-hook adapter: speak Claude's reply aloud.

A thin surface adapter over the shared mouth (speak.py) — the same trunk that
eve_voice.py and bus_speak.py feed. The mouth is built once in speak.py; this is
one more leaf, this time for Claude Code.

How it works: registered as a Claude Code `Stop` hook, Claude Code pipes the hook
JSON (which contains `transcript_path`, NOT the reply text) on stdin when an
assistant turn finishes. This script reads the LAST assistant *text* from the
transcript JSONL and feeds it to `speak.py --spoken`, spawned DETACHED so TTS
playback never blocks the session.

Transcript shape (validated against a real CC transcript 2026-05-29): each line is
{type, message:{role, content:[...]}, ...}; assistant text lives in
message.content[] blocks of {"type":"text","text":...}. Tool-only turns have no
text block -> nothing is spoken (we walk back to the last turn that has text).

settings.json — register PROJECT-scoped (a global Stop hook would make EVERY
Claude Code session talk, including unrelated ones):

  {"hooks": {"Stop": [{"matcher": "", "hooks": [
    {"type": "command", "async": true,
     "command": "<venv-python> <repo>/integrations/cc_speak_hook.py"}]}]}}

Fast path (added 2026-07-13): the hook first tries the resident warm-voice daemon
(integrations/speak_daemon.py, localhost TCP) — model already loaded, speech
starts in ~1s instead of paying the per-process Kokoro cold start. If the daemon
isn't running, the hook auto-starts it detached AND speaks THIS reply via the
old one-shot path (cold, same as before) — so the first reply of a session is
slow exactly once, every later one is fast. Set CC_SPEAK_NO_DAEMON=1 to keep the
pure one-shot behavior.

Env:
  CC_SPEAK_VOICE     Kokoro voice           (default af_heart — Eve's voice)
  CC_SPEAK_PYTHON    python with kokoro_onnx (default: the repo .venv interpreter)
  CC_SPEAK_MAXCHARS  cap spoken length      (default 1200; 0 = uncapped)
  CC_SPEAK_NO_DAEMON disable the warm-daemon fast path (one-shot only)
  SPEAK_DAEMON_PORT  daemon port            (default 48765, must match daemon)
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEAK = ROOT / "speak.py"
# Windows venv interpreter that has kokoro_onnx (speak.cmd uses this one). Prefer
# pythonw.exe — the windowless interpreter — so the detached TTS process never
# pops a console window; fall back to python.exe.
_VENV_SCRIPTS = ROOT / ".venv" / "Scripts"
DEFAULT_PY = _VENV_SCRIPTS / "pythonw.exe"
if not DEFAULT_PY.exists():
    DEFAULT_PY = _VENV_SCRIPTS / "python.exe"


def last_assistant_text(transcript_path: str) -> str:
    """Return the concatenated text of the most recent assistant turn that has
    any text block. Skips tool-only turns. Tolerant of both the nested
    ({message:{role,content}}) and flat ({role,content}) shapes."""
    text = ""
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                msg = d.get("message") if isinstance(d.get("message"), dict) else d
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content")
                parts = []
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                            parts.append(b["text"])
                joined = "\n".join(parts).strip()
                if joined:  # keep advancing to the latest turn that actually has text
                    text = joined
    except FileNotFoundError:
        return ""
    return text


def _daemon_speak(text: str, voice: str) -> str:
    """Hand the utterance to the resident warm daemon.
    Returns "done" (spoken / nothing speakable), "warming" (a daemon exists but
    can't take this one — fall back, do NOT spawn another), or "down" (no
    daemon — fall back AND spawn one for next time).

    Key distinction: a CONNECT failure means no daemon (spawn); a connected-
    but-unanswered request means one is loading its model (the load holds the
    GIL, so it may not even manage a "warming" reply — don't spawn a second).
    The request we leave behind in its backlog is killed daemon-side by the
    stale-drop ("ts" below), so it is never spoken late/twice."""
    try:
        port = int(os.getenv("SPEAK_DAEMON_PORT", "48765"))
    except ValueError:
        port = 48765
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=0.5)
    except Exception:
        return "down"
    try:
        with s:
            s.settimeout(1.5)   # warm acks arrive in ~10ms; don't stall a cold session
            s.sendall(json.dumps({"text": text, "voice": voice, "spoken": True,
                                  "ts": time.time()}).encode("utf-8") + b"\n")
            resp = s.recv(16)
            if resp in (b"ok", b"empty"):
                return "done"
            return "warming"    # warming/stale/err/anything: a daemon exists
    except Exception:
        return "warming"        # connected but no answer: it's loading — don't spawn


def _tee_avatar(raw_text: str) -> None:
    """Best-effort tee to the eve-avatar orb's spool (the face).

    On THIS machine Kokoro (af_heart) is the MOUTH — it plays straight to the
    speakers and never makes an audio file. So we tee the reply to the orb
    WITHOUT an audioUrl: the page performs the `*stage directions*` as gestures
    + caption + a synthetic breath pulse, and does NOT play audio — no duet with
    Kokoro. This is the desktop analog of the laptop's default (speaker speaks,
    orb performs cues), the tee the laptop's eve_speak_fallback.py already does.

    Fail-open by contract: a dead/absent orb must NEVER cost Eve her voice, so
    every error here is swallowed. Inert where no orb is checked out (the spool
    parent dir doesn't exist) — same guard shape as the termux launcher branch.
    Override the location with EVE_AVATAR_SPOOL. Wired 2026-07-19 (the desktop
    half of the double-held carry item; the note asked for a WAV-capable port,
    but the honest desktop design is NO audio in the spool at all)."""
    try:
        spool = os.getenv("EVE_AVATAR_SPOOL")
        if spool:
            spool_dir = Path(spool)
        else:
            for cand in (Path("D:/FromGitHubEtc/eve-avatar/spool"),
                         Path("C:/FromGithubEtc/eve-avatar/spool")):
                if cand.parent.exists():
                    spool_dir = cand
                    break
            else:
                return  # no orb checked out here — inert, no error
        if not spool_dir.parent.exists():
            return
        spool_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "id": f"eve-{int(time.time())}-{os.getpid()}",
            "text": raw_text,          # raw, *asides* intact → the page extracts cues
            # no spokenText / no audioUrl: Kokoro is the mouth, the orb is the face
        }
        # Atomic write so the poller never reads a half-written latest.json.
        tmp = spool_dir / f".latest.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, spool_dir / "latest.json")
    except Exception:
        pass  # a broken face never touches the voice


def _avatar_spool_dir() -> Path | None:
    """The orb's spool dir if an orb is checked out here, else None (inert)."""
    spool = os.getenv("EVE_AVATAR_SPOOL")
    if spool:
        d = Path(spool)
        return d if d.parent.exists() else None
    for cand in (Path("D:/FromGitHubEtc/eve-avatar/spool"),
                 Path("C:/FromGithubEtc/eve-avatar/spool")):
        if cand.parent.exists():
            return cand
    return None


def _page_worker(voice: str) -> None:
    """Detached child (EVE_AVATAR_MOUTH=page): synth af_heart to a WAV the ORB
    plays, so the browser's analyser drives the pulse to the real voice — the
    face finally moves *with* the voice, not on a synthetic timeline. Reads the
    RAW reply on stdin (stars intact → cues), synth is `--spoken` (stripped →
    the audio says no stage directions), then tees latest.json with the audioUrl.
    The orb is the mouth in this mode: local playback is skipped upstream, so no
    duet. Blocking synth is fine — this whole process is detached."""
    raw = sys.stdin.read()
    if not raw.strip():
        return
    spool_dir = _avatar_spool_dir()
    if spool_dir is None:
        return
    try:
        spool_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        wav = spool_dir / f"utterance-{ts}-{os.getpid()}.wav"
        # sys.executable IS the venv python (parent launched us with it) → has kokoro.
        r = subprocess.run(
            [sys.executable, str(SPEAK), "--spoken", "--voice", voice,
             "--save", str(wav), "-"],
            input=raw.encode("utf-8"),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if r.returncode != 0 or not wav.exists():
            return  # synth failed — leave the last performance up, don't tee a dead url
        entry = {
            "id": f"eve-{ts}-{os.getpid()}",
            "text": raw,                       # raw → the page extracts cues
            "audioUrl": f"spool/{wav.name}",   # relative to the served root
        }
        tmp = spool_dir / f".latest.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, spool_dir / "latest.json")
        # Keep the spool from growing forever: retain the newest few utterance wavs.
        wavs = sorted(spool_dir.glob("utterance-*.wav"), key=lambda p: p.stat().st_mtime)
        for old in wavs[:-6]:
            try:
                old.unlink()
            except OSError:
                pass
    except Exception:
        pass  # page-mouth is opt-in; a failure here means one silent reply, logged nowhere


def _start_daemon(py: str) -> None:
    """Spawn the daemon detached so the NEXT reply finds it warm. A second
    spawn is harmless — the port bind is the single-instance guard. Its stderr
    goes to speak_daemon.log at the repo root (append), not DEVNULL — when the
    voice goes quiet, the log is the first place to look."""
    daemon = Path(__file__).resolve().parent / "speak_daemon.py"
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    else:
        kwargs["start_new_session"] = True
    try:
        log = open(ROOT / "speak_daemon.log", "ab")
    except Exception:
        log = subprocess.DEVNULL
    try:
        subprocess.Popen([py, str(daemon)], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=log,
                         **kwargs)
    except Exception:
        pass


def main() -> None:
    # Detached page-mouth worker mode (spawned by this same script below).
    if len(sys.argv) >= 2 and sys.argv[1] == "--page-worker":
        _page_worker(sys.argv[2] if len(sys.argv) > 2 else "af_heart")
        return

    try:
        hook = json.load(sys.stdin)
    except Exception:
        return  # not invoked as a hook / bad input — say nothing
    tp = hook.get("transcript_path")
    if not tp:
        return
    text = last_assistant_text(tp)
    if not text:
        return  # tool-only turn or empty — nothing to speak

    # Tee the RAW reply (stars intact) to the orb before the spoken-length cap
    # mutates it — the face performs the full caption + all cues. Best-effort.
    _tee_avatar(text)

    try:
        cap = int(os.getenv("CC_SPEAK_MAXCHARS", "1200"))
    except ValueError:
        cap = 1200
    if cap and len(text) > cap:
        text = text[:cap]

    py = os.getenv("CC_SPEAK_PYTHON") or (str(DEFAULT_PY) if DEFAULT_PY.exists() else sys.executable)
    voice = os.getenv("CC_SPEAK_VOICE", "af_heart")

    # Page-mouth mode (EVE_AVATAR_MOUTH=page): the ORB plays the voice, so the
    # browser analyser can pulse the face to the real audio. Synth happens in a
    # detached worker (spawned from this same script), which tees the wav+cue
    # spool when done. We RETURN here — no local Kokoro playback — so the orb is
    # the sole mouth and there is no duet. Falls back to normal playback only if
    # no orb is checked out (so page-mouth on a machine without the orb still
    # talks). Verified end-to-end 2026-07-19.
    if os.getenv("EVE_AVATAR_MOUTH", "").lower() == "page" and _avatar_spool_dir() is not None:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        else:
            kwargs["start_new_session"] = True
        try:
            p = subprocess.Popen([py, os.path.abspath(__file__), "--page-worker", voice],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, **kwargs)
            p.stdin.write(text.encode("utf-8"))
            p.stdin.close()
        except Exception:
            pass
        return  # orb is the mouth; do not also play locally

    # Fast path: resident warm daemon (speech in ~1s, playback serialized,
    # latest-wins). "warming" = one is already loading its model — fall through
    # to one-shot for THIS reply but don't spawn a second. "down" = start one
    # for next time and fall through.
    if not os.getenv("CC_SPEAK_NO_DAEMON"):
        status = _daemon_speak(text, voice)
        if status == "done":
            return
        if status == "down":
            _start_daemon(py)

    # Spawn DETACHED so a long playback never freezes the Claude Code session.
    kwargs = {}
    if os.name == "nt":
        # CREATE_NO_WINDOW: run the TTS process silently — no console window pops up.
        # (DETACHED_PROCESS gave python.exe its own visible console; this suppresses it.)
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    else:
        kwargs["start_new_session"] = True
    try:
        p = subprocess.Popen(
            [py, str(SPEAK), "--spoken", "--voice", voice, "-"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            **kwargs,
        )
        p.stdin.write(text.encode("utf-8"))
        p.stdin.close()
    except Exception:
        pass  # never let a TTS hiccup surface as a hook failure
    # Do NOT wait — return immediately so the session is not blocked.


if __name__ == "__main__":
    main()
