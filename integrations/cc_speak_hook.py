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

Env:
  CC_SPEAK_VOICE     Kokoro voice           (default af_heart — Eve's voice)
  CC_SPEAK_PYTHON    python with kokoro_onnx (default: the repo .venv interpreter)
  CC_SPEAK_MAXCHARS  cap spoken length      (default 1200; 0 = uncapped)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
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


def main() -> None:
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

    cap = int(os.getenv("CC_SPEAK_MAXCHARS", "1200"))
    if cap and len(text) > cap:
        text = text[:cap]

    py = os.getenv("CC_SPEAK_PYTHON") or (str(DEFAULT_PY) if DEFAULT_PY.exists() else sys.executable)
    voice = os.getenv("CC_SPEAK_VOICE", "af_heart")

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
