#!/usr/bin/env python3
"""Bus-speech bridge — speak inbound SecuredChat messages aloud.

Reads `chat.py watch --json` (merged with stderr) on stdin. For each inbound
message: (1) print a text notification to stdout (so an upstream Monitor still
surfaces it), and (2) speak a short summary via speak.py — "<sender> says: <first
N words>". Bodies can be long; we speak a summary, not the whole thing.

Wire it as:
  SECUREDCHAT_BUS=... python <SecuredChat>/cli/chat.py --room R --identity ME \
    watch --addressed-to-me --exclude-self --since <id> --poll 30 --json 2>&1 \
    | python integrations/bus_speak.py

speak.py cold-starts (~2.5s) per message — fine for low-cadence bus relay. If the
channel ever gets chatty, swap to a resident speak-server.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
SPEAK = ROOT / "speak.py"
VOICE = "af_sarah"
MAX_WORDS = 18


def speak(text: str) -> None:
    try:
        subprocess.run(
            [str(VENV_PY), str(SPEAK), text, "--voice", VOICE],
            capture_output=True, timeout=180,
        )
    except Exception:
        pass  # never let a speech failure kill the bridge


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                m = json.loads(line)
            except Exception:
                continue
            sender = str(m.get("from", "someone"))
            body = str(m.get("body", "")).replace("\n", " ").replace("\r", " ")
            mid = str(m.get("id", ""))[:8]
            # text notification (stdout -> upstream Monitor surfaces it)
            print(f"[bus] from={sender} id={mid}: {body[:180]}", flush=True)
            # spoken summary
            words = body.split()
            summary = " ".join(words[:MAX_WORDS])
            if len(words) > MAX_WORDS:
                summary += ", and more"
            speak(f"{sender} says: {summary}")
        elif "securedchat:" in line and ("not found" in line or "ambiguous" in line):
            print(f"[bus-ALERT] cursor issue -> {line[:200]}", flush=True)


if __name__ == "__main__":
    main()
