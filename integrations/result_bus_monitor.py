#!/usr/bin/env python3
"""Result + bus speaking monitor — hear results, see them in chat, relay to a peer.

One long-running process that loads Kokoro ONCE (in-process, like bus_speak.py)
and does THREE things in a single poll loop:

  1. RESULT WATCH  — tails one or more compute result logs (e.g. a long GPU
     sweep's stdout). For each NEW line matching the result pattern:
        a. print "[RESULT] ..." to stdout  -> an upstream Monitor surfaces it in chat
        b. speak it aloud via Kokoro (Eve) -> you HEAR results as they land
        c. relay it over the SecuredChat bus to a peer (default termux-claude)
           -> the phone's termux PCLA session receives every result
  2. BUS SPEECH    — reads `chat.py watch --json` (inbound, addressed-to-me,
     exclude-self) on a daemon thread; prints + speaks "<sender> says: ..."
     (this is the "speaking background monitor" half — re-wired).
  3. Never dies on a speech/bus glitch: errors go to stderr, the loop survives.

Why one process: two Kokoro loads contend for the audio device and memory; a
single in-process model is the proven shape from bus_speak.py. Run on CPU by
default (set ONNX_PROVIDER=CUDAExecutionProvider for GPU, but NOT while a GPU
compute job is running — they fight over the card).

Config (env / args):
  positional args         result log files / globs to tail (or RESULT_LOGS)
  RESULT_LOGS             comma/space-separated log files / globs
  RESULT_PATTERN          regex; matching lines are treated as results
  RESULT_PEER            bus recipient identity        (default termux-claude)
  RESULT_POLL            poll seconds                  (default 5)
  RESULT_VOICE / BUS_VOICE  Kokoro voice               (default af_sarah)
  RESULT_RELAY           "1"/"0" enable bus relay      (default 1)
  RESULT_SPEAK           "1"/"0" enable speech         (default 1)
  RESULT_INBOUND         "1"/"0" enable inbound bus speech (default 1)
  RESULT_MAX_WORDS       max words spoken per item     (default 28)
  SECUREDCHAT_BUS        bus path (default /d/FromGitHubEtc/securedchat-bus)
  SECUREDCHAT_ROOM       room (default prometheus-relay)
  SECUREDCHAT_IDENTITY   my identity (default windows-claude)
  CHAT_PY                path to SecuredChat cli/chat.py
                         (default /d/FromGitHubEtc/SecuredChat/cli/chat.py)

Wire it as a persistent Monitor (Claude Code) so its stdout shows in chat:
  python integrations/result_bus_monitor.py \
      /d/FromGitHubEtc/Researches/d-axis-sweep/d10_cert_run4.log
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # import speak.py from the repo root

import speak  # noqa: E402 — reuse load_kokoro + generate + VOICE_LANG

# ---- config -----------------------------------------------------------------
VOICE = os.getenv("RESULT_VOICE") or os.getenv("BUS_VOICE", "af_sarah")
LANG = os.getenv("BUS_LANG") or speak.VOICE_LANG.get(VOICE[:2], "en-us")
MAX_WORDS = int(os.getenv("RESULT_MAX_WORDS", "28"))
PEER = os.getenv("RESULT_PEER", "termux-claude")
POLL = float(os.getenv("RESULT_POLL", "5"))
RELAY = os.getenv("RESULT_RELAY", "1") != "0"
SPEAK = os.getenv("RESULT_SPEAK", "1") != "0"
INBOUND = os.getenv("RESULT_INBOUND", "1") != "0"

# Drive-letter forward-slash form (D:/...) is unambiguous for native-Windows
# python regardless of launcher cwd; git-bash /d/... mis-resolves there.
BUS = os.getenv("SECUREDCHAT_BUS", "D:/FromGitHubEtc/securedchat-bus")
ROOM = os.getenv("SECUREDCHAT_ROOM", "prometheus-relay")
IDENTITY = os.getenv("SECUREDCHAT_IDENTITY", "windows-claude")
CHAT_PY = os.getenv("CHAT_PY", "D:/FromGitHubEtc/SecuredChat/cli/chat.py")

# Lines that count as "results" worth surfacing/speaking/relaying.
DEFAULT_PATTERN = (
    r"(K=\d|c8=\d|===.*===|RESULT|OVERTURN|CERTIFIED|FULL-EXACT|LB\(|"
    r"CONFIRMED|done\b|DONE\b|PASS\b|FAIL\b|abort)"
)
PATTERN = re.compile(os.getenv("RESULT_PATTERN", DEFAULT_PATTERN))

# Noise filter: skip a result line that is ONLY a banner with no payload? No —
# banners (=== ... done ===) are meaningful milestones. Keep them.

_KOKORO = None
_SD = None
# One audio device, two producer threads (result-watch + inbound-bus). Without
# serialization, concurrent _SD.play/_SD.wait deadlocks PortAudio and wedges the
# caller forever (observed 2026-05-31: monitor caught the self-test, then a
# concurrent inbound speak collided and the result-watch loop never polled
# again). The lock serializes playback; the worker thread keeps audio OFF the
# poll loop so a hung device can never block result detection/relay.
_SPEAK_LOCK = threading.Lock()


def _ensure_loaded() -> bool:
    global _KOKORO, _SD
    if not SPEAK:
        return False
    if _KOKORO is None:
        try:
            _KOKORO = speak.load_kokoro()
            import sounddevice as sd
            _SD = sd
            print("MONITOR_READY: result_bus_monitor loaded Kokoro (in-process speech)",
                  flush=True)
        except Exception as e:
            print(f"MONITOR_WARN: speech unavailable ({type(e).__name__}: {e}); text-only",
                  flush=True)
            _KOKORO = False  # tried + failed; don't retry
    return bool(_KOKORO)


def speak_text(text: str) -> None:
    """Fire-and-forget: speech runs on its own daemon thread, serialized by
    _SPEAK_LOCK. The caller (poll loop / inbound reader) NEVER blocks on audio,
    so a stuck device can't wedge result detection or relay."""
    if not _ensure_loaded():
        return

    def _worker():
        try:
            with _SPEAK_LOCK:
                samples, sr = speak.generate(_KOKORO, text, VOICE, 1.0, LANG)
                _SD.play(samples, sr)
                _SD.wait()
        except Exception as e:
            print(f"[monitor] speech failed: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)

    threading.Thread(target=_worker, daemon=True).start()


def _summary(text: str) -> str:
    words = text.split()
    s = " ".join(words[:MAX_WORDS])
    if len(words) > MAX_WORDS:
        s += ", and more"
    return s


def relay_to_peer(body: str, kind: str = "result") -> None:
    """Send `body` over the bus to PEER. Body via stdin (no arg-quoting)."""
    if not RELAY:
        return
    cmd = [sys.executable, CHAT_PY, "--bus", BUS, "--room", ROOM,
           "--identity", IDENTITY, "send", "--to", PEER, "--kind", kind]
    try:
        r = subprocess.run(cmd, input=body, text=True, capture_output=True,
                           timeout=60)
        if r.returncode != 0:
            print(f"[monitor] relay rc={r.returncode}: {r.stderr.strip()[:200]}",
                  file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[monitor] relay failed: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)


# ---- result watch -----------------------------------------------------------
def _resolve_logs(args) -> list[str]:
    raw = list(args)
    env = os.getenv("RESULT_LOGS", "")
    if env:
        raw += re.split(r"[,\s]+", env.strip())
    out = []
    for r in raw:
        if not r:
            continue
        hits = glob.glob(r)
        out.extend(hits if hits else [r])  # keep literal path so it's tailed once it appears
    # de-dupe, preserve order
    seen, uniq = set(), []
    for p in out:
        ap = str(Path(p))
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq


def watch_results(log_files: list[str], stop: threading.Event) -> None:
    """Tail each log file; emit new result lines. Pre-seed to EOF so only lines
    appended AFTER startup are emitted (no backlog replay)."""
    offsets: dict[str, int] = {}
    # Pre-seed existing files to current size (emit only future appends).
    for p in log_files:
        try:
            offsets[p] = os.path.getsize(p) if os.path.exists(p) else 0
        except OSError:
            offsets[p] = 0
    print(f"[monitor] watching {len(log_files)} log(s); peer={PEER} relay={RELAY} "
          f"speak={SPEAK}", flush=True)

    while not stop.is_set():
        for p in log_files:
            try:
                if not os.path.exists(p):
                    continue
                size = os.path.getsize(p)
                last = offsets.get(p, 0)
                if size < last:  # truncated / rotated
                    last = 0
                if size <= last:
                    continue
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(last)
                    new = f.read()
                    offsets[p] = f.tell()
            except OSError:
                continue
            for line in new.splitlines():
                line = line.strip()
                if not line or not PATTERN.search(line):
                    continue
                name = Path(p).name
                # Order: surface (chat) -> relay (phone) -> speak (audio, last
                # and non-blocking). Critical outputs land before the risky one.
                print(f"[RESULT] ({name}) {line}", flush=True)
                relay_to_peer(f"[RESULT from {IDENTITY} / {name}] {line}", kind="result")
                speak_text(f"New result: {_summary(line)}")
        stop.wait(POLL)


# ---- inbound bus speech (the "speaking monitor again" half) ------------------
def watch_inbound(stop: threading.Event) -> None:
    """Stream inbound bus messages via `chat.py watch --json`; print + speak."""
    # Find the latest existing id so we only hear messages arriving AFTER start.
    since = None
    try:
        r = subprocess.run(
            [sys.executable, CHAT_PY, "--bus", BUS, "--room", ROOM,
             "--identity", IDENTITY, "recv", "--addressed-to-me",
             "--exclude-self", "--json"],
            text=True, capture_output=True, timeout=60)
        ids = [json.loads(l).get("id") for l in r.stdout.splitlines()
               if l.strip().startswith("{")]
        ids = [i for i in ids if i]
        if ids:
            since = ids[-1]
    except Exception as e:
        print(f"[monitor] inbound pre-seed failed: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)

    cmd = [sys.executable, "-u", CHAT_PY, "--bus", BUS, "--room", ROOM,
           "--identity", IDENTITY, "watch", "--addressed-to-me",
           "--exclude-self", "--poll", str(max(15, int(POLL))), "--json"]
    if since:
        cmd += ["--since", since]
    print(f"[monitor] inbound bus watch started (since={str(since)[:8]})", flush=True)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
    except Exception as e:
        print(f"[monitor] inbound watch failed to start: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        return
    try:
        for line in proc.stdout:
            if stop.is_set():
                break
            line = line.strip()
            if not line.startswith("{"):
                if "securedchat:" in line and ("not found" in line or "ambiguous" in line):
                    print(f"[bus-ALERT] cursor issue -> {line[:200]}", flush=True)
                continue
            try:
                m = json.loads(line)
            except Exception:
                continue
            sender = str(m.get("from", "someone"))
            body = str(m.get("body", "")).replace("\n", " ").replace("\r", " ")
            mid = str(m.get("id", ""))[:8]
            print(f"[bus] from={sender} id={mid}: {body[:180]}", flush=True)
            speak_text(f"{sender} says: {_summary(body)}")
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


def main() -> int:
    log_files = _resolve_logs(sys.argv[1:])
    if not log_files and not INBOUND:
        print("MONITOR_WARN: no result logs given and inbound disabled — nothing to do",
              flush=True)
        return 2
    _ensure_loaded()  # load up front so the first item isn't delayed by model load

    stop = threading.Event()
    threads = []
    if INBOUND:
        t = threading.Thread(target=watch_inbound, args=(stop,), daemon=True)
        t.start()
        threads.append(t)
    try:
        if log_files:
            watch_results(log_files, stop)
        else:
            while not stop.is_set():
                stop.wait(POLL)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
