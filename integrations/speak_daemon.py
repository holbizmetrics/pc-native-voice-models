#!/usr/bin/env python3
"""speak_daemon.py — resident warm-voice TTS daemon: load Kokoro ONCE, speak many.

The known lever from ROADMAP.md ("resident/daemon mode — build when interactive
use asks for it"): interactive use asked on 2026-07-13 (the Eve Claude Code
surface — per-reply cold start was the whole felt delay). A fresh process pays
venv start + model load (~3-12s CPU) on EVERY utterance; this daemon pays it
once and each request then costs only synthesis (~1s to first audio, streamed).

Hardened same day by a 2-lens blind audit + live repro. The design points that
matter (each one traces to a caught defect):

  - TCP on 127.0.0.1:SPEAK_DAEMON_PORT (default 48765), localhost-only.
    One JSON object per connection, newline-terminated:
      {"text": "...", "voice": "af_heart", "speed": 1.0, "spoken": true,
       "ts": <sender time.time()>}
    plus {"cmd": "ping"} -> pong|warming and {"cmd": "quit"}.
  - STALE-DROP is the load-bearing warm-window guard: the port binds before the
    model loads (bind-first IS the single-instance guard), so requests sent
    during the load sit in the socket backlog and would be spoken LATE and
    TWICE once the model lands (reproduced live; Windows even drains the
    backlog LIFO, replaying the OLDEST reply). A "warming" reply alone cannot
    close this: the model load holds the GIL, so handle() may only run AFTER
    warm-up. Any request older than STALE_MAX seconds is dropped with b"stale".
  - LATEST-WINS playback: a new utterance gracefully stops the current one
    (stop writing 40ms slices, close the stream — never kill the process, the
    RODE-endpoint lesson). Identical text to what's playing is deduped (two
    rapid Stop-hook firings extract the same reply; replaying it from the
    start is an audible stutter). Honest bound: an interrupt cannot land while
    speak.generate() is synthesizing a sentence (blocking ONNX call) — worst
    case a few seconds on CPU, then the cut lands.
  - SELF-HEAL over zombie: if the model load fails, if the playback worker
    dies, or if 3 consecutive utterances fail (classic after OS sleep/resume
    changes the audio device), the daemon EXITS — freeing the port so the next
    hook firing starts a fresh daemon — instead of holding the port while
    acking "ok" forever mute.
  - Graceful quit: stops playback and joins the worker before exiting, so the
    output stream is closed properly even mid-utterance.
  - No auth (accepted, single-user desktop): any local process can make the
    box speak or quit the daemon; port squatting could intercept reply text.
    Recorded as an OPEN item in the audit ledger — named pipe / token if the
    threat model ever changes.

Run it (usually you don't — cc_speak_hook.py auto-starts it detached, logging
to speak_daemon.log next to this repo's root):
    python integrations/speak_daemon.py            # foreground, logs to stderr
    KOKORO_GPU=1 python integrations/speak_daemon.py   # amortized => GPU wins (~0.5s/gen)

Env:
  SPEAK_DAEMON_PORT   listen port                  (default 48765)
  KOKORO_GPU=1        opt into CUDA — worth it HERE (load amortized), unlike one-shot CLI
  SPEAK_TIMING=1      log per-utterance timing
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import speak  # noqa: E402  (the shared mouth — model load, chunking, stripping)

try:
    PORT = int(os.getenv("SPEAK_DAEMON_PORT", "48765"))
except ValueError:
    PORT = 48765
STALE_MAX = 8.0          # seconds; older requests are backlog ghosts — drop them
MAX_CONSEC_FAILURES = 3  # then exit so the next hook start heals us
CONN_DEADLINE = 5.0      # total seconds one connection may occupy the accept loop

# keep stderr alive on cp1252 consoles (standing Windows class: non-ASCII print crash)
try:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _log(msg: str) -> None:
    print(f"[daemon {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class Player:
    """Single playback worker with latest-wins replacement.

    submit() parks the newest utterance and interrupts the current one; the
    worker thread speaks whatever is parked. Interruption is graceful: the
    40 ms slice loop stops writing and closes the stream (never a process
    kill — see feedback_never_force_kill_audio_stream_process).
    """

    def __init__(self, kokoro):
        # Import HERE, not in the worker: if PortAudio/device init fails we want
        # the daemon to die loudly (port freed -> next hook heals cold) instead
        # of a mute worker-less daemon acking "ok" forever.
        import sounddevice as sd
        self._sd = sd
        self.kokoro = kokoro
        self._pending: dict | None = None
        self._current_text: str = ""
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_current = threading.Event()
        self._shutdown = False
        self._consec_failures = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, req: dict) -> str:
        """Park req as the newest utterance. Returns 'ok' or 'dup'."""
        with self._lock:
            if req["text"] == self._current_text and not self._stop_current.is_set():
                return "dup"   # double Stop-hook firing on the same reply — don't stutter-restart
            self._pending = req
        self._stop_current.set()   # cut the current utterance, if any
        self._wake.set()
        return "ok"

    def stop(self, join_timeout: float = 2.0) -> None:
        """Graceful shutdown: interrupt playback, let the worker close its
        stream, and join it — never exit with a live OutputStream."""
        self._shutdown = True
        self._stop_current.set()
        self._wake.set()
        self._thread.join(timeout=join_timeout)

    def alive(self) -> bool:
        return self._thread.is_alive()

    def _take(self) -> dict | None:
        with self._lock:
            req, self._pending = self._pending, None
        return req

    def _run(self) -> None:
        while not self._shutdown:
            self._wake.wait()
            self._wake.clear()
            req = self._take()
            if req is None or self._shutdown:
                continue
            self._stop_current.clear()
            with self._lock:
                # a submit() may have landed between _take() and clear() — its
                # interrupt flag was just wiped; loop so the newer one wins
                if self._pending is not None:
                    self._wake.set()
                    continue
                self._current_text = req["text"]
            try:
                self._speak_cancellable(req)
                self._consec_failures = 0
            except BaseException as e:   # incl. SystemExit (zh voice w/o misaki[zh])
                self._consec_failures += 1
                _log(f"utterance failed ({self._consec_failures}/{MAX_CONSEC_FAILURES}): "
                     f"{type(e).__name__}: {e}")
                if self._consec_failures >= MAX_CONSEC_FAILURES:
                    _log("too many consecutive failures (dead audio device?) - "
                         "exiting so the next hook start heals us")
                    os._exit(1)
            finally:
                with self._lock:
                    self._current_text = ""

    def _speak_cancellable(self, req: dict) -> None:
        sd = self._sd
        text = req["text"]
        voice = req.get("voice", "af_heart")
        speed = float(req.get("speed", 1.0))
        lang = speak.lang_for(voice, req.get("lang"))
        t0 = time.time()
        first = True
        stream = None
        try:
            for sent in speak.chunk_for_streaming(text):
                if self._stop_current.is_set():
                    break
                samples, sr = speak.generate(self.kokoro, sent, voice, speed, lang)
                samples = samples.astype(np.float32)
                if stream is None:
                    stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
                    stream.start()
                if first and os.getenv("SPEAK_TIMING"):
                    _log(f"time-to-first-audio {time.time() - t0:.2f}s")
                first = False
                # 40 ms slices so an interrupt lands fast during playback
                # (during generate() above it cannot — blocking ONNX call)
                frame = max(1, int(sr * 0.04))
                for i in range(0, len(samples), frame):
                    if self._stop_current.is_set():
                        return
                    stream.write(samples[i:i + frame])
        finally:
            if stream is not None:
                try:
                    stream.stop(); stream.close()
                except Exception:
                    pass
            _log(f"spoke {len(text)} chars in {time.time() - t0:.2f}s"
                 + (" (interrupted)" if self._stop_current.is_set() else ""))


def handle(conn: socket.socket, player: Player | None) -> bool:
    """Handle one connection. Returns False if the daemon should exit.

    player=None means the model is still loading: text requests get b"warming"
    and are NEVER queued. Requests older than STALE_MAX get b"stale" and are
    dropped — the backlog-ghost guard (see module docstring)."""
    conn.settimeout(2.0)
    t_conn = time.time()
    buf = b""
    try:
        while b"\n" not in buf and len(buf) < 1_000_000:
            if time.time() - t_conn > CONN_DEADLINE:
                return True   # slow client: drop, keep serving others
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
        req = json.loads(buf.decode("utf-8", errors="replace").strip() or "{}")
        if req.get("cmd") == "ping":
            conn.sendall(b"pong" if player is not None else b"warming")
            return True
        if req.get("cmd") == "quit":
            conn.sendall(b"bye")
            return False
        if player is None:
            conn.sendall(b"warming")
            return True
        if not player.alive():
            _log("playback worker is dead - exiting so the next hook start heals us")
            os._exit(1)
        ts = req.get("ts")
        if isinstance(ts, (int, float)) and time.time() - ts > STALE_MAX:
            _log(f"dropped stale request ({time.time() - ts:.1f}s old, backlog ghost)")
            conn.sendall(b"stale")
            return True
        text = (req.get("text") or "").strip()
        if not text:
            conn.sendall(b"empty")
            return True
        if req.get("spoken", True):
            text = speak.strip_markdown(text, drop_actions=True).strip()
            if not text:
                conn.sendall(b"empty")   # stage-directions-only reply: nothing to say
                return True
        req["text"] = text
        player.submit(req)               # 'dup' still acks ok — the text IS being spoken
        conn.sendall(b"ok")
    except Exception as e:
        _log(f"bad request: {type(e).__name__}: {e}")
        try:
            conn.sendall(b"err")
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return True


def main() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # NO SO_REUSEADDR on purpose: the failed bind IS the single-instance guard.
    try:
        srv.bind(("127.0.0.1", PORT))
    except OSError:
        # another daemon already owns the port — nothing to do
        sys.exit(0)
    srv.listen(8)

    # Load the model in the BACKGROUND so the accept loop can answer "warming"
    # whenever the GIL lets it; the stale-drop in handle() covers the rest.
    state: dict = {"player": None}

    def _warm() -> None:
        _log(f"loading Kokoro (provider selection per speak.py; KOKORO_GPU={os.getenv('KOKORO_GPU', '')!r})")
        t0 = time.time()
        try:
            kokoro = speak.load_kokoro()
            state["player"] = Player(kokoro)
        except BaseException as e:   # incl. SystemExit from missing model files
            _log(f"model/audio init FAILED ({type(e).__name__}: {e}) - exiting so "
                 f"clients stop seeing 'warming' forever")
            os._exit(1)
        _log(f"model warm in {time.time() - t0:.1f}s - serving on 127.0.0.1:{PORT}")

    threading.Thread(target=_warm, daemon=True).start()
    _log(f"listening on 127.0.0.1:{PORT} (warming)")

    while True:
        try:
            conn, _ = srv.accept()
        except OSError as e:   # transient WSAECONNRESET-class accept errors
            _log(f"accept error ({type(e).__name__}: {e}) - continuing")
            continue
        if not handle(conn, state["player"]):
            break
    player = state["player"]
    if player is not None:
        player.stop()
    _log("quit requested - bye")


if __name__ == "__main__":
    main()
