#!/usr/bin/env python3
"""speak_daemon.py — resident warm-voice TTS daemon: load Kokoro ONCE, speak many.

The known lever from ROADMAP.md ("resident/daemon mode — build when interactive
use asks for it"): interactive use asked on 2026-07-13 (the Eve Claude Code
surface — per-reply cold start was the whole felt delay). A fresh process pays
venv start + model load (~3-4s CPU) on EVERY utterance; this daemon pays it once
and then each request costs only synthesis (~1s to first audio, streamed).

Shape:
  - TCP on 127.0.0.1:SPEAK_DAEMON_PORT (default 48765). Localhost-only.
  - One JSON object per connection, newline-terminated:
      {"text": "...", "voice": "af_heart", "speed": 1.0, "spoken": true}
    or {"cmd": "ping"} -> "pong", {"cmd": "quit"} -> daemon exits.
    Replies "ok" once the utterance is accepted.
  - LATEST-WINS playback: a new utterance gracefully stops the current one
    (stop writing slices, close the stream cleanly — never kill the process,
    the RODE-endpoint lesson) and replaces anything queued. For a dialog,
    freshness beats completeness; the old overlapping-processes behavior is
    strictly worse on both axes.
  - Single instance by construction: a second daemon fails to bind and exits.

Run it (usually you don't — cc_speak_hook.py auto-starts it detached):
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

PORT = int(os.getenv("SPEAK_DAEMON_PORT", "48765"))

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
    40 ms slice loop just stops writing and closes the stream (never a process
    kill — see feedback_never_force_kill_audio_stream_process).
    """

    def __init__(self, kokoro):
        self.kokoro = kokoro
        self._pending: dict | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_current = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, req: dict) -> None:
        with self._lock:
            self._pending = req
        self._stop_current.set()   # cut the current utterance, if any
        self._wake.set()

    def _take(self) -> dict | None:
        with self._lock:
            req, self._pending = self._pending, None
        return req

    def _run(self) -> None:
        import sounddevice as sd
        while True:
            self._wake.wait()
            self._wake.clear()
            req = self._take()
            if req is None:
                continue
            self._stop_current.clear()
            try:
                self._speak_cancellable(sd, req)
            except Exception as e:
                _log(f"utterance failed: {type(e).__name__}: {e}")

    def _speak_cancellable(self, sd, req: dict) -> None:
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
                # 40 ms slices so an interrupt lands fast, mid-sentence
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


def handle(conn: socket.socket, player: Player) -> bool:
    """Handle one connection. Returns False if the daemon should exit."""
    conn.settimeout(5.0)
    buf = b""
    try:
        while b"\n" not in buf and len(buf) < 1_000_000:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
        req = json.loads(buf.decode("utf-8", errors="replace").strip() or "{}")
        if req.get("cmd") == "ping":
            conn.sendall(b"pong")
            return True
        if req.get("cmd") == "quit":
            conn.sendall(b"bye")
            return False
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
        player.submit(req)
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

    _log(f"loading Kokoro (provider selection per speak.py; KOKORO_GPU={os.getenv('KOKORO_GPU', '')!r})")
    t0 = time.time()
    kokoro = speak.load_kokoro()
    _log(f"model warm in {time.time() - t0:.1f}s — listening on 127.0.0.1:{PORT}")

    player = Player(kokoro)
    while True:
        conn, _ = srv.accept()
        if not handle(conn, player):
            break
    _log("quit requested — bye")


if __name__ == "__main__":
    main()
