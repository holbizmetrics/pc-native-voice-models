# Roadmap — pc-native-voice-models

**Philosophy:** build small → ship → *use* → learn → iterate. Exit criteria are behaviors ("do I reach for it?"), not feature counts. Honest status only — no "phase complete" that doesn't match runnable code.

---

## Done (shipped, runnable)

- **v1 — `speak.py` (talk-only TTS).** Type/pipe/file text → streamed speech on CPU. Kokoro 82M via ONNX, no GPU, no network at runtime. 54 voices, 8 languages auto-detected from voice prefix, save-to-WAV. Operator-confirmed working live 2026-05-21.
- **Bus integration — speech output + presence** (`integrations/bus_speak.py` + presence beat). The voice model is a **consumer of the SecuredChat bus**: inbound messages get spoken aloud, and the session advertises itself online via a respawn-wrapped presence beat. Tested live 2026-05-23 (phone-sent message spoken on PC). Setup runbook: [`integrations/README.md`](integrations/README.md). Made the voice model load-bearing in the PCLA ecosystem, not a standalone toy.
- **Research record:** landscape survey, Kokoro CPU benchmark (RTF 0.48), streaming wrapper (TTFA ~1.5s), Sesame spike (ruled out for non-verbal). In `Researches/pc-native-voice-models/`.

---

## Now — Dogfood v1 + close the cheap gap

**The load-bearing step.** v1 only matters if it gets used. Before building more, run it on real text in real workflows.

- **Use it.** Exit criterion: do you actually reach for `speak.py` instead of reading on screen / using a cloud TTS? A week of real use tells us what to prioritize next better than guessing.
- **Mandarin** (cheap completeness) — `pip install misaki[zh]` unlocks the 9th language (zf_/zm_ voices). ~5 min. The only "incomplete" piece of v1.

**Exit:** operator reaches for it ≥ a few times in real use, OR names a concrete friction that redirects the roadmap.

---

## Next — Make it *yours* (voice-cloning)

The honest gap: v1 is the operator's *tool* over *Kokoro's* voices. Voice-cloning makes the **voice** yours.

- **Clone your own voice** from a few seconds of audio (OpenVoice V2 or XTTS v2). The model stays pre-trained; the *voice* becomes custom.
- This is the direct answer to "is this really mine?" — the voice is, even if the model isn't.
- **Ethics gate before shipping:** consent + watermarking strategy if cloning anyone but yourself.

**Exit:** `speak.py "..." --voice me` produces a recognizable clone of the operator's own voice.

---

## After — v2 non-verbal (the open research question)

Natural laughs/sighs/backchannel. Sesame CSM-1B ruled out (not human-grade + GPU-only). One data point isn't enough to conclude "open-source can't do it on PC."

- **Orpheus 3B spike** (ungated, no token) — does it laugh better than Sesame? Settles the research question with a 2nd data point.
- If still no → **Bark-precache** (pre-generate laugh clips offline, splice into Kokoro). Lower ceiling, reliable.
- Frame the answer either way: "human-grade open non-verbal on PC is hard-but-possible" vs "not there yet in 2026."

**Exit:** a clear verdict on whether natural non-verbal is achievable PC-native, with a working path if yes.

---

## Polish — anytime, after v1 is in real use

- **First-chunk latency** — break the opening sentence at its first clause so first audio fires <1s (currently ~1.5s). ~15 min. Only worth it if the latency actually bothers you in use.
- **GPU full-mode** — swap `onnxruntime` → `onnxruntime-gpu` for the dual-mode target. Only if a use case needs faster-than-CPU throughput.

---

## Eventually / deferred (not load-bearing yet)

- Cross-app surfaces (CLI is enough for now; MCP/hotkey/browser only if a real need shows up)
- Streaming ASR (the "understand" side) — v3+ territory; Whisper-tier is already strong, low differentiation
- Emotion/tone control beyond voice selection
- Real-time interruption / barge-in (needs the ASR side first)

---

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-05-20 | Scope (c) integration-first, talk-only v1, dual-mode CPU+GPU | (b) trained-from-scratch infeasible on consumer HW; (a) architectural = months; ship something usable in weeks |
| 2026-05-21 | Kokoro = v1 workhorse | RTF 0.48 CPU, MOS 4.2, 54 voices, Apache, no torch |
| 2026-05-21 | Sesame CSM-1B ruled out for non-verbal | laughs not human-grade (operator ear) + RTF ~14 CPU (GPU-only) |
| 2026-05-21 | v1 shipped (`speak.py`) | clean speech path proven end-to-end, operator-confirmed |
