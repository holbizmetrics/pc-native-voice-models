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
- ~~Mandarin~~ **DONE 2026-05-23** — added via `misaki[zh]`; speak.py routes Chinese (zf_/zm_) through misaki G2P → kokoro `is_phonemes`. All 9 languages now work. (Turned out misaki[zh] alone wasn't enough — kokoro-onnx phonemizes via espeak only, so the integration needed the misaki→phoneme→is_phonemes path, not just the package.)

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

- ~~First-chunk latency~~ **DONE 2026-05-23** — `chunk_for_streaming()` peels the opening clause when the first sentence is long + has an early comma/semicolon/colon. Measured: 2.28s → 1.24s TTFA (46% faster, now under the 1.5s target) for comma-having openings; no-op (no regression) for comma-less ones.
- ~~GPU full-mode~~ **DONE 2026-05-24** — onnxruntime-gpu + CUDA 12 / cuDNN 9 pip wheels (no system CUDA install). `_register_cuda_dlls` adds the wheels' `nvidia/*/bin` dirs to the DLL path (`preload_dlls()` alone misses cuDNN 9 sub-libraries → silent CPU fallback). Warm-gen **2.56s → 0.50s (~5.2×)** on an RTX 3060. **But GPU is opt-in, not default** (`KOKORO_GPU=1` / `ONNX_PROVIDER=CUDAExecutionProvider`): a fresh process pays CUDA cold-start, so **time-to-first-audio is ~5.6s warm-disk / ~18.8s cold-disk vs ~3.2s on CPU** — GPU loses for one-shot CLI, wins only where load is amortized (resident bus monitor → per-msg ~0.5s; long streamed text). CPU is the CLI default; the bus monitor opts into GPU explicitly. DirectML ruled out (Kokoro F0 ConvTranspose fails on DmlExecutionProvider). `SPEAK_TIMING=1` prints time-to-first-audio.

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
| 2026-05-24 | GPU = CUDA (not DirectML) | DirectML fails on Kokoro's F0 ConvTranspose; CUDA via pip wheels works, ~5.2× warm. cuDNN 9 needs nvidia/*/bin on the DLL path, not just preload_dlls() |
| 2026-05-24 | GPU opt-in, CPU is CLI default | Measured time-to-first-audio: CPU ~3.2s vs CUDA ~5.6s warm-disk / ~18.8s cold-disk. CUDA cold-start (context + cuDNN autotune) per fresh process loses for one-shots; only amortizes for the load-once resident monitor / long text |
