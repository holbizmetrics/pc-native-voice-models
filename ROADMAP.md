# Roadmap — pc-native-voice-models

**Philosophy:** build small → ship → *use* → learn → iterate. Exit criteria are behaviors ("do I reach for it?"), not feature counts. Honest status only — no "phase complete" that doesn't match runnable code.

---

## Done (shipped, runnable)

- **v1 — `speak.py` (talk-only TTS).** Type/pipe/file text → streamed speech on CPU. Kokoro 82M via ONNX, no GPU, no network at runtime. 54 voices, 8 languages auto-detected from voice prefix, save-to-WAV. Operator-confirmed working live 2026-05-21.
- **Bus integration — speech output + presence** (`integrations/bus_speak.py` + presence beat). The voice model is a **consumer of the SecuredChat bus**: inbound messages get spoken aloud, and the session advertises itself online via a respawn-wrapped presence beat. Tested live 2026-05-23 (phone-sent message spoken on PC). Setup runbook: [`integrations/README.md`](integrations/README.md). Made the voice model load-bearing in the PCLA ecosystem, not a standalone toy.
- **Research record:** landscape survey, Kokoro CPU benchmark (RTF 0.48), streaming wrapper (TTFA ~1.5s), Sesame spike (ruled out for non-verbal). In `Researches/pc-native-voice-models/`.
- **v1.1 — recording, archiving, reading mode, and the first real audit (shipped 2026-05-26).** Operator's first deep dogfood run drove a same-day cluster:
  - `speak.py --record PATH` — play AND save in one pass (wav/flac/ogg/mp3, libsndfile-native — no ffmpeg). Streamed write; partial-file valid on interrupt.
  - `speak.py --read` — **reading mode**: prints the text word-by-word IN SYNC with the speech (one sentence per line). Composes with `--record`.
  - `eve_voice.py EVE_ARCHIVE=1` — every Eve reply archived to `eve-archive/eve_<ts>.{mp3,txt}` (audio + full-fidelity transcript).
  - `bus_speak.py BUS_VOICE`/`BUS_LANG`/`BUS_MAX_WORDS` env config (was hardcoded `af_sarah`/`en-us`/`18`). Lang auto-derives from voice prefix matching `speak.lang_for`.
  - **TRIAD+KG audit of `speak.py`:** H1 (zh-voice streaming hang when `misaki[zh]` missing — test-confirmed) + M1–M4 (doc honesty, queue backpressure) fixed; L1–L4 deferred as low.
  - Polish: bare `speak.py` errors instantly (no 4 s model load first); usage shows long-form `--file`/`--record` (was inconsistent).
  - **Measured this session (RTX 3060 host):** silent-gen RTF ≈ 0.37 CPU (~2.7× faster than real-time); model load ~4.4 s.

---

## Now — Dogfood v1 + close the cheap gap

**The load-bearing step.** v1 only matters if it gets used. Before building more, run it on real text in real workflows.

- **Use it.** Exit criterion: do you actually reach for `speak.py` instead of reading on screen / using a cloud TTS? A week of real use tells us what to prioritize next better than guessing.
- ~~Mandarin~~ **DONE 2026-05-23** — added via `misaki[zh]`; speak.py routes Chinese (zf_/zm_) through misaki G2P → kokoro `is_phonemes`. All 9 languages now work. (Turned out misaki[zh] alone wasn't enough — kokoro-onnx phonemizes via espeak only, so the integration needed the misaki→phoneme→is_phonemes path, not just the package.)

**Exit:** operator reaches for it ≥ a few times in real use, OR names a concrete friction that redirects the roadmap.

**✓ EXIT FIRED 2026-05-26** — operator used it on a real ~1-hour text → mp3 in one shot ("no problems whatsoever"), AND during the same session named concrete frictions (no recording flag, no progress display, doc/CLI inconsistencies, zh-streaming hang) which crystallized into the **v1.1** cluster above. Bus-to-voice bridge also revived and witnessed externally (`af_nicole`, `BUS_MAX_WORDS=50`). Advancing to **Next: voice-cloning**.

---

## Next — Make it *yours* (voice-cloning) — IN PROGRESS

The honest gap: v1 is the operator's *tool* over *Kokoro's* voices. Voice-cloning makes the **voice** yours. (Kokoro can't clone — its 54 voices are fixed — so this is a separate engine, not a Kokoro voice.)

- **Engine chosen: OpenVoice V2** (not XTTS v2). MIT-licensed, so a clone path stays usable even if this project ever ships commercially; XTTS v2's CPML license forbids commercial use. Lighter for the CPU constraint too. Decided up front because switching engines later would mean re-cloning every registered voice.
- **Ethics gate + scaffold landed 2026-06-23** (office box, no model run): `clone.py` holds the consent gate, the voice registry (`voices/<name>/`), and the watermark policy (self-test green, 4 gate cases). `speak.py --voice me` routes a registered clone voice through the gate instead of `kokoro.create()`; Kokoro voices fall through unchanged (verified).
- **Ethics rule:** clone yourself freely; cloning anyone else needs an explicit consent record on file (`voices/<name>/consent.json`) AND forces a watermark on the output. **Honest limit:** the gate enforces honest *process*, not *identity* — `kind: self` is self-asserted, so it's an honesty rail for a cooperating user, not a security control.
- **Engine WIRED + proven end-to-end 2026-07-10 (home box, CPU):** `.venv-openvoice` (dedicated venv — MeloTTS pins + modern faster-whisper; OpenVoice installed `--no-deps` past its rotten 2024 pins) + `checkpoints_v2` from HF (the official S3 zip is dead). `clone.synth` bridges to `clone_worker.py` as a SUBPROCESS, so speak.py's venv stays torch-free. Embedding via direct `extract_se` (librosa) — the `se_extractor` path shells out to ffmpeg, which this repo deliberately doesn't have; per-voice embedding cached (`voices/<name>/se.pth`). `clone.py register <name> <sample>` enrolls a voice. Pipeline proven on a Kokoro-generated stand-in voice (`standin`): synth via `speak.py --voice standin` for save AND live playback; provenance tags verified in-file.
- **Perceptual watermark LIVE with verify-after-embed:** wavmark embeds a 4-char tag (`PNVM`, 32 bits in the first 16000-sample chunk — fits any clip ≥ ~0.75s). Found live: OpenVoice *silently skips* the mark on too-short audio while reporting success — so the worker now decodes the written file and **refuses to emit** a forced-watermark output whose mark can't be verified. The gate's "output WILL be watermarked" is now enforced, not promised.
- **Timing (CPU, warm):** ~20-25s wall per utterance (venv + model load dominates; MeloTTS render ~4s, conversion ~2s). Fine for validation; a resident/daemon mode or CUDA opt-in is the known lever if real use wants it snappier.

**Exit:** `speak.py "..." --voice me` produces a recognizable clone of the operator's own voice.

**✓ EXIT FIRED 2026-07-10** — operator registered his own sample and confirmed by ear same day: *"I would say it works … more than fine. Didn't even expect it to work that well, because that sample I gave it was VERY short."* The short-sample surprise is by design: OpenVoice extracts a single global timbre vector, which stabilizes after ~10s of audio — the trade-off is it captures tone color, not the speaker's rhythm/expressiveness (that ceiling is the engine's, not the sample's). Named friction from first real use: **every utterance reloads all models (~20s wall)** — a fresh worker process per call. Known lever: resident/daemon mode (load once, ~3-5s per utterance); build when interactive use asks for it. Advancing to **After: v2 non-verbal**.

---

## After — v2 non-verbal (locating the one real wall)

Natural laughs/sighs/backchannel — the only capability not yet shown contingent (see README "The bet"). The job here is **wall-location, not wall-breaking:** find out whether human-grade non-verbal is an *absolute* wall on a PC (genuinely needs the compute) or merely an *access* problem (reachable by a non-default route). Sesame CSM-1B is one data point (ruled out: not human-grade + GPU-only); one isn't enough to conclude "open-source can't."

Keep the two axes separate so any result stays honest:
- **Absolute** — can a PC-runnable model *generate* human-grade non-verbal de novo? (The Orpheus spike probes this.)
- **Access** — can human-grade non-verbal appear *in the output* by a non-default technique, even if de-novo generation is walled? (The splice route probes this.)

- **Orpheus 3B spike** (ungated, no token) — does it move the *absolute* boundary past Sesame? A 2nd data point on the wall, not a "settle it" claim.
- **If de-novo stays walled → Bark-precache** (pre-generate laugh clips offline, splice into Kokoro). Reframed: this is the **access route around the wall**, not a "lower-ceiling compromise" — measure how close the spliced output gets, and never conflate "the model generated it" (walled) with "the delivered audio is human-grade, via technique X" (access).
- Report the verdict on the right axis: *absolute* = "PC-native de-novo non-verbal is / isn't there in 2026"; *access* = "human-grade non-verbal in PC output is / isn't reachable by splicing."

**Exit:** a clear verdict **on each axis** — is the wall absolute, and is the access route good enough — with a working path for whichever lands.

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
| 2026-05-25 | Reframe "the bet" as absolute-vs-access per capability | Monolithic "fundamental or contingent?" has no single answer; naturalness + latency are contingent (closed), non-verbal is the lone wall candidate. v2 = wall-*location* (absolute axis = de-novo generation; access axis = splice route), not wall-breaking |
| 2026-05-26 | v1.1 cluster — record/archive/reading-mode + TRIAD+KG audit | Operator's first deep dogfood (real ~1hr text → mp3) fired the "Now" exit and named real frictions same-day. H1 zh-streaming hang was found by audit AND test-confirmed — the bridge would have silently hung on missing `misaki[zh]`. M4 unbounded queue would have buffered the full audio in RAM for long inputs (~200-330 MB for an hour) |
| 2026-05-26 | Bus speech env-configurable (`BUS_VOICE`/`BUS_LANG`/`BUS_MAX_WORDS`) | Same pattern as `eve_voice.py`'s `EVE_VOICE`; LANG auto-derives from voice prefix matching `speak.lang_for`. Default unchanged so existing wiring keeps working. Drove the bus-monitor revival with `af_nicole` and the 18→50 cap fix once a real test exposed mango truncation |
| 2026-06-06 | `speak.py` input edge hardened: utf-8-sig default + `--encoding` fallback chain + `strip_html`/`--html`/`.html` auto-detect | Real use case `--file book.html --record book.mp3` needs non-UTF-8 files to not crash AND HTML to not be read aloud as markup. No charset auto-detect dep (it guesses); strip_html is stdlib-only like strip_markdown |
| 2026-06-06 | `book2audio.py` — document → chaptered audiobook | Generalized from a one-off render of a 700KB HTML book. Reliable spine = numbered size-chunks (any doc); `--by-heading` naming is best-effort (heading conventions vary — verify-first caught a default pattern over-matching prose 35×). GPU default since load-once batch is where CUDA amortizes. Resumable |
| 2026-06-23 | Cloning engine = OpenVoice V2, not XTTS v2 | License is the deciding factor: OpenVoice V2 is MIT (free commercial use, verified from source); XTTS v2's CPML forbids commercial use, which would silently foreclose the sellable-voice-agent path. OpenVoice is also lighter on CPU. Both pull torch (this repo is torch-free today), so the engine goes behind its own optional install. Decided now, not later, because re-cloning every voice on a different engine is the expensive way to find out |
| 2026-06-23 | Ethics gate first, before any cloning | Clone yourself freely; cloning anyone else needs a consent record on file AND forces a watermark. Built the gate + registry + watermark policy (`clone.py`) and the `--voice me` routing (`speak.py`) on the office box — all the parts that run without a model. Honest limit named in the gate itself: it enforces honest *process* (consent recorded, output watermarked), not *identity* (`kind: self` is self-asserted) — a local tool can't verify whose voice a sample is |
| 2026-07-10 | Clone engine = subprocess worker in a dedicated venv | OpenVoice+MeloTTS pull torch + a 2024-pinned dep tree that conflicts with both the Kokoro venv and the coqui venv. `clone_worker.py` runs in `.venv-openvoice`; `clone.synth` shells out to it — speak.py's venv stays torch-free, dep stacks never share a process |
| 2026-07-10 | Embedding via direct `extract_se`, not `se_extractor` | `se_extractor.get_se` → whisper `load_audio` → **ffmpeg subprocess** (absent by design — repo is ffmpeg-free since v1.1). Direct librosa extraction serves the curated-30s-clip case; VAD segmentation only pays on long noisy refs. Also keeps whisper models off the required path |
| 2026-07-10 | Watermark = 4-char tag + verify-after-embed, refuse on failure | Found live: OpenVoice embeds 4 chars per ~1.5s of audio and **silently skips** the mark when the clip is too short — success reported, output unmarked. A long message needed ≥12s audio. Short tag (`PNVM`, one chunk) fits any clip ≥ ~0.75s; the worker decodes the written file and hard-fails rather than ship an unmarked forced-watermark clone |
| 2026-07-10 | CPU-first (no CUDA install) | Same logic as the 2026-05-24 Kokoro call: one-shot CLI is where CUDA cold-start loses. Measured warm: ~20-25s wall/utterance, render ~4s. Device switch stays free (torch is device-portable) — CUDA opt-in or a resident daemon are the levers if real use wants speed |
| 2026-07-13 | Warm-voice daemon (Kokoro) SHIPPED — `integrations/speak_daemon.py` + hook fast path | Interactive use asked (Eve CC surface: per-reply cold start was the whole felt delay). Hook tries daemon (ack ~10ms, first audio ~1-3s) → auto-starts it detached → one-shot fallback for the reply that arrives while it warms — first reply slow exactly once. Latest-wins playback (graceful in-process interrupt, never a process kill) replaces the old overlapping detached one-shots. Same-session fix: `--spoken` now drops INLINE stage directions too ("*smiles* Good —" was being SPOKEN as "smiles Good"; sentence-boundary heuristic keeps real emphasis) — corpus `test_spoken_actions.py` 8 drop + 3 keep green. Operator-confirmed by ear on both paths. Kokoro only — the OpenVoice clone engine (~20s reload) still has no daemon |
