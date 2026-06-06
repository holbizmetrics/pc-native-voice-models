# PC-Native Voice Models

> Voice models with full pain-point coverage (latency, naturalness, non-verbal cues like laughs) — running on a normal consumer PC.

**Status:** **v1 shipped (talk-only TTS).** `speak.py` turns text into streamed speech on CPU via Kokoro. Scope resolved in [`docs/SCOPE-DECISION.md`](docs/SCOPE-DECISION.md); landscape + spike findings in the [research thread](https://github.com/holbizmetrics/Researches/tree/main/pc-native-voice-models).

## v1 — Usage

```bash
# setup (once)
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt kokoro-onnx
# download Kokoro weights into models/ (kokoro-v1.0.onnx + voices-v1.0.bin)

# speak
python speak.py "Hello, this is my own voice model, running on the CPU."
python speak.py "..." --voice af_bella --speed 1.1
python speak.py --file story.txt        # speak a text file
echo "piped text works too" | python speak.py
python speak.py "save instead of play" --save out.wav
python speak.py --list-voices          # 54 voices, 8 languages
python speak.py -h                      # full help with examples
```

Streams sentence-by-sentence — first words start ~1s after enter, gapless after that (generator runs ~2× faster than playback on CPU). No GPU required.

**`speak` from anywhere (launcher):** `speak.cmd` lets you run `speak "..."` without typing the venv path. Add the repo dir to your user PATH once:

```powershell
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";D:\FromGitHubEtc\pc-native-voice-models", "User")
# open a NEW terminal, then:
speak "Now it works from anywhere."
speak --file notes.txt
speak "你好" --voice zf_xiaobei
```

(`speak.cmd` resolves its own location, so keep it in the repo dir; just put that dir on PATH rather than copying the file.)

### GPU mode (NVIDIA, optional — for the resident monitor / long text)

CPU is the **default**, and it's the right choice for one-shot `speak.py` calls. A fresh GPU process pays the full CUDA cold-start every launch (context init + cuDNN first-conv autotune), so GPU is actually *slower* to first audio for a single short utterance. Measured on an RTX 3060 (time-to-first-audio):

| path | first audio |
|---|---|
| CPU | ~3.2s |
| CUDA, warm disk | ~5.6s |
| CUDA, cold disk (post-reboot) | ~18.8s |

GPU **wins** only where the model loads once and is reused: the resident bus monitor (per-message gen ~2.5s → ~0.5s) and long streamed text (every chunk after the first is ~5× faster). So GPU is **opt-in**, not the default.

Install the CUDA runtime (no system CUDA, no admin — it's all pip wheels):

```bash
.venv/Scripts/pip uninstall onnxruntime -y
.venv/Scripts/pip install onnxruntime-gpu==1.23.2 \
  nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 \
  nvidia-curand-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12
```

Then enable GPU per-invocation (only worth it for long text):

```bash
KOKORO_GPU=1 python speak.py "long passage where the 5x warm-gen pays off..."
# equivalently: ONNX_PROVIDER=CUDAExecutionProvider
```

speak.py adds the wheels' `nvidia/*/bin` dirs to the DLL search path itself (cuDNN 9 lazily loads sub-libraries that `preload_dlls()` alone misses). `KOKORO_CPU=1` forces CPU even if a GPU env var is set. Time the first audio yourself with `SPEAK_TIMING=1`.

> **DirectML doesn't work** for Kokoro — its F0 `ConvTranspose` op fails on the DmlExecutionProvider. CUDA is the only working GPU path.

### Languages

Pick a voice — the language **auto-derives from the voice prefix**, no `--lang` needed:

```bash
python speak.py "Bonjour le monde." --voice ff_siwis      # French, auto
python speak.py "Hola, qué tal." --voice ef_dora          # Spanish, auto
python speak.py "こんにちは。" --voice jf_alpha             # Japanese, auto
```

| Language | Voice prefix | auto `--lang` | works in base v1 |
|---|---|---|---|
| US English | `af_` `am_` | en-us | ✓ |
| British English | `bf_` `bm_` | en-gb | ✓ |
| Spanish | `ef_` `em_` | es | ✓ |
| French | `ff_` | fr-fr | ✓ |
| Hindi | `hf_` `hm_` | hi | ✓ |
| Italian | `if_` `im_` | it | ✓ |
| Portuguese (BR) | `pf_` `pm_` | pt-br | ✓ |
| Japanese | `jf_` `jm_` | ja | ✓ |
| Mandarin | `zf_` `zm_` | zh | ✓ via `misaki[zh]` (auto-routed; speak.py runs misaki G2P → kokoro `is_phonemes`) |

`--lang` is still accepted as an explicit override.

**What works (v1):** clean speech, 54 voices, **9 languages auto-detected from voice** (incl. Mandarin via misaki[zh]), streaming low-latency playback, save-to-WAV, CPU-only.
**Not in v1:** non-verbal cues (laughs/sighs). Sesame CSM-1B was tested and ruled out — not human-grade laughs + ~14× realtime on CPU (see research thread `SESAME-SPIKE-RESULT`). Natural non-verbal remains an open research question, deferred to v2.

## book2audio.py — document → chaptered audiobook

Turn a long `.html` / `.txt` / `.md` into a folder of ordered, size-capped mp3s. Loads Kokoro **once** and defaults to **GPU** (this is a load-once batch — the one case where CUDA's cold-start is amortized away). Built on `speak.py`'s text helpers (HTML/Markdown stripping, encoding-robust read).

```bash
python book2audio.py book.html                          # numbered ~16-min mp3s (reliable)
python book2audio.py book.html --by-heading --voice af_nicole
python book2audio.py book.html --by-heading --dry-run   # preview the split first
python book2audio.py notes.md --out audio --max-min 10 --cpu
```

- **Reliable:** numbered size-chunks split at paragraph/sentence boundaries — works on any document.
- **Best-effort:** `--by-heading` names pieces after detected headings (Chapter/Prologue/Part/`#`/… ; `--heading-regex` to override) and skips a front table-of-contents when it detects one. Heading conventions vary between books, so it **always falls back to size-chunks** and you should eyeball `--dry-run` before trusting chapter-accurate naming.
- **Resumable:** pieces already on disk are skipped, so an interrupted run continues where it stopped. Each piece streams to disk (bounded memory).

## The bet

State-of-art voice quality (ElevenLabs, OpenAI Realtime, Sesame, Hume) requires datacenter inference. Consumer-PC voice (Piper, Coqui, Bark, Whisper variants) makes serious quality compromises — latency spikes, robotic prosody, no non-verbal capability.

**Don't ask the gap question monolithically.** "Is datacenter-vs-PC voice *fundamental* or *contingent*?" has no single answer — it splits per capability. Decompose, and tag each as either an **absolute wall** (it genuinely needs the compute) or an **access** problem (reachable on a PC, just not by the *default* tooling):

| Capability | Verdict | Evidence |
|---|---|---|
| **Naturalness** | contingent | Kokoro hits MOS 4.2 at 82M params on CPU — there's no quality wall at this size |
| **Latency** | contingent, *already routed around* | streaming + first-chunk peel gets felt-latency under target with no faster compute |
| **Non-verbal** (laughs/sighs) | the one real **wall candidate** | Sesame ruled out (not human-grade + GPU-only); still unproven either way (v2) |

So the repo isn't betting on one yes/no. It **closes the contingent gaps** (done: naturalness + latency) and **locates the absolute wall** precisely (non-verbal, v2). Where a capability turns out genuinely walled, the honest move is the **access route** — reach it by a non-default technique (e.g. pre-generated splices) and *measure the gap* — not pretend the wall isn't there. A capability that "reduces to a known compute limit" is not worthless; conflating *walled* with *worthless* is the category error to avoid.

## Pain points to address

In approximate order of operator-felt importance:

1. **Latency / immediate response** — round-trip from speech-end to model-speech-start. Datacenter ~500ms; consumer typically 2-5s+.
2. **Naturalness** — prosody, rhythm, breathing, conversational pacing. Robotic delivery is the most consistent giveaway.
3. **Non-verbal cues** — laughs, sighs, "um/uh", interruption handling, backchannel ("mhm"). Sesame's work demonstrates this is solvable; most consumer models drop it entirely.
4. **Understanding (ASR side)** — accent robustness, noise robustness, code-switching, fast streaming with partial results.
5. **Multi-voice / voice cloning** — with appropriate ethical guardrails.
6. **Multi-language** — same model, multiple languages, ideally code-switching mid-sentence.
7. **Emotion / tone control** — explicit control over delivered emotion when needed.
8. **Robustness** — silence, noise, music-in-background, multiple speakers.

## Constraint: "Normal PC"

Target spec (to refine in scope decision):

- CPU-only feasibility for at minimum a degraded-but-usable mode
- Consumer GPU (8-16GB VRAM) for full mode
- RAM budget: 8-16GB
- Latency budget per pain point (TBD per component)

## Phase 1 — Landscape survey (next moves)

See [`docs/SCOPE-DECISION.md`](docs/SCOPE-DECISION.md) for the three competing framings of "from scratch" and the v1 scope sequencing (talk-only / understand-only / both-minimal).

Survey targets:
- Consumer SOTA: Piper, Coqui, Kokoro, StyleTTS2, OpenVoice, Bark, Sesame open releases, whisper.cpp, distil-whisper, faster-whisper
- Datacenter SOTA: ElevenLabs, OpenAI Realtime, Sesame full, Hume, Resemble
- Existing research on the gap: TTS distillation, ASR quantization, on-device deployment
- Non-verbal cue research: Sesame papers, conversational-AI laugh/pause/breathing modeling, EmoTTS

## Origin

Filed 2026-05-20 by operator (Holger). Cross-tracked in [Researches/pc-native-voice-models](https://github.com/holbizmetrics/Researches/tree/main/pc-native-voice-models) — research thread there links here as the build-side of the same effort.

## License

MIT — see [LICENSE](LICENSE).
