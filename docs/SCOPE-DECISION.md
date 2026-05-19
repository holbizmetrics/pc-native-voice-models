# Scope Decision — pc-native-voice-models

**Status:** RESOLVED 2026-05-20 (operator: Holger, recommended by windows-claude PCLA S53). Implementation begins with Phase 1 landscape survey.

## Decision 1: Scope of "from scratch" → **(c) integration-first, with graduation option to (a)**

Pick best open-weight components (Whisper for ASR when v2 lands; Coqui/Piper/Kokoro/StyleTTS2/Bark/Sesame open releases for TTS), build the orchestration + non-verbal layer + PC-native optimization on top.

**Graduation trigger:** if integration hits a wall on a specific pain point (e.g., non-verbal cues can't be added on top of an existing model), graduate to (a) architectural-from-scratch for that component only. Don't pre-commit to (a) for the full pipeline.

**Rejected:**
- **(b) trained-from-random** — years of compute + petabyte data; not feasible without datacenter. Only viable as research-via-failure documentation, deferred indefinitely.
- **(a) full architectural-from-scratch upfront** — months of design + training before any shippable v1. Too long without prior empirical evidence of where integration fails.

## Decision 2: v1 capability → **Talk-only (TTS)**

Focus v1 on the output side: TTS pipeline with latency + naturalness + non-verbal cues. Assume text input.

**Rationale:**
- Most visible quality differentiation against consumer SOTA (consumer TTS audibly trails datacenter; consumer ASR via whisper.cpp / faster-whisper is "already-solved-enough" for many use cases)
- Non-verbal cues (laughs, sighs, backchannel) are output-side work — TTS is where they live
- Simpler integration surface (no microphone, no noise handling, no streaming ASR pipeline)
- ASR (understand side) deferred to v2

## Decision 3: Compute target → **Dual-mode (CPU-only minimum + Consumer GPU full)**

| Mode | Hardware | What works |
|---|---|---|
| **CPU-only minimum** | Modern x86 / Apple Silicon, 16GB RAM, no GPU | Degraded-but-usable TTS — slower latency, possibly smaller model variant |
| **Consumer GPU full** | NVIDIA 8-16GB VRAM OR Apple Silicon GPU | Full quality + low latency + full non-verbal cues |

Both modes are first-class targets. "Normal PC" in 2026 means both exist.

## Decision 4: TTS latency budget → **≤500ms (GPU full mode) / ≤1.5s (CPU minimum mode)**

Measured: time-to-first-audio-sample from text-input-end.

- GPU full mode target ≤500ms matches datacenter SOTA (ElevenLabs ~500-700ms typical, OpenAI Realtime ~300-500ms typical)
- CPU minimum mode target ≤1.5s acknowledges the compute reality; degraded but usable

**Per-pain-point latency budgets within the 500ms / 1.5s window:** TBD in Phase 1 landscape survey output.

---

## Resolved table

| Decision | Value |
|---|---|
| Scope of "from scratch" | (c) integration-first, graduate to (a) on specific failed pain points |
| v1 capability | Talk-only (TTS) |
| Compute target | Dual-mode: CPU-only minimum + Consumer GPU full |
| TTS latency budget | ≤500ms GPU full / ≤1.5s CPU minimum |

## Next moves

1. **Phase 1 landscape survey** — output at `Researches/pc-native-voice-models/LANDSCAPE-2026-05.md`
   - Consumer SOTA TTS: Piper, Coqui-XTTS, Kokoro, StyleTTS2, OpenVoice, Bark, Sesame open
   - Datacenter SOTA TTS: ElevenLabs, OpenAI Realtime, Sesame full, Hume, Resemble — for capability-gap analysis
   - PC-native TTS research: distillation, quantization, streaming
   - Non-verbal cue research: Sesame papers + EmoTTS + backchannel modeling
2. **Component selection** — from landscape survey, pick 1-2 candidate base components for v1
3. **Initial benchmark** — run candidate on this PC, measure latency + RAM + quality subjectively
4. **v1 scaffold** — Python project skeleton, audio I/O, candidate component integration

## Cross-references

- Research-side thread: [Researches/pc-native-voice-models](https://github.com/holbizmetrics/Researches/tree/main/pc-native-voice-models)
- Pain-points list: [`../README.md`](../README.md)
