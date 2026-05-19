# PC-Native Voice Models

> Voice models with full pain-point coverage (latency, naturalness, non-verbal cues like laughs) — running on a normal consumer PC.

**Status:** initial-research / pre-build. Scope decision pending — see [`docs/SCOPE-DECISION.md`](docs/SCOPE-DECISION.md).

## The bet

State-of-art voice quality (ElevenLabs, OpenAI Realtime, Sesame, Hume) requires datacenter inference. Consumer-PC voice (Piper, Coqui, Bark, Whisper variants) makes serious quality compromises — latency spikes, robotic prosody, no non-verbal capability.

The research question: is the gap between datacenter-quality voice and PC-runnable voice **fundamental** (it requires the compute) or **contingent** (the field optimized for capabilities first, compute efficiency under fixed budget hasn't been closed)?

If contingent: this repo builds the closure. If fundamental: this repo documents *why* with empirical receipts, which is still useful to the field.

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
