# Scope Decision — pc-native-voice-models

**Status:** OPEN. Two decisions pending operator + Phase 1 landscape survey output.

## Decision 1: What does "from scratch" mean here?

Three framings, with different research-value and feasibility profiles:

| Scope | What it means | Feasibility | Research value | Time to v1 |
|---|---|---|---|---|
| **(a) Architecturally from scratch** | Novel architecture; allowed to bootstrap from existing pre-trained audio components (wav2vec / Whisper encoders / Encodec / etc.) and re-fine-tune | Months of design + training; feasible for components, harder for full pipeline | HIGH (genuinely novel) | 6-12 months |
| **(b) Trained from scratch** | Standard architectures, but trained from random init on operator's own data | Years of compute + petabyte data; not feasible without datacenter | Only if framed as research-via-failure (document why not feasible) | N/A (infeasible) |
| **(c) Integration from scratch** | Pick best open-weight components (Whisper for ASR, Coqui/Piper/Kokoro/StyleTTS2/Bark for TTS), build the orchestration + non-verbal layer + PC-native optimization on top | Weeks to months; very feasible | MED-HIGH (novelty is in integration + PC-native optimization + non-verbal layer) | 4-12 weeks |

**Likely:** (a) or (c). Decision pending Phase 1 landscape survey.

**My (initial) lean:** start with (c) for v1 (ship something usable in 4-12 weeks), evaluate whether (a)-style components are necessary for specific pain points that (c) can't address.

## Decision 2: v1 capability scope

Three sequencing options:

| Option | What v1 ships | Pros | Cons |
|---|---|---|---|
| **Talk-only first** | TTS pipeline with latency + naturalness + non-verbal cues; assume text input | Demonstrates naturalness gains visibly; no microphone integration; no noise handling; simpler first ship | Doesn't validate the full loop; can't test end-to-end latency |
| **Understand-only first** | ASR pipeline with accent + noise + streaming + interruption | Whisper-tier already strong → less white space for novel research | Less visible quality gain; the "voice model" framing implies output, not just input |
| **Both, minimal** | Full loop, each component minimal; demonstrates the integration question | Validates end-to-end latency; tests interruption handling; ships a real demo | More moving parts; harder to scope-lock |

**My (initial) lean:** Talk-only first. The TTS-side has more visible quality differentiation against consumer SOTA, and the non-verbal cue research (laughs, sighs, backchannel) is most clearly *output* work. ASR can come in v2 once TTS is working.

## Decisions pending operator

- Scope-of-from-scratch: (a) architectural, (b) trained-from-random, or (c) integration?
- v1 capability: talk-only, understand-only, or both-minimal?
- Compute target: CPU-only minimum mode? Consumer-GPU expected mode? VRAM ceiling?
- Latency targets per pain point (per-pain-point budget)?

## Landscape survey targets (Phase 1)

Before resolving the above, survey:

1. **Consumer SOTA TTS:** Piper, Coqui-XTTS, Kokoro, StyleTTS2, OpenVoice, Bark, Sesame open releases
2. **Consumer SOTA ASR:** whisper.cpp, distil-whisper, faster-whisper, Conformer variants
3. **Datacenter SOTA:** ElevenLabs, OpenAI Realtime, Sesame full, Hume, Resemble — capabilities that exist that consumer hasn't reproduced, and why
4. **PC-native compute research:** TTS distillation papers, ASR quantization, GGUF/llama.cpp-style optimization for audio, MLX/CUDA/CoreML pipelines for audio
5. **Non-verbal cue research:** Sesame papers + conversational-AI work on laugh/pause/breathing modeling + EmoTTS work + work on backchannel modeling
6. **Latency-specific work:** streaming TTS, partial-output ASR, full-duplex audio handling

## Cross-references

- Research-side thread: [Researches/pc-native-voice-models](https://github.com/holbizmetrics/Researches/tree/main/pc-native-voice-models)
- Pain-points list: [`../README.md`](../README.md)
