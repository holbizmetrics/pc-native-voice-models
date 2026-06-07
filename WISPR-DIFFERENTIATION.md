# Our Own Wispr Flow — Pain Points & Durable Differentiation

*Research run 2026-06-08. Honest status only — claims here are graded by whether
they survive commoditization, not by whether they sound good. Companion to
`dictate.py` (the local speech-to-text tool, the inverse of `speak.py`).*

---

## TL;DR

- `dictate.py` is our **local, own-the-stack dictation tool** — push-to-talk → Whisper (local) → cleaned → typed into any app. Built + proven live 2026-06-08 (ASR, mic, push-to-talk, `--clean`, clipboard injection in a real browser; pre-roll + VAD added same day).
- **Almost every Wispr Flow pain point is either (a) already fixed by being local, or (b) commoditizes within ~1 year.** Neither is where we compete.
- The **only two durable differentiators**:
  1. **Tuned to you** — your dictionary, your voice, your workflows. (Most-confirmed gap in the whole haul — 4+ independent sources.)
  2. **The ear of *your* local agent (Eve)** — not a dictation endpoint, an organ of your own agent. The one thing no competitor, even agentic-Wispr, can structurally be.
- **Deepest "why":** kill the **translation tax** — the friction between thought and text — so you express more, *especially to your AI* (one power user: 3–4× more context per AI prompt).
- **Honest:** "local" is **table-stakes**, not a moat (Handy/Spokenly already ship it). And frontier labs (OpenAI/Google/Anthropic) will commoditize voice-to-AI — so our durable position is a **defensible niche** (own-it / privacy / control / *your* agent), not the mass market.

---

## Pain point → does it commoditize? → our answer

| Wispr pain point | Root cause | Commoditizes? | Our durable answer |
|---|---|---|---|
| Privacy: audio (and reportedly active-window screenshots) sent to cloud; "Privacy Mode" is policy, not verifiable | cloud architecture | No — structural to a SaaS | **Local by construction**; nothing leaves the machine; *verifiable* because you own the code |
| Outages / "works 60% after you pay" / Trustpilot 2.7 | their servers, their incentives | No | No server to fail; behavior is stable day-to-day |
| Latency (network round-trip) | cloud | No | Local inference (~0.4–0.5s measured on the 3060) |
| No offline mode | cloud | No | Fully offline |
| $15/mo, subscription fatigue | SaaS model | No | Free, unlimited, owned |
| Model silently degrades after cost-cutting "model swaps" | their cost incentives ≠ yours | No | **Pinned model** — only changes when *you* change it |
| Real-time AI cleanup of prose | the one feature Wispr has | **Yes (~1yr)** — LLMs commoditize it; native Android already catching up | Don't lean on this; `--clean` is parity, not edge |
| Generic "real-world actions" | — | **Yes** — Wispr's own roadmap, this year | Don't bet here; the labs are coming too |
| First-word clipping | OS mic-activation lag after hotkey | (bug, not strategy) | **Fixed**: always-on pre-roll ring buffer (`PREROLL_S`) captures the word *before* the press |
| "Doesn't know your dictionary" (jargon: AWS lambda→"Alice's lamb", etc.) | generic model, no personalization | Partially (everyone will add some) | **Tuned-to-you dictionary** — *durable bet #1* |
| Pause = loses the thread | auto-segments on silence | — | Our push-to-talk holds the whole utterance until release; pauses don't break it |
| Punctuation/spacing varies per app (Teams, Reddit) | post-injection / app-dependent cleanup | — | `--clean` runs *our* side before injection → consistent across apps |

---

## The two durable bets (everything else commoditizes)

**1. Tuned to you.** The most-repeated concrete complaint across every source is *"it doesn't know my dictionary."* A bias/correction layer for your terms (PCLA, Kokoro, Λ, repo names, people) kills a whole error class the generic cloud model can't be bothered to fix per-user. Cheap, daily payoff, hard for a mass-market SaaS to match per-person. **Highest-confidence next build.**

**2. The ear of your agent (Eve).** Wispr dead-ends at "text in your app." Ours doesn't have to: the same slot that holds `--clean` can hand text to **Eve**, get a reply, speak it back via `speak.py` — `dictate → understand → Eve → speak`, a voice *conversation*, not transcription. In the voice-agent north-star: speak.py = mouth, bus = nervous system, Eve = heart — **this is the ear.** Even as Wispr adds "actions," those are *their* cloud agent's actions; this is *your* local agent that knows you. That distinction is the only one that survives.

---

## The deepest "why": the translation tax

The real value of voice input isn't speed — it's removing the gap between thought and text that makes people skip the reply, oversimplify, or stay silent. One power user: AI prompts went 100–200 → 600–700 words (**3–4× context per interaction**), wiki 30min → 5min. North-star UX, in his words: *"the product has dissolved into the act of having a thought."* For our use, *3–4× context to the model* is the prize.

---

## Honest edges (where we lose or aren't better)

- **"Local" is not a moat** — Handy (Windows), Spokenly (Mac/iOS) already ship local dictation. It's the price of entry, not the win.
- **Frontier-lab threat** — when OpenAI/Google/Anthropic make voice-to-their-AI excellent, the mass market for differentiated dictation closes. Our durable spot is the *own-it / your-agent* niche.
- **Noisy real-life environments** — Whisper degrades in noise like any ASR. The "dictate while cooking with a kid around" dream is unsolved across the board, us included.
- **Mobile / iOS** — no system-wide hotkey path; we're a desktop/Windows tool. No mobile story.
- **Mic quality & contention** — input quality matters (a headset helps everyone, us too); and "one mic, two consumers" (can't dictate while the mic is locked by a meeting) is a constraint we likely share.
- **RAM** — the model sits resident (base.en light, large-v3 ~3GB); we're only leaner than Wispr's ~800MB-idle if we stay small or add unload-on-idle. Not an automatic win.
- **Polish** — Wispr is a funded product (UI, sync, integrations, context-awareness). We're a CLI MVP. And its community support looked genuinely good in one thread — the "scripted bot" complaint is inconsistent, not universal.

---

## Built vs owed

**Built + verified (2026-06-08):** `dictate.py` — push-to-talk (`pynput`), always-on mic + pre-roll ring buffer (first-word fix), faster-whisper ASR (CPU int8 default, `--gpu`), `vad_filter` (silence-hallucination guard), `--clean` deterministic layer (fillers, voice commands, tidy), clipboard-paste / `--type` injection, `--once` test mode. ASR proven on real audio; clean_text unit-checked; injection proven live in browser; pre-roll proven mechanically.

**Owed:** live hold-key re-test confirming pre-roll prevents first-word clipping in real speech; **tuned-to-you dictionary (durable bet #1)**; **Eve loop (durable bet #2)**; model eval (Parakeet / large-v3 for accuracy + accents); optional local-LLM `--clean` backend (true "what you meant" rewrite); `dictate.cmd` launcher + quit-hotkey for terminal-free use.

---

## Sources

Web: Medium (Ryan Shrott "Trust Gap"; Aadityasinh Jadeja "460,000 words"); ModelPiper, VocAI, get-whisper (privacy incident); Weesper Neon Flow, Spokenly, Voicescriber (offline/pricing/limits). Reddit: r/WisprFlow (LordSarcaus 6-week review + Wispr's own reply on roadmap), r/ProductivityApps (No-Presentation298 clunky-execution; the Spokenly/Handy/Parakeet thread; the Aqua Voice/mahasen "first-word" thread). Google AI Overview (free-tier cap, mobile recording caps, model-swap accuracy drops). Sources skew toward Wispr-*alternative* blogs (competitive bias noted); load-bearing facts (cloud-only, $15/mo, audio-to-cloud, roadmap) are corroborated by Wispr's own docs/community.
