#!/usr/bin/env python3
"""eve_voice.py — give FVPA's Eve a voice (the FVPA -> TTS half of the agent loop).

  your text -> Claude with the FVPA system prompt (Eve) -> reply -> spoken aloud
  via the local voice model (speak.py).

This is step 1 of the callable empathy agent: the empathy engine (holbizmetrics/FVPA)
gets a voice. STT (the "ear") and a callable interface come next.

System prompt = an FVPA version (default 6.0) + selected EUP emotional-unlock tiers
(default tier1_core), read live from the local FVPA checkout.

API key: env ANTHROPIC_API_KEY, else a gitignored <repo>/.eve_key file. Never paste
the key into chat.

Config (env):
  FVPA_DIR     path to the FVPA checkout          (default D:\\FromGitHubEtc\\FVPA)
  EVE_FVPA     FVPA version stem                  (default fvpa_6_0)
  EVE_TIERS    comma-sep EUP tiers                (default tier1_core)
  EVE_MODEL    Claude model                       (default claude-sonnet-4-6;
                                                   set claude-opus-4-7 for max empathy)
  EVE_VOICE    Eve's Kokoro voice                 (default af_heart — warm)
  EVE_MAX_TOKENS  reply cap                       (default 800)
  EVE_ARCHIVE  set 1 to keep every reply          (off by default)
                 -> <dir>/eve_<timestamp>.mp3 + .txt transcript, recorded as Eve speaks
  EVE_ARCHIVE_DIR  where archives are written     (default <repo>/eve-archive)

Usage:
  python integrations/eve_voice.py "I had a rough day."
  python integrations/eve_voice.py --dry-run          # assemble prompt, no API call
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # import speak.py from the repo root
import speak  # noqa: E402

FVPA = Path(os.getenv("FVPA_DIR", r"D:\FromGitHubEtc\FVPA")) / "API" / "prompts"
EVE_FVPA = os.getenv("EVE_FVPA", "fvpa_6_0")
EVE_TIERS = [t.strip() for t in os.getenv("EVE_TIERS", "tier1_core").split(",") if t.strip()]
EVE_MODEL = os.getenv("EVE_MODEL", "claude-sonnet-4-6")
EVE_VOICE = os.getenv("EVE_VOICE", "af_heart")
EVE_MAX_TOKENS = int(os.getenv("EVE_MAX_TOKENS", "800"))
EVE_ARCHIVE = bool(os.getenv("EVE_ARCHIVE"))  # truthy -> save each reply (audio + transcript)
EVE_ARCHIVE_DIR = Path(os.getenv("EVE_ARCHIVE_DIR", str(ROOT / "eve-archive")))


def _api_key() -> str:
    k = os.getenv("ANTHROPIC_API_KEY")
    if k:
        return k.strip()
    f = ROOT / ".eve_key"
    if f.is_file():
        return f.read_text(encoding="utf-8").strip()
    sys.exit("No Anthropic API key found.\n"
             "  Either: set ANTHROPIC_API_KEY in your environment,\n"
             f"  or: put the key (one line) in {f}  (already gitignored).\n"
             "  Do NOT paste the key into chat.")


def build_system() -> str:
    parts = [(FVPA / f"{EVE_FVPA}.txt").read_text(encoding="utf-8")]
    for t in EVE_TIERS:
        p = FVPA / "eup_tiers" / f"{t}.txt"
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8"))
        else:
            print(f"[eve] WARN: tier not found: {p}", file=sys.stderr)
    return "\n\n".join(parts)


def _for_speech(text: str) -> str:
    """Markdown -> clean speech. Delegates stripping to speak.strip_markdown with
    drop_actions=True, which removes the standalone *stage-direction* lines Eve
    emits ("*settling in*", "*tears, if I had them*") so they aren't read aloud
    verbatim — then turns paragraph breaks into spoken pauses."""
    text = speak.strip_markdown(text, drop_actions=True)
    out = []
    for para in re.split(r"\n{2,}", text):            # each paragraph -> a spoken unit
        para = re.sub(r"[ \t\n]+", " ", para).strip()
        if not para:
            continue
        if out and not re.search(r"[.!?…]$", out[-1]):  # add a pause-period only if needed
            out[-1] += "."
        out.append(para)
    return " ".join(out).strip()


def eve_reply(user_text: str, system: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_api_key())
    msg = client.messages.create(
        model=EVE_MODEL, max_tokens=EVE_MAX_TOKENS, system=system,
        messages=[{"role": "user", "content": user_text}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def main() -> None:
    args = sys.argv[1:]
    dry = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]
    user_text = " ".join(args).strip()
    if not user_text and not dry and not sys.stdin.isatty():
        user_text = sys.stdin.read().strip()

    system = build_system()
    print(f"[eve] FVPA={EVE_FVPA} tiers={','.join(EVE_TIERS)} model={EVE_MODEL} "
          f"voice={EVE_VOICE} | system={len(system)} chars", file=sys.stderr)

    if dry:
        print("[eve] --dry-run: system prompt assembled, speak.py imported OK. "
              "No API call made.", file=sys.stderr)
        return
    if not user_text:
        sys.exit('usage: eve_voice.py "your message to Eve"')

    reply = eve_reply(user_text, system)
    print(f"\nEve: {reply}\n")

    kokoro = speak.load_kokoro()
    lang = speak.VOICE_LANG.get(EVE_VOICE[:2], "en-us")

    record_path = None
    if EVE_ARCHIVE:
        from datetime import datetime
        EVE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        record_path = EVE_ARCHIVE_DIR / f"eve_{stamp}.mp3"
        record_path.with_suffix(".txt").write_text(reply, encoding="utf-8")  # full-fidelity text
        print(f"[eve] archiving reply (audio + transcript) -> {record_path}", file=sys.stderr)

    speak.speak_streaming(kokoro, _for_speech(reply), EVE_VOICE, 1.0, lang,
                          record_path=record_path)


if __name__ == "__main__":
    main()
