#!/usr/bin/env python3
"""clone_worker.py — the heavy half of voice cloning; runs INSIDE .venv-openvoice.

clone.py (the gate + registry, importable anywhere) shells out to THIS script with
the OpenVoice venv's interpreter, because speak.py lives in the torch-free Kokoro
venv and the two dependency stacks must never share a process. Contract:

    .venv-openvoice/Scripts/python.exe clone_worker.py \
        --text "..." --ref voices/me/ref.wav --voice-dir voices/me \
        --out out.wav [--watermark] [--speed 1.0] [--lang EN_NEWEST]

Pipeline (OpenVoice V2):
    1. MeloTTS renders the text in a base speaker            (base timbre)
    2. ToneColorConverter transfers timbre from the ref clip  (your timbre)
    3. wavmark perceptual watermark, when --watermark         (in-signal, survives re-encode)
    4. provenance tags written into the output file           (always — honest origin)

The tone-color embedding of a voice is cached at <voice-dir>/se.pth after the first
extraction, so later calls skip the (slow) whisper/VAD segmentation of the ref clip.

Exit code 0 + final line "CLONE-WORKER-OK <out-path>" on success; nonzero + stderr
otherwise. stdout is machine-readable-last-line; progress goes to stderr.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
CKPT = REPO / "models" / "openvoice" / "checkpoints_v2"

# MeloTTS language -> (melo speaker key, base speaker-embedding file)
LANG_BASE = {
    "EN_NEWEST": ("EN-Newest", "en-newest.pth"),
    "EN": ("EN-US", "en-us.pth"),
    "ES": ("ES", "es.pth"),
    "FR": ("FR", "fr.pth"),
    "JP": ("JP", "jp.pth"),
    "KR": ("KR", "kr.pth"),
    "ZH": ("ZH", "zh.pth"),
}


def log(msg: str) -> None:
    print(f"[clone-worker] {msg}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref", required=True, help="reference sample (the voice to clone)")
    ap.add_argument("--voice-dir", required=True, help="registry folder; se.pth cache lives here")
    ap.add_argument("--out", required=True, help="output audio path (wav)")
    ap.add_argument("--watermark", action="store_true",
                    help="embed the wavmark perceptual watermark (forced for non-self clones)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--lang", default="EN_NEWEST", choices=sorted(LANG_BASE))
    args = ap.parse_args()

    ref = Path(args.ref)
    if not ref.is_file():
        sys.exit(f"reference sample not found: {ref}")
    if not (CKPT / "converter" / "checkpoint.pth").is_file():
        sys.exit(f"OpenVoice checkpoints missing under {CKPT} — run the checkpoint download first")

    import torch  # noqa: E402  (heavy imports after arg/file validation)
    from openvoice.api import ToneColorConverter
    from melo.api import TTS

    device = "cpu"
    log("loading ToneColorConverter…")
    converter = ToneColorConverter(str(CKPT / "converter" / "config.json"), device=device)
    converter.load_ckpt(str(CKPT / "converter" / "checkpoint.pth"))
    if not args.watermark:
        # upstream's enable_watermark kwarg is unreachable (its __init__ forwards kwargs
        # to a base class that rejects it), so the switch is thrown post-construction:
        # convert() watermarks unconditionally whenever watermark_model is set.
        converter.watermark_model = None

    # Target speaker embedding — cached per registered voice after first extraction.
    se_cache = Path(args.voice_dir) / "se.pth"
    if se_cache.is_file():
        log(f"target embedding from cache: {se_cache}")
        target_se = torch.load(se_cache, map_location=device)
    else:
        log("extracting target embedding from reference (first run for this voice)…")
        # Direct extraction (librosa-based) instead of se_extractor.get_se: the VAD /
        # whisper segmentation path shells out to ffmpeg (absent here by design — this
        # repo is deliberately ffmpeg-free) and only pays off on long noisy recordings.
        # A curated ~30s reference is exactly the clean case direct extraction serves.
        target_se = converter.extract_se([str(ref)])
        torch.save(target_se, se_cache)
        log(f"embedding cached: {se_cache}")

    speaker_key, ses_file = LANG_BASE[args.lang]
    source_se = torch.load(str(CKPT / "base_speakers" / "ses" / ses_file), map_location=device)

    log(f"MeloTTS base render ({args.lang}/{speaker_key})…")
    melo = TTS(language=args.lang, device=device)
    spk_id = melo.hps.data.spk2id[speaker_key]
    with tempfile.TemporaryDirectory() as td:
        base_wav = str(Path(td) / "base.wav")
        melo.tts_to_file(args.text, spk_id, base_wav, speed=args.speed)

        log("tone-color conversion…")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        converter.convert(audio_src_path=base_wav, src_se=source_se, tgt_se=target_se,
                          output_path=str(out), message=WATERMARK_TAG)

    if args.watermark:
        _verify_watermark(out, converter)
    _stamp_provenance(out)
    print(f"CLONE-WORKER-OK {out}")
    return 0


# OpenVoice embeds 32 bits (= 4 chars) per 16000-sample chunk, one chunk every other
# second — so a LONG message needs long audio and silently doesn't fit short clips.
# A 4-char tag fits in the first chunk of any clip >= ~0.75s.
WATERMARK_TAG = "PNVM"


def _verify_watermark(path: Path, converter) -> None:
    """The gate's 'output WILL be watermarked' must be true or the output must not
    ship: OpenVoice SKIPS the mark on too-short audio while still reporting success
    (found live 2026-07-10). Decode the written file and refuse on a missing mark."""
    import soundfile
    audio, _sr = soundfile.read(str(path))
    decoded = converter.detect_watermark(audio, n_repeat=1)
    if decoded != WATERMARK_TAG:
        path.unlink(missing_ok=True)
        sys.exit(f"forced watermark could not be embedded/verified (decoded {decoded!r}; "
                 f"audio too short?) — refusing to emit an unmarked non-self clone")
    log("watermark verified in output")


def _stamp_provenance(path: Path) -> None:
    """Honest-origin metadata in the file itself (libsndfile string tags). Always on —
    this is provenance, distinct from the perceptual watermark the gate can force."""
    try:
        import soundfile as sf
        data, sr = sf.read(str(path))
        with sf.SoundFile(str(path), "w", samplerate=sr,
                          channels=1 if data.ndim == 1 else data.shape[1]) as f:
            f.title = "synthetic voice clone"
            f.software = "pc-native-voice-models"
            f.comment = ("AI-generated synthetic speech (OpenVoice V2). "
                         "Synthetic voice clone - not a recording of the named person.")
            f.write(data)
    except Exception as e:  # provenance must never eat a successful synthesis
        log(f"provenance stamp skipped: {e}")


if __name__ == "__main__":
    sys.exit(main())
