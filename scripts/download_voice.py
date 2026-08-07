"""Download a Piper voice from the rhasspy/piper-voices HuggingFace repo.

    python scripts/download_voice.py                      # default = PIPER_VOICE
    python scripts/download_voice.py cs_CZ-kasandra-medium
    python scripts/download_voice.py cs_CZ-jirka-medium

Places <name>.onnx and <name>.onnx.json into voices/ (gitignored).
Voice name format: <lang>_<REGION>-<name>-<quality>, e.g. cs_CZ-kasandra-medium.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = ROOT / "voices"
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def _parse(name: str) -> str:
    # cs_CZ-kasandra-medium -> cs/cs_CZ/kasandra/medium
    lang_region, voice, quality = name.split("-", 2)
    lang = lang_region.split("_", 1)[0]
    return f"{lang}/{lang_region}/{voice}/{quality}"


def _fetch(url: str, dest: Path) -> None:
    print(f"  -> {dest.name}")
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:  # noqa: S310
        while True:
            block = r.read(1 << 16)
            if not block:
                break
            f.write(block)


def main(argv: list[str]) -> int:
    # Resolve the requested voice (fall back to configured default).
    if argv:
        name = argv[0]
    else:
        from config.settings import settings

        name = settings.piper_voice

    sub = _parse(name)
    base = f"{HF_BASE}/{sub}/{name}"
    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    onnx = VOICES_DIR / f"{name}.onnx"
    cfg = VOICES_DIR / f"{name}.onnx.json"
    if onnx.exists() and cfg.exists():
        print(f"Already present: {onnx}")
        return 0

    print(f"Downloading voice '{name}' from HuggingFace...")
    _fetch(f"{base}.onnx.json", cfg)
    _fetch(f"{base}.onnx", onnx)
    print(f"Done: {onnx} ({onnx.stat().st_size // (1024*1024)} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
