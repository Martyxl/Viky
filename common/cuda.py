"""Make bundled NVIDIA CUDA libraries loadable on Windows (M7 fix).

faster-whisper (ctranslate2) needs cuBLAS/cuDNN DLLs at runtime. When the
`nvidia-cublas-cu12` / `nvidia-cudnn-cu12` pip wheels are installed, their DLLs
live under site-packages/nvidia/*/bin but are NOT on PATH, so CUDA inference
fails with e.g. "cublas64_12.dll is not found". Call `ensure_cuda_dlls()` before
loading a CUDA model to register those directories. No-op off Windows or when
the wheels aren't present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from common.logging import get_logger

log = get_logger("common.cuda")
_DONE = False


def ensure_cuda_dlls() -> None:
    global _DONE
    if _DONE or sys.platform != "win32":
        return
    _DONE = True
    for site in sys.path:
        nvidia = Path(site) / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            if bin_dir.is_dir():
                try:
                    os.add_dll_directory(str(bin_dir.resolve()))
                    os.environ["PATH"] = str(bin_dir.resolve()) + os.pathsep + os.environ.get("PATH", "")
                    log.debug("registered CUDA DLL dir: %s", bin_dir)
                except OSError as exc:  # noqa: PERF203
                    log.debug("skip DLL dir %s: %s", bin_dir, exc)
