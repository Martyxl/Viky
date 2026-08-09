"""Launch the Viky app: start the web server and open the orb UI in a window.

    python scripts/viky_ui.py

Opens the UI in Edge/Chrome "app mode" (no browser chrome) when available, so it
looks like a native app. Double-click launch: use Viky.bat.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402

# Common Windows install locations for app-mode browsers.
_BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def _open_app_window(url: str) -> None:
    exe = next((p for p in _BROWSERS if Path(p).exists()), None)
    if exe is None:
        exe = shutil.which("chrome") or shutil.which("msedge")
    if exe:
        try:
            subprocess.Popen([exe, f"--app={url}", "--window-size=520,760"])
            return
        except Exception:  # noqa: BLE001
            pass
    webbrowser.open(url)


def main() -> int:
    import uvicorn

    host = settings.webui_host
    port = settings.webui_port
    url = f"http://{host}:{port}"
    print(f"Viky UI: {url}")

    # Open the window a moment after the server starts.
    threading.Timer(2.0, lambda: _open_app_window(url)).start()

    uvicorn.run("webui.app:app", host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
