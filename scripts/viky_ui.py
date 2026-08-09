"""Launch the Viky app: start the web server and open the orb UI in a window.

    python scripts/viky_ui.py

Robust relaunch: if Viky is already running on the port, just open the window;
if the port is held by a stale/dead process, free it first. Opens the UI in
Edge/Chrome "app mode" so it looks like a native app. Double-click: Viky.bat.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402

_BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def _open_app_window(url: str) -> None:
    exe = next((p for p in _BROWSERS if Path(p).exists()), None) or shutil.which("chrome") or shutil.which("msedge")
    if exe:
        try:
            subprocess.Popen([exe, f"--app={url}", "--window-size=520,780"])
            return
        except Exception:  # noqa: BLE001
            pass
    webbrowser.open(url)


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _is_viky(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.5) as r:
            return b"webui" in r.read()
    except Exception:  # noqa: BLE001
        return False


def _pids_on_port(port: int) -> list[int]:
    pids: set[int] = set()
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if f":{port} " in line and "LISTENING" in line:
                parts = line.split()
                if parts and parts[-1].isdigit():
                    pids.add(int(parts[-1]))
    except Exception:  # noqa: BLE001
        pass
    return list(pids)


def _free_port(port: int) -> None:
    for pid in _pids_on_port(port):
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    time.sleep(1.0)


def main() -> int:
    import uvicorn

    host = settings.webui_host
    port = settings.webui_port

    # Use HTTPS if a cert exists (required for phone microphone access).
    cert = ROOT / "certs" / "viky.crt"
    key = ROOT / "certs" / "viky.key"
    ssl = cert.exists() and key.exists()
    scheme = "https" if ssl else "http"
    local = f"{scheme}://127.0.0.1:{port}"

    if _port_busy(port):
        if _is_viky(port):
            print("Viky už běží — otevírám okno.")
            _open_app_window(local)
            return 0
        print(f"Port {port} drží zaseknutý proces — uvolňuji…")
        _free_port(port)
        if _port_busy(port):
            print(f"Port {port} je pořád obsazený. Zavři starou Viky a zkus znovu.")
            return 1

    print(f"Viky UI: {scheme}://{host}:{port}"
          + ("" if ssl else "  (bez HTTPS — mikrofon na telefonu nepůjde; spusť scripts/make_cert.py)"))
    threading.Timer(2.0, lambda: _open_app_window(local)).start()
    ssl_kw = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)} if ssl else {}
    try:
        uvicorn.run("webui.app:app", host=host, port=port, log_level="warning", **ssl_kw)
    finally:
        os._exit(0)  # force-exit so no lingering thread keeps the port held
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
