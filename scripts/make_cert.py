"""Generate a self-signed HTTPS certificate so phones can use the mic.

Browsers only allow microphone access over HTTPS (or localhost). To talk to
Viky from a phone on the LAN, the UI must be served over https. This creates a
self-signed cert covering localhost + this machine's LAN IPs.

    python scripts/make_cert.py

Writes certs/viky.crt and certs/viky.key. The phone will show a one-time
"not trusted" warning — accept it (Advanced -> proceed). Re-run if your IP
changes.
"""

from __future__ import annotations

import datetime
import ipaddress
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CERT_DIR = ROOT / "certs"


def _lan_ips() -> list[str]:
    ips = {"127.0.0.1"}
    try:
        out = subprocess.run(["ipconfig"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "IPv4" in line and ":" in line:
                ip = line.split(":")[-1].strip()
                # skip WSL/Hyper-V virtual ranges (172.x) — keep home LAN IPs
                if ip and not ip.startswith("172."):
                    ips.add(ip)
    except Exception:  # noqa: BLE001
        pass
    try:
        ips.add(socket.gethostbyname(socket.gethostname()))
    except Exception:  # noqa: BLE001
        pass
    return sorted(ips)


def main() -> int:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    CERT_DIR.mkdir(exist_ok=True)
    ips = _lan_ips()
    print("Certifikát pro:", ", ".join(ips), "+ localhost")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Viky")])
    san = [x509.DNSName("localhost")]
    for ip in ips:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )

    (CERT_DIR / "viky.key").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    (CERT_DIR / "viky.crt").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print(f"Hotovo: {CERT_DIR / 'viky.crt'}")
    print(f"        {CERT_DIR / 'viky.key'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
