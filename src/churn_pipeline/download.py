"""Download Source A and verify it byte-for-byte.

The URL is pinned to a commit, and the file is checked against a known SHA-256.
If the upstream file changes, the run fails loudly instead of quietly training
on different data.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from . import config


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_ibm_csv(path: Path) -> str:
    """Raise if the file is missing or does not match the pinned checksum."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `churn-pipeline download` first."
        )
    digest = sha256_file(path)
    if digest != config.IBM_CSV_SHA256:
        raise ValueError(
            f"Checksum mismatch for {path}: got {digest}, "
            f"expected {config.IBM_CSV_SHA256}. Refusing to continue."
        )
    return digest


def download_ibm_csv(dest: Path, force: bool = False) -> Path:
    """Download into a temp file, verify it, then atomically rename into place."""
    if dest.exists() and not force:
        verify_ibm_csv(dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(config.IBM_CSV_URL, timeout=60) as resp:
            tmp.write_bytes(resp.read())
        verify_ibm_csv(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)
    return dest
