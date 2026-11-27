"""Resolve optional, user-supplied reference libraries outside the source tree."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = Path(os.environ.get("XPL_REFERENCE_DIR") or ROOT / "references").expanduser().resolve()
if not REFERENCE_DIR.is_dir():
    raise FileNotFoundError(f"XPL_REFERENCE_DIR must name an existing directory: {REFERENCE_DIR}")
