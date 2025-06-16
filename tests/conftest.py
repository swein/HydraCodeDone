"""Pytest configuration: ensure project src/ added to sys.path for imports."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
