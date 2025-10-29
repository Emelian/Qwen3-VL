#!/usr/bin/env python3
"""CLI wrapper for qwenvl.generate_predicts."""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwenvl.generate_predicts import main


if __name__ == "__main__":
    main()
