"""Backward-compatible UI entrypoint.

Use `python launch.py` from project root for the primary startup path.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from launch import main


if __name__ == "__main__":
    main()
