#!/usr/bin/env python3
"""
Chained headless runs: horizontal, vertical, smiley, 2×2 columns.

Doesn't pass --render — add that inside each demo if you want eyes on.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"


def _run(args: list[str]) -> None:
    cmd = [sys.executable, str(_SCRIPTS / args[0])] + args[1:]
    print("\n===", " ".join(cmd), "===\n", flush=True)
    r = subprocess.run(cmd, cwd=_ROOT)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()

    _run(["demo_horizontal_line.py", "--max-steps", "25000"])
    _run(["demo_vertical_line.py", "--cubes", "3", "--max-steps", "30000"])
    _run(["demo_smiley.py", "--max-steps", "100000"])
    _run(["demo_2x2_columns.py", "--max-steps", "70000"])
    print("\n=== All selected demos finished OK ===")


if __name__ == "__main__":
    main()
