#!/usr/bin/env python3
"""Fire every test script in sequence; --skip-slow skips the long sim rollouts."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.main()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip MuJoCo integration rollouts (instruction → policy → sim).",
    )
    args = p.parse_args()

    scripts = [
        _ROOT / "scripts" / "test_horizontal_verification.py",
        _ROOT / "scripts" / "test_vertical_stack.py",
        _ROOT / "scripts" / "test_instruction_parse.py",
        _ROOT / "scripts" / "test_instruction_plan_binding.py",
        _ROOT / "scripts" / "test_instruction_consistency.py",
        _ROOT / "scripts" / "test_instruction_edge_cases.py",
    ]
    if not args.skip_slow:
        scripts.append(_ROOT / "scripts" / "test_integration_instruction_tasks.py")

    for path in scripts:
        print(f"\n=== {path.name} ===", flush=True)
        _run(path)
    print("\n=== ALL SELECTED TESTS PASSED ===")


if __name__ == "__main__":
    main()
