#!/usr/bin/env python3
"""Stack geometry + vertical_stack placement profile smoke."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import robosuite  # noqa: E402

import line_assembly  # noqa: E402, F401
from line_assembly.layout_verify import check_vertical_adjacent_stack  # noqa: E402


def _test_pure_vertical_geometry():
    hz = 0.02
    h = (0.02, 0.02, hz)
    pos = np.array([[0.0, 0.0, 0.79 + i * 2 * hz] for i in range(4)])
    r = check_vertical_adjacent_stack(pos, h)
    assert r.ok, r.messages
    pos_bad = pos.copy()
    pos_bad[2, 0] += 0.08
    r2 = check_vertical_adjacent_stack(pos_bad, h)
    assert not r2.ok
    print("pure vertical geometry: OK")


def _test_vertical_stack_spawn():
    env = robosuite.make(
        "LineLayoutBlocksEnv",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        horizon=50,
        cube_count=4,
        placement_profile="vertical_stack",
        vertical_stack_eval_gap=0.0,
    )
    env.reset()
    ev = env.evaluate_vertical_stack()
    assert ev["ok"], ev
    env.close()
    print("vertical_stack spawn: OK")


def main():
    _test_pure_vertical_geometry()
    _test_vertical_stack_spawn()
    print("\nAll vertical stack tests passed.")


if __name__ == "__main__":
    main()
