# Line assembly (CS188) — RoboSuite

Custom **`LineLayoutBlocksEnv`**: a Panda arm and colored cubes on a table. Policies build **horizontal rows**, **vertical stacks**, and **preset patterns** (smiley, grids) using Cartesian PID under **OSC_POSE**.

## Setup

```bash
cd /path/to/cs188-final
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m robosuite.scripts.setup_macros
```

**Controller:** Demos use **`OSC_POSE`**. The default Panda controller is **`JOINT_VELOCITY`**, which does not match “move in Cartesian space” actions—always pass the same controller config as the demo scripts when you `make` the env yourself.

## Quick commands

| What | Command |
|------|---------|
| Load env, random actions (no task) | `python scripts/smoke_test_env.py` |
| Horizontal row + PID | `python scripts/demo_horizontal_line.py --render` |
| Vertical stack | `python scripts/demo_vertical_line.py --render` |
| Phrase → policy (regex parser) | `python scripts/demo_from_instruction.py --text "sort by color" --render` |
| Interactive presets (row, stack, smiley, grid, …) | `python scripts/play_task_menu.py` |
| All pattern demos (headless) | `python scripts/run_all_demos.py` |
| All tests | `python scripts/run_all_tests.py` |
| Skip slow integration rollouts | `python scripts/run_all_tests.py --skip-slow` |

**Optional — GPT custom shape** (needs `OPENAI_API_KEY` in env or `.env`):

```bash
python scripts/demo_custom_from_prompt.py --render "a simple smiley face"
```

## What lives where

- **`line_assembly/`** — env (`line_layout_blocks_env.py`), verifiers (`layout_verify.py`), policies (`*_policy.py`), patterns (`layout_patterns.py`), NL bridge (`instruction_bridge.py`).
- **`scripts/`** — demos, tests, menu, `run_all_tests.py`.
- **`gpt_api.py`** — Chat API helper for bitmap-from-prompt (used by `demo_custom_from_prompt.py`).