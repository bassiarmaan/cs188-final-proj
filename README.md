# Line assembly (CS188) — custom RoboSuite environment

This repo defines **`LineLayoutBlocksEnv`**: a Panda arm on a table with several colored cubes respawned each reset. The goal of the course project is to later arrange cubes into **horizontal**, **diagonal**, and **vertical** structures; this milestone only provides a **tested simulation environment**.

## Match to course starters (same “scene,” different task)

- **Table & robot:** Defaults follow RoboSuite’s **`Lift`** task: `0.8 × 0.8` table, standard friction, table height `0.8`, Panda base placement like `Lift` (see `line_assembly/line_layout_blocks_env.py` constants `LIFT_*`).
- **Spawn region:** Default `placement_profile="calibration"` uses the same XY bounds and `z_offset` as the **`2_calibration_starter`** `UniformRandomSampler` in `test.py` (`CALIBRATION_*` constants), so the workspace overlaps assignments that use that calibration script.
- **`placement_profile="lift_default"`:** Uses RoboSuite Lift’s tight one-cube box when `cube_count == 1`; for multiple cubes it automatically uses the same wider region as `calibration` so placement stays feasible.
- **`placement_profile="table_uniform"`:** Uniform samples over most of the table (inset margin), handy for spread-out starts.
- **Task:** This is **not** the Lift task (no lift reward). Reward is still a placeholder until you define line-assembly success.

The **CA1 PID starter** uses other envs (e.g. `NutAssembly`) for homework; only the **Panda + table manipulation** vibe overlaps—use this repo when you want **multicolor cubes + line-assembly** with Lift/calibration-style geometry.

## Setup

```bash
cd /path/to/cs188-final
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m robosuite.scripts.setup_macros
```

## Smoke test

Offscreen (no window):

```bash
python scripts/smoke_test_env.py
```

On-screen (requires a working MuJoCo display):

```bash
python scripts/smoke_test_env.py --render
```

**Important:** `smoke_test_env.py` only sends **uniform random** actions so the env loads; the arm will **not** pick or line up blocks.

To **watch the arm build a horizontal row** (CA1-style PID in Cartesian space), run:

```bash
python scripts/demo_horizontal_line.py --render
```

That script uses **`OSC_POSE`** (`load_controller_config(default_controller="OSC_POSE")`). The default Panda controller is **`JOINT_VELOCITY`**, which ignores Cartesian “move toward this point” commands—same pitfall if you `make` an env without passing `controller_configs`.

## Horizontal line task (current milestone)

- **Verification:** [`line_assembly/layout_verify.py`](line_assembly/layout_verify.py) implements `check_horizontal_adjacent_line`: same \(y,z\), sorted along \(x\), neighbor spacing ≈ one block width (\(2 \times\) half-extent along \(x\)).
- **Environment:** `env.evaluate_horizontal_line()` returns a dict (`ok`, spans, pitches, messages). Each `step` adds `info["horizontal_line"]` and `info["horizontal_line_ok"]`.
- **Reward:** Pass `horizontal_reward_mode="sparse"` to `robosuite.make(...)` for reward `1.0` when the layout checks pass, else `0.0` (default is `"none"` = always `0` for now).
- **Deterministic spawn:** `placement_profile="horizontal_line"` places all cubes in a valid row on reset (good for debugging policies / videos).
- **Spacing / eval:** The demo policy leaves a small **gap** between cubes (`inter_cube_gap` ≈ 8 mm) so they do not collide when released; set env `horizontal_line_eval_gap` to the same value so `evaluate_horizontal_line()` expects that center-to-center pitch (`2·half_width + gap`).
- **Demo policy:** [`scripts/demo_horizontal_line.py`](scripts/demo_horizontal_line.py) runs a waypoint + PID controller (like CA1) under **OSC_POSE**; use `--render` to watch. Picks are ordered by **palette index** (`assembly_cube_i` uses `rgba_cycle[i % K]`), then by initial **x** within the same color, so the finished row reads **left-to-right in color order** (red → green → blue → … as defined in the env). Pass `pick_sort_mode='x'` to `HorizontalLineAssemblyPolicy` to restore old left-to-right-by-position pick order. After each place, it runs **`ASCEND_CLEAR`**: straight up above the slot before moving to the next pick, so the arm does not sweep through cubes already in the row. When all cubes are placed, **`FINAL_PARK`** moves the gripper high above the **center of the row** (default ~0.42 m above table cube height) so the layout stays in view; with `--render`, **`--hold-seconds`** (default 3) keeps the window open at the end.

- **Tests:**

```bash
python scripts/test_horizontal_verification.py
```

## Vertical stack (column along +z)

- **Verification:** `check_vertical_adjacent_stack` in [`line_assembly/layout_verify.py`](line_assembly/layout_verify.py) — centers share ~constant \(x,y\), sorted bottom-to-top by \(z\), neighbor spacing ≈ \(2 \times\) half-height (plus optional gap).
- **Environment:** `env.evaluate_vertical_stack()` returns `ok`, `x_span`, `y_span`, `pitches_z`, etc. Each `step` also fills `info["vertical_stack"]` and `info["vertical_stack_ok"]`.
- **Eval gap:** Set `vertical_stack_eval_gap` to match your policy’s vertical gap between cube centers (same idea as `horizontal_line_eval_gap`).
- **Deterministic spawn:** `placement_profile="vertical_stack"` stacks cubes in a valid column on reset.
- **Demo:** [`scripts/demo_vertical_line.py`](scripts/demo_vertical_line.py) uses **`OSC_POSE`** and `VerticalLineAssemblyPolicy` (see `line_assembly/vertical_line_policy.py`).

```bash
python scripts/demo_vertical_line.py --render
```

```bash
python scripts/test_vertical_stack.py
```

## Language → plan → control (scaffolding for your agent)

Course-style goal: **parse** diverse instructions, **emit** a structured plan, **run** a low-level controller in RoboSuite with a clear pipeline you can cite in a report.

| Stage | Code | Tests |
|--------|------|--------|
| **Language → task** | `parse_instruction()` in [`line_assembly/instruction_bridge.py`](line_assembly/instruction_bridge.py) | [`scripts/test_instruction_parse.py`](scripts/test_instruction_parse.py) |
| **Task → subgoals** | `build_subgoal_plan()` | [`scripts/test_instruction_plan_binding.py`](scripts/test_instruction_plan_binding.py) |
| **Task → policy binding** | `control_binding_for_plan()`, `instantiate_policy()` | same + [`scripts/test_instruction_consistency.py`](scripts/test_instruction_consistency.py) |
| **End-to-end sim** | `run_parsed_instruction()` | [`scripts/test_integration_instruction_tasks.py`](scripts/test_integration_instruction_tasks.py) |

**Executable today (3+ distinct behaviors):** vertical **stack**, horizontal **row** (west-to-east pick order), **sort-by-color** row (palette order + x tie-break). **Sort-by-size** is recognized and gets a plan + binding with `sim_supported=False` until the env exposes different block scales—good for “future work” without blocking demos.

**One-shot demo from a string:**

```bash
python scripts/demo_from_instruction.py --text "stack three blocks" --print-json
python scripts/demo_from_instruction.py --text "sort by color" --render
```

### Interactive task bank (terminal + render)

[`scripts/play_task_menu.py`](scripts/play_task_menu.py) loops on `task>` input: choose a **preset** (horizontal row, vertical stack, **smiley**, then 2×2 / 3×3 grids) and optional **color** (`blue`, `red`, … or `default` multicolor). Uses **OSC_POSE** and the same PID pick-place stack as the demos.

```bash
python scripts/play_task_menu.py
```

Examples at the prompt: `h`, `hc`, `v`, **`smiley`** (flat pattern — good to try before `g2` / `g3`), then `g2`, `g3`, or `blue v` / `v blue`. Non-interactive: `python scripts/play_task_menu.py --one-shot "red smiley"`. Headless: `--no-render`. Override pick order: `--pick color` or `--pick x`.

**One demo per pattern** (recommended order: smiley before the heavy grids):

```bash
python scripts/demo_smiley.py --render
python scripts/demo_2x2_columns.py --render
python scripts/demo_3x3_columns.py --render   # table_uniform spawn (18 cubes)
```

Each supports `--color blue` (etc.), `--pick x`, `--max-steps`, and `--hold-seconds`.

Run them in that order automatically (headless): `python scripts/run_all_demos.py` (add `--skip-3x3` to skip the long 18-cube grid).

**Run everything (use `--skip-slow` to omit full rollouts):**

```bash
python scripts/run_all_tests.py
python scripts/run_all_tests.py --skip-slow
```

### Ideas to grow complexity while staying doable

1. **Swap the parser** — keep `ParsedInstruction` / `Subgoal` / tests; replace `parse_instruction` with an LLM + JSON schema or a small intent classifier.
2. **Richer plans** — split “perceive” into fake vision vs sim-state branches; log plans to a trace file for the write-up.
3. **New tasks** — add `TaskType`, geometry in `layout_verify.py`, a policy module, then extend the phrase tables + one integration scenario each.
4. **Variable-size blocks** — new `BoxObject` sizes in the env → implement `SORT_BY_SIZE` execution and flip `sim_supported` in the binding.
5. **Metrics** — record steps-to-success, parse confidence, and verifier `messages` for ablations.

## Credits / originality

- Built with **RoboSuite** APIs (`TableArena`, `ManipulationTask`, `UniformRandomSampler`, `SingleArmEnv`).
- A friend’s public example project ([RoboStack](https://github.com/aditya-r123/RoboStack)) motivated the *idea* of a multicolor block table task; **this code was written separately** for this course submission.

## Package layout

- `line_assembly/line_layout_blocks_env.py` — environment definition
- `line_assembly/__init__.py` — registers the env with RoboSuite
- `scripts/smoke_test_env.py` — load / step sanity check
- `line_assembly/instruction_bridge.py` — NL → task → plan → control hooks
- `scripts/run_all_tests.py` — aggregate local tests
