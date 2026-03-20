# CS188 Final Project — In-Depth Technical Report

**Repository:** Line assembly in a custom RoboSuite environment (`LineLayoutBlocksEnv`)  
**High-level story:** A Franka Panda manipulates multicolored cubes on a table. We built (1) a simulation environment aligned with course geometry conventions, (2) geometric *verifiers* that judge whether a horizontal line or vertical stack is correct, (3) hand-crafted **state-machine policies** that use **Cartesian PID** under **OSC_POSE** to pick and place cubes, and (4) a **language → task → plan → controller** bridge that maps natural phrases to a small set of executable behaviors.  

This document is intentionally longer than necessary: it explains *why* each layer exists, how PID and policy testing are set up, how evaluation metrics are defined, and how the “open-ended language” story relates to a **finite task bank** (with an optional LLM front-end sketched for the write-up).

---

## Table of contents

1. [Rubric mapping (graded sections)](#1-rubric-mapping-graded-sections)  
2. [Problem definition (expanded)](#2-problem-definition-expanded)  
3. [Method](#3-method)  
4. [Why the method is technically sound](#4-why-the-method-is-technically-sound)  
5. [PID controller: design, role, and tuning](#5-pid-controller-design-role-and-tuning)  
6. [Policy architecture and testing setup](#6-policy-architecture-and-testing-setup)  
7. [Evaluation metrics, verifiers, and reward](#7-evaluation-metrics-verifiers-and-reward)  
8. [Natural language, preset tasks, and the “structured spec” illusion](#8-natural-language-preset-tasks-and-the-structured-spec-illusion)  
9. [Testing matrix and how to reproduce results](#9-testing-matrix-and-how-to-reproduce-results)  
10. [Results presentation (tables you can paste into the PDF)](#10-results-presentation-tables-you-can-paste-into-the-pdf)  
11. [Success criteria: what improved, what did not, and why](#11-success-criteria-what-improved-what-did-not-and-why)  
12. [Discussion, limitations, and reflections](#12-discussion-limitations-and-reflections)  
13. [Team contributions](#13-team-contributions-template)  
14. [Appendix: file map](#appendix-file-map)

---

## 1. Rubric mapping (graded sections)

| Rubric item | Where this report covers it | Primary code anchors |
|-------------|----------------------------|----------------------|
| **Problem clearly defined (1.5%)** | [§2](#2-problem-definition-expanded) | `README.md`, `line_assembly/line_layout_blocks_env.py` |
| **Method clearly described (1%)** | [§3](#3-method), [§6](#6-policy-architecture-and-testing-setup) | Policies, `instruction_bridge.py` |
| **Method technically sound (2%)** | [§4](#4-why-the-method-is-technically-sound), [§5](#5-pid-controller-design-role-and-tuning) | `pid_position.py`, OSC_POSE + phase machines |
| **Evaluation well presented (1%)** | [§7](#7-evaluation-metrics-verifiers-and-reward), [§10](#10-results-presentation-tables-you-can-paste-into-the-pdf) | `layout_verify.py`, `evaluate_*` in env |
| **Met success criteria (1%)** | [§11](#11-success-criteria-what-improved-what-did-not-and-why) | Tests + integration rollouts |
| **Discussion / reflections (1%)** | [§12](#12-discussion-limitations-and-reflections) | — |
| **Team contribution (0.5%)** | [§13](#13-team-contributions-template) | — |

---

## 2. Problem definition (expanded)

### 2.1 What is the task?

We want a **tabletop manipulation agent** in simulation that can rearrange **multiple colored cubes** into **goal layouts** inspired by “line assembly” coursework:

- **Horizontal line:** cubes form an adjacent row along an axis (default **+x**), with consistent **y** and **z** (within tolerances), and center-to-center spacing consistent with **twice the half-extent** plus an optional **gap** (so cubes do not interpenetrate after release).
- **Vertical stack:** cubes share approximately the same **(x, y)** and increase in **z** with consistent vertical pitch (again with optional gap).
- **Extensions (demo-level):** additional **preset patterns** (e.g. smiley, 2×2 column grid) built from the same pick-place machinery.

The *course framing* often adds: parse **diverse natural-language instructions**, produce a **plan**, and **execute** in sim with measurable success. This repo implements that pipeline in a **deliberately modular** way so the language front-end can stay dumb (regex) today and become smarter (LLM / classifier) tomorrow without rewriting the low level.

### 2.2 What is given?

- **Simulator:** RoboSuite / MuJoCo, with a **Panda** arm, **parallel jaw gripper**, and a **table arena** whose defaults are aligned with the **Lift** task geometry for familiarity (`table` size, height, base placement — see env constants in code).
- **Objects:** Several **box** objects (cubes) with a configurable **`rgba_cycle`** so you can run **multicolor** (“rainbow”) or **monochrome** (all blue, all red, …) episodes via `line_assembly/color_presets.py`.
- **State:** Full simulation state is available (for policies we read **`robot0_eef_pos`**, cube body positions via env helpers, and gripper contact checks).
- **Controller interface:** Actions must be chosen compatible with the selected controller. Our policies assume **`OSC_POSE`**: the first three action dimensions behave like **delta end-effector pose** commands (position part used heavily; orientation deltas kept at zero in the demos).

### 2.3 What outcome do we expect?

At the end of an episode (or policy termination):

1. **Geometric success:** `evaluate_horizontal_line()` and/or `evaluate_vertical_stack()` return **`ok: true`** when the layout matches the verifier’s tolerances.
2. **Behavioral success:** The policy’s **`finished`** flag becomes true after a terminal **park** phase (so we know the controller *believes* it completed the scripted sequence).
3. **Pipeline success (language path):** A phrase maps to a **`TaskType`**, a **`ControlBinding`** selects env overrides + policy parameters, and **`run_parsed_instruction`** completes without timeout.

Failure modes are also first-class: unknown phrases raise **`ValueError`**; **sort-by-size** is recognized at the plan/binding layer but marked **`sim_supported=False`** until the environment exposes variable block scales.

---

## 3. Method

### 3.1 Layered decomposition

We split the system into layers that correspond to how robotics stacks are usually described in reports (and how you can swap components for milestones):

```mermaid
flowchart LR
  NL[Natural language or menu text]
  P[parse_instruction / future LLM]
  T[TaskType + ParsedInstruction]
  PL[build_subgoal_plan: Subgoals]
  B[control_binding_for_plan: binding]
  POL[instantiate_policy]
  SIM[RoboSuite step loop]
  V[layout verifiers + info dicts]

  NL --> P --> T --> PL --> B --> POL --> SIM --> V
```

1. **Environment (`LineLayoutBlocksEnv`)**  
   Registers with RoboSuite, spawns cubes per **`placement_profile`** (random calibration-style region, deterministic row, deterministic stack, etc.), and after each step attaches **evaluation snapshots** to `info`.

2. **Verification (`layout_verify.py`)**  
   Pure geometry: given cube centers and half-extents, check adjacency pitches and cross-axis spread. No learning — deterministic pass/fail with diagnostic **`messages`**.

3. **Low-level control assumption**  
   **`OSC_POSE`** so that a **world-frame PID on `robot0_eef_pos`** produces meaningful motion. (Using default **`JOINT_VELOCITY`** without reconfiguring the controller is a known footgun called out in the README.)

4. **Policies**  
   Finite-state machines (FSMs) with phases like **HOVER_SOURCE → DOWN_SOURCE → CLOSE_GRIP → LIFT → …**. Each phase sets a ** Cartesian target** for the gripper; **`PIDPosition`** converts position error into **clipped OSC deltas**. Gripping uses timed steps plus optional **contact-based** “grasp streak” confirmation (`env.is_gripping_cube`).

5. **Instruction bridge (`instruction_bridge.py`)**  
   Maps text → **`TaskType`** (stack, horizontal line, sort-by-color, sort-by-size) → human-readable **`Subgoal`** list (for logging / figures) → **`ControlBinding`** (which policy class, kwargs like **`pick_sort_mode`**, env overrides like **`cube_count`**, **`horizontal_line_eval_gap`**, etc.).

6. **Demos / menus**  
   - **`demo_from_instruction.py`**: single string, optional JSON dump of plan/binding.  
   - **`play_task_menu.py`**: interactive **preset task bank** (horizontal, horizontal-by-color, vertical, smiley, 2×2 grid) plus **color preset** and **cube count** parsing.

### 3.2 Why not end-to-end RL (in this milestone)?

The project milestone emphasized a **reliable sim environment** and **interpretable control**. Model-free RL could learn contact-rich behavior but costs sample complexity and obscures *why* a motion succeeded. Here, **explicit pick-place logic + PID** gives repeatable demos, tight alignment with CA1-style homework, and **transparent failure attribution** (PID thresholds, grasp timers, placement pitch mismatch, etc.).

---

## 4. Why the method is technically sound

### 4.1 Separating “success definition” from “how we get there”

The **verifier** encodes what “line” and “stack” mean independent of the policy. That avoids circular evaluation where the same heuristic both acts and judges. It also lets tests **inject** cube poses (`set_cube_free_joint`) to confirm the geometry layer flips from pass to fail when you break **y** alignment or spacing.

### 4.2 OSC_POSE + Cartesian PID is a standard composite

Operational space control (here via RoboSuite’s controller config) abstracts part of the inverse kinematics/Jacobian behavior so that commanding **delta** motions in the end-effector frame tracks **position targets** reasonably for tabletop reaches. A **PID** on measured **`eef_pos`** provides closed-loop correction against model error, contact, and integration drift *within each phase*.

**Caveat (intellectual honesty):** This is not a full dynamics-aware controller; it is a **pragmatic** layer appropriate for coursework sim where contact is damped and targets are slow-moving relative to control frequency.

### 4.3 Finite-state pick-place decomposes a hard problem

Continuous “do manipulation” is high-dimensional. A phase machine reduces it to a sequence of **short-horizon regulation problems** (go to hover, descend, close gripper, lift transit, align above slot, descend, open). Each subproblem is easier to tune and diagnose than a single monolithic trajectory.

### 4.4 Language layer as *routing*, not *physics*

Parsing (regex today) only selects among **known executable bundles**. That is sound engineering for a class timeline: correctness is concentrated in **policies + verifiers**. The language module’s obligation is **typed routing**, not magic.

---

## 5. PID controller: design, role, and tuning

### 5.1 Implementation

**File:** `line_assembly/pid_position.py`  

**Class:** `PIDPosition(kp, ki, kd, target)`

- **State:** integral of error (3-vector), previous error (for finite-difference derivative), **`last_error_norm`** (L2 norm of error — used as a cheap “are we close?” signal).
- **`update(current_pos, dt)`** returns a **3D control vector**  
  \[
  u = k_p e + k_i \int e\,dt + k_d \frac{e - e_{prev}}{dt}
  \]
  with **`ki` often zero** in policies (integral windup in contact-rich sim can hurt).

### 5.2 How policies use PID

In **`HorizontalLineAssemblyPolicy.get_action`** (and analogously vertical / generic):

1. Read **`eef = obs["robot0_eef_pos"]`**.
2. Compute **`self._eef_target()`** from the current **FSM phase** and the active **pick/place job** (cube index + slot XYZ).
3. **`self.pid.set_target(...)`** then **`ctrl = self.pid.update(eef, self.dt)`**.
4. **Phase-shaped gating:** e.g. reduce **`ctrl[2]`** during **DOWN_SLOT** / **OPEN_GRIP** so the arm does not **ram the table**; reduce **xy** during **ASCEND_CLEAR** so the robot primarily **lifts** before translating (prevents sweeping through an existing row).
5. Write **`action[0:3] = clip(ctrl, -1, 1)`**, **`action[3:6] = 0`** (orientation untouched in demos).
6. **Advance phases** when **`pid.error_norm() < threshold`** for **`settle_steps`** consecutive checks (hysteresis / dwell time).

### 5.3 Thresholds: two-speed convergence

Policies use **`pos_threshold`** (tight) in many approach phases and **`pos_threshold_coarse`** (looser) during **LIFT / HOVER_SLOT / DOWN_SLOT / ASCEND / FINAL_PARK**. Reason: transit phases should not spend thousands of steps chasing millimeters that do not affect collision safety; place phases still need accuracy but tolerate slightly more error to avoid brittle stagnation.

### 5.4 Knobs you can cite as “ablation levers”

Document these as **engineering hyperparameters** (not learned):

| Parameter (conceptual) | Role |
|------------------------|------|
| `kp`, `kd` | Stiffness / damping of Cartesian tracking |
| `pos_threshold*` | When the FSM considers a waypoint “reached” |
| `settle_steps*` | Anti-chatter: require stability before advancing |
| `gripper_close_steps`, `grasp_streak` | Time + contact confirmation for a secure grasp |
| `hover_z`, `grasp_z_slack` | Approach geometry |
| `inter_cube_gap` | Must match verifier’s **`horizontal_line_eval_gap` / `vertical_stack_eval_gap`** |
| `clear_height_above_table`, `ASCEND_CLEAR` | Collision avoidance between successive places |

The README notes that **`play_task_menu`** presets may pass **stricter overrides** for harder patterns (grids, smiley).

### 5.5 What PID does *not* solve

- **Grasp success under slip:** partially mitigated by timers + `is_gripping_cube`, but not guaranteed.
- **Global planning among obstacles:** the FSM assumes a structured workspace (tabletop, scattered cubes).
- **Orientation-sensitive placements:** demos keep orientation deltas at zero; some real tasks would need non-zero `action[3:6]` or a different controller parameterization.

---

## 6. Policy architecture and testing setup

### 6.1 Policies in this repo

| Policy | File | Primary behavior |
|--------|------|-------------------|
| `HorizontalLineAssemblyPolicy` | `line_assembly/horizontal_line_policy.py` | Row along +x; pick order **`x`** or **`color`** |
| `VerticalLineAssemblyPolicy` | `line_assembly/vertical_line_policy.py` | Stack along +z; shared (x,y), increasing z |
| `GenericAssemblyPolicy` | `line_assembly/generic_assembly_policy.py` | Arbitrary slot list (patterns like smiley / grid) |

All are **deterministic given env reset conditions** (same cube poses → same job ordering), aside from physics noise.

### 6.2 “Policy testing” in three tiers

We deliberately test at **increasing integration depth**:

#### Tier A — Pure math / geometry (no physics rollouts)

- **`test_horizontal_verification.py`**:  
  - `pick_indices_in_palette_order` unit logic.  
  - `check_horizontal_adjacent_line` on synthetic positions (good row, broken **y**, bad pitch).  
- **`test_vertical_stack.py`**:  
  - `check_vertical_adjacent_stack` synthetic cases.  

These validate **success criteria** independent of PID or MuJoCo contacts.

#### Tier B — Environment + verifier wiring (short MuJoCo)

- Spawn with **`placement_profile="horizontal_line"`** → immediate **`evaluate_horizontal_line()["ok"]`**.
- **`horizontal_reward_mode="sparse"`** → reward **1.0** when layout ok (sanity that reward hooks read the same verifier).
- **`placement_profile="vertical_stack"`** → vertical verifier ok.
- **Manual pose hack:** random calibration spawn, then **`set_cube_free_joint`** to snap a perfect line → verifier passes.

These tests prove: **env state reflects geometry**, **`info`** mirrors **`evaluate_*`**, and **`reward()`** can be tied to **`ok`**.

#### Tier C — Full closed-loop instruction → policy → sim

**`test_integration_instruction_tasks.py`** (slow):

- Builds real env with **`OSC_POSE`**, **`placement_profile="calibration"`**.
- For each phrase:

  | Phrase | Expected binding flavor | Assert |
  |--------|-------------------------|--------|
  | `"stack two blocks"` | vertical policy, 2 cubes | `evaluate_vertical_stack()["ok"]` |
  | `"put four cubes in a row"` | horizontal policy, sort by **x** | `evaluate_horizontal_line()["ok"]` |
  | `"sort by color"` | horizontal policy, **`pick_sort_mode="color"`** | `evaluate_horizontal_line()["ok"]` |

Also asserts **`summary["finished_policy"]`** so we distinguish “layout accidentally ok” from “policy completed”.

**Why two cubes for stack integration?** Comment in test: **more reliable** under physics variance than tall stacks in automated CI-style runs.

#### Tier D — Menu / pattern battery (optional, longest)

**`test_all_formations.py`** runs tasks like **`smiley`** and **`g2`** headless with a step budget. This is closer to **system demo testing** than unit testing.

### 6.3 Aggregate runner

**`scripts/run_all_tests.py`** runs, in order:

1. `test_horizontal_verification.py`  
2. `test_vertical_stack.py`  
3. `test_instruction_parse.py`  
4. `test_instruction_plan_binding.py`  
5. `test_instruction_consistency.py`  
6. `test_instruction_edge_cases.py`  
7. Unless `--skip-slow`: `test_integration_instruction_tasks.py`

**`--skip-slow`** exists because MuJoCo rollouts are **minutes of CPU** on some machines and painful in CI.

### 6.4 What a “PID test” is in practice

We do **not** have a separate `test_pid.py` that asserts overshoot < 5%. Instead, PID is **indirectly** tested:

- If gains/thresholds were nonsense, **Tier C integration** would fail (timeouts, misaligned placements, verifier false negatives).
- If PID never converged, FSM phases would **stall** → `finished_policy` false or step cap hit.

For a class report, that is acceptable if you state it explicitly: **PID tuning is validated by end-effect task success**, not by Nyquist analysis.

---

## 7. Evaluation metrics, verifiers, and reward

### 7.1 Horizontal line report (`evaluate_horizontal_line`)

Backed by `check_horizontal_adjacent_line` → serialized by `report_to_info_dict`.

**Key fields (conceptual):**

| Field | Meaning |
|-------|---------|
| `ok` | Boolean pass |
| `y_span`, `z_span` | Spread off the primary axis (here row along **x**) |
| `pitches` | Center-to-center differences along **x** between sorted cubes |
| `pitch_errors` | How far pitches deviate outside acceptable band around nominal |
| `messages` | Human-readable reasons for failure |

**Nominal pitch** used by the env:

\[
\text{nominal\_pitch} = 2 \cdot h_x + \text{horizontal\_line\_eval\_gap}
\]

This **must** match what policies implement as **`2 * half_extent + inter_cube_gap`**. If they diverge, you can get **false negatives** (“policy did what we wanted but verifier disagrees”).

### 7.2 Vertical stack report (`evaluate_vertical_stack`)

Same idea along **z** with **`vertical_stack_eval_gap`**.

### 7.3 Per-step `info` (always logged)

After each `env.step`, **`info`** includes at least:

- `horizontal_line_ok`, `horizontal_line`  
- `vertical_stack_ok`, `vertical_stack`  

So you can plot **ok vs time** post-hoc or inspect **messages** at the final step.

### 7.4 Reward modes

Default reward is **0** (placeholder). With **`horizontal_reward_mode="sparse"`**, reward is **1.0** iff horizontal verifier passes — useful if you later train a value function or logging critic.

---

## 8. Natural language, preset tasks, and the “structured spec” illusion

### 8.1 What the code does today

**`parse_instruction`** in `instruction_bridge.py` is intentionally **regex / keyword** driven:

- It scans for **clusters of phrases** associated with each **`TaskType`**.
- **Precedence matters:** e.g. “sort by size” wins over color/stack/line ambiguity; vertical/stack cues can override ambiguous “line up vertically” wording.
- It extracts small integers (“three blocks” → **3**) when present. Number words in idioms such as **“Put one on top of the other”** are not treated as a cube count (the substring **“one on top”** is excluded for **one** so `suggested_cube_count` stays unset).

Then **`control_binding_for_plan`** returns a **`ControlBinding`**:

- **`policy_id`**: `"vertical_line_assembly"` or `"horizontal_line_assembly"` or `"none"`.
- **`policy_kwargs`**: e.g. **`pick_sort_mode`** (`"x"` vs `"color"`), **`inter_cube_gap`**.
- **`env_overrides`**: e.g. **`cube_count`**, gap fields, **`placement_profile`**.

**`instantiate_policy`** imports the concrete policy class and constructs it. **`run_parsed_instruction`** runs the sim loop until **`policy.finished`** or **`max_steps`**.

### 8.2 The architecture you described for the report (LLM / Chat → preset task)

Your team’s *conceptual* story (and a very reasonable final-project narrative) is:

1. Maintain a **closed task library** — behaviors we know how to execute well:
   - **Alternate / multicolor** appearance vs **single-color** (“all blue”) — implemented via **`rgba_cycle`** + `color_presets.cycle_for_preset`.
   - **Horizontal row** with **spatial pick order** vs **palette (color) pick order**.
   - **Vertical stack** with optional **count**.
   - **Structured patterns** (smiley, 2×2 grid) exposed only through the **menu** path today, but conceptually the same idea: **preset + parameters**.

2. Take a **free-form natural language prompt** from a user.

3. Send that prompt to a **chat model** (or small classifier) with a **constrained output schema**, e.g. JSON:
   ```json
   {
     "task": "vertical_stack",
     "cube_count": 4,
     "color_preset": "blue",
     "pick_order": "palette"
   }
   ```

4. **Map JSON → the same datatypes** this repo already uses:
   - Either directly fabricate a **`ControlBinding`** / env kwargs,  
   - Or emit a synthetic phrase that **`parse_instruction`** already understands (brittle),  
   - Or best: **replace `parse_instruction`** with `parse_instruction_llm` that returns **`ParsedInstruction`** unchanged.

5. **Execute** the corresponding policy in RoboSuite.

**Why this looks like “general intelligence” but is actually a small menu**

From the user’s perspective, the robot “understands anything.” In reality, **generalization is bounded** by:

- The **enumerated** `TaskType` set (plus menu-only patterns).
- The **parameter tensor** each task accepts (count, color preset, pick order).
- The **verifier** only checks **geometric families** we coded (line, stack; patterns may not have a dedicated verifier and rely on **demonstrated** success).

This is not cheating — it is how many deployed robots work: **skill libraries + routers**.

### 8.3 Honesty clause graders appreciate

If you implement the Chat router, say:

- **Strength:** robust paraphrase → same `TaskType`.  
- **Weakness:** hallucinated task labels must be **schema-validated**; out-of-distribution requests should **clarify** or **refuse**.  
- **Safety / sim:** irrelevant here, but worth one sentence.

If you **only** use regex (current repo), say: **“We used a deterministic parser for repeatability; LLM routing is planned / optional.”**

---

## 9. Testing matrix and how to reproduce results

### 9.1 Commands

From repo root (after venv + `pip install -r requirements.txt` + robosuite macros):

```bash
# Fast suite (skips long rollouts)
python scripts/run_all_tests.py --skip-slow

# Full suite including NL → policy → sim integration
python scripts/run_all_tests.py

# Instruction-only smoke + JSON plan
python scripts/demo_from_instruction.py --text "sort by color" --print-json

# Visual confirmation
python scripts/demo_horizontal_line.py --render
python scripts/demo_vertical_line.py --render
```

### 9.2 Expected high-level outcomes

When dependencies are installed correctly:

- **Parse tests** print counts of covered phrases and precedence checks.
- **Plan/binding tests** print **`plan + binding: OK`**.
- **Integration tests** print lines like **`integration [vertical]: OK steps=...`** per scenario.
- **Final banner:** **`ALL SELECTED TESTS PASSED`**.

### 9.3 Observed run: `python scripts/run_all_tests.py` (full suite, no `--skip-slow`)

**Date:** 2026-03-19. **Command:** `python scripts/run_all_tests.py` from repository root. **Result:** exit code 0; all listed scripts completed successfully.

```
=== test_horizontal_verification.py ===
pick order by color: OK
pure geometry: OK
horizontal_line placement + info: OK
sparse horizontal reward: OK
manual snap to line: OK

All horizontal verification tests passed.

=== test_vertical_stack.py ===
pure vertical geometry: OK
vertical_stack spawn: OK

All vertical stack tests passed.

=== test_instruction_parse.py ===
instruction parse: OK (17 phrases + precedence + unknown)

=== test_instruction_plan_binding.py ===
plan + binding: OK

=== test_instruction_consistency.py ===
instruction consistency: OK (all TaskTypes wired)

=== test_instruction_edge_cases.py ===
instruction edge cases: OK

=== test_integration_instruction_tasks.py ===
integration [vertical]: OK steps=719
integration [horizontal]: OK steps=1015
integration [color_row]: OK steps=807

All integration instruction tasks passed.

=== ALL SELECTED TESTS PASSED ===
```

**Wall time (this machine):** approximately 16 s end-to-end for the full aggregate run above.

---

## 10. Results presentation (tables you can paste into the PDF)

### 10.1 Verifier unit tests (Tier A/B)

| Test script | What it proves | Representative assertion |
|-------------|----------------|--------------------------|
| `test_horizontal_verification.py` | Row geometry + spawn profiles + sparse reward | `r.ok` on good layout; `r2.ok` false when **y** broken |
| `test_vertical_stack.py` | Column geometry + vertical spawn profile | `evaluate_vertical_stack()["ok"]` on spawn |

### 10.2 Instruction parsing coverage (illustrative)

| Input phrase (examples) | Parsed `TaskType` | Notes |
|-------------------------|-------------------|--------|
| “Stack three blocks” | `STACK_VERTICAL` | Count **3** propagated to env overrides |
| “Line up the cubes in a row” | `LINE_HORIZONTAL` | |
| “Sort by color” | `SORT_BY_COLOR` | Same horizontal verifier; different pick order |
| “Sort by size …” | `SORT_BY_SIZE` | `sim_supported=False` until variable sizes exist |
| “Put one on top of the other” | `STACK_VERTICAL` | No numeric count (idiom; not “1 cube”) |
| Nonsense string | — | `ValueError` |

*(Full list lives in `scripts/test_instruction_parse.py`.)*

### 10.3 Integration rollouts (Tier C)

Observed on **2026-03-19** with `python scripts/run_all_tests.py` (see §9.3). Each scenario asserts `finished_policy` and the relevant verifier `ok` in `test_integration_instruction_tasks.py`.

| Scenario | Steps (`summary["steps"]`) | `finished_policy` | Verifier `ok` | Notes |
|----------|----------------------------|-------------------|---------------|-------|
| stack two blocks (`integration [vertical]`) | 719 | true | true | |
| put four cubes in a row (`integration [horizontal]`) | 1015 | true | true | |
| sort by color (`integration [color_row]`) | 807 | true | true | |

### 10.4 Optional: menu pattern stress

| Task token | Cubes | Policy family | Typical check |
|------------|-------|-----------------|---------------|
| `h` | env default | Horizontal | `horizontal_line_ok` at end |
| `hc` / `sort` | ” | Horizontal + palette picks | same |
| `v` / `stack` | configurable | Vertical | `vertical_stack_ok` |
| `smiley` | pattern-derived | Generic slots | visual / step count |
| `g2` | pattern-derived | Generic slots | visual / step count |

---

## 11. Success criteria: what improved, what did not, and why

### 11.1 What unambiguously improved / succeeded

- **Environment reproducibility:** Lift/calibration-aligned geometry reduces “works on my laptop” drift vs course starters.
- **Measurable success:** From “we eyeball the sim” → **`evaluate_*` + automated integration tests**.
- **Controller correctness path:** Documented requirement to use **`OSC_POSE`** with Cartesian policies.
- **Composable language interface:** Even if parsing is simple, the **typed** pipeline (`ParsedInstruction`, `Subgoal`, `ControlBinding`) is extensible.

### 11.2 What did not improve (yet) — and defensible reasons

- **True open-vocabulary manipulation:** Without perception from pixels + a general planner, arbitrary instructions (“build a bridge”, “spell HELP”) are **out of scope** unless added as new **`TaskType`s** with new verifiers.
- **Sort by size execution:** Recognized at parse/plan level but **not sim-supported**; all cubes share extents today. This is a **clean** “future work” item rather than a silent bug (`instantiate_policy` raises **`NotImplementedError`** if mis-wired).
- **PID optimality:** Gains are hand-tuned for typical spawns; extreme random placements might fail grabs.

### 11.3 If integration tests ever fail intermittently

Blame candidates (in order):

1. **Physics variance** on grasp / slip — mitigate with seeds, more `settle_steps`, or slightly looser thresholds.  
2. **Cube count** too high for vertical policy timing budget.  
3. **Mismatch** between **`inter_cube_gap`** and **`*_eval_gap`**.  
4. **Wrong controller** (forgot `load_controller_config(default_controller="OSC_POSE")`).

---

## 12. Discussion, limitations, and reflections

### 12.1 What went well

- **Clear separation of concerns:** verifiers vs policies vs parsing.  
- **Test pyramid:** geometry unit tests catch math errors before debugging MuJoCo.  
- **Demo ergonomics:** menu + one-shot scripts make video capture straightforward.

### 12.2 What went wrong / was painful (typical for this kind of project)

- **Controller mismatch confusion:** Cartesian PID silently “does nothing useful” under the wrong action parameterization — we mitigated via README warnings and demo scripts that always pass **`OSC_POSE`**.  
- **Contact & grasp reliability:** Simulation grippers are finicky; policies add **timers** and **contact streaks** as heuristics.  
- **Time budgets:** integration tests need large **`max_steps`**; without `--skip-slow`, iteration slows.

### 12.3 Limitations (be explicit in the conclusion)

- **No exteroceptive perception pipeline** (camera → object poses) in the core loop; policies read **sim ground truth** poses.  
- **No learned error recovery**; failures propagate or stall the FSM.  
- **Orientation** mostly ignored in action space.  
- **Language** coverage is finite; LLM front-ends need **guardrails**.

### 12.4 Broader lesson

This project demonstrates a **skill-based** manipulation stack: **verify geometric goals**, **execute with a structured controller**, **route language to skills**. That pattern scales better in practice than pretending one monolithic model “just knows” how to move.

---

## 13. Team contributions (template)

*Replace the bracketed text with your team’s actual division of labor. The grader expects explicit names + responsibilities.*

| Teammate | Primary contributions | Artifacts / evidence |
|----------|----------------------|----------------------|
| **[Name A]** | Environment + verifiers + reward/info hooks | `line_layout_blocks_env.py`, `layout_verify.py`, geometry tests |
| **[Name B]** | PID policies (horizontal / vertical / generic) + tuning | `*_policy.py`, `pid_position.py`, demo scripts |
| **[Name C]** | Instruction bridge + integration tests + README | `instruction_bridge.py`, `test_integration_*`, `run_all_tests.py` |
| **[Name D]** | Report, figures, video capture, LLM router experiment (if any) | this document, appendix logs |

If everyone paired: say so, but still list **who led** each file area.

---

## Appendix: file map

| Path | Role |
|------|------|
| `line_assembly/line_layout_blocks_env.py` | Custom RoboSuite env, placement, `evaluate_*`, `info` |
| `line_assembly/layout_verify.py` | Pure geometry verifiers |
| `line_assembly/pid_position.py` | PID on xyz |
| `line_assembly/horizontal_line_policy.py` | Row FSM + PID |
| `line_assembly/vertical_line_policy.py` | Stack FSM + PID |
| `line_assembly/generic_assembly_policy.py` | Arbitrary slot patterns |
| `line_assembly/layout_patterns.py` | Smiley / grid slot builders |
| `line_assembly/color_presets.py` | `rgba_cycle` presets |
| `line_assembly/instruction_bridge.py` | NL → task → plan → binding → runner |
| `scripts/run_all_tests.py` | Aggregated test driver |
| `scripts/test_integration_instruction_tasks.py` | End-to-end sim tests |
| `scripts/play_task_menu.py` | Interactive preset task launcher |
| `scripts/demo_from_instruction.py` | One-shot NL demo |
| `README.md` | Setup + conceptual overview |

---

*End of report draft — trim sections for page limits, paste command outputs into §9–§10, and personalize §13.*
