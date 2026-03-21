"""
Panda + table + a pile of colored cubes. Lift-sized table and calib-hw-style spawn box
so it doesn't feel totally random compared to class starters. Task is messing with blocks,
not the Lift reward.
"""

from __future__ import annotations

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

from .layout_verify import (
    check_horizontal_adjacent_line,
    check_vertical_adjacent_stack,
    report_to_info_dict,
    vertical_stack_report_to_dict,
)

# Stolen from Lift defaults so numbers look familiar
LIFT_TABLE_FULL_SIZE = (0.8, 0.8, 0.05)
LIFT_TABLE_FRICTION = (1.0, 5e-3, 1e-4)
LIFT_TABLE_OFFSET = (0.0, 0.0, 0.8)
LIFT_CUBE_HALF_EXTENTS = (0.02, 0.02, 0.02)

# Same XY box as the calibration homework script
CALIBRATION_X_RANGE = (-0.05, 0.18)
CALIBRATION_Y_RANGE = (-0.1, 0.1)
CALIBRATION_Z_OFFSET = 0.01

# Top-right table section for custom-from-prompt spawn
# X must stay right (>=0.12) so arm can reach both spawn and slots (avoids kinematic freeze)
SPAWN_TOP_RIGHT_X_RANGE = (0.12, 0.24)
SPAWN_TOP_RIGHT_Y_RANGE = (-0.02, 0.16)


def _default_color_cycle():
    # Red first — handy if you grep for "red block" stuff
    return (
        (1.0, 0.0, 0.0, 1.0),  # red
        (0.10, 0.90, 0.10, 1.0),  # green
        (0.10, 0.10, 0.90, 1.0),  # blue
        (0.90, 0.90, 0.10, 1.0),  # yellow
        (0.90, 0.10, 0.90, 1.0),  # magenta
    )


class LineLayoutBlocksEnv(SingleArmEnv):
    """Respawns cubes each reset; policies live elsewhere."""

    def __init__(
        self,
        robots="Panda",
        cube_count: int = 6,
        cube_half_extents=LIFT_CUBE_HALF_EXTENTS,
        rgba_cycle=None,
        table_full_size=LIFT_TABLE_FULL_SIZE,
        table_friction=LIFT_TABLE_FRICTION,
        table_offset=LIFT_TABLE_OFFSET,
        placement_profile: str = "calibration",
        workspace_inset: float = 0.06,
        robot_base_yaw: float = 0.0,
        horizontal_reward_mode: str = "none",
        horizontal_line_eval_gap: float = 0.0,
        vertical_stack_eval_gap: float = 0.0,
        **kwargs,
    ):
        self.cube_count = int(cube_count)
        if self.cube_count < 1:
            raise ValueError("cube_count must be >= 1")

        self.cube_half_extents = tuple(float(x) for x in cube_half_extents)
        self.rgba_cycle = tuple(rgba_cycle) if rgba_cycle is not None else _default_color_cycle()
        self.table_full_size = tuple(table_full_size)
        self.table_friction = tuple(table_friction)
        self.table_offset = tuple(table_offset)
        self.placement_profile = placement_profile
        self.workspace_inset = float(workspace_inset)
        self.robot_base_yaw = float(robot_base_yaw)
        if horizontal_reward_mode not in ("none", "sparse"):
            raise ValueError("horizontal_reward_mode must be 'none' or 'sparse'")
        self.horizontal_reward_mode = horizontal_reward_mode
        self.horizontal_line_eval_gap = float(horizontal_line_eval_gap)
        self.vertical_stack_eval_gap = float(vertical_stack_eval_gap)

        self._cubes: list[BoxObject] = []
        self._cube_placer: UniformRandomSampler | None = None
        self.cube_body_ids: list[int] = []

        super().__init__(robots=robots, **kwargs)

    # --- build model ---
    def _load_model(self):
        self._cubes = []
        super()._load_model()

        self._place_robot_like_lift()

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        self._cubes = self._make_cubes()
        self._cube_placer = self._make_placer()
        self._cube_placer.reset()
        self._cube_placer.add_objects(self._cubes)

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self._cubes,
        )

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_ids = [
            self.sim.model.body_name2id(cube.root_body) for cube in self._cubes
        ]

    def _place_robot_like_lift(self):
        # Copy Lift's base pose; yaw is optional
        robot_model = self.robots[0].robot_model
        if self.robot_base_yaw != 0.0:
            robot_model.set_base_ori(np.array([0.0, 0.0, self.robot_base_yaw]))
        xpos = robot_model.base_xpos_offset["table"](self.table_full_size[0])
        robot_model.set_base_xpos(xpos)

    def _make_cubes(self) -> list[BoxObject]:
        cubes: list[BoxObject] = []
        for idx in range(self.cube_count):
            rgba = self.rgba_cycle[idx % len(self.rgba_cycle)]
            cube = BoxObject(
                name=f"assembly_cube_{idx}",
                size=np.array(self.cube_half_extents, dtype=np.float64),
                rgba=rgba,
            )
            cubes.append(cube)
        return cubes

    def _make_placer(self) -> UniformRandomSampler:
        ref = np.array(self.table_offset, dtype=np.float64)
        z_lift = float(self.cube_half_extents[2])

        if self.placement_profile == "calibration":
            return UniformRandomSampler(
                name="assembly_cube_calib_region",
                x_range=list(CALIBRATION_X_RANGE),
                y_range=list(CALIBRATION_Y_RANGE),
                rotation=0.0,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=ref,
                z_offset=CALIBRATION_Z_OFFSET,
            )
        if self.placement_profile == "lift_default":
            # Lift's tiny box only works for one cube; spill to calib region if we have more
            if self.cube_count > 1:
                return UniformRandomSampler(
                    name="assembly_cube_lift_multi_fallback",
                    x_range=list(CALIBRATION_X_RANGE),
                    y_range=list(CALIBRATION_Y_RANGE),
                    rotation=0.0,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=ref,
                    z_offset=CALIBRATION_Z_OFFSET,
                )
            return UniformRandomSampler(
                name="assembly_cube_lift_tight",
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=ref,
                z_offset=0.01,
            )
        if self.placement_profile == "table_uniform":
            half_x = self.table_full_size[0] / 2.0
            half_y = self.table_full_size[1] / 2.0
            inset = self.workspace_inset
            return UniformRandomSampler(
                name="assembly_cube_table_uniform",
                x_range=[-half_x + inset, half_x - inset],
                y_range=[-half_y + inset, half_y - inset],
                rotation=None,
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=tuple(ref.tolist()),
                z_offset=z_lift,
            )
        if self.placement_profile == "horizontal_line":
            # dummy sampler — we hand-place in _reset_internal
            return UniformRandomSampler(
                name="assembly_cube_unused_placeholder",
                x_range=[0.0, 0.0],
                y_range=[0.0, 0.0],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=np.array(self.table_offset, dtype=np.float64),
                z_offset=float(self.cube_half_extents[2]),
            )
        if self.placement_profile == "compact_row":
            # dummy sampler — we hand-place in _reset_internal (centered row)
            return UniformRandomSampler(
                name="assembly_cube_compact_row_placeholder",
                x_range=[0.0, 0.0],
                y_range=[0.0, 0.0],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=np.array(self.table_offset, dtype=np.float64),
                z_offset=float(self.cube_half_extents[2]),
            )
        if self.placement_profile == "spawn_top_right":
            # dummy sampler — we hand-place in _reset_internal (deterministic grid)
            return UniformRandomSampler(
                name="assembly_cube_spawn_top_right_placeholder",
                x_range=[0.0, 0.0],
                y_range=[0.0, 0.0],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=np.array(self.table_offset, dtype=np.float64),
                z_offset=float(self.cube_half_extents[2]),
            )
        if self.placement_profile == "vertical_stack":
            return UniformRandomSampler(
                name="assembly_cube_vertical_placeholder",
                x_range=[0.0, 0.0],
                y_range=[0.0, 0.0],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                reference_pos=np.array(self.table_offset, dtype=np.float64),
                z_offset=float(self.cube_half_extents[2]),
            )
        raise ValueError(
            "placement_profile must be 'calibration', 'lift_default', 'table_uniform', "
            "'horizontal_line', 'compact_row', 'spawn_top_right', or 'vertical_stack'"
        )

    # --- cube geometry helpers ---
    def _cube_center_z_on_table(self) -> float:
        table_top_z = self.table_offset[2] - 0.5 * self.table_full_size[2]
        return float(table_top_z + self.cube_half_extents[2])

    def get_cube_world_positions(self) -> np.ndarray:
        # Nx3 world positions
        return np.stack(
            [self.sim.data.body_xpos[bid].copy() for bid in self.cube_body_ids],
            axis=0,
        )

    def set_cube_free_joint(self, cube_index: int, pos_xyz, quat_wxyz) -> None:
        # for tests / cheating layouts
        obj = self._cubes[cube_index]
        qpos = np.concatenate([np.asarray(pos_xyz, dtype=np.float64).ravel()[:3], np.asarray(quat_wxyz, dtype=np.float64).ravel()[:4]])
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)

    def evaluate_horizontal_line(self) -> dict:
        pos = self.get_cube_world_positions()
        hx = float(self.cube_half_extents[0])
        nom = 2.0 * hx + float(self.horizontal_line_eval_gap)
        rep = check_horizontal_adjacent_line(
            pos, self.cube_half_extents, nominal_pitch=nom
        )
        return report_to_info_dict(rep)

    def evaluate_vertical_stack(self) -> dict:
        pos = self.get_cube_world_positions()
        hz = float(self.cube_half_extents[2])
        nom = 2.0 * hz + float(self.vertical_stack_eval_gap)
        rep = check_vertical_adjacent_stack(
            pos, self.cube_half_extents, nominal_pitch_z=nom
        )
        return vertical_stack_report_to_dict(rep)

    def is_gripping_cube(self, cube_index: int) -> bool:
        """True if the gripper appears to be grasping the given cube (contact-based)."""
        return self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=self._cubes[cube_index],
        )

    def _spawn_cubes_horizontal_line_layout(
        self, x_start: float = -0.06, y: float = 0.0, center: bool = False
    ) -> None:
        pitch = 2.0 * float(self.cube_half_extents[0])
        z = self._cube_center_z_on_table()
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if center and self.cube_count > 0:
            span = (self.cube_count - 1) * pitch
            x_start = -span / 2.0
        for i, cube in enumerate(self._cubes):
            p = np.array([x_start + i * pitch, y, z], dtype=np.float64)
            self.sim.data.set_joint_qpos(cube.joints[0], np.concatenate([p, quat]))
        self.sim.forward()

    def _spawn_cubes_top_right_grid(self) -> None:
        """Deterministic grid in top-right region, within Panda workspace (no RandomizationError)."""
        hx = float(self.cube_half_extents[0])
        pitch_x = 2.0 * hx + 0.02   # column spacing: 2cm gap between cubes
        pitch_y = 2.0 * hx + 0.05   # row spacing: 5cm gap for easier pickup
        z = self._cube_center_z_on_table()
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        x_lo, x_hi = SPAWN_TOP_RIGHT_X_RANGE
        y_lo, y_hi = SPAWN_TOP_RIGHT_Y_RANGE
        n = self.cube_count
        ncol = max(1, int(np.ceil(np.sqrt(n * (x_hi - x_lo) * pitch_y / ((y_hi - y_lo) * pitch_x)))))
        nrow = (n + ncol - 1) // ncol
        span_x = (ncol - 1) * pitch_x
        span_y = (nrow - 1) * pitch_y
        cx = (x_lo + x_hi) / 2.0
        cy = (y_lo + y_hi) / 2.0
        x0 = cx - span_x / 2.0
        y0 = cy - span_y / 2.0
        for i, cube in enumerate(self._cubes):
            col, row = i % ncol, i // ncol
            p = np.array([x0 + col * pitch_x, y0 + row * pitch_y, z], dtype=np.float64)
            self.sim.data.set_joint_qpos(cube.joints[0], np.concatenate([p, quat]))
        self.sim.forward()

    def _spawn_cubes_vertical_stack_layout(self, x: float = 0.02, y: float = 0.0) -> None:
        hz = float(self.cube_half_extents[2])
        z0 = self._cube_center_z_on_table()
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for i, cube in enumerate(self._cubes):
            zc = z0 + i * (2.0 * hz)
            p = np.array([x, y, zc], dtype=np.float64)
            self.sim.data.set_joint_qpos(cube.joints[0], np.concatenate([p, quat]))
        self.sim.forward()

    # --- reset ---
    def _reset_internal(self):
        super()._reset_internal()

        if self.placement_profile == "horizontal_line":
            self._spawn_cubes_horizontal_line_layout()
        elif self.placement_profile == "compact_row":
            self._spawn_cubes_horizontal_line_layout(center=True)
        elif self.placement_profile == "spawn_top_right":
            self._spawn_cubes_top_right_grid()
        elif self.placement_profile == "vertical_stack":
            self._spawn_cubes_vertical_stack_layout()
        else:
            placements = self._cube_placer.sample()
            for _key, (pos, quat, obj) in placements.items():
                qpos = np.concatenate([pos, quat])
                self.sim.data.set_joint_qpos(obj.joints[0], qpos)
            self.sim.forward()

    # --- reward (placeholder) ---
    def reward(self, action=None):
        if self.horizontal_reward_mode == "sparse":
            return 1.0 if self.evaluate_horizontal_line()["ok"] else 0.0
        return 0.0

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        ev_h = self.evaluate_horizontal_line()
        info["horizontal_line_ok"] = ev_h["ok"]
        info["horizontal_line"] = ev_h
        ev_v = self.evaluate_vertical_stack()
        info["vertical_stack_ok"] = ev_v["ok"]
        info["vertical_stack"] = ev_v
        return reward, done, info
