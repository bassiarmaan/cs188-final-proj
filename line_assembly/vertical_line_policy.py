"""Tower stack along +z. Same phase machine idea as horizontal_line_policy; needs OSC_POSE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .horizontal_line_policy import (
    Phase,
    _PICK_PLACE_SEQUENCE,
    pick_indices_in_palette_order,
)
from .layout_patterns import pattern_anchor_clear_of_cubes
from .pid_position import PIDPosition


@dataclass
class VerticalCubeJob:
    cube_index: int
    slot_xyz: np.ndarray


class VerticalLineAssemblyPolicy:
    def __init__(
        self,
        env,
        *,
        inter_cube_gap: float = 0.008,
        place_eef_above_center: float = 0.020,
        kp: float = 92.0,
        ki: float = 0.0,
        kd: float = 12.0,
        hover_z: float = 0.10,
        grasp_z_slack: float = 0.003,
        pos_threshold: float = 0.017,
        pos_threshold_coarse: float = 0.038,
        gripper_close_steps: int = 170,
        gripper_open_steps: int = 60,
        settle_steps: int = 14,
        settle_steps_place: int = 34,
        ascend_clear_above_placed_z: float = 0.34,
        final_park_above_top: float = 0.36,
        final_park_settle_steps: int = 25,
        pick_sort_mode: str = "color",
    ):
        self.env = env
        self.inter_cube_gap = float(inter_cube_gap)
        self.place_eef_above_center = float(place_eef_above_center)
        self.hover_z = hover_z
        self.grasp_z_slack = grasp_z_slack
        self.pos_threshold = pos_threshold
        self.pos_threshold_coarse = pos_threshold_coarse
        self.gripper_close_steps = gripper_close_steps
        self.gripper_open_steps = gripper_open_steps
        self.settle_steps = settle_steps
        self.settle_steps_place = settle_steps_place
        self.ascend_clear_above_placed_z = float(ascend_clear_above_placed_z)
        self.final_park_above_top = float(final_park_above_top)
        self.final_park_settle_steps = int(final_park_settle_steps)
        if pick_sort_mode not in ("color", "x"):
            raise ValueError("pick_sort_mode must be 'color' or 'x'")
        self.pick_sort_mode = pick_sort_mode
        self.dt = 1.0 / float(env.control_freq)

        hz = float(env.cube_half_extents[2])
        positions = env.get_cube_world_positions()
        z_table = env._cube_center_z_on_table()
        hx0 = float(env.cube_half_extents[0])
        hy0 = float(env.cube_half_extents[1])
        xc, yc = pattern_anchor_clear_of_cubes(
            env, half_extent_x=hx0 * 1.8, half_extent_y=hy0 * 1.8
        )

        n = positions.shape[0]
        eff_gap = float(self.inter_cube_gap)
        if n >= 7:
            eff_gap = min(eff_gap, 0.005)
        self.inter_cube_gap = eff_gap
        self.env.vertical_stack_eval_gap = eff_gap
        self.stack_pitch = 2.0 * hz + eff_gap

        if self.pick_sort_mode == "x":
            pick_order = np.argsort(positions[:, 0])
        else:
            pick_order = pick_indices_in_palette_order(
                n, positions[:, 0], len(env.rgba_cycle)
            )
        self.pick_order = pick_order.copy()

        self.jobs = [
            VerticalCubeJob(
                int(ci),
                np.array(
                    [xc, yc, z_table + si * self.stack_pitch], dtype=np.float64
                ),
            )
            for si, ci in enumerate(pick_order)
        ]
        self.job_idx = 0
        self.phase = Phase.HOVER_SOURCE
        self.phase_counter = 0
        self.grasp_streak = 0
        self._lift_target: np.ndarray | None = None
        self._ascend_xy = np.zeros(2, dtype=np.float64)
        self._ascend_z_target = 1.0
        self._final_park_target = np.zeros(3, dtype=np.float64)
        self._done = False
        self.pid = PIDPosition(kp, ki, kd, np.zeros(3))
        self.pid.reset(self._eef_target())

    @property
    def finished(self) -> bool:
        return self._done

    def _cube_pos(self, idx: int) -> np.ndarray:
        return self.env.get_cube_world_positions()[idx].copy()

    def _slot_release_xyz(self, job: VerticalCubeJob) -> np.ndarray:
        s = job.slot_xyz
        return np.array(
            [s[0], s[1], s[2] + self.place_eef_above_center], dtype=np.float64
        )

    def _eef_target(self) -> np.ndarray:
        if self.phase == Phase.ASCEND_CLEAR:
            return np.array(
                [self._ascend_xy[0], self._ascend_xy[1], self._ascend_z_target],
                dtype=np.float64,
            )
        if self.phase == Phase.FINAL_PARK:
            return self._final_park_target.copy()
        job = self.jobs[self.job_idx]
        c = self._cube_pos(job.cube_index)
        s = job.slot_xyz
        if self.phase == Phase.HOVER_SOURCE:
            zt = float(self.env._cube_center_z_on_table())
            z_h = max(c[2] + self.hover_z, zt + 0.28)
            return np.array([c[0], c[1], z_h])
        if self.phase == Phase.DOWN_SOURCE:
            return np.array([c[0], c[1], c[2] - self.grasp_z_slack])
        if self.phase == Phase.CLOSE_GRIP:
            return np.array([c[0], c[1], c[2] - self.grasp_z_slack])
        if self.phase == Phase.LIFT:
            if self._lift_target is not None:
                return self._lift_target.copy()
            zt = float(self.env._cube_center_z_on_table())
            z_up = max(c[2] + self.hover_z + 0.06, zt + 0.30)
            return np.array([c[0], c[1], z_up])
        if self.phase == Phase.HOVER_SLOT:
            rel = self._slot_release_xyz(job)
            zt = float(self.env._cube_center_z_on_table())
            # Extra margin over the slot so the wrist clears the tower on high placements.
            z_h = max(rel[2] + self.hover_z + 0.24, zt + 0.40, float(s[2]) + 0.10)
            return np.array([rel[0], rel[1], z_h])
        if self.phase == Phase.DOWN_SLOT:
            return self._slot_release_xyz(job)
        return self._slot_release_xyz(job)

    def _advance_phase(self):
        assert self.phase not in (
            Phase.OPEN_GRIP,
            Phase.ASCEND_CLEAR,
            Phase.FINAL_PARK,
        )
        prev = self.phase
        i = _PICK_PLACE_SEQUENCE.index(self.phase)
        self.phase = _PICK_PLACE_SEQUENCE[i + 1]
        self.phase_counter = 0
        self.grasp_streak = 0
        if prev == Phase.CLOSE_GRIP and self.phase == Phase.LIFT:
            c = self._cube_pos(self.jobs[self.job_idx].cube_index)
            zt = float(self.env._cube_center_z_on_table())
            z_up = max(float(c[2]) + self.hover_z + 0.07, zt + 0.30)
            self._lift_target = np.array(
                [float(c[0]), float(c[1]), z_up],
                dtype=np.float64,
            )
        if prev == Phase.OPEN_GRIP:
            self._lift_target = None
        self.pid.reset(self._eef_target())

    def _advance_after_open_grip(self):
        self.phase_counter = 0
        self.grasp_streak = 0
        self._lift_target = None
        placed = self.jobs[self.job_idx]
        slot_xy = placed.slot_xyz[:2].copy()
        top_z = float(placed.slot_xyz[2])
        self.job_idx += 1
        if self.job_idx >= len(self.jobs):
            self._start_final_park()
            return
        self._ascend_xy = slot_xy
        self._ascend_z_target = top_z + self.ascend_clear_above_placed_z
        self.phase = Phase.ASCEND_CLEAR
        self.pid.reset(self._eef_target())

    def _finish_ascend_clear(self):
        self.phase = Phase.HOVER_SOURCE
        self.phase_counter = 0
        self.pid.reset(self._eef_target())

    def _start_final_park(self):
        cx = float(np.mean([j.slot_xyz[0] for j in self.jobs]))
        cy = float(self.jobs[0].slot_xyz[1])
        top_z = float(self.jobs[-1].slot_xyz[2])
        z = top_z + self.final_park_above_top
        self._final_park_target = np.array([cx, cy, z], dtype=np.float64)
        self.phase = Phase.FINAL_PARK
        self.phase_counter = 0
        self.pid.reset(self._final_park_target)

    def get_action(self, obs: dict) -> np.ndarray:
        low, high = self.env.action_spec
        dim = low.shape[0]
        action = np.zeros(dim, dtype=np.float64)

        if self._done:
            action[6:] = -1.0
            return np.clip(action, low, high)

        eef = obs["robot0_eef_pos"]
        self.pid.set_target(self._eef_target())

        ctrl = self.pid.update(eef, self.dt)
        if self.phase in (Phase.DOWN_SLOT, Phase.OPEN_GRIP):
            ctrl[2] *= 0.42
        elif self.phase == Phase.DOWN_SOURCE:
            ctrl[2] *= 0.65
        if self.phase == Phase.ASCEND_CLEAR:
            ctrl[0] *= 0.2
            ctrl[1] *= 0.2
        elif self.phase == Phase.FINAL_PARK:
            ctrl[0] *= 0.35
            ctrl[1] *= 0.35
        elif self.phase in (Phase.LIFT, Phase.HOVER_SLOT, Phase.DOWN_SLOT):
            ctrl[0] *= 0.18
            ctrl[1] *= 0.18

        action[0:3] = np.clip(ctrl, -1.0, 1.0)
        action[3:6] = 0.0

        thr = (
            self.pos_threshold_coarse
            if self.phase
            in (
                Phase.LIFT,
                Phase.HOVER_SLOT,
                Phase.DOWN_SLOT,
                Phase.ASCEND_CLEAR,
                Phase.FINAL_PARK,
            )
            else self.pos_threshold
        )
        err_ok = self.pid.error_norm() < thr
        grip_open = -1.0
        grip_close = 1.0

        settle_need = (
            self.settle_steps_place
            if self.phase in (Phase.DOWN_SLOT, Phase.OPEN_GRIP)
            else self.settle_steps
        )

        if self.phase in (Phase.HOVER_SOURCE, Phase.DOWN_SOURCE):
            g = grip_open
            if err_ok:
                self.phase_counter += 1
                if self.phase_counter >= settle_need:
                    self._advance_phase()
            else:
                self.phase_counter = 0
        elif self.phase == Phase.CLOSE_GRIP:
            g = grip_close
            self.phase_counter += 1
            job = self.jobs[self.job_idx]
            if self.env.is_gripping_cube(job.cube_index):
                self.grasp_streak += 1
            else:
                # Decay instead of zeroing so brief sim flicker does not force a "blind" lift.
                self.grasp_streak = max(0, self.grasp_streak - 4)
            if self.grasp_streak >= 26:
                self._advance_phase()
            elif self.phase_counter >= self.gripper_close_steps and self.grasp_streak >= 14:
                self._advance_phase()
            elif self.phase_counter >= self.gripper_close_steps + 120:
                self._advance_phase()
        elif self.phase in (Phase.LIFT, Phase.HOVER_SLOT, Phase.DOWN_SLOT):
            g = grip_close
            if err_ok:
                self.phase_counter += 1
                if self.phase_counter >= settle_need:
                    self._advance_phase()
            else:
                self.phase_counter = 0
        elif self.phase == Phase.ASCEND_CLEAR:
            g = grip_open
            if err_ok:
                self.phase_counter += 1
                if self.phase_counter >= self.settle_steps:
                    self._finish_ascend_clear()
            else:
                self.phase_counter = 0
        elif self.phase == Phase.FINAL_PARK:
            g = grip_open
            if err_ok:
                self.phase_counter += 1
                if self.phase_counter >= self.final_park_settle_steps:
                    self._done = True
            else:
                self.phase_counter = 0
        else:
            g = grip_open
            self.phase_counter += 1
            if self.phase_counter >= self.gripper_open_steps:
                self._advance_after_open_grip()

        g_dim = dim - 6
        if g_dim == 1:
            action[6] = g
        else:
            action[6:] = g

        return np.clip(action, low, high)
