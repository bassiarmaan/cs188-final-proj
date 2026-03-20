"""
CA1 vibes: state machine + PID on eef pos, line the cubes up.

Won't work on default JOINT_VELOCITY — need OSC_POSE (delta pos in action[:6], gripper 6:).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .pid_position import PIDPosition


class Phase(Enum):
    HOVER_SOURCE = auto()
    DOWN_SOURCE = auto()
    CLOSE_GRIP = auto()
    LIFT = auto()
    HOVER_SLOT = auto()
    DOWN_SLOT = auto()
    OPEN_GRIP = auto()
    ASCEND_CLEAR = auto()  # go up before scooting sideways (don't bowl over the row)
    FINAL_PARK = auto()  # park camera-friendly above the line


# OPEN_GRIP jumps out via _advance_after_open_grip, not listed here
_PICK_PLACE_SEQUENCE = (
    Phase.HOVER_SOURCE,
    Phase.DOWN_SOURCE,
    Phase.CLOSE_GRIP,
    Phase.LIFT,
    Phase.HOVER_SLOT,
    Phase.DOWN_SLOT,
    Phase.OPEN_GRIP,
)


@dataclass
class CubeJob:
    cube_index: int
    slot_xyz: np.ndarray


def pick_indices_in_palette_order(
    n_cubes: int, pos_x: np.ndarray, palette_size: int
) -> np.ndarray:
    # cube i has color i % palette_size (same as env). Sort by color band then by x so the row reads rainbow-ish
    palette_idx = (np.arange(n_cubes, dtype=np.int64) % int(palette_size)).astype(np.float64)
    px = np.asarray(pos_x, dtype=np.float64).ravel()[:n_cubes]
    # lexsort: last key is primary → palette, then x
    return np.lexsort((px, palette_idx))


class HorizontalLineAssemblyPolicy:
    def __init__(
        self,
        env,
        *,
        inter_cube_gap: float = 0.008,
        place_eef_above_center: float = 0.016,
        kp: float = 85.0,
        ki: float = 0.0,
        kd: float = 10.0,
        hover_z: float = 0.10,
        grasp_z_slack: float = 0.003,
        pos_threshold: float = 0.022,
        pos_threshold_coarse: float = 0.045,
        gripper_close_steps: int = 120,
        gripper_open_steps: int = 55,
        settle_steps: int = 12,
        settle_steps_place: int = 22,
        clear_height_above_table: float = 0.20,
        final_park_height_above_table: float = 0.42,
        final_park_settle_steps: int = 25,
        pick_sort_mode: str = "color",
    ):
        self.env = env
        self.inter_cube_gap = float(inter_cube_gap)
        self.place_eef_above_center = float(place_eef_above_center)
        self.env.horizontal_line_eval_gap = self.inter_cube_gap

        self.hover_z = hover_z
        self.grasp_z_slack = grasp_z_slack
        self.pos_threshold = pos_threshold
        self.pos_threshold_coarse = pos_threshold_coarse
        self.gripper_close_steps = gripper_close_steps
        self.gripper_open_steps = gripper_open_steps
        self.settle_steps = settle_steps
        self.settle_steps_place = settle_steps_place
        self.clear_height_above_table = float(clear_height_above_table)
        self.final_park_height_above_table = float(final_park_height_above_table)
        self.final_park_settle_steps = int(final_park_settle_steps)
        if pick_sort_mode not in ("color", "x"):
            raise ValueError("pick_sort_mode must be 'color' or 'x'")
        self.pick_sort_mode = pick_sort_mode
        self.dt = 1.0 / float(env.control_freq)

        hx = float(env.cube_half_extents[0])
        # Center-to-center spacing with a gap so placed cubes do not interpenetrate / collide
        self.slot_pitch = 2.0 * hx + self.inter_cube_gap

        positions = env.get_cube_world_positions()
        z_table = env._cube_center_z_on_table()
        y_line = float(np.clip(np.mean(positions[:, 1]), -0.08, 0.08))
        x0 = float(np.min(positions[:, 0]) - 0.05)
        x0 = max(x0, -0.16)

        n = positions.shape[0]
        if self.pick_sort_mode == "x":
            pick_order = np.argsort(positions[:, 0])
        else:
            pick_order = pick_indices_in_palette_order(
                n, positions[:, 0], len(env.rgba_cycle)
            )
        self.pick_order = pick_order.copy()

        self.jobs = [
            CubeJob(
                int(ci),
                np.array([x0 + si * self.slot_pitch, y_line, z_table], dtype=np.float64),
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

    def _slot_release_xyz(self, job: CubeJob) -> np.ndarray:
        # release a bit above center so we don't ram the table
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
            return np.array([c[0], c[1], c[2] + self.hover_z])
        if self.phase == Phase.DOWN_SOURCE:
            return np.array([c[0], c[1], c[2] - self.grasp_z_slack])
        if self.phase == Phase.CLOSE_GRIP:
            return np.array([c[0], c[1], c[2] - self.grasp_z_slack])
        if self.phase == Phase.LIFT:
            if self._lift_target is not None:
                return self._lift_target.copy()
            return np.array([c[0], c[1], c[2] + self.hover_z + 0.06])
        if self.phase == Phase.HOVER_SLOT:
            # come in high over the row so we don't clothesline the line
            rel = self._slot_release_xyz(job)
            return np.array([rel[0], rel[1], rel[2] + self.hover_z + 0.08])
        if self.phase == Phase.DOWN_SLOT:
            return self._slot_release_xyz(job)
        return self._slot_release_xyz(job)

    def _advance_phase(self):
        assert self.phase != Phase.OPEN_GRIP
        assert self.phase != Phase.ASCEND_CLEAR
        assert self.phase != Phase.FINAL_PARK
        prev = self.phase
        i = _PICK_PLACE_SEQUENCE.index(self.phase)
        self.phase = _PICK_PLACE_SEQUENCE[i + 1]
        self.phase_counter = 0
        self.grasp_streak = 0
        if prev == Phase.CLOSE_GRIP and self.phase == Phase.LIFT:
            c = self._cube_pos(self.jobs[self.job_idx].cube_index)
            self._lift_target = np.array(
                [float(c[0]), float(c[1]), float(c[2]) + self.hover_z + 0.07],
                dtype=np.float64,
            )
        if prev == Phase.OPEN_GRIP:
            self._lift_target = None
        self.pid.reset(self._eef_target())

    def _advance_after_open_grip(self):
        # pop up then on to the next cube
        self.phase_counter = 0
        self.grasp_streak = 0
        self._lift_target = None
        slot_xy = self.jobs[self.job_idx].slot_xyz[:2].copy()
        self.job_idx += 1
        if self.job_idx >= len(self.jobs):
            self._start_final_park()
            return
        z_table = self.env._cube_center_z_on_table()
        self._ascend_xy = slot_xy
        self._ascend_z_target = float(z_table + self.clear_height_above_table)
        self.phase = Phase.ASCEND_CLEAR
        self.pid.reset(self._eef_target())

    def _finish_ascend_clear(self):
        self.phase = Phase.HOVER_SOURCE
        self.phase_counter = 0
        self.pid.reset(self._eef_target())

    def _start_final_park(self):
        z_table = self.env._cube_center_z_on_table()
        xs = [float(j.slot_xyz[0]) for j in self.jobs]
        cx = float(np.mean(xs))
        cy = float(self.jobs[0].slot_xyz[1])
        z = z_table + self.final_park_height_above_table
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
        # don't drill through the table on place
        if self.phase in (Phase.DOWN_SLOT, Phase.OPEN_GRIP):
            ctrl[2] *= 0.42
        elif self.phase == Phase.DOWN_SOURCE:
            ctrl[2] *= 0.65
        elif self.phase == Phase.ASCEND_CLEAR:
            # mostly go up; don't wander in xy while ascending
            ctrl[0] *= 0.2
            ctrl[1] *= 0.2
        elif self.phase == Phase.FINAL_PARK:
            ctrl[0] *= 0.35
            ctrl[1] *= 0.35

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
                self.grasp_streak = 0
            if self.grasp_streak >= 28:
                self._advance_phase()
            elif self.phase_counter >= self.gripper_close_steps:
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
        else:  # OPEN_GRIP
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
