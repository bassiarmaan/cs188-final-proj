"""Cheap geometry checks: did they actually make a line / a stack? World frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HorizontalLineReport:
    ok: bool
    y_span: float
    z_span: float
    pitches: np.ndarray
    pitch_errors: np.ndarray
    messages: list[str] = field(default_factory=list)


def check_horizontal_adjacent_line(
    positions_xyz: np.ndarray,
    half_extents_xyz,
    *,
    axis: int = 0,
    nominal_pitch: float | None = None,
    y_span_max: float = 0.04,
    z_span_max: float = 0.055,
    min_pitch_frac: float = 0.72,
    max_pitch_frac: float = 1.75,
) -> HorizontalLineReport:
    # row along +x by default; bump nominal_pitch if you left a gap between blocks on purpose
    msgs: list[str] = []
    pos = np.asarray(positions_xyz, dtype=np.float64).reshape(-1, 3)
    n = pos.shape[0]
    hx = float(np.asarray(half_extents_xyz, dtype=np.float64).reshape(3)[axis])
    if nominal_pitch is None:
        nominal_pitch = 2.0 * hx
    nominal_pitch = float(nominal_pitch)

    if n == 0:
        return HorizontalLineReport(False, 0.0, 0.0, np.array([]), np.array([]), ["no cubes"])
    if n == 1:
        return HorizontalLineReport(True, 0.0, 0.0, np.array([]), np.array([]), ["only one cube"])

    # how sloppy the row is in the other axes
    axes = [0, 1, 2]
    off = [a for a in axes if a != axis]
    span_y = float(np.ptp(pos[:, off[0]]))
    span_z = float(np.ptp(pos[:, off[1]]))

    if span_y > y_span_max:
        msgs.append(f"y_span {span_y:.4f} > {y_span_max}")
    if span_z > z_span_max:
        msgs.append(f"z_span {span_z:.4f} > {z_span_max}")

    order = np.argsort(pos[:, axis])
    xs = pos[order, axis]
    pitches = np.diff(xs)
    if np.any(pitches <= 1e-4):
        msgs.append("non-increasing along axis (overlapping or duplicate ordering)")

    lo = nominal_pitch * min_pitch_frac
    hi = nominal_pitch * max_pitch_frac
    pitch_err = np.maximum(0.0, lo - pitches) + np.maximum(0.0, pitches - hi)
    bad_pitch = (pitches < lo) | (pitches > hi)
    if np.any(bad_pitch):
        msgs.append(
            f"pitch out of [{lo:.4f},{hi:.4f}] (nominal {nominal_pitch:.4f}): {pitches.tolist()}"
        )

    ok = len(msgs) == 0
    return HorizontalLineReport(
        ok=ok,
        y_span=span_y,
        z_span=span_z,
        pitches=pitches,
        pitch_errors=pitch_err,
        messages=msgs,
    )


def report_to_info_dict(rep: HorizontalLineReport) -> dict:
    return {
        "ok": rep.ok,
        "y_span": rep.y_span,
        "z_span": rep.z_span,
        "pitches": rep.pitches.tolist(),
        "pitch_errors": rep.pitch_errors.tolist(),
        "messages": list(rep.messages),
    }


@dataclass
class VerticalStackReport:
    ok: bool
    x_span: float
    y_span: float
    pitches_z: np.ndarray
    pitch_errors: np.ndarray
    messages: list[str] = field(default_factory=list)


def check_vertical_adjacent_stack(
    positions_xyz: np.ndarray,
    half_extents_xyz,
    *,
    nominal_pitch_z: float | None = None,
    x_span_max: float = 0.055,
    y_span_max: float = 0.10,
    min_pitch_frac: float = 0.72,
    max_pitch_frac: float = 1.75,
) -> VerticalStackReport:
    # same x/y, step up z; nominal_pitch_z accounts for gap if you use one
    msgs: list[str] = []
    pos = np.asarray(positions_xyz, dtype=np.float64).reshape(-1, 3)
    n = pos.shape[0]
    hz = float(np.asarray(half_extents_xyz, dtype=np.float64).reshape(3)[2])
    if nominal_pitch_z is None:
        nominal_pitch_z = 2.0 * hz
    nominal_pitch_z = float(nominal_pitch_z)

    if n == 0:
        return VerticalStackReport(False, 0.0, 0.0, np.array([]), np.array([]), ["no cubes"])
    if n == 1:
        return VerticalStackReport(True, 0.0, 0.0, np.array([]), np.array([]), ["only one cube"])

    span_x = float(np.ptp(pos[:, 0]))
    span_y = float(np.ptp(pos[:, 1]))
    if span_x > x_span_max:
        msgs.append(f"x_span {span_x:.4f} > {x_span_max}")
    if span_y > y_span_max:
        msgs.append(f"y_span {span_y:.4f} > {y_span_max}")

    order = np.argsort(pos[:, 2])
    zs = pos[order, 2]
    pitches = np.diff(zs)
    if np.any(pitches <= 1e-4):
        msgs.append("non-increasing along z (overlapping or duplicate z order)")

    lo = nominal_pitch_z * min_pitch_frac
    hi = nominal_pitch_z * max_pitch_frac
    pitch_err = np.maximum(0.0, lo - pitches) + np.maximum(0.0, pitches - hi)
    bad_pitch = (pitches < lo) | (pitches > hi)
    if np.any(bad_pitch):
        msgs.append(
            f"z pitch out of [{lo:.4f},{hi:.4f}] (nominal {nominal_pitch_z:.4f}): {pitches.tolist()}"
        )

    ok = len(msgs) == 0
    return VerticalStackReport(
        ok=ok,
        x_span=span_x,
        y_span=span_y,
        pitches_z=pitches,
        pitch_errors=pitch_err,
        messages=msgs,
    )


def vertical_stack_report_to_dict(rep: VerticalStackReport) -> dict:
    return {
        "ok": rep.ok,
        "x_span": rep.x_span,
        "y_span": rep.y_span,
        "pitches_z": rep.pitches_z.tolist(),
        "pitch_errors": rep.pitch_errors.tolist(),
        "messages": list(rep.messages),
    }
