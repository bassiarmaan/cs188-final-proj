"""World XYZ targets for smiley bitmap + column grids (used by GenericAssemblyPolicy)."""

from __future__ import annotations

import numpy as np

# fudge factor workspace so we don't park patterns off the table
_WORKSPACE_X = (-0.26, 0.26)
_WORKSPACE_Y = (-0.14, 0.14)

# Bitmap: '1' = cube. Default 5x7 smiley; overwritten by set_smiley_rows() for custom designs (e.g. 6x8 from GPT).
SMILEY_ROWS = (
    "0100010",
    "0001000",
    "0000000",
    "1110111",
    "0000000",
)


def set_smiley_rows(rows: tuple[str, ...]) -> None:
    """Set SMILEY_ROWS to a custom bitmap. Validates: same-length rows, only 0 and 1."""
    global SMILEY_ROWS
    if not rows:
        raise ValueError("Bitmap must have at least one row")
    ncols = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != ncols:
            raise ValueError(f"Row {i} has length {len(row)}, expected {ncols}")
        if not all(c in "01" for c in row):
            raise ValueError(f"Row {i} contains invalid chars (only 0 and 1 allowed)")
    SMILEY_ROWS = tuple(rows)


def smiley_cell_offsets() -> list[tuple[int, int]]:
    # (col, row) for each 1 in the ascii art
    out: list[tuple[int, int]] = []
    for r, row in enumerate(SMILEY_ROWS):
        for c, ch in enumerate(row):
            if ch == "1":
                out.append((c, r))
    return out


def _smiley_half_extents_xy(xy_pitch: float) -> tuple[float, float]:
    # how fat the smiley is — used so we can scoot it away from the spawn blob
    ncols = len(SMILEY_ROWS[0])
    nrows = len(SMILEY_ROWS)
    c_mid = (ncols - 1) / 2.0
    r_mid = (nrows - 1) / 2.0
    offs = smiley_cell_offsets()
    hx = max(abs(c - c_mid) for c, _ in offs) * xy_pitch
    hy = max(abs(r - r_mid) for _, r in offs) * xy_pitch
    return float(hx), float(hy)


def _grid_footprint_half_extents(nx: int, ny: int, xy_pitch: float) -> tuple[float, float]:
    # grid radius in xy for anchor math
    return (
        float((nx - 1) * 0.5 * xy_pitch),
        float((ny - 1) * 0.5 * xy_pitch),
    )


def grid_cell_visit_order(nx: int, ny: int) -> list[tuple[int, int]]:
    # 2×2 overrides with robot-aware order inside grid_stack_world_slots; others: row-major in x then y.
    return [(ix, iy) for ix in range(nx) for iy in range(ny)]


def _robot_base_xy(env) -> np.ndarray:
    """World XY of the first manipulator base; origin if lookup fails."""
    try:
        r = env.robots[0]
        if hasattr(r, "base_pos"):
            p = np.asarray(r.base_pos, dtype=np.float64).ravel()
            return np.array([float(p[0]), float(p[1])], dtype=np.float64)
        m = r.robot_model
        if hasattr(m, "base_xpos"):
            p = np.asarray(m.base_xpos, dtype=np.float64).ravel()
            return np.array([float(p[0]), float(p[1])], dtype=np.float64)
        sim = env.sim
        prefix = getattr(m, "naming_prefix", "robot0_")
        bid = sim.model.body_name2id(prefix + "base")
        p = sim.data.get_body_xpos(bid)
        return np.array([float(p[0]), float(p[1])], dtype=np.float64)
    except Exception:
        return np.zeros(2, dtype=np.float64)


def _grid_visit_order_rows_far_first(
    env,
    nx: int,
    ny: int,
    cx: float,
    cy: float,
    xy_pitch: float,
) -> list[tuple[int, int]]:
    """Visit by row (iy): farther-from-robot row first; within a row, farther ix first."""
    rb = _robot_base_xy(env)

    def cell_xy(ix: int, iy: int) -> tuple[float, float]:
        x = cx + (ix - (nx - 1) / 2.0) * xy_pitch
        y = cy + (iy - (ny - 1) / 2.0) * xy_pitch
        return float(x), float(y)

    def dist(ix: int, iy: int) -> float:
        x, y = cell_xy(ix, iy)
        return float(np.hypot(x - rb[0], y - rb[1]))

    rows = list(range(ny))
    rows.sort(
        key=lambda iy: (
            -max(dist(ix, iy) for ix in range(nx)),
            iy,
        )
    )
    out: list[tuple[int, int]] = []
    for iy in rows:
        cols = list(range(nx))
        cols.sort(key=lambda ix: (-dist(ix, iy), ix))
        for ix in cols:
            out.append((ix, iy))
    return out


def _footprint_disjoint(
    cx: float,
    cy: float,
    hx: float,
    hy: float,
    xmin_p: float,
    xmax_p: float,
    ymin_p: float,
    ymax_p: float,
    eps: float = 0.003,
) -> bool:
    # pattern box vs padded mess of initial cubes
    pl, pr = cx - hx, cx + hx
    pb, pt = cy - hy, cy + hy
    return (
        pr < xmin_p - eps
        or pl > xmax_p + eps
        or pt < ymin_p - eps
        or pb > ymax_p + eps
    )


def _min_dist_centers_to_footprint(
    cx: float,
    cy: float,
    hx: float,
    hy: float,
    pos_xy: np.ndarray,
) -> float:
    # scoring for "how far is this anchor from the clutter"
    dmin = float("inf")
    for i in range(pos_xy.shape[0]):
        px, py = float(pos_xy[i, 0]), float(pos_xy[i, 1])
        dx = 0.0
        if px < cx - hx:
            dx = (cx - hx) - px
        elif px > cx + hx:
            dx = px - (cx + hx)
        dy = 0.0
        if py < cy - hy:
            dy = (cy - hy) - py
        elif py > cy + hy:
            dy = py - (cy + hy)
        d = float(np.hypot(dx, dy))
        dmin = min(dmin, d)
    return dmin


def pattern_anchor_clear_of_cubes(
    env,
    *,
    half_extent_x: float,
    half_extent_y: float,
    margin: float = 0.045,
) -> tuple[float, float]:
    # try to not build the smiley on top of the random pile — nudge to empty side / grid search / fallback
    pos = env.get_cube_world_positions()
    pos_xy = pos[:, :2]
    xmin = float(np.min(pos[:, 0]))
    xmax = float(np.max(pos[:, 0]))
    ymin = float(np.min(pos[:, 1]))
    ymax = float(np.max(pos[:, 1]))

    ch = float(max(env.cube_half_extents[0], env.cube_half_extents[1]))
    m = float(margin)
    xmin_p = xmin - m - ch
    xmax_p = xmax + m + ch
    ymin_p = ymin - m - ch
    ymax_p = ymax + m + ch

    wx_lo, wx_hi = _WORKSPACE_X
    wy_lo, wy_hi = _WORKSPACE_Y
    hx = float(half_extent_x)
    hy = float(half_extent_y)

    def fits(cx: float, cy: float) -> bool:
        return (
            wx_lo + hx <= cx <= wx_hi - hx
            and wy_lo + hy <= cy <= wy_hi - hy
        )

    ymid = float(np.clip(0.5 * (ymin + ymax), wy_lo + hy, wy_hi - hy))
    xmid = float(np.clip(0.5 * (xmin + xmax), wx_lo + hx, wx_hi - hx))

    gap = 0.003
    cx_l = xmin_p - hx - gap
    if fits(cx_l, ymid) and _footprint_disjoint(
        cx_l, ymid, hx, hy, xmin_p, xmax_p, ymin_p, ymax_p
    ):
        return cx_l, ymid

    cx_r = xmax_p + hx + gap
    if fits(cx_r, ymid) and _footprint_disjoint(
        cx_r, ymid, hx, hy, xmin_p, xmax_p, ymin_p, ymax_p
    ):
        return cx_r, ymid

    cy_s = ymin_p - hy - gap
    if fits(xmid, cy_s) and _footprint_disjoint(
        xmid, cy_s, hx, hy, xmin_p, xmax_p, ymin_p, ymax_p
    ):
        return xmid, cy_s

    cy_n = ymax_p + hy + gap
    if fits(xmid, cy_n) and _footprint_disjoint(
        xmid, cy_n, hx, hy, xmin_p, xmax_p, ymin_p, ymax_p
    ):
        return xmid, cy_n

    # brute force a grid of centers if the tidy sides don't work
    nxg, nyg = 15, 11
    best_disjoint: tuple[float, float] | None = None
    best_d_disc = -1.0

    x_span = (wx_hi - hx) - (wx_lo + hx)
    y_span = (wy_hi - hy) - (wy_lo + hy)
    for i in range(nxg):
        cx = (wx_lo + hx) + (i / max(nxg - 1, 1)) * x_span if nxg > 1 else wx_lo + hx
        for j in range(nyg):
            cy = (wy_lo + hy) + (j / max(nyg - 1, 1)) * y_span if nyg > 1 else wy_lo + hy
            if not fits(cx, cy):
                continue
            disc = _footprint_disjoint(cx, cy, hx, hy, xmin_p, xmax_p, ymin_p, ymax_p)
            if not disc:
                continue
            d = _min_dist_centers_to_footprint(cx, cy, hx, hy, pos_xy)
            if d > best_d_disc:
                best_d_disc = d
                best_disjoint = (float(cx), float(cy))

    if best_disjoint is not None:
        return best_disjoint

    # give up on perfect clearance — hug the left or right edge opposite where cubes landed
    xm = float(np.mean(pos[:, 0]))
    ym = float(np.mean(pos[:, 1]))
    cy = float(np.clip(ym, wy_lo + hy, wy_hi - hy))
    edge_inset = 0.02
    if xm >= -0.02:
        cx = wx_lo + hx + edge_inset
    else:
        cx = wx_hi - hx - edge_inset
    cx = float(np.clip(cx, wx_lo + hx, wx_hi - hx))
    return cx, cy


def smiley_world_slots(
    env,
    *,
    xy_pitch_scale: float = 1.15,
) -> list[np.ndarray]:
    # one layer, z = table. anchor tries to avoid the random heap
    hx = float(env.cube_half_extents[0])
    xy_pitch = (2.0 * hx + 0.012) * float(xy_pitch_scale)
    half_x, half_y = _smiley_half_extents_xy(xy_pitch)
    cx, cy = pattern_anchor_clear_of_cubes(env, half_extent_x=half_x, half_extent_y=half_y)
    z_table = env._cube_center_z_on_table()

    ncols = len(SMILEY_ROWS[0])
    nrows = len(SMILEY_ROWS)
    c_mid = (ncols - 1) / 2.0
    r_mid = (nrows - 1) / 2.0

    slots: list[np.ndarray] = []
    for c, r in smiley_cell_offsets():
        x = cx + (c - c_mid) * xy_pitch
        y = cy + (r - r_mid) * xy_pitch
        slots.append(np.array([x, y, z_table], dtype=np.float64))
    return slots


def grid_stack_world_slots(
    env,
    nx: int,
    ny: int,
    stack_height: int,
    *,
    xy_pitch_scale: float = 1.48,
    inter_cube_gap_z: float = 0.01,
) -> list[np.ndarray]:
    # Multi-layer grids: layer-first (all bottoms, then all second layers, …) so the arm rarely
    # carries a cube over a two-high neighbor. 2×2 cell order: far row/column from robot base first.
    hx = float(env.cube_half_extents[0])
    hz = float(env.cube_half_extents[2])
    xy_pitch = (2.0 * hx + 0.012) * float(xy_pitch_scale)
    stack_pitch = 2.0 * hz + float(inter_cube_gap_z)

    half_x, half_y = _grid_footprint_half_extents(nx, ny, xy_pitch)
    cx, cy = pattern_anchor_clear_of_cubes(env, half_extent_x=half_x, half_extent_y=half_y)
    z_table = env._cube_center_z_on_table()

    if nx == 2 and ny == 2:
        visit = _grid_visit_order_rows_far_first(env, nx, ny, cx, cy, xy_pitch)
    else:
        visit = grid_cell_visit_order(nx, ny)

    slots: list[np.ndarray] = []
    if stack_height > 1:
        for iz in range(stack_height):
            for ix, iy in visit:
                x = cx + (ix - (nx - 1) / 2.0) * xy_pitch
                y = cy + (iy - (ny - 1) / 2.0) * xy_pitch
                z = z_table + iz * stack_pitch
                slots.append(np.array([x, y, z], dtype=np.float64))
    else:
        for ix, iy in visit:
            x = cx + (ix - (nx - 1) / 2.0) * xy_pitch
            y = cy + (iy - (ny - 1) / 2.0) * xy_pitch
            z = z_table
            slots.append(np.array([x, y, z], dtype=np.float64))
    return slots


def smiley_cube_count() -> int:
    return len(smiley_cell_offsets())


def grid_cube_count(nx: int, ny: int, stack_height: int) -> int:
    return nx * ny * stack_height
