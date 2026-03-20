"""rgba_cycle shortcuts — one tuple repeated = all cubes same color."""

from __future__ import annotations

# one entry => i%1 => everything matches
SOLID_BLUE = ((0.08, 0.12, 0.95, 1.0),)
SOLID_RED = ((0.95, 0.12, 0.08, 1.0),)
SOLID_GREEN = ((0.12, 0.85, 0.15, 1.0),)
SOLID_YELLOW = ((0.95, 0.88, 0.12, 1.0),)
SOLID_MAGENTA = ((0.90, 0.12, 0.85, 1.0),)


def default_multicolor():
    # matches env default cycle
    return (
        (1.0, 0.0, 0.0, 1.0),
        (0.10, 0.90, 0.10, 1.0),
        (0.10, 0.10, 0.90, 1.0),
        (0.90, 0.90, 0.10, 1.0),
        (0.90, 0.10, 0.90, 1.0),
    )


def cycle_for_preset(name: str):
    n = (name or "default").strip().lower()
    if n in ("default", "multi", "rainbow", "sort"):
        return default_multicolor()
    if n in ("blue", "all_blue"):
        return SOLID_BLUE
    if n in ("red",):
        return SOLID_RED
    if n in ("green",):
        return SOLID_GREEN
    if n in ("yellow",):
        return SOLID_YELLOW
    if n in ("magenta", "purple"):
        return SOLID_MAGENTA
    raise ValueError(
        f"Unknown color preset {name!r}; try default, blue, red, green, yellow, magenta"
    )
