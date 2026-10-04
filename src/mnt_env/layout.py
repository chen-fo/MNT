"""Bounded layout sampling using only the owning environment's RNG."""

import numpy as np


def integer_array(value, shape, name):
    array = np.asarray(value)
    if array.size == 0 and shape == (0, 2):
        return np.empty(shape, dtype=np.int32)
    if array.shape != shape or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array of shape {shape}")
    if np.any(array < np.iinfo(np.int32).min) or np.any(array > np.iinfo(np.int32).max):
        raise ValueError(f"{name} contains out-of-range integers")
    return array.astype(np.int32, copy=True)


def sample_layout(config, rng, options):
    # Parallel wrappers may pass unrelated reset metadata. Only consume layout
    # keys; validate their values strictly without rejecting wrapper options.
    if not isinstance(options, dict):
        raise ValueError("Layout options must be a dictionary")
    size, count = config.size, config.num_allies + config.num_enemies

    def coordinates(value, shape, name):
        result = integer_array(value, shape, name)
        if np.any(result < 0) or np.any(result >= size):
            raise ValueError(f"{name} must be inside the map")
        return result

    positions = (coordinates(options["positions"], (count, 2), "positions")
                 if "positions" in options else None)
    mines = None
    if "mines" in options:
        value = np.asarray(options["mines"])
        if value.size == 0:
            mines = np.empty((0, 2), dtype=np.int32)
        elif value.ndim == 2 and value.shape[1] == 2:
            mines = coordinates(value, value.shape, "mines")
        else:
            raise ValueError("mines must have shape (m, 2), or be []")
        if len({tuple(p) for p in mines}) != len(mines):
            raise ValueError("Mine positions must be unique")
    excluded = {tuple(p) for group in (positions, mines) if group is not None for p in group}
    if "target" in options:
        target = coordinates(options["target"], (2,), "target")
    else:
        candidates = [(x, y) for x in range(2, size - 2) for y in range(2, size - 2)
                      if (x, y) not in excluded]
        if not candidates:
            raise ValueError("No free interior target cell; supply a valid explicit target")
        target = np.asarray(candidates[int(rng.integers(len(candidates)))], dtype=np.int32)
    if tuple(target) in excluded:
        raise ValueError("Target cannot overlap an initial agent or mine")

    def choose_cells(number, occupied, name):
        candidates = [(x, y) for x in range(size) for y in range(size) if (x, y) not in occupied]
        if number > len(candidates):
            raise ValueError(f"Not enough free cells for {name}")
        indices = rng.choice(len(candidates), size=number, replace=False)
        return np.asarray([candidates[int(i)] for i in indices], dtype=np.int32).reshape(number, 2)

    if positions is None:
        occupied = {tuple(target)} | ({tuple(p) for p in mines} if mines is not None else set())
        positions = choose_cells(count, occupied, "agents")
    if mines is None:
        mines = choose_cells(config.num_mines, {tuple(target)} | {tuple(p) for p in positions}, "mines")
    for index, point in enumerate(positions[:config.num_allies]):
        if any(np.array_equal(point, other) for j, other in enumerate(positions) if j != index):
            raise ValueError("An ally cannot overlap any agent at reset; enemy-enemy overlap is allowed")
        if any(np.array_equal(point, mine) for mine in mines):
            raise ValueError("An ally cannot start on a mine")
    if "headings" in options:
        headings = integer_array(options["headings"], (count,), "headings")
        if np.any(headings < 0) or np.any(headings > 7):
            raise ValueError("headings must be in [0, 7]")
    else:
        headings = rng.integers(0, 8, size=count, dtype=np.int32)
    return positions, headings, target, mines
