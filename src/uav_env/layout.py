"""Validated UAV layouts sampled only from the owning environment's RNG."""

import numpy as np


MAX_LAYOUT_ATTEMPTS = 1000


def integer_array(value, shape, name):
    """Copy an exactly shaped integer array without silently rounding values."""
    array = np.asarray(value)
    # A mixed Python sequence such as [True, 2] otherwise coerces to integers.
    has_boolean = any(isinstance(item, (bool, np.bool_))
                      for item in np.asarray(value, dtype=object).flat)
    if array.shape != shape or array.dtype.kind not in "iu" or has_boolean:
        raise ValueError(f"{name} must be an integer array of shape {shape}")
    limits = np.iinfo(np.int32)
    if np.any(array < limits.min) or np.any(array > limits.max):
        raise ValueError(f"{name} contains out-of-range integers")
    return array.astype(np.int32, copy=True)


def sample_layout(config, rng, options):
    """Return distinct starts outside capture range and a reachable empty target.

    Reachability concerns empty-grid navigation within the task deadline, not
    guaranteed escape against an adversary. Unknown options are wrapper metadata.
    Random sampling is bounded; crowded configurations may exhaust the budget.
    """
    if not isinstance(options, dict):
        raise ValueError("Layout options must be a dictionary")
    size = config.size
    count = config.num_evaders + config.num_pursuers

    def coordinates(value, shape, name):
        result = integer_array(value, shape, name)
        if np.any(result < 0) or np.any(result >= size):
            raise ValueError(f"{name} must be inside the map")
        return result

    def distances(points, others):
        # int64 avoids overflow when subtracting validated int32 coordinates.
        return np.abs(points.astype(np.int64)[:, None, :]
                      - others.astype(np.int64)[None, :, :]).sum(axis=2)

    positions = (coordinates(options["positions"], (count, 2), "positions")
                 if "positions" in options else None)
    target = (coordinates(options["target"], (2,), "target")
              if "target" in options else None)
    cells = np.indices((size, size), dtype=np.int32).reshape(2, -1).T

    if positions is not None:
        if len({tuple(point) for point in positions}) != count:
            raise ValueError("Initial agent positions must be distinct")
        evaders = positions[:config.num_evaders]
        pursuers = positions[config.num_evaders:]
        if np.any(distances(evaders, pursuers) <= 1):
            raise ValueError("Initial evaders and pursuers must be outside capture range")
        if target is None:
            free = ~np.any(np.all(cells[:, None, :] == positions[None, :, :], axis=2), axis=1)
            reachable = np.all(distances(cells, evaders) <= config.max_steps, axis=1)
            candidates = cells[free & reachable]
            if not len(candidates):
                raise ValueError("No free target reachable by every evader within max_steps")
            target = candidates[int(rng.integers(len(candidates)))].copy()
        elif np.any(np.all(positions == target, axis=1)):
            raise ValueError("Target cannot overlap an initial agent")
        if np.any(distances(evaders, target[None, :]) > config.max_steps):
            raise ValueError("Every evader must be able to reach the target within max_steps")
        return positions, target

    for _ in range(MAX_LAYOUT_ATTEMPTS):
        candidate_target = (target if target is not None
                            else cells[int(rng.integers(len(cells)))])
        target_distance = distances(cells, candidate_target[None, :])[:, 0]
        evader_cells = cells[(target_distance > 0) & (target_distance <= config.max_steps)]
        if len(evader_cells) < config.num_evaders:
            if target is not None:
                raise ValueError("Not enough evader starts within max_steps of the target")
            continue
        evaders = evader_cells[rng.choice(len(evader_cells), size=config.num_evaders, replace=False)]
        pursuer_cells = cells[(target_distance > 0)
                             & np.all(distances(cells, evaders) > 1, axis=1)]
        if len(pursuer_cells) < config.num_pursuers:
            continue
        pursuers = pursuer_cells[rng.choice(len(pursuer_cells), size=config.num_pursuers, replace=False)]
        return np.concatenate((evaders, pursuers)), candidate_target.copy()

    raise ValueError(
        f"Could not sample a valid layout after {MAX_LAYOUT_ATTEMPTS} attempts; "
        "reduce agent counts, increase size/max_steps, or supply a valid fixed layout"
    )
