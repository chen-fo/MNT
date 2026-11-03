"""Reset layout validity, bounded failures, and task parameter validation."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from uav_env import UAVConfig, UAVEnv
from uav_env import layout


def assert_safe_start(env):
    positions = env.positions
    assert len(set(map(tuple, positions))) == len(positions)
    assert not any(np.array_equal(position, env.target) for position in positions)
    assert np.all((positions >= 0) & (positions < env.config.size))
    for evader in positions[:env.config.num_evaders]:
        assert np.abs(evader - env.target).sum() <= env.config.max_steps
        for pursuer in positions[env.config.num_evaders:]:
            assert np.abs(evader - pursuer).sum() > 1


@pytest.mark.parametrize("kwargs", [{}, {"size": 3, "num_evaders": 1, "num_pursuers": 1},
                                    {"size": 8, "num_evaders": 1, "num_pursuers": 2, "max_steps": 1}])
def test_random_starts_are_valid_and_have_static_routes(kwargs):
    env = UAVEnv(**kwargs)
    for seed in range(50):
        env.reset(seed=seed)
        assert_safe_start(env)


def test_fixed_and_partial_layouts_and_metadata():
    env = UAVEnv(size=8, num_evaders=1, num_pursuers=1)
    env.reset(seed=4, options={"positions": [[1, 1], [6, 6]], "target": [5, 5], "framework_meta": True})
    assert env.positions.tolist() == [[1, 1], [6, 6]]
    assert env.target.tolist() == [5, 5]
    env.reset(seed=4, options={"positions": [[1, 1], [6, 6]]})
    assert env.positions.tolist() == [[1, 1], [6, 6]]
    assert_safe_start(env)
    env.reset(seed=4, options={"target": [0, 0]})
    assert env.target.tolist() == [0, 0]
    assert_safe_start(env)


@pytest.mark.parametrize("options", [
    {"positions": [[1, 1], [1, 1]]},
    {"positions": [[1, 1], [1, 2]]},
    {"positions": [[1, 1], [6, 6]], "target": [1, 1]},
    {"positions": [[-1, 1], [6, 6]]},
    {"positions": [[1, 1], [8, 6]]},
    {"positions": [[1.0, 1.0], [6.0, 6.0]]},
    {"positions": [[True, False], [False, True]]},
    {"positions": [[True, 1], [6, 6]]},
    {"positions": [[1, 1]]},
    {"target": [1.5, 2.0]},
    {"target": [False, True]},
    {"target": [True, 2]},
    {"target": [8, 0]},
    {"target": [1, 2, 3]},
    {"positions": np.array([[2 ** 64 - 1, 1], [6, 6]], dtype=np.uint64)},
    [],
])
def test_invalid_layout_is_rejected(options):
    env = UAVEnv(size=8, num_evaders=1, num_pursuers=1)
    with pytest.raises(ValueError):
        env.reset(seed=1, options=options)


def test_reject_fixed_layout_beyond_deadline():
    env = UAVEnv(size=8, num_evaders=1, num_pursuers=1, max_steps=1)
    with pytest.raises(ValueError):
        env.reset(options={"positions": [[1, 1], [6, 6]], "target": [5, 5]})


def test_no_free_target_within_deadline_fails_instead_of_looping():
    env = UAVEnv(size=5, num_evaders=2, num_pursuers=1, max_steps=1)
    with pytest.raises(ValueError):
        env.reset(options={"positions": [[0, 0], [4, 4], [2, 2]]})


def test_physically_fitting_but_capture_unsafe_layout_has_bounded_failure(monkeypatch):
    # Nine cells can fit eight agents and the target, but capture exclusion
    # leaves at most five pursuer cells around a corner evader on this map.
    monkeypatch.setattr(layout, "MAX_LAYOUT_ATTEMPTS", 3)
    env = UAVEnv(size=3, num_evaders=1, num_pursuers=7)
    with pytest.raises(ValueError, match="after 3 attempts"):
        env.reset(seed=0)


def test_numpy_config_scalars_are_normalized_without_integer_overflow():
    config = UAVConfig(size=np.uint8(16), num_evaders=np.int8(2),
                       num_pursuers=np.uint8(2), vision_radius=np.uint8(4),
                       max_steps=np.uint8(100), target_reward=np.float32(10),
                       failure_penalty=np.float64(10), capture_reward=np.float32(10))
    data = asdict(config)
    for key in ("size", "num_evaders", "num_pursuers", "vision_radius", "max_steps"):
        assert type(data[key]) is int
    for key in ("target_reward", "failure_penalty", "capture_reward"):
        assert type(data[key]) is float
    assert json.loads(json.dumps(data)) == data
    env = UAVEnv(config=config)
    env.reset(seed=42)
    assert env.state_space.contains(env.state())


@pytest.mark.parametrize("kwargs", [
    {"size": 2}, {"size": True}, {"num_evaders": 0}, {"num_pursuers": 0},
    {"num_evaders": 2.1}, {"vision_radius": -1}, {"vision_radius": True},
    {"max_steps": 0}, {"target_reward": -1}, {"target_reward": np.inf},
    {"failure_penalty": 0}, {"capture_reward": -1}, {"capture_reward": np.nan},
    {"size": 3, "num_evaders": 8, "num_pursuers": 1},
])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        UAVConfig(**kwargs)
