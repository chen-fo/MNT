"""Deterministic UAV pursuit rules, local sensing, and transition safety."""

import pickle
import random

import numpy as np
import pytest

from uav_env import UAVEnv
from uav_env.env import ACTIVE, CAPTURED, EVACUATED, TIMED_OUT


def arena(evaders, pursuers, target=(10, 10), **kwargs):
    kwargs.setdefault("size", 12)
    env = UAVEnv(num_evaders=len(evaders), num_pursuers=len(pursuers), **kwargs)
    env.reset(seed=71, options={"positions": evaders + pursuers, "target": target})
    return env


def actions(env, **overrides):
    return {agent: overrides.get(agent, 0) for agent in env.agents}


@pytest.mark.parametrize("action,point", [(0, [5, 5]), (1, [5, 4]),
                                         (2, [6, 5]), (3, [5, 6]), (4, [4, 5])])
def test_absolute_cardinal_moves_and_stay_for_both_teams(action, point):
    env = arena([[5, 5]], [[1, 1]])
    _, rewards, terminated, _, _ = env.step(actions(env, evader_0=action, pursuer_0=action))
    assert env.positions[0].tolist() == point
    assert env.positions[1].tolist() == [point[0] - 4, point[1] - 4]
    assert rewards == {"evader_0": 0.0, "pursuer_0": 0.0}
    assert not any(terminated.values())


def test_boundary_blocks_both_teams_without_failure():
    env = arena([[0, 0]], [[11, 11]])
    _, rewards, terminated, _, info = env.step(actions(env, evader_0=4, pursuer_0=2))
    assert env.positions.tolist() == [[0, 0], [11, 11]]
    assert env.statuses.tolist() == [ACTIVE, ACTIVE]
    assert all(r == 0 for r in rewards.values()) and not any(terminated.values())
    assert info["evader_0"]["events"] == ["blocked_boundary"]
    assert info["pursuer_0"]["events"] == ["blocked_boundary"]


@pytest.mark.parametrize("pursuer,action", [([5, 3], 3), ([7, 5], 4),
                                           ([5, 7], 1), ([3, 5], 2)])
def test_each_of_four_neighbor_cells_captures(pursuer, action):
    env = arena([[5, 5]], [pursuer])
    _, rewards, terminated, truncated, info = env.step(actions(env, pursuer_0=action))
    assert rewards == {"evader_0": -10.0, "pursuer_0": 10.0}
    assert all(terminated.values()) and not any(truncated.values())
    assert env.statuses.tolist() == [CAPTURED, ACTIVE]
    assert info["evader_0"]["events"] == ["captured"]
    assert env.agents == []


def test_same_cell_captures_but_diagonal_neighbor_does_not():
    env = arena([[4, 5]], [[6, 5]])
    _, rewards, _, _, _ = env.step(actions(env, evader_0=2, pursuer_0=4))
    assert env.positions.tolist() == [[5, 5], [5, 5]]
    assert rewards == {"evader_0": -10.0, "pursuer_0": 10.0}
    env = arena([[5, 5]], [[7, 6]])
    _, rewards, terminated, _, _ = env.step(actions(env, pursuer_0=4))
    assert env.positions[1].tolist() == [6, 6]
    assert all(r == 0 for r in rewards.values()) and not any(terminated.values())


def test_capture_uses_both_units_after_simultaneous_movement():
    env = arena([[5, 5]], [[5, 3]])
    _, rewards, terminated, _, _ = env.step(actions(env, evader_0=3, pursuer_0=3))
    assert env.positions.tolist() == [[5, 6], [5, 4]]
    assert all(r == 0 for r in rewards.values()) and not any(terminated.values())


def test_many_to_many_capture_counts_each_victim_once_and_splits_credit():
    env = arena([[4, 4], [6, 4]], [[5, 2], [5, 6], [10, 10]], target=(10, 8))
    _, rewards, _, _, info = env.step(actions(
        env, evader_0=2, evader_1=4, pursuer_0=3, pursuer_1=1))
    assert rewards == {"evader_0": -10.0, "evader_1": -10.0,
                       "pursuer_0": 10.0, "pursuer_1": 10.0, "pursuer_2": 0.0}
    assert info["evader_0"]["events"] == ["captured"]
    assert info["evader_1"]["events"] == ["captured"]
    assert info["pursuer_0"]["reward_components"]["capture"] == 10.0


def test_capture_reward_and_failure_penalty_are_independently_configurable():
    env = arena([[5, 5]], [[5, 3], [7, 5]], capture_reward=6.0, failure_penalty=9.0)
    _, rewards, _, _, _ = env.step(actions(env, pursuer_0=3, pursuer_1=4))
    assert rewards == {"evader_0": -9.0, "pursuer_0": 3.0, "pursuer_1": 3.0}


def test_same_team_units_may_overlap_without_dying():
    env = arena([[4, 5], [6, 5]], [[9, 8], [9, 10]])
    _, rewards, terminated, _, _ = env.step(actions(
        env, evader_0=2, evader_1=4, pursuer_0=3, pursuer_1=1))
    assert env.positions.tolist() == [[5, 5], [5, 5], [9, 9], [9, 9]]
    assert np.all(env.statuses == ACTIVE)
    assert all(r == 0 for r in rewards.values()) and not any(terminated.values())


def test_arrival_wins_over_simultaneous_same_cell_capture():
    env = arena([[4, 5]], [[6, 5]], target=(5, 5))
    _, rewards, terminated, _, info = env.step(actions(env, evader_0=2, pursuer_0=4))
    assert env.positions.tolist() == [[5, 5], [5, 5]]
    assert env.statuses.tolist() == [EVACUATED, ACTIVE]
    assert rewards == {"evader_0": 10.0, "pursuer_0": 0.0}
    assert info["evader_0"]["events"] == ["evacuated"]
    assert all(terminated.values())


def test_arrival_wins_over_a_pursuer_already_guarding_a_neighbor():
    env = arena([[4, 5]], [[5, 4]], target=(5, 5))
    _, rewards, _, _, _ = env.step(actions(env, evader_0=2))
    assert env.statuses[0] == EVACUATED
    assert rewards == {"evader_0": 10.0, "pursuer_0": 0.0}


def test_multiple_evaders_can_arrive_together_before_capture():
    env = arena([[4, 5], [6, 5]], [[5, 3]], target=(5, 5))
    _, rewards, _, _, _ = env.step(actions(env, evader_0=2, evader_1=4, pursuer_0=3))
    assert env.statuses.tolist() == [EVACUATED, EVACUATED, ACTIVE]
    assert rewards == {"evader_0": 10.0, "evader_1": 10.0, "pursuer_0": 0.0}


@pytest.mark.parametrize("exit_status", ["evacuated", "captured"])
def test_individual_exit_does_not_end_teammate_or_repeat_rewards(exit_status):
    env = (arena([[4, 5], [2, 2]], [[9, 9]], target=(5, 5)) if exit_status == "evacuated"
           else arena([[5, 5], [2, 2]], [[7, 5]]))
    first_actions = actions(env, **({"evader_0": 2} if exit_status == "evacuated"
                                   else {"pursuer_0": 4}))
    obs, rewards, terminated, _, info = env.step(first_actions)
    assert terminated["evader_0"] and not terminated["evader_1"]
    assert rewards["evader_1"] == 0.0 and not env.episode_done
    assert info["evader_0"]["status"] == exit_status
    assert "evader_0" not in env.agents
    assert not obs["evader_0"]["local_map"].any()
    np.testing.assert_array_equal(obs["evader_0"]["self"], [0, 0, 0, 0, 1, 0])
    obs, rewards, _, _, info = env.step(actions(env))
    assert "evader_0" not in obs and "evader_0" not in rewards and "evader_0" not in info


@pytest.mark.parametrize("arrive", [False, True])
def test_exact_deadline_and_last_step_arrival(arrive):
    env = arena([[4, 5]], [[9, 9]], target=(5, 5), max_steps=2)
    env.step(actions(env))
    assert env.agents
    _, rewards, terminated, truncated, info = env.step(actions(env, evader_0=2 if arrive else 0))
    assert env.steps == 2 and not env.agents
    assert all(terminated.values()) and not any(truncated.values())
    assert env.statuses[0] == (EVACUATED if arrive else TIMED_OUT)
    assert rewards == {"evader_0": 10.0 if arrive else -10.0, "pursuer_0": 0.0}
    assert info["evader_0"]["events"] == ["evacuated" if arrive else "timeout"]
    with pytest.raises(RuntimeError):
        env.step({})


def test_last_step_capture_is_not_also_timeout():
    env = arena([[5, 5]], [[7, 5]], target=(5, 6), max_steps=1)
    _, rewards, _, _, info = env.step(actions(env, pursuer_0=4))
    assert env.statuses[0] == CAPTURED
    assert info["evader_0"]["events"] == ["captured"]
    assert rewards == {"evader_0": -10.0, "pursuer_0": 10.0}


def test_success_reward_has_no_environment_time_discount():
    env = arena([[4, 5]], [[9, 9]], target=(5, 5), target_reward=17.0)
    for _ in range(10):
        env.step(actions(env))
    _, rewards, _, _, _ = env.step(actions(env, evader_0=2))
    assert rewards["evader_0"] == 17.0


def test_local_map_channels_axes_and_self_features():
    env = arena([[5, 5], [5, 6]], [[7, 5]], target=(6, 4), vision_radius=2)
    obs = env.observe("evader_0")
    assert obs["local_map"].shape == (4, 5, 5)
    np.testing.assert_array_equal(obs["self"], [5, 5, 1, -1, 0, 0])
    assert obs["local_map"][1, 3, 2] == 1  # Southward friend.
    assert obs["local_map"][2, 2, 4] == 1  # Eastward pursuer, exactly at radius.
    assert obs["local_map"][3, 1, 3] == 1  # Northeast target.
    assert obs["local_map"].sum() == 3  # Self is not an extra friendly unit.
    other = env.observe("pursuer_0")
    assert other["self"][-1] == 1
    assert other["local_map"][1].sum() == 0
    assert other["local_map"][2].sum() == 2  # Role-relative opponents.


def test_square_vision_includes_diagonal_and_has_no_unit_occlusion():
    env = arena([[5, 5]], [[7, 7], [7, 5], [8, 5]], vision_radius=3)
    grid = env.observe("evader_0")["local_map"]
    assert grid[2, 5, 5] == 1  # Diagonal within square radius.
    assert grid[2, 3, 5] == grid[2, 3, 6] == 1  # Near unit does not occlude far unit.


def test_invisible_positions_do_not_leak_into_observation_or_info():
    first = arena([[5, 5]], [[8, 5]], vision_radius=2)
    second = arena([[5, 5]], [[10, 8]], vision_radius=2)
    for key in ("local_map", "self"):
        np.testing.assert_array_equal(first.observe("evader_0")[key], second.observe("evader_0")[key])
    _, _, _, _, info1 = first.step(actions(first))
    _, _, _, _, info2 = second.step(actions(second))
    assert info1["evader_0"] == info2["evader_0"]
    assert set(info1["evader_0"]) == {"status", "events", "reward_components", "steps", "episode_done"}


def test_local_map_keeps_counts_for_overlapping_units():
    env = arena([[5, 5], [4, 6], [5, 7]], [[7, 4], [8, 5]], vision_radius=2)
    env.step(actions(env, evader_1=2, evader_2=1, pursuer_0=3, pursuer_1=4))
    grid = env.observe("evader_0")["local_map"]
    assert grid[1, 3, 2] == 2
    assert grid[2, 2, 4] == 2
    assert env.observation_space("evader_0").contains(env.observe("evader_0"))


def test_view_near_boundary_and_goal_outside_view():
    env = arena([[0, 0]], [[6, 6]], size=8, target=(5, 5), vision_radius=2)
    obs = env.observe("evader_0")
    assert np.all(obs["local_map"][0, :2, :] == 1)
    assert np.all(obs["local_map"][0, :, :2] == 1)
    assert np.all(obs["local_map"][0, 2:, 2:] == 0)
    assert not obs["local_map"][3].any()
    np.testing.assert_array_equal(obs["self"][2:4], [5, 5])


def test_zero_vision_radius_still_has_own_navigation_features():
    env = arena([[5, 5]], [[9, 9]], vision_radius=0)
    obs = env.observe("evader_0")
    assert obs["local_map"].shape == (4, 1, 1) and not obs["local_map"].any()
    np.testing.assert_array_equal(obs["self"], [5, 5, 5, 5, 0, 0])


def test_state_observation_and_debug_are_copies():
    env = arena([[5, 5]], [[9, 9]])
    before = env.state()
    np.testing.assert_array_equal(before, [10, 10, 0, 0, 5, 5, ACTIVE, 1, 9, 9, ACTIVE])
    state = env.state()
    assert env.state_space.contains(state)
    state.fill(-123)
    obs = env.observe("evader_0")
    obs["self"].fill(-123)
    obs["local_map"].fill(-123)
    debug = env.get_debug_info()
    debug["positions"].fill(-123)
    debug["target"].fill(-123)
    debug["active"].fill(False)
    np.testing.assert_array_equal(env.state(), before)
    assert env.observation_space("evader_0").contains(env.observe("evader_0"))


def test_rng_is_local_and_reproducible_when_environments_interleave():
    random.seed(314)
    np.random.seed(271)
    before = pickle.dumps((random.getstate(), np.random.get_state()))
    env = UAVEnv()
    env.reset(seed=18)
    first = env.state()
    env.reset()
    second = env.state()
    env.reset(seed=18)
    np.testing.assert_array_equal(env.state(), first)
    other = UAVEnv()
    other.reset(seed=999)
    env.reset()
    np.testing.assert_array_equal(env.state(), second)
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before


@pytest.mark.parametrize("bad", [True, np.bool_(False), 2.9, -1, 5, "2", np.array([2]),
                                  np.array(2.0), np.array(True)])
def test_bad_actions_rejected_without_mutation(bad):
    env = arena([[5, 5]], [[9, 9]])
    before = env.state()
    # Put the valid pursuer first, to detect validation that starts moving too early.
    with pytest.raises(ValueError):
        env.step({"pursuer_0": 1, "evader_0": bad})
    np.testing.assert_array_equal(env.state(), before)


def test_numpy_integer_scalar_arrays_are_accepted():
    env = arena([[5, 5]], [[9, 9]])
    env.step({"evader_0": np.array(2, dtype=np.int64), "pursuer_0": np.int32(4)})
    assert env.positions.tolist() == [[6, 5], [8, 9]]


def test_reset_required_and_exact_action_keys_required():
    env = UAVEnv()
    with pytest.raises(RuntimeError):
        env.step({})
    with pytest.raises(RuntimeError):
        env.state()
    env.reset(seed=2)
    before = env.state()
    for bad in ({"evader_0": 0}, dict(actions(env), unknown=0), []):
        with pytest.raises(ValueError):
            env.step(bad)
        np.testing.assert_array_equal(env.state(), before)


def test_action_dictionary_order_does_not_affect_outcome():
    first = arena([[5, 5]], [[5, 3], [7, 5]])
    second = arena([[5, 5]], [[5, 3], [7, 5]])
    a = actions(first, pursuer_0=3, pursuer_1=4)
    result1 = first.step(a)
    result2 = second.step(dict(reversed(list(a.items()))))
    assert result1[1:] == result2[1:]
    np.testing.assert_array_equal(first.state(), second.state())


def test_ansi_render_is_available():
    env = arena([[5, 5]], [[9, 9]], render_mode="ansi")
    assert isinstance(env.render(), str) and env.render()
    env.close()
