"""Task rules verified with deterministic layouts, without any training code."""

import random
import pickle

import numpy as np
import pytest

from mnt_env import MntEnv
from mnt_env.env import ACTIVE, DEAD, EVACUATED, TIMED_OUT


def arena(allies, enemies, headings=None, target=(10, 10), mines=(), **kwargs):
    # Rule fixtures deliberately use a long range; production defaults are tested separately.
    kwargs.setdefault("shoot_range", 8)
    env = MntEnv(num_allies=len(allies), num_enemies=len(enemies), num_mines=0, size=12, **kwargs)
    options = {"positions": allies + enemies, "target": target, "mines": list(mines),
               "headings": headings if headings is not None else [0] * (len(allies) + len(enemies))}
    env.reset(seed=71, options=options)
    return env


def actions(env, **overrides):
    return {a: overrides.get(a, 2 if a.startswith("ally_") else 6) for a in env.agents}


def test_many_to_many_contact_rewards_are_individual():
    env = arena([[4, 5], [6, 5], [5, 6]], [[5, 5], [5, 5]], headings=[2, 6, 0, 0, 0])
    _, rewards, done, truncated, info = env.step(actions(env))
    assert rewards == {"ally_0": -10., "ally_1": -10., "ally_2": -10., "enemy_0": 15., "enemy_1": 15.}
    assert all(done.values()) and not any(truncated.values())
    assert env.statuses.tolist() == [DEAD, DEAD, DEAD, ACTIVE, ACTIVE]
    assert env.agents == []
    assert info["enemy_0"]["reward_components"]["collision_kill"] == 15.


def test_only_same_cell_counts_as_collision():
    env = arena([[1, 3], [2, 3], [3, 3]], [[9, 9]])
    _, rewards, _, _, _ = env.step(actions(env))
    assert all(env.alive) and all(r == 0 for r in rewards.values())
    # Swapping cells is legal under endpoint-only collision semantics.
    env = arena([[4, 5], [5, 5]], [[9, 9]], headings=[2, 6, 0])
    env.step(actions(env))
    assert all(env.alive)


def test_ally_collision_has_no_enemy_credit():
    env = arena([[4, 5], [6, 5]], [[9, 9]], headings=[2, 6, 0])
    _, rewards, _, _, _ = env.step(actions(env))
    assert rewards == {"ally_0": -10., "ally_1": -10., "enemy_0": 0.}


def test_enemies_can_stack_and_stand_on_mines():
    env = arena([[1, 8]], [[4, 5], [6, 5]], headings=[0, 2, 6], mines=[(5, 5)])
    _, rewards, _, _, _ = env.step(actions(env, enemy_0=2, enemy_1=2))
    np.testing.assert_array_equal(env.positions[1:], [[5, 5], [5, 5]])
    assert all(env.alive) and all(r == 0 for r in rewards.values())


def test_projectile_passes_mines_and_kills_only_first_ally():
    env = arena([[6, 5], [8, 5]], [[2, 4]], headings=[0, 0, 2], mines=[(4, 4)])
    _, rewards, done, _, _ = env.step(actions(env))
    assert rewards == {"ally_0": -10., "ally_1": 0., "enemy_0": 10.}
    assert done["ally_0"] and not done["ally_1"]
    assert "ally_0" not in env.agents
    _, next_rewards, _, _, _ = env.step(actions(env))
    assert "ally_0" not in next_rewards  # No repeated penalties or dead actions.


def test_enemy_tank_blocks_projectiles_without_taking_damage():
    env = arena([[6, 5]], [[2, 4], [4, 4]], headings=[0, 2, 0])
    _, rewards, _, _, _ = env.step(actions(env))
    assert all(env.alive) and all(r == 0 for r in rewards.values())
    assert env.get_debug_info()["events"][0]["blockers"] == ["enemy_1"]


@pytest.mark.parametrize("swap", [False, True])
def test_simultaneous_shooters_split_one_kill_without_order_bias(swap):
    enemies = [[2, 4], [6, 8]]
    headings = [2, 0]
    if swap:
        enemies.reverse()
        headings.reverse()
    env = arena([[6, 5]], enemies, headings=[0] + headings)
    _, rewards, _, _, _ = env.step(actions(env))
    assert rewards == {"ally_0": -10., "enemy_0": 5., "enemy_1": 5.}


def test_all_rays_see_same_snapshot_and_stacked_shooters_can_fire():
    env = arena([[6, 5], [8, 5]], [[2, 4], [2, 4]], headings=[0, 0, 2, 2])
    _, rewards, _, _, _ = env.step(actions(env))
    assert rewards == {"ally_0": -10., "ally_1": 0., "enemy_0": 5., "enemy_1": 5.}
    assert env.statuses[1] == ACTIVE


@pytest.mark.parametrize("fire,final_point", [(5, [6, 0]), (6, [6, 4]), (7, [6, 8])])
def test_three_fire_directions_do_not_turn_or_move_tank(fire, final_point):
    initial = [final_point[0], final_point[1] + 1]
    env = arena([initial], [[2, 4]], headings=[0, 2], shoot_range=4)
    _, rewards, _, _, _ = env.step(actions(env, enemy_0=fire))
    assert rewards["enemy_0"] == 10.
    assert env.positions[1].tolist() == [2, 4] and env.headings[1] == 2


def test_shoot_range_boundary_and_movement_before_shooting():
    env = arena([[7, 5]], [[2, 4]], headings=[0, 2], shoot_range=4)
    env.step(actions(env))
    assert env.alive[0]  # Five cells away after movement.
    env = arena([[6, 4]], [[2, 4]], headings=[0, 2])
    env.step(actions(env))
    assert env.alive[0]  # Moved off the firing ray this frame.


@pytest.mark.parametrize("fire", [5, 6, 7])
@pytest.mark.parametrize("distance,killed", [(3, True), (4, False)])
def test_default_three_cell_range_includes_endpoint_and_excludes_fourth_cell(fire, distance, killed):
    env = MntEnv(size=12, num_allies=1, num_enemies=1, num_mines=0)
    assert env.config.shoot_range == 3
    assert env.config.sonar_range == 8
    point = [2 + distance, 4 + (fire - 6) * distance]
    env.reset(seed=71, options={"positions": [[point[0], point[1] + 1], [2, 4]],
                               "headings": [0, 2], "target": [10, 10], "mines": []})
    _, rewards, _, _, _ = env.step({"ally_0": 2, "enemy_0": fire})
    assert bool(env.alive[0]) is not killed
    assert rewards["ally_0"] == (-10. if killed else 0.)
    assert rewards["enemy_0"] == (10. if killed else 0.)
    env.close()


def test_evacuate_before_shooting_and_others_continue():
    env = arena([[6, 5], [9, 8]], [[2, 4]], headings=[0, 0, 2], target=(6, 4), gamma=1.)
    _, rewards, done, _, info = env.step(actions(env))
    assert rewards == {"ally_0": 50., "ally_1": 0., "enemy_0": 0.}
    assert env.statuses[0] == EVACUATED
    assert done["ally_0"] and not done["ally_1"] and not env.episode_done
    assert info["ally_0"]["status"] == "evacuated"


def test_collision_on_target_prevents_evacuation():
    env = arena([[4, 5]], [[6, 5]], headings=[2, 6], target=(5, 5))
    _, rewards, _, _, _ = env.step(actions(env, enemy_0=2))
    assert rewards == {"ally_0": -10., "enemy_0": 10.}


def test_collision_beats_mine_and_does_not_double_penalize():
    env = arena([[4, 5]], [[5, 5]], headings=[2, 0], mines=[(5, 5)])
    _, rewards, _, _, info = env.step(actions(env))
    assert rewards == {"ally_0": -10., "enemy_0": 10.}
    assert info["ally_0"]["events"] == ["enemy_collision"]


@pytest.mark.parametrize("cause", ["out_of_bounds", "mine", "shot", "timeout"])
def test_all_failure_causes_use_same_penalty(cause):
    common = dict(failure_penalty=7.)
    if cause == "out_of_bounds":
        env = arena([[0, 0]], [[8, 8]], **common)
    elif cause == "mine":
        env = arena([[5, 5]], [[8, 8]], mines=[(5, 4)], **common)
    elif cause == "shot":
        env = arena([[6, 5]], [[2, 4]], headings=[0, 2], **common)
    else:
        env = arena([[5, 5]], [[8, 8]], max_steps=1, **common)
    _, rewards, _, truncated, info = env.step(actions(env))
    assert rewards["ally_0"] == -7.
    assert info["ally_0"]["events"] == [cause]
    assert not any(truncated.values())


def test_deadline_is_exactly_100_and_last_step_arrival_succeeds():
    for escape in (False, True):
        env = arena([[5, 5]], [[1, 1]], headings=[0, 2 if escape else 0],
                    target=(5, 1) if escape else (10, 10), gamma=1.)
        assert env.config.max_steps == 100
        for step in range(1, 101):
            action = 4 if step <= 96 else 2  # 24 squares, then four northward steps.
            _, rewards, done, truncated, _ = env.step(actions(env, ally_0=action))
            assert bool(env.agents) == (step < 100)
        assert env.steps == 100 and all(done.values()) and not any(truncated.values())
        assert rewards["ally_0"] == (50. if escape else -10.)
        assert rewards["enemy_0"] == 0.
        assert env.statuses[0] == (EVACUATED if escape else TIMED_OUT)
        with pytest.raises(RuntimeError, match="ended"):
            env.step({})


def test_enemy_boundary_is_blocked_without_failure():
    env = arena([[5, 5]], [[0, 0]], headings=[0, 0])
    _, rewards, _, _, _ = env.step(actions(env, enemy_0=2))
    assert env.positions[1].tolist() == [0, 0] and env.alive[1]
    assert rewards["enemy_0"] == 0.


def test_sonar_is_type_blind_occluded_and_limited_to_eight_cells():
    def observation(mines=(), enemies=((1, 1),), allies=((10, 10),)):
        env = MntEnv(size=22, num_allies=len(allies), num_enemies=len(enemies), num_mines=0)
        obs, _ = env.reset(seed=2, options={"positions": list(allies) + list(enemies),
            "headings": [0] * (len(allies) + len(enemies)), "mines": list(mines), "target": [18, 18]})
        return obs["ally_0"]
    mine = observation(mines=[(10, 7)])
    enemy = observation(enemies=[(10, 7)])
    ally = observation(allies=[(10, 10), (10, 7)])
    hidden_enemy = observation(mines=[(10, 7)], enemies=[(10, 5)])
    for obs in (enemy, ally, hidden_enemy):
        np.testing.assert_array_equal(obs, mine)
    assert mine.shape == (9,) and mine[5] == pytest.approx(1 / 3)
    assert observation(mines=[(10, 2)])[5] == .125
    assert observation(mines=[(10, 1)])[5] == 0.
    env = arena([[5, 2]], [[9, 9]])
    assert env.observe("ally_0")[5] == pytest.approx(1 / 3)  # Wall, same signal.


def test_info_is_local_and_global_interfaces_return_copies():
    env = arena([[5, 5]], [[9, 9]])
    obs, _, _, _, info = env.step(actions(env))
    assert set(info["ally_0"]) == {"status", "events", "steps", "episode_done", "reward_components"}
    assert env.observation_space("ally_0").contains(obs["ally_0"])
    state = env.state()
    assert env.state_space.contains(state)
    state.fill(-123)
    debug = env.get_debug_info()
    debug["positions"].fill(-123)
    debug["events"].clear()
    assert env.state_space.contains(env.state()) and np.all(env.positions >= 0)


def test_rng_is_local_and_reproducible_when_environments_interleave():
    random.seed(314)
    np.random.seed(271)
    before = pickle.dumps((random.getstate(), np.random.get_state()))
    env = MntEnv()
    env.reset(seed=18)
    first = env.state()
    env.reset()
    second = env.state()
    env.reset(seed=18)
    np.testing.assert_array_equal(env.state(), first)
    other = MntEnv()
    other.reset(seed=999)
    env.reset()
    np.testing.assert_array_equal(env.state(), second)
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before


@pytest.mark.parametrize("bad", [2.9, True, -1, 5, "2", np.array([2])])
def test_invalid_ally_action_is_rejected_without_mutating_state(bad):
    env = arena([[5, 5]], [[9, 9]])
    before = env.state()
    with pytest.raises(ValueError):
        env.step(actions(env, ally_0=bad))
    np.testing.assert_array_equal(env.state(), before)


def test_incomplete_action_dict_and_step_before_reset_are_rejected():
    env = MntEnv()
    with pytest.raises(RuntimeError, match="reset"):
        env.step({})
    env.reset(seed=2)
    with pytest.raises(ValueError, match="every"):
        env.step({"ally_0": 2})


def test_layout_handles_small_maps_empty_mines_and_impossible_layouts():
    for seed in range(50):
        env = MntEnv(size=5, num_allies=1, num_enemies=1, num_mines=0)
        env.reset(seed=seed, options={"mines": []})
        assert env.target.tolist() == [2, 2]
        assert not any(np.array_equal(p, env.target) for p in env.positions)
    with pytest.raises(ValueError, match="target cell"):
        env.reset(options={"positions": [[2, 2], [0, 0]], "mines": []})
    with pytest.raises(ValueError, match="Not enough"):
        MntEnv(size=5, num_allies=25)
    with pytest.raises(ValueError, match="integer"):
        MntEnv(num_allies=2.1)
    with pytest.raises(ValueError, match="headings"):
        env.reset(options={"headings": [-1, 8]})


def test_action_dictionary_order_does_not_affect_outcome():
    first = arena([[6, 5]], [[2, 4], [6, 8]], headings=[0, 2, 0])
    second = arena([[6, 5]], [[2, 4], [6, 8]], headings=[0, 2, 0])
    a = actions(first)
    result1 = first.step(a)
    result2 = second.step(dict(reversed(list(a.items()))))
    assert result1[1:] == result2[1:]
    np.testing.assert_array_equal(first.state(), second.state())
