"""Official API/seed checks plus complete multi-agent rollouts."""

import numpy as np
from pettingzoo.test import parallel_api_test, parallel_seed_test

from mnt_env import MntEnv, parallel_env


def test_official_parallel_api():
    parallel_api_test(parallel_env(), num_cycles=150)


def test_official_parallel_seed():
    parallel_seed_test(parallel_env, num_cycles=100)


def test_random_rollouts_stay_in_spaces_and_rewards_stay_individual():
    env = MntEnv(num_allies=3, num_enemies=4)
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        while env.agents:
            prior = env.agents.copy()
            action = {a: env.action_space(a).sample() for a in prior}
            obs, rewards, terminated, truncated, info = env.step(action)
            assert set(obs) == set(rewards) == set(terminated) == set(truncated) == set(info) == set(prior)
            for a in prior:
                assert env.observation_space(a).contains(obs[a])
                assert isinstance(rewards[a], float) and np.isfinite(rewards[a])
                assert not truncated[a]
            assert env.state_space.contains(env.state())
            assert env.steps <= 100
    env.close()
