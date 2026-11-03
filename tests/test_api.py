"""Official multi-agent API/seed checks and complete UAV rollouts."""

from functools import partial

import numpy as np
import pytest
from pettingzoo.test import api_test, parallel_api_test, parallel_seed_test
from pettingzoo.utils.conversions import parallel_to_aec

from uav_env import UAVEnv, parallel_env


@pytest.mark.parametrize("evaders,pursuers", [(1, 1), (2, 2), (1, 2), (2, 1)])
def test_official_parallel_api_and_seed(evaders, pursuers):
    factory = partial(parallel_env, num_evaders=evaders, num_pursuers=pursuers)
    parallel_api_test(factory(), num_cycles=150)
    parallel_seed_test(factory, num_cycles=100)


@pytest.mark.parametrize("evaders,pursuers", [(1, 1), (2, 2)])
@pytest.mark.filterwarnings("ignore:Observation is not a NumPy array:UserWarning")
@pytest.mark.filterwarnings("ignore:Observation space for each agent probably should be gymnasium.spaces.box or gymnasium.spaces.discrete:UserWarning")
def test_official_aec_conversion(evaders, pursuers):
    # Dict observations are intentional; the AEC checker recommends arrays but
    # accepts arbitrary Gymnasium spaces and verifies returned observations.
    env = parallel_to_aec(parallel_env(num_evaders=evaders, num_pursuers=pursuers))
    api_test(env, num_cycles=150)


@pytest.mark.parametrize("evaders,pursuers", [(1, 1), (2, 2), (3, 4)])
def test_random_rollouts_remain_in_spaces(evaders, pursuers):
    env = UAVEnv(num_evaders=evaders, num_pursuers=pursuers)
    for seed in range(20):
        observations, info = env.reset(seed=seed)
        assert set(observations) == set(info) == set(env.possible_agents)
        while env.agents:
            prior = env.agents.copy()
            action = {agent: env.action_space(agent).sample() for agent in prior}
            observations, rewards, terminated, truncated, info = env.step(action)
            assert set(observations) == set(rewards) == set(terminated) == set(truncated) == set(info) == set(prior)
            assert env.agents == [agent for agent in prior if not terminated[agent]]
            for agent in prior:
                assert env.observation_space(agent).contains(observations[agent])
                assert isinstance(rewards[agent], float) and np.isfinite(rewards[agent])
                assert not truncated[agent]
            assert env.state_space.contains(env.state())
            assert env.steps <= env.config.max_steps
    env.close()
