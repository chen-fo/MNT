"""Observation-only controllers, with optional MTAPRO frozen-team integration."""
from dataclasses import fields

import numpy as np

from mnt_env import MntConfig


def checkpoint_config(metadata):
    if metadata.get("environment", {}).get("id") != "mnt":
        raise ValueError("The checkpoint must declare environment.id = mnt")
    config = dict(metadata.get("environment_config", {}))
    if not config:
        raise ValueError("Checkpoint has no environment_config; cannot reproduce its task")
    supported = {f.name for f in fields(MntConfig)}
    for key in ("ally_sonar_mode", "enemy_sonar_mode"):
        if key not in supported and key in config:
            if config.pop(key) != "blind":
                raise ValueError("This MNT branch only supports blind sonar checkpoints")
    return MntConfig(**config)


class RandomController:
    def __init__(self, agents, action_count):
        self.agents, self.action_count = list(agents), action_count

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, observations, active_agents):
        return {a: int(self.rng.integers(self.action_count))
                for a in self.agents if a in active_agents}


class TeamController:
    """No env/state/debug argument: the actors only receive their own observations."""
    def __init__(self, path, *, device="cpu", deterministic=False):
        try:
            import torch
            from mtapro.policies.checkpoints import load_team_policy
        except ImportError as exc:
            raise RuntimeError("Viewing MTAPRO models requires Torch and editable MTAPRO in this environment") from exc
        self.torch = torch
        self.policy, self.metadata = load_team_policy(path, device=device)
        self.config = checkpoint_config(self.metadata)
        self.deterministic = deterministic
        self.memory = None

    def bind(self, env, agents):
        self.agents = list(agents)
        spec = self.policy.spec
        space = env.observation_space(self.agents[0])
        if (spec["team_size"] != len(self.agents)
                or spec["action_count"] != env.action_space(self.agents[0]).n
                or not np.array_equal(spec["observation_low"], space.low)
                or not np.array_equal(spec["observation_high"], space.high)):
            raise ValueError("Checkpoint team size, observation bounds or action space do not match this side")
        self.obs_size = space.shape[0]

    def reset(self, seed):
        self.memory = None

    def act(self, observations, active_agents):
        obs = np.stack([observations.get(a, np.zeros(self.obs_size, dtype=np.float32))
                        for a in self.agents])
        alive = np.asarray([a in active_agents for a in self.agents], dtype=bool)
        available = np.ones((len(self.agents), self.policy.actions), dtype=bool)
        available[~alive, 1:] = False
        actions, self.memory = self.policy.act(
            obs, self.memory, deterministic=self.deterministic, available=available, alive=alive)
        return {a: int(actions[i]) for i, a in enumerate(self.agents) if alive[i]}
