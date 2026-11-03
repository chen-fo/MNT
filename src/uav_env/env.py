"""A partially observed, simultaneous UAV pursuit and evacuation environment."""

from copy import deepcopy
from dataclasses import asdict
from numbers import Integral

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from .config import UAVConfig
from .layout import sample_layout


ACTIVE, EVACUATED, CAPTURED, TIMED_OUT = range(4)
STATUS_NAMES = ("active", "evacuated", "captured", "timed_out")
# Absolute world directions: stay, up, right, down, left. y increases downward.
DELTAS = np.asarray([(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)], dtype=np.int32)
REWARD_KEYS = ("target", "failure", "capture")


class UAVEnv(ParallelEnv):
    metadata = {"name": "uav_pursuit_v0", "render_modes": ["human", "ansi"],
                "is_parallelizable": True}

    def __init__(self, config=None, render_mode=None, **kwargs):
        if config is not None and kwargs:
            raise ValueError("Pass either UAVConfig or keyword task parameters")
        self.config = config if config is not None else UAVConfig(**kwargs)
        if not isinstance(self.config, UAVConfig):
            raise TypeError("config must be UAVConfig")
        if render_mode not in (None, "human", "ansi"):
            raise ValueError("render_mode must be None, 'human' or 'ansi'")
        self.render_mode = render_mode
        c = self.config
        self.possible_agents = ([f"evader_{i}" for i in range(c.num_evaders)]
                                + [f"pursuer_{i}" for i in range(c.num_pursuers)])
        self._indices = {name: i for i, name in enumerate(self.possible_agents)}
        self.agents = []
        count = len(self.possible_agents)
        self.teams = np.asarray([0] * c.num_evaders + [1] * c.num_pursuers, dtype=np.int32)
        self.action_spaces = {name: spaces.Discrete(5) for name in self.possible_agents}
        width = 2 * c.vision_radius + 1
        self_low = np.asarray([0, 0, -(c.size - 1), -(c.size - 1), 0, 0], dtype=np.float32)
        self_high = np.asarray([c.size - 1] * 4 + [c.max_steps, 1], dtype=np.float32)
        self.observation_spaces = {}
        for name, i in self._indices.items():
            friends, opponents = ((c.num_evaders, c.num_pursuers) if self.teams[i] == 0
                                  else (c.num_pursuers, c.num_evaders))
            high = np.broadcast_to(np.asarray([1, friends - 1, opponents, 1], dtype=np.float32)
                                   [:, None, None], (4, width, width)).copy()
            self.observation_spaces[name] = spaces.Dict({
                "local_map": spaces.Box(np.zeros_like(high), high, dtype=np.float32),
                "self": spaces.Box(self_low.copy(), self_high.copy(), dtype=np.float32),
            })
        # target_x, target_y, steps, then [team, x, y, status] in fixed ID order.
        state_high = np.asarray([c.size - 1, c.size - 1, c.max_steps]
                                + [1, c.size - 1, c.size - 1, TIMED_OUT] * count, dtype=np.float32)
        self.state_space = spaces.Box(np.zeros_like(state_high), state_high, dtype=np.float32)
        self.positions = np.zeros((count, 2), dtype=np.int32)
        self.previous_positions = self.positions.copy()
        self.statuses = np.full(count, ACTIVE, dtype=np.int32)
        self.target = np.zeros(2, dtype=np.int32)
        self.steps = 0
        self.np_random = None
        self._initialized = False
        self.episode_done = False
        self.end_reason = None
        self._events = []
        self._agent_events = {name: [] for name in self.possible_agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    @property
    def active(self):
        """Physical participation; pursuers can remain active at episode end."""
        return self.statuses == ACTIVE

    def _require_reset(self):
        if not self._initialized:
            raise RuntimeError("Call reset() before using the environment")

    def _in_bounds(self, point):
        return bool(np.all(point >= 0) and np.all(point < self.config.size))

    def reset(self, seed=None, options=None):
        self._initialized = False
        self.agents = []
        if seed is not None or self.np_random is None:
            self.np_random, actual_seed = seeding.np_random(seed)
            children = np.random.SeedSequence(actual_seed).spawn(2 * len(self.possible_agents))
            for i, name in enumerate(self.possible_agents):
                self.action_space(name).seed(int(children[2 * i].generate_state(1)[0]))
                self.observation_space(name).seed(int(children[2 * i + 1].generate_state(1)[0]))
        positions, target = sample_layout(self.config, self.np_random,
                                          {} if options is None else options)
        self.positions[:] = positions
        self.previous_positions[:] = positions
        self.target[:] = target
        self.statuses.fill(ACTIVE)
        self.steps = 0
        self.episode_done = False
        self.end_reason = None
        self._events = []
        self._agent_events = {name: [] for name in self.possible_agents}
        self.agents = self.possible_agents.copy()
        self._initialized = True
        observations = {name: self.observe(name) for name in self.agents}
        infos = {name: self._info(name) for name in self.agents}
        if self.render_mode == "human":
            self.render()
        return observations, infos

    def step(self, actions):
        self._require_reset()
        if self.episode_done:
            raise RuntimeError("Episode has ended; call reset() before step()")
        if not isinstance(actions, dict) or set(actions) != set(self.agents):
            raise ValueError("Submit exactly one action for every current env.agents entry")
        # Normalize legitimate integer scalars, validating everything before mutation.
        normalized = {}
        for name, action in actions.items():
            if isinstance(action, np.ndarray):
                if action.shape != () or action.dtype.kind not in "iu":
                    raise ValueError(f"Invalid action for {name}: {action!r}")
                action = action.item()
            if (isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral)
                    or not self.action_space(name).contains(action)):
                raise ValueError(f"Invalid action for {name}: {action!r}")
            normalized[name] = int(action)

        acting = self.agents.copy()
        self.steps += 1
        self.previous_positions[:] = self.positions
        self._events = []
        self._agent_events = {name: [] for name in self.possible_agents}
        components = {name: {key: 0.0 for key in REWARD_KEYS} for name in acting}

        # Every endpoint depends only on its own previous position, not update order.
        for name in acting:
            i = self._indices[name]
            point = self.previous_positions[i] + DELTAS[normalized[name]]
            if self._in_bounds(point):
                self.positions[i] = point
            else:
                self._agent_events[name].append("blocked_boundary")

        # The goal is a safe exit even when a pursuer occupies or neighbors it.
        for i in np.flatnonzero(self.active & (self.teams == 0)):
            if np.array_equal(self.positions[i], self.target):
                name = self.possible_agents[i]
                self.statuses[i] = EVACUATED
                components[name]["target"] = float(self.config.target_reward)
                self._agent_events[name].append("evacuated")
                self._events.append({"kind": "evacuated", "agent": name})

        pursuers = np.flatnonzero(self.active & (self.teams == 1))
        for i in np.flatnonzero(self.active & (self.teams == 0)):
            distances = np.abs(self.positions[pursuers] - self.positions[i]).sum(axis=1)
            captors = pursuers[distances <= 1]
            if not len(captors):
                continue
            victim = self.possible_agents[i]
            self.statuses[i] = CAPTURED
            components[victim]["failure"] = -float(self.config.failure_penalty)
            self._agent_events[victim].append("captured")
            share = float(self.config.capture_reward) / len(captors)
            for j in captors:
                name = self.possible_agents[j]
                components[name]["capture"] += share
                self._agent_events[name].append("capture")
            self._events.append({"kind": "capture", "victim": victim,
                                 "captors": [self.possible_agents[j] for j in captors]})

        # Resolve arrival/capture on the last step before failing remaining evaders.
        remaining = np.flatnonzero(self.active & (self.teams == 0))
        if not len(remaining):
            self.end_reason = "all_evaders_resolved"
        elif self.steps >= self.config.max_steps:
            for i in remaining:
                name = self.possible_agents[i]
                self.statuses[i] = TIMED_OUT
                components[name]["failure"] = -float(self.config.failure_penalty)
                self._agent_events[name].append("timeout")
                self._events.append({"kind": "timeout", "agent": name})
            self.end_reason = "deadline"
        self.episode_done = self.end_reason is not None
        observations = {name: self.observe(name) for name in acting}
        rewards = {name: float(sum(components[name].values())) for name in acting}
        terminations = {name: bool(self.episode_done or not self.active[self._indices[name]])
                        for name in acting}
        # The deadline defines task failure, rather than an external rollout cutoff.
        truncations = {name: False for name in acting}
        infos = {name: self._info(name, components[name]) for name in acting}
        self.agents = [name for name in acting if not terminations[name]]
        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def observe(self, agent):
        self._require_reset()
        i = self._indices[agent]
        radius = self.config.vision_radius
        width = 2 * radius + 1
        local_map = np.zeros((4, width, width), dtype=np.float32)
        own = np.zeros(6, dtype=np.float32)
        own[4:] = [self.steps, self.teams[i]]
        if not self.active[i]:
            return {"local_map": local_map, "self": own}

        own[:2] = self.positions[i]
        own[2:4] = self.target - self.positions[i]
        x, y = self.positions[i]
        xs = np.arange(x - radius, x + radius + 1)
        ys = np.arange(y - radius, y + radius + 1)
        local_map[0] = ((ys[:, None] < 0) | (ys[:, None] >= self.config.size)
                        | (xs[None, :] < 0) | (xs[None, :] >= self.config.size))
        for j in np.flatnonzero(self.active):
            if j == i:
                continue
            dx, dy = self.positions[j] - self.positions[i]
            if max(abs(int(dx)), abs(int(dy))) <= radius:
                channel = 1 if self.teams[j] == self.teams[i] else 2
                local_map[channel, radius + dy, radius + dx] += 1
        dx, dy = self.target - self.positions[i]
        if max(abs(int(dx)), abs(int(dy))) <= radius:
            local_map[3, radius + dy, radius + dx] = 1
        return {"local_map": local_map, "self": own}

    def _info(self, agent, components=None):
        return {"status": STATUS_NAMES[self.statuses[self._indices[agent]]],
                "events": self._agent_events[agent].copy(), "steps": self.steps,
                "episode_done": self.episode_done,
                "reward_components": dict(components or {key: 0.0 for key in REWARD_KEYS})}

    def state(self):
        self._require_reset()
        units = np.column_stack((self.teams, self.positions, self.statuses))
        return np.concatenate((self.target, [self.steps], units.ravel())).astype(np.float32)

    def get_debug_info(self):
        self._require_reset()
        return {"config": asdict(self.config), "agent_order": self.possible_agents.copy(),
                "positions": self.positions.copy(), "previous_positions": self.previous_positions.copy(),
                "statuses": self.statuses.copy(), "active": self.active.copy(),
                "target": self.target.copy(), "steps": self.steps, "end_reason": self.end_reason,
                "events": deepcopy(self._events)}

    def render(self):
        self._require_reset()
        if self.render_mode is None:
            return None
        lines = [f"step {self.steps}/{self.config.max_steps}  E=evader P=pursuer T=target"]
        cells = {}
        for i in np.flatnonzero(self.active):
            counts = cells.setdefault(tuple(self.positions[i]), [0, 0])
            counts[self.teams[i]] += 1
        for y in range(self.config.size):
            row = []
            for x in range(self.config.size):
                token = "T" if np.array_equal(self.target, [x, y]) else "."
                evaders, pursuers = cells.get((x, y), (0, 0))
                if evaders:
                    token += f"E{evaders}"
                if pursuers:
                    token += f"P{pursuers}"
                row.append(f"{token:>6}")
            lines.append("".join(row))
        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
            return None
        return output

    def close(self):
        pass


def parallel_env(**kwargs):
    return UAVEnv(**kwargs)
