"""Simultaneous, individually rewarded minefield evacuation and interception."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from numbers import Integral

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from .config import MntConfig
from .layout import sample_layout


ACTIVE, EVACUATED, DEAD, TIMED_OUT = range(4)
STATUS_NAMES = ("active", "evacuated", "dead", "timed_out")
MOVE_OFFSETS = (-2, -1, 0, 1, 2)
FIRE_OFFSETS = (-1, 0, 1)
DELTAS = np.asarray([(0, -1), (1, -1), (1, 0), (1, 1),
                     (0, 1), (-1, 1), (-1, 0), (-1, -1)], dtype=np.int32)
REWARD_KEYS = ("target", "failure", "collision_kill", "shot_kill")


class MntEnv(ParallelEnv):
    metadata = {"name": "mnt_v0", "render_modes": ["human", "ansi"], "is_parallelizable": True}

    def __init__(self, config=None, render_mode=None, **kwargs):
        if config is not None and kwargs:
            raise ValueError("Pass either MntConfig or keyword task parameters")
        self.config = config if config is not None else MntConfig(**kwargs)
        if not isinstance(self.config, MntConfig):
            raise TypeError("config must be MntConfig")
        if render_mode not in (None, "human", "ansi"):
            raise ValueError("render_mode must be None, 'human' or 'ansi'")
        self.render_mode = render_mode
        c = self.config
        self.possible_agents = ([f"ally_{i}" for i in range(c.num_allies)]
                                + [f"enemy_{i}" for i in range(c.num_enemies)])
        self._indices = {agent: i for i, agent in enumerate(self.possible_agents)}
        self.agents = []
        count = len(self.possible_agents)
        self.teams = np.asarray([0] * c.num_allies + [1] * c.num_enemies, dtype=np.int32)
        self.action_spaces = {name: spaces.Discrete(5 if self.teams[i] == 0 else 8)
                              for name, i in self._indices.items()}
        low = np.asarray([0, -(c.size - 1), -(c.size - 1)] + [0] * 5 + [0], dtype=np.float32)
        high = np.asarray([7, c.size - 1, c.size - 1] + [1] * 5 + [c.max_steps], dtype=np.float32)
        self.observation_spaces = {name: spaces.Box(low, high, dtype=np.float32) for name in self.possible_agents}
        # Flattened global state: mine grid, target, clock, then fixed-ID unit rows.
        state_high = ([1] * c.size ** 2 + [c.size - 1, c.size - 1, c.max_steps]
                      + [1, c.size - 1, c.size - 1, 7, TIMED_OUT] * count)
        self.state_space = spaces.Box(np.zeros(len(state_high), dtype=np.float32),
                                      np.asarray(state_high, dtype=np.float32), dtype=np.float32)
        self.positions = np.zeros((count, 2), dtype=np.int32)
        self.previous_positions = self.positions.copy()
        self.headings = np.zeros(count, dtype=np.int32)
        self.statuses = np.full(count, ACTIVE, dtype=np.int32)
        self.mines_map = np.zeros((c.size, c.size), dtype=bool)
        self.target = np.zeros(2, dtype=np.int32)
        self.steps = 0
        self._initialized = False
        self.episode_done = False
        self.end_reason = None
        self.np_random = None
        self._events = []
        self._agent_events = {a: [] for a in self.possible_agents}
        self.path = np.zeros((count, c.max_steps + 1, 2), dtype=np.int32)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    @property
    def alive(self):
        """Units physically in play; surviving enemies may remain at episode end."""
        return self.statuses == ACTIVE

    def reset(self, seed=None, options=None):
        self._initialized = False
        self.agents = []
        if seed is not None or self.np_random is None:
            self.np_random, actual_seed = seeding.np_random(seed)
            children = np.random.SeedSequence(actual_seed).spawn(2 * len(self.possible_agents))
            for i, name in enumerate(self.possible_agents):
                self.action_space(name).seed(int(children[2 * i].generate_state(1)[0]))
                self.observation_space(name).seed(int(children[2 * i + 1].generate_state(1)[0]))
        positions, headings, target, mines = sample_layout(self.config, self.np_random,
                                                          {} if options is None else options)
        self.positions[:] = positions
        self.previous_positions[:] = positions
        self.headings[:] = headings
        self.target[:] = target
        self.mines_map.fill(False)
        self.mines_map[mines[:, 0], mines[:, 1]] = True
        self.statuses.fill(ACTIVE)
        self.steps = 0
        self.path.fill(0)
        self.path[:, 0] = positions
        self.episode_done = False
        self.end_reason = None
        self._events = []
        self._agent_events = {a: [] for a in self.possible_agents}
        self.agents = self.possible_agents.copy()
        self._initialized = True
        observations = {a: self.observe(a) for a in self.agents}
        infos = {a: self._info(a) for a in self.agents}
        if self.render_mode == "human":
            self.render()
        return observations, infos

    def _require_reset(self):
        if not self._initialized:
            raise RuntimeError("Call reset() before using the environment")

    def _in_bounds(self, point):
        return bool(np.all(point >= 0) and np.all(point < self.config.size))

    def _occupancy(self):
        cells = defaultdict(list)
        for index in np.flatnonzero(self.alive):
            cells[tuple(self.positions[index])].append(int(index))
        return dict(cells)

    def step(self, actions):
        self._require_reset()
        if self.episode_done:
            raise RuntimeError("Episode has ended; call reset() before step()")
        if not isinstance(actions, dict) or set(actions) != set(self.agents):
            raise ValueError("Submit exactly one action for every current env.agents entry")
        # Validate every action before mutating any state (no float truncation).
        for name, action in actions.items():
            if (isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral)
                    or not self.action_space(name).contains(action)):
                raise ValueError(f"Invalid action for {name}: {action!r}")
        acting = self.agents.copy()
        self.steps += 1
        self.previous_positions[:] = self.positions
        self._events = []
        self._agent_events = {a: [] for a in self.possible_agents}
        components = {a: {k: 0.0 for k in REWARD_KEYS} for a in acting}

        def fail(index, reason):
            name = self.possible_agents[index]
            self.statuses[index] = TIMED_OUT if reason == "timeout" else DEAD
            components[name]["failure"] = -float(self.config.failure_penalty)
            self._agent_events[name].append(reason)

        # Movement: shooters neither turn nor move. A blocked enemy move may turn.
        for name in acting:
            i, action = self._indices[name], int(actions[name])
            if action >= 5:
                continue
            self.headings[i] = (self.headings[i] + MOVE_OFFSETS[action]) % 8
            point = self.positions[i] + DELTAS[self.headings[i]]
            if self._in_bounds(point):
                self.positions[i] = point
            elif self.teams[i] == 0:
                fail(i, "out_of_bounds")
                self._events.append({"kind": "out_of_bounds", "agent": name})
            else:
                self._agent_events[name].append("blocked_boundary")

        # All co-located allies fail. Enemy contact takes priority over ally contact.
        for cell, indices in self._occupancy().items():
            allies = [i for i in indices if self.teams[i] == 0]
            enemies = [i for i in indices if self.teams[i] == 1]
            if allies and (enemies or len(allies) > 1):
                for i in allies:
                    fail(i, "enemy_collision" if enemies else "ally_collision")
                if enemies:
                    share = len(allies) * self.config.failure_penalty / len(enemies)
                    for i in enemies:
                        name = self.possible_agents[i]
                        components[name]["collision_kill"] += share
                        self._agent_events[name].append("collision_kill")
                self._events.append({"kind": "collision", "cell": list(cell),
                    "victims": [self.possible_agents[i] for i in allies],
                    "attackers": [self.possible_agents[i] for i in enemies]})

        for i in np.flatnonzero(self.alive & (self.teams == 0)):
            if self.mines_map[tuple(self.positions[i])]:
                fail(i, "mine")
                self._events.append({"kind": "mine", "agent": self.possible_agents[i]})

        # Safe arrival removes the ally before any shot is traced, including step 100.
        for i in np.flatnonzero(self.alive & (self.teams == 0)):
            if np.array_equal(self.positions[i], self.target):
                name = self.possible_agents[i]
                self.statuses[i] = EVACUATED
                components[name]["target"] = float(self.config.target_reward * self.config.gamma ** self.steps)
                self._agent_events[name].append("evacuated")
                self._events.append({"kind": "evacuated", "agent": name})

        # Snapshot stays unchanged for all rays: a tank hit by one ray still blocks
        # every other ray in this frame. Mines never block projectiles.
        cells = self._occupancy()
        hits = defaultdict(list)
        for name in acting:
            i, action = self._indices[name], int(actions[name])
            if self.teams[i] != 1 or not self.alive[i] or action < 5:
                continue
            bearing = (self.headings[i] + FIRE_OFFSETS[action - 5]) % 8
            shot = {"kind": "shot", "shooter": name, "bearing": int(bearing),
                    "blockers": [], "victim": None}
            self._agent_events[name].append("fired")
            # A shot starts in the adjacent cell; co-located firing tanks do not
            # block each other's muzzle. Every later occupied cell blocks it.
            for distance in range(1, self.config.shoot_range + 1):
                point = self.positions[i] + DELTAS[bearing] * distance
                if not self._in_bounds(point):
                    break
                occupants = cells.get(tuple(point), [])
                if occupants:
                    shot["blockers"] = [self.possible_agents[j] for j in occupants]
                    for j in occupants:
                        if self.teams[j] == 0:
                            hits[j].append(i)
                            shot["victim"] = self.possible_agents[j]
                    break
            self._events.append(shot)
        for victim, shooters in hits.items():
            fail(victim, "shot")
            share = self.config.failure_penalty / len(shooters)
            for i in shooters:
                name = self.possible_agents[i]
                components[name]["shot_kill"] += share
                self._agent_events[name].append("shot_kill")

        if self.steps >= self.config.max_steps:
            for i in np.flatnonzero(self.alive & (self.teams == 0)):
                fail(i, "timeout")
                self._events.append({"kind": "timeout", "agent": self.possible_agents[i]})
            self.end_reason = "deadline"
        elif not np.any(self.alive & (self.teams == 0)):
            self.end_reason = "no_allies_remaining"
        elif not np.any(self.alive & (self.teams == 1)):
            self.end_reason = "no_enemies_remaining"
        self.episode_done = self.end_reason is not None
        self.path[:, self.steps] = self.positions
        observations = {a: self.observe(a) for a in acting}
        rewards = {a: float(sum(components[a].values())) for a in acting}
        terminations = {a: bool(self.episode_done or not self.alive[self._indices[a]]) for a in acting}
        # The deadline belongs to the task, not an external rollout cutoff.
        truncations = {a: False for a in acting}
        infos = {a: self._info(a, components[a]) for a in acting}
        self.agents = [a for a in acting if not terminations[a]]
        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def observe(self, agent):
        self._require_reset()
        i = self._indices[agent]
        obs = np.zeros(9, dtype=np.float32)
        obs[-1] = self.steps
        if not self.alive[i]:
            return obs
        obs[0] = self.headings[i]
        obs[1:3] = self.target - self.positions[i]
        cells = self._occupancy()
        for ray, offset in enumerate(MOVE_OFFSETS):
            bearing = (self.headings[i] + offset) % 8
            for distance in range(1, self.config.sonar_range + 1):
                point = self.positions[i] + DELTAS[bearing] * distance
                if (not self._in_bounds(point) or self.mines_map[tuple(point)]
                        or tuple(point) in cells):
                    obs[3 + ray] = 1.0 / distance
                    break
        return obs

    def _info(self, agent, components=None):
        # Only this agent's own status and reward events. Global positions,
        # identities of targets/blockers and the mine map are debug-only.
        return {"status": STATUS_NAMES[self.statuses[self._indices[agent]]],
                "events": self._agent_events[agent].copy(), "steps": self.steps,
                "episode_done": self.episode_done,
                "reward_components": dict(components or {k: 0.0 for k in REWARD_KEYS})}

    def state(self):
        self._require_reset()
        units = np.column_stack((self.teams, self.positions, self.headings, self.statuses))
        return np.concatenate((self.mines_map.ravel(), self.target, [self.steps], units.ravel())).astype(np.float32)

    def get_debug_info(self):
        self._require_reset()
        return {"config": asdict(self.config), "agent_order": self.possible_agents.copy(),
                "positions": self.positions.copy(), "previous_positions": self.previous_positions.copy(),
                "headings": self.headings.copy(), "statuses": self.statuses.copy(),
                "alive": self.alive.copy(), "target": self.target.copy(),
                "mines": np.argwhere(self.mines_map).astype(np.int32), "steps": self.steps,
                "end_reason": self.end_reason, "events": deepcopy(self._events)}

    def render(self):
        self._require_reset()
        if self.render_mode is None:
            return None
        cells, lines = self._occupancy(), [f"step {self.steps}/{self.config.max_steps}"]
        for y in range(self.config.size):
            row = []
            for x in range(self.config.size):
                occupants = cells.get((x, y), [])
                token = "T" if np.array_equal(self.target, [x, y]) else "*" if self.mines_map[x, y] else "."
                if occupants:
                    ally_count = sum(self.teams[i] == 0 for i in occupants)
                    enemy_count = len(occupants) - ally_count
                    token += (f"A{ally_count}" if ally_count else "") + (f"E{enemy_count}" if enemy_count else "")
                row.append(f"{token:>5}")
            lines.append("".join(row))
        output = "\n".join(lines)
        if self.render_mode == "human":
            print(output)
            return None
        return output

    def close(self):
        pass


def parallel_env(**kwargs):
    return MntEnv(**kwargs)
