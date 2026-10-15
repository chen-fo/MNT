"""Single-match bookkeeping shared by windowed and headless viewers."""
from collections import deque


class MatchSession:
    def __init__(self, env, controllers, seed=0):
        self.env, self.controllers, self.seed = env, controllers, seed
        self.episode = 0
        self.reset()

    @property
    def finished(self):
        return not self.env.agents

    def reset(self):
        episode_seed = self.seed + self.episode
        self.observations, self.infos = self.env.reset(seed=episode_seed)
        for i, controller in enumerate(self.controllers):
            controller.reset(episode_seed * 2 + i)
        self.episode += 1
        self.actions = {}
        self.returns = dict.fromkeys(self.env.possible_agents, 0.0)
        self.rewards = self.returns.copy()
        self.events = deque(maxlen=5)

    def step(self):
        if self.finished:
            return
        self.actions = {}
        for controller in self.controllers:
            self.actions.update(controller.act(self.observations, self.env.agents))
        obs, rewards, _, _, infos = self.env.step(self.actions)
        self.observations = obs
        self.rewards = dict.fromkeys(self.env.possible_agents, 0.0)
        self.rewards.update(rewards)
        self.infos.update(infos)
        for agent, reward in rewards.items():
            self.returns[agent] += float(reward)
        for agent, info in infos.items():
            for event in info["events"]:
                if event != "fired":
                    self.events.append(f"{self.env.steps}: {agent} {event}")

    def result(self):
        return {"episode": self.episode, "seed": self.seed + self.episode - 1,
                "steps": self.env.steps, "end_reason": self.env.end_reason,
                "returns": self.returns.copy(),
                "agents": {a: {"status": v["status"], "events": v["events"]}
                           for a, v in self.infos.items()}}
