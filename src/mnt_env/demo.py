"""Run random independent policies to inspect the environment; no training."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .env import MntEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON report, e.g. outputs/demo.json")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    env = MntEnv(**config, render_mode="human" if args.render else None)
    records = []
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            observations, infos = env.reset(seed=seed)
            # Separate policy RNGs; no consumption of the environment's RNG.
            streams = np.random.SeedSequence(seed).spawn(len(env.possible_agents))
            rngs = {a: np.random.default_rng(s) for a, s in zip(env.possible_agents, streams)}
            returns = {a: 0.0 for a in env.possible_agents}
            while env.agents:
                actions = {a: int(rngs[a].integers(env.action_space(a).n)) for a in env.agents}
                observations, rewards, terminated, truncated, infos = env.step(actions)
                for agent, reward in rewards.items():
                    returns[agent] += reward
            record = {"seed": seed, "steps": env.steps, "end_reason": env.end_reason,
                      "returns": returns, "statuses": {
                          a: env._info(a)["status"] for a in env.possible_agents}}
            records.append(record)
            print(json.dumps(record, ensure_ascii=False))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"config": asdict(env.config), "episodes": records},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
