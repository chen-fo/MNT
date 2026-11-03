"""Run independent random policies or a deterministic pursuit scenario."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .env import UAVEnv


def _scenario(name):
    config = {"size": 5, "num_evaders": 1, "num_pursuers": 1,
              "vision_radius": 2, "max_steps": 10}
    options = {"positions": [[1, 2], [3, 2]], "target": [2, 2]}
    actions = {"evader_0": 2, "pursuer_0": 4}
    if name == "capture":
        options["target"] = [4, 4]
    elif name == "timeout":
        config["max_steps"] = 1
        actions = {"evader_0": 0, "pursuer_0": 0}
    return config, options, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="JSON task configuration")
    source.add_argument("--scenario", choices=("evacuation", "capture", "timeout"),
                        help="Fixed 1v1 layout and actions demonstrating the selected outcome")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    if args.scenario:
        config, options, fixed_actions = _scenario(args.scenario)
    else:
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        options, fixed_actions = None, None
    env = UAVEnv(**config, render_mode="human" if args.render else None)
    records = []
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            _, infos = env.reset(seed=seed, options=options)
            streams = np.random.SeedSequence(seed).spawn(len(env.possible_agents))
            rngs = {a: np.random.default_rng(s) for a, s in zip(env.possible_agents, streams)}
            returns = {a: 0.0 for a in env.possible_agents}
            statuses = {a: info["status"] for a, info in infos.items()}
            while env.agents:
                if fixed_actions is None:
                    actions = {a: int(rngs[a].integers(env.action_space(a).n)) for a in env.agents}
                else:
                    actions = {a: fixed_actions[a] for a in env.agents}
                _, rewards, _, _, infos = env.step(actions)
                for agent, reward in rewards.items():
                    returns[agent] += reward
                statuses.update({a: info["status"] for a, info in infos.items()})
            record = {"seed": seed, "steps": env.steps, "end_reason": env.end_reason,
                      "returns": returns, "statuses": statuses}
            records.append(record)
            print(json.dumps(record, ensure_ascii=False))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"config": asdict(env.config),
                "scenario": args.scenario, "episodes": records}, ensure_ascii=False, indent=2),
                encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
