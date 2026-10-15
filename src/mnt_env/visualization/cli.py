"""Local interactive viewing or short headless inference, never training."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from mnt_env import MntConfig, MntEnv
from .policies import RandomController, TeamController
from .session import MatchSession


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="JSON MntConfig; otherwise use checkpoint config or defaults")
    p.add_argument("--subject", type=Path, help="Frozen MTAPRO ally team .pt (otherwise random)")
    p.add_argument("--opponent", type=Path, help="Frozen MTAPRO enemy team .pt (otherwise random)")
    p.add_argument("--device", default="cpu", help="Inference device; CPU is sufficient for local viewing")
    p.add_argument("--deterministic", action="store_true", help="Use maximum-probability actions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", type=float, default=5, help="Environment steps per second, 0.25..60")
    p.add_argument("--paused", action="store_true")
    p.add_argument("--headless", action="store_true", help="No GUI; run --episodes then exit")
    p.add_argument("--episodes", type=int, default=1, help="Number of complete headless episodes")
    p.add_argument("--output", type=Path, help="Optional JSON record of completed matches")
    return p


def make_session(args):
    models = [TeamController(path, device=args.device, deterministic=args.deterministic) if path else None
              for path in (args.subject, args.opponent)]
    if args.config:
        config = MntConfig(**json.loads(args.config.read_text(encoding="utf-8-sig")))
    else:
        config = next((m.config for m in models if m is not None), MntConfig())
    for model in models:
        if model is not None and model.config != config:
            raise ValueError("Environment configs differ between checkpoints or --config")
    if any(models):
        import torch
        torch.manual_seed(args.seed)
        torch.set_num_threads(1)
    env = MntEnv(config=config)
    controllers = []
    for model, prefix, count, actions in zip(models, ("ally", "enemy"),
                                            (config.num_allies, config.num_enemies), (5, 8)):
        names = [f"{prefix}_{i}" for i in range(count)]
        if model is None:
            model = RandomController(names, actions)
        else:
            model.bind(env, names)
        controllers.append(model)
    print(f"MNT {config.num_allies}v{config.num_enemies}, sonar={config.sonar_range}, shots={config.shoot_range}")
    print(f"Subject: {args.subject or 'random'}\nOpponent: {args.opponent or 'random'}")
    print("Viewer reports native environment rewards (private training shaping is excluded).")
    return MatchSession(env, controllers, args.seed)


def watch(session, args, record):
    from .renderer import Renderer
    renderer = Renderer(session.env.config.size)
    pg = renderer.pg
    clock = pg.time.Clock()
    paused, speed, selected = args.paused, args.speed, 0
    last_step = time.monotonic()
    recorded = False
    running = True
    try:
        while running:
            single = False
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                    elif event.key == pg.K_SPACE:
                        paused = not paused
                        last_step = time.monotonic()
                    elif event.key == pg.K_RIGHT:
                        single, paused = True, True
                    elif event.key == pg.K_TAB:
                        selected = (selected+1) % len(session.env.possible_agents)
                    elif event.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                        speed = min(60, speed*2)
                    elif event.key in (pg.K_MINUS, pg.K_KP_MINUS):
                        speed = max(.25, speed/2)
                    elif event.key == pg.K_n:
                        session.reset()
                        recorded = False
                        last_step = time.monotonic()
            if not running:
                break
            now = time.monotonic()
            if single or (not paused and now-last_step >= 1/speed):
                session.step()
                last_step = now
            if session.finished and not recorded:
                record(session.result())
                recorded = True
            renderer.draw(session.env.get_debug_info(), observations=session.observations,
                          infos=session.infos, rewards=session.rewards, returns=session.returns,
                          actions=session.actions, events=session.events, episode=session.episode,
                          selected=selected, paused=paused, speed=speed, finished=session.finished)
            clock.tick(30)
    finally:
        renderer.close()


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.seed < 0 or args.episodes < 1 or not .25 <= args.speed <= 60:
        p.error("Require seed >= 0, episodes >= 1 and speed in [0.25, 60]")
    session = make_session(args)
    results = []

    def record(result):
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"config": asdict(session.env.config),
                                               "subject": str(args.subject) if args.subject else "random",
                                               "opponent": str(args.opponent) if args.opponent else "random",
                                               "deterministic": args.deterministic,
                                               "episodes": results}, indent=2), encoding="utf-8")
    try:
        if args.headless:
            for episode in range(args.episodes):
                if episode:
                    session.reset()
                while not session.finished:
                    session.step()
                record(session.result())
        else:
            watch(session, args, record)
    finally:
        session.env.close()


if __name__ == "__main__":
    main()
