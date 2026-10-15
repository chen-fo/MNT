"""Viewer regression tests; no training and no desktop window required."""
from dataclasses import asdict
import numpy as np
import pytest

from mnt_env import MntConfig, MntEnv
from mnt_env.visualization.geometry import shot_segments
from mnt_env.visualization.policies import RandomController, TeamController, checkpoint_config
from mnt_env.visualization.session import MatchSession


def make_session(allies=2, enemies=2):
    env = MntEnv(num_allies=allies, num_enemies=enemies)
    return MatchSession(env, [RandomController([f"ally_{i}" for i in range(allies)], 5),
                              RandomController([f"enemy_{i}" for i in range(enemies)], 8)], seed=19)


@pytest.mark.parametrize("counts", [(1, 1), (2, 2), (3, 3), (1, 3)])
def test_complete_match_and_seeded_reset(counts):
    a, b = make_session(*counts), make_session(*counts)
    while not a.finished:
        a.step()
        b.step()
        np.testing.assert_array_equal(a.env.state(), b.env.state())
        assert a.returns == b.returns
    assert a.result() == b.result()
    assert a.env.steps <= 100
    a.reset()
    assert a.episode == 2 and not a.finished
    assert not any(a.returns.values())


def test_shot_geometry_uses_actual_blocker():
    env = MntEnv(size=8, num_allies=1, num_enemies=2, num_mines=0, shoot_range=5)
    env.reset(seed=0, options={"positions": [[6, 6], [2, 2], [2, 4]],
                               "headings": [0, 4, 0], "target": [7, 7], "mines": []})
    env.step({"ally_0": 2, "enemy_0": 6, "enemy_1": 6})
    segments = shot_segments(env.get_debug_info())
    assert len(segments) == 2
    np.testing.assert_array_equal(segments[0][0], [2, 2])
    np.testing.assert_array_equal(segments[0][1], [2, 4])
    assert not segments[0][2]


def test_config_rejects_typed_sonar():
    with pytest.raises(ValueError, match="blind"):
        checkpoint_config({"environment": {"id": "mnt"},
                           "environment_config": {**asdict(MntConfig()), "ally_sonar_mode": "typed"}})


def test_rendering_is_read_only(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pg = pytest.importorskip("pygame")
    from mnt_env.visualization.renderer import Renderer
    a, b = make_session(), make_session()
    renderer = Renderer(a.env.config.size)
    try:
        while not a.finished:
            a.step()
            renderer.draw(a.env.get_debug_info(), observations=a.observations, infos=a.infos,
                          rewards=a.rewards, returns=a.returns, actions=a.actions, events=a.events,
                          episode=1, selected=3, finished=a.finished)
            b.step()
            np.testing.assert_array_equal(a.env.state(), b.env.state())
            assert a.returns == b.returns
        pg.image.save(renderer.screen, str(tmp_path / "viewer.png"))
    finally:
        renderer.close()


def test_frozen_team_loading_and_memory(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mtapro")
    from mtapro.policies.team import TeamPolicy
    from mtapro.policies.checkpoints import team_policy_payload
    torch.set_num_threads(1)
    env = MntEnv(num_allies=1, num_enemies=3)
    obs, _ = env.reset(seed=2)
    space = env.observation_space("enemy_0")
    spec = {"team_size": 3, "action_count": 8,
            "observation_low": space.low.tolist(), "observation_high": space.high.tolist()}
    path = tmp_path / "opponent.pt"
    model = TeamPolicy(spec, hidden=8)
    torch.save(team_policy_payload(model, {"environment": {"id": "mnt"},
                                          "environment_config": asdict(env.config)}), path)
    controller = TeamController(path)
    controller.bind(env, ["enemy_0", "enemy_1", "enemy_2"])
    controller.reset(0)
    actions = controller.act(obs, env.agents)
    assert set(actions) == {"enemy_0", "enemy_1", "enemy_2"}
    assert all(0 <= a < 8 for a in actions.values())
    assert len(controller.memory) == 3
    before = [p.clone() for p in controller.policy.parameters()]
    # Removed agents retain a fixed model slot but cannot emit actions.
    controller.act({"enemy_2": obs["enemy_2"]}, ["enemy_2"])
    assert all(torch.equal(a, b) for a, b in zip(before, controller.policy.parameters()))
    assert not controller.policy.training
    controller.reset(1)
    assert controller.memory is None
    with pytest.raises(ValueError, match="match"):
        controller.bind(env, ["ally_0"])
