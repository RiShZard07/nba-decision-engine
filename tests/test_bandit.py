import numpy as np
import pytest

from nba import bandit as bd


@pytest.fixture(scope="module")
def env():
    return bd.linear_environment(n=1500, dim=5, n_actions=4, seed=11, noise_std=0.3)


def _uniform_regret(env, rounds, seed):
    return bd.simulate_online(bd.UniformRandom(env.n_actions, seed), env, rounds, seed).final_regret


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_linucb_beats_uniform_and_is_sublinear(env, seed):
    res = bd.simulate_online(bd.LinUCB(env.n_actions, env.dim, alpha=1.0, seed=seed), env, 3000, seed)
    assert res.final_regret < 0.25 * _uniform_regret(env, 3000, seed)
    assert bd.regret_summary(res.cumulative_regret)["sublinear"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_thompson_beats_uniform_and_is_sublinear(env, seed):
    player = bd.LinearThompsonSampling(env.n_actions, env.dim, v=0.5, seed=seed)
    res = bd.simulate_online(player, env, 3000, seed)
    assert res.final_regret < 0.25 * _uniform_regret(env, 3000, seed)
    assert bd.regret_summary(res.cumulative_regret)["sublinear"]


def test_oracle_has_zero_regret(env):
    res = bd.simulate_online(bd.Oracle(env), env, 500, seed=0)
    assert res.final_regret == 0.0
    assert res.action_counts.sum() == 500


def test_simulation_is_deterministic(env):
    a = bd.simulate_online(bd.LinUCB(env.n_actions, env.dim, seed=3), env, 400, seed=9)
    b = bd.simulate_online(bd.LinUCB(env.n_actions, env.dim, seed=3), env, 400, seed=9)
    assert np.array_equal(a.cumulative_regret, b.cumulative_regret)
    assert np.array_equal(a.cumulative_reward, b.cumulative_reward)


def test_update_moves_estimate_toward_reward():
    player = bd.LinUCB(n_actions=2, dim=3, alpha=0.1, seed=0)
    x = np.array([1.0, 0.0, 0.0])
    before = player.theta()[0, 0]
    for _ in range(20):
        player.update(x, 0, 1.0)
    after = player.theta()[0, 0]
    assert before == 0.0 and 0.9 < after <= 1.0
    assert player.select(x) == 0


def test_select_returns_valid_actions(env):
    player = bd.LinearThompsonSampling(env.n_actions, env.dim, seed=0)
    actions = {player.select(env.contexts[i]) for i in range(50)}
    assert actions <= set(range(env.n_actions))


def test_regret_summary_distinguishes_linear_from_sqrt():
    t = np.arange(1, 3001, dtype=float)
    assert not bd.regret_summary(t)["sublinear"]
    assert bd.regret_summary(t)["ratio"] == pytest.approx(1.0)
    assert bd.regret_summary(np.sqrt(t))["sublinear"]
    assert bd.regret_summary(np.sqrt(t))["ratio"] == pytest.approx(0.32, abs=0.02)


def test_customer_environment_shapes(population):
    env = bd.customer_environment(population)
    assert env.contexts.shape == (len(population), 14)
    assert env.expected.shape == (len(population), 5)
    rng = np.random.default_rng(0)
    rewards = [env.sample_reward(0, a, rng) for a in range(5)]
    assert all(np.isfinite(rewards)) and max(rewards) <= 3.5
    assert env.regret(0, int(env.expected[0].argmax())) == 0.0
