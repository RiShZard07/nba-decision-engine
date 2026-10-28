import numpy as np
import pytest

from nba import ope
from nba import policy as pol
from nba import simulate as sim


def _oracle_probs(df):
    return pol.ConstrainedPolicy("oracle", pol.oracle_scorer, pol.Constraints(), 0).action_probs(df)


def _run(seed, n=5000):
    df = sim.generate_customers(n, seed)
    log = sim.simulate_logged(df, seed + 50)
    probs = _oracle_probs(df)
    q_hat = ope.fit_outcome_model(sim.feature_matrix(log), log["action"].to_numpy(),
                                  log["reward"].to_numpy(), sim.N_ACTIONS, seed=seed)
    results = ope.evaluate_policy(log, probs, q_hat, n_boot=100, seed=seed)
    return results, ope.true_policy_value(df, probs)


def test_ips_on_logging_policy_reproduces_observed_mean(population, logged):
    probs = sim.logging_policy_probs(population)
    results = ope.evaluate_policy(logged, probs, n_boot=20)
    observed = logged["reward"].mean()
    assert results["ips"].estimate == pytest.approx(observed, abs=1e-9)
    assert results["snips"].estimate == pytest.approx(observed, abs=1e-9)
    assert results["ips"].effective_sample_size == pytest.approx(len(logged))
    check = ope.logging_policy_check(logged, probs)
    assert check["ips_abs_error"] == pytest.approx(0.0, abs=1e-9)
    assert check["snips_abs_error"] == pytest.approx(0.0, abs=1e-9)


def test_importance_weights_are_clipped(population, logged):
    probs = _oracle_probs(population)
    actions, prop = logged["action"].to_numpy(), logged["propensity"].to_numpy()
    raw = ope.importance_weights(probs, actions, prop, clip=np.inf)
    assert raw.max() > 5.0
    assert ope.importance_weights(probs, actions, prop, clip=5.0).max() <= 5.0
    assert set(np.unique(np.round(raw[raw > 0], 6))) <= set(np.round(1.0 / np.unique(prop), 6))


def test_dr_has_lower_error_than_ips_across_seeds():
    ips_err, dr_err = [], []
    for seed in (21, 22, 23):
        results, truth = _run(seed)
        ips_err.append(abs(results["ips"].estimate - truth))
        dr_err.append(abs(results["dr"].estimate - truth))
    assert np.mean(dr_err) < np.mean(ips_err)


def test_bootstrap_ci_contains_truth_at_a_reasonable_rate():
    hits = 0
    for seed in range(31, 37):
        results, truth = _run(seed, n=4000)
        hits += results["dr"].contains(truth)
    assert hits >= 4


def test_dm_and_dr_require_outcome_model(population, logged):
    probs = _oracle_probs(population)
    without = ope.evaluate_policy(logged, probs, n_boot=20)
    assert set(without) == {"ips", "snips"}
    q_hat = ope.fit_outcome_model(sim.feature_matrix(logged), logged["action"].to_numpy(),
                                  logged["reward"].to_numpy(), sim.N_ACTIONS)
    with_q = ope.evaluate_policy(logged, probs, q_hat, n_boot=20)
    assert set(with_q) == set(ope.ESTIMATORS)
    for r in with_q.values():
        assert r.ci_low <= r.estimate <= r.ci_high
        assert set(r.to_dict()) == {"estimator", "estimate", "ci_low", "ci_high", "effective_sample_size"}


def test_outcome_model_is_cross_fitted_and_calibrated(logged):
    q_hat = ope.fit_outcome_model(sim.feature_matrix(logged), logged["action"].to_numpy(),
                                  logged["reward"].to_numpy(), sim.N_ACTIONS, n_folds=3)
    assert q_hat.shape == (len(logged), sim.N_ACTIONS) and np.isfinite(q_hat).all()
    fitted = q_hat[np.arange(len(logged)), logged["action"].to_numpy()]
    assert abs(fitted.mean() - logged["reward"].mean()) < 2.0


def test_true_policy_value_matches_policy_value(population):
    actions = sim.oracle_actions(population)
    probs = np.eye(sim.N_ACTIONS)[actions]
    assert ope.true_policy_value(population, probs) == pytest.approx(sim.policy_value(population, actions))
