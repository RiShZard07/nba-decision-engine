import numpy as np
import pandas as pd

from nba import simulate as sim


def test_generation_is_deterministic():
    a = sim.generate_customers(500, seed=3)
    b = sim.generate_customers(500, seed=3)
    c = sim.generate_customers(500, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not a["monthly_spend"].equals(c["monthly_spend"])


def test_feature_ranges(population):
    assert population["tenure_months"].between(1, 120).all()
    assert population["monthly_spend"].between(15, 250).all()
    assert set(population["plan_tier"].unique()) <= {0, 1, 2}
    assert population["usage_trend"].between(-1, 1).all()
    assert population["churn_risk_score"].between(0, 1).all()
    assert list(population.columns[1:]) == list(sim.FEATURES)


def test_logged_propensity_matches_policy_probs(population, logged):
    probs = sim.logging_policy_probs(population)
    rows = np.arange(len(logged))
    assert np.allclose(logged["propensity"].to_numpy(), probs[rows, logged["action"].to_numpy()])
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (logged["propensity"] > 0).all()


def test_propensity_bookkeeping_reconstructs_each_arm(logged):
    """E[1(a_i = a) / p_i] must be 1 for every action if propensities are right."""
    w = 1.0 / logged["propensity"].to_numpy()
    for a in range(sim.N_ACTIONS):
        assert abs((w * (logged["action"].to_numpy() == a)).mean() - 1.0) < 0.2


def test_counterfactual_tables_are_consistent(population):
    probs = sim.retention_probabilities(population)
    assert probs.shape == (len(population), sim.N_ACTIONS)
    assert ((probs >= 0.01) & (probs <= 0.99)).all()
    expected = probs * sim.quarterly_margin(population)[:, None] - sim.COSTS[None, :]
    assert np.allclose(expected, sim.expected_rewards(population))
    assert np.all(sim.true_net_values(population)[:, 0] == 0)
    tau = sim.treatment_effects(population)
    assert all((tau[:, a] < 0).any() and (tau[:, a] > 0).any() for a in range(1, sim.N_ACTIONS))


def test_oracle_dominates_other_action_vectors(population):
    oracle = sim.policy_value(population, sim.oracle_actions(population))
    assert oracle >= sim.policy_value(population, sim.heuristic_actions(population))
    assert oracle >= sim.policy_value(population, np.zeros(len(population), int))


def test_logged_reward_is_margin_minus_cost(logged):
    expected = logged["margin"] - sim.COSTS[logged["action"].to_numpy()]
    assert np.allclose(logged["reward"], expected.round(2))
    assert ((logged["retained"] == 1) == (logged["margin"] > 0)).all()


def test_outcome_rate_matches_designed_probabilities(population, logged):
    rows = np.arange(len(logged))
    p = sim.retention_probabilities(population)[rows, logged["action"].to_numpy()]
    assert abs(logged["retained"].mean() - p.mean()) < 0.02
    assert np.corrcoef(population["churn_risk_score"], sim.true_churn_probability(population))[0, 1] > 0.8
