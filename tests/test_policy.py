import numpy as np
import pytest

from nba import policy as pol
from nba import simulate as sim

LABELS = {"none", "budget", "budget_fallback", "below_threshold", "ineligible", "contact_cap"}


def _oracle_policy(constraints, seed=0):
    return pol.ConstrainedPolicy("oracle", pol.oracle_scorer, constraints, seed)


def test_budget_cap_never_exceeded(population):
    budget = {1: 50, 2: 120, 3: 0, 4: 30}
    decision = _oracle_policy(pol.Constraints(budget=budget)).decide(population)
    counts = decision.action_counts(sim.N_ACTIONS)
    for a, cap in budget.items():
        assert counts[a] <= cap
    assert counts[1] == 50 and counts[3] == 0
    assert decision.budget_used == {1: 50, 2: 120, 3: 0, 4: 30}


def test_contact_cap_honored(population):
    decision = _oracle_policy(pol.Constraints(contact_cap=1)).decide(population)
    capped = population["offers_last_90d"].to_numpy() >= 1
    assert (decision.actions[capped] == sim.NO_ACTION).all()
    assert (decision.actions[~capped] != sim.NO_ACTION).any()


def test_ineligible_customers_never_actioned(population):
    decision = _oracle_policy(pol.Constraints()).decide(population)
    for rule in pol.default_eligibility():
        taken = decision.actions == rule.action
        assert rule.predicate(population)[taken].all()
    assert (decision.binding == "ineligible").sum() > 0


def test_no_action_default_below_threshold(population):
    decision = _oracle_policy(pol.Constraints(min_net_value=1e9)).decide(population)
    assert (decision.actions == sim.NO_ACTION).all()
    assert set(decision.binding) == {"below_threshold"}
    mild = _oracle_policy(pol.Constraints(min_net_value=0.0)).decide(population)
    chosen = mild.net_values[np.arange(len(population)), mild.actions]
    assert (chosen[mild.actions != sim.NO_ACTION] >= 0.0).all()


def test_decisions_are_deterministic_and_one_hot(population):
    constraints = pol.Constraints(budget={1: 100, 2: 100, 3: 100, 4: 100})
    a = _oracle_policy(constraints, seed=4).decide(population)
    b = _oracle_policy(constraints, seed=4).decide(population)
    assert np.array_equal(a.actions, b.actions)
    probs = _oracle_policy(constraints, seed=4).action_probs(population)
    assert probs.shape == (len(population), sim.N_ACTIONS)
    assert np.allclose(probs.sum(axis=1), 1.0) and set(np.unique(probs)) == {0.0, 1.0}


def test_binding_labels_and_budget_fallback(population):
    decision = _oracle_policy(pol.Constraints(budget={2: 0})).decide(population)
    assert set(decision.binding) <= LABELS
    fallback = decision.binding == "budget_fallback"
    assert fallback.sum() > 0
    assert (decision.actions[fallback] != 2).all()
    taken = decision.net_values[fallback, decision.actions[fallback]]
    assert (decision.net_values[fallback, 2] > taken).all() and (taken >= 0).all()


def test_check_constraints_detects_violations(population):
    constraints = pol.Constraints(budget={1: 10})
    decision = _oracle_policy(constraints).decide(population)
    assert pol.check_constraints(decision, population, constraints)["budget_ok"]
    tampered = pol.Decision(decision.actions.copy(), decision.net_values, decision.binding, decision.budget_used)
    tampered.actions[:] = 1
    audit = pol.check_constraints(tampered, population, constraints)
    assert not audit["budget_ok"] and not audit["eligibility_ok"] and not audit["contact_cap_ok"]


def test_oracle_policy_beats_logging_under_constraints(population):
    constraints = pol.Constraints(budget=pol.budget_from_fractions(len(population), {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}))
    oracle_value = sim.policy_value(population, _oracle_policy(constraints).actions(population))
    logging_value = (sim.expected_rewards(population) * pol.logging_policy().action_probs(population)).sum(axis=1).mean()
    assert oracle_value > logging_value
    assert constraints.budget == {1: 600, 2: 600, 3: 600, 4: 600}


def test_protected_looking_features_are_rejected(population):
    pol.assert_no_protected_features()
    with pytest.raises(ValueError, match="protected-looking"):
        pol.assert_no_protected_features(("tenure_months", "customer_age"))
    extra = population.assign(gender="x")  # extra columns are ignored by construction
    decision = _oracle_policy(pol.Constraints()).decide(extra)
    assert np.array_equal(decision.actions, _oracle_policy(pol.Constraints()).actions(population))


def test_uplift_scorer_uses_margin_and_cost(population, fitted_t):
    net = pol.uplift_scorer(fitted_t)(population)
    uplift = fitted_t.predict_uplift(sim.feature_matrix(population))
    expected = uplift * sim.quarterly_margin(population)[:, None] - sim.COSTS[None, :]
    assert np.allclose(net[:, 1:], expected[:, 1:]) and np.all(net[:, 0] == 0)
    with pytest.raises(KeyError):
        pol.uplift_scorer(fitted_t)(population.drop(columns=["monthly_spend"]))
