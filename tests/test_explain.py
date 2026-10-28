import json

import numpy as np
import pytest

from nba import policy as pol
from nba import simulate as sim
from nba.explain import Explanation, explain_recommendation, feature_contributions


@pytest.fixture(scope="module")
def decision(population, fitted_t):
    return pol.ConstrainedPolicy("t", pol.uplift_scorer(fitted_t), pol.Constraints(), 0).decide(population)


def test_explanation_structure(population, fitted_t, decision):
    cid = int(population["customer_id"][np.flatnonzero(decision.actions != sim.NO_ACTION)[0]])
    exp = explain_recommendation(population, cid, fitted_t, decision, top_k=3)
    assert isinstance(exp, Explanation)
    assert exp.action in sim.ACTIONS and exp.action != "no_action"
    assert exp.cost == sim.COSTS[sim.ACTIONS.index(exp.action)]
    assert len(exp.top_features) == 3
    contributions = [abs(f["contribution"]) for f in exp.top_features]
    assert contributions == sorted(contributions, reverse=True)
    assert {f["feature"] for f in exp.top_features} <= set(sim.FEATURES)
    assert exp.binding_constraint in {"none", "budget_fallback"}
    assert exp.expected_net_value >= 0.0


def test_text_mentions_action_and_numbers(population, fitted_t, decision):
    cid = int(population["customer_id"][np.flatnonzero(decision.actions != sim.NO_ACTION)[0]])
    exp = explain_recommendation(population, cid, fitted_t, decision)
    text = exp.to_text()
    assert exp.action in text and f"Customer {cid}" in text
    assert f"{exp.expected_net_value:.2f}" in text and "Binding constraint" in text
    assert chr(0x2014) not in text


def test_no_action_explanation_names_the_reason(population, fitted_t):
    blocked = pol.ConstrainedPolicy("t", pol.uplift_scorer(fitted_t), pol.Constraints(min_net_value=1e9), 0)
    exp = explain_recommendation(population, 0, fitted_t, blocked.decide(population))
    assert exp.action == "no_action" and exp.binding_constraint == "below_threshold"
    assert "no action" in exp.to_text() and "threshold" in exp.to_text()
    assert exp.alternatives[0]["action"] != "no_action"


def test_to_dict_is_json_serializable_and_contributions_sum_sensibly(population, fitted_t, decision):
    exp = explain_recommendation(population, 5, fitted_t, decision)
    payload = json.loads(json.dumps(exp.to_dict()))
    assert payload["customer_id"] == 5
    everything = feature_contributions(fitted_t, population, 5, 1, top_k=len(sim.FEATURES))
    assert len(everything) == len(sim.FEATURES)
    assert all(np.isfinite(f["contribution"]) for f in everything)
