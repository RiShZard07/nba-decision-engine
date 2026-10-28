"""Policies that turn per-action scores into actions under business constraints.

The model produces a net value matrix (expected uplift times margin minus
cost). Everything else, eligibility, budgets, contact caps and the minimum
value threshold, is applied here, outside the model, so a rule change never
requires retraining.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from nba import simulate as sim

Scorer = Callable[[pd.DataFrame], np.ndarray]
Predicate = Callable[[pd.DataFrame], np.ndarray]

# Names that must never drive a decision. The synthetic data has none of them;
# the check exists so that adding one to FEATURES fails loudly.
PROTECTED_LOOKING: tuple[str, ...] = (
    "age", "customer_age", "gender", "sex", "race", "ethnicity", "religion", "nationality",
    "disability", "zip_code", "postcode", "marital_status",
)


def assert_no_protected_features(features: tuple[str, ...] = sim.FEATURES) -> None:
    """Raise if a model input is, or starts with, a protected-looking name."""
    prefixes = tuple(p + "_" for p in PROTECTED_LOOKING)
    hits = [f for f in features if f.lower() in PROTECTED_LOOKING or f.lower().startswith(prefixes)]
    if hits:
        raise ValueError(f"protected-looking features are not allowed as inputs: {hits}")


@dataclass(frozen=True)
class EligibilityRule:
    """An action may only go to customers for whom `predicate` is True."""

    action: int
    description: str
    predicate: Predicate


def default_eligibility() -> list[EligibilityRule]:
    """Rules a subscription business would plausibly impose."""
    return [
        EligibilityRule(1, "tenure of at least 3 months", lambda d: d["tenure_months"].to_numpy() >= 3),
        EligibilityRule(2, "device at least 12 months old", lambda d: d["device_age_months"].to_numpy() >= 12),
        EligibilityRule(3, "not already on the top plan tier", lambda d: d["plan_tier"].to_numpy() < 2),
        EligibilityRule(4, "at least one recent support contact", lambda d: d["support_contacts_90d"].to_numpy() >= 1),
    ]


@dataclass
class Constraints:
    """Budget caps are absolute counts per action for the decision period."""

    budget: dict[int, int] = field(default_factory=dict)
    contact_cap: int = 2
    min_net_value: float = 0.0
    eligibility: list[EligibilityRule] = field(default_factory=default_eligibility)


@dataclass
class Decision:
    """Output of a constrained policy for a whole population."""

    actions: np.ndarray
    net_values: np.ndarray
    binding: np.ndarray
    budget_used: dict[int, int]

    def action_counts(self, n_actions: int) -> np.ndarray:
        return np.bincount(self.actions, minlength=n_actions)


def uplift_scorer(model, costs: np.ndarray = sim.COSTS) -> Scorer:
    """Net value = predicted uplift times quarterly margin minus action cost."""

    def score(df: pd.DataFrame) -> np.ndarray:
        uplift = model.predict_uplift(sim.feature_matrix(df))
        net = uplift * sim.quarterly_margin(df)[:, None] - costs[None, :]
        net[:, sim.NO_ACTION] = 0.0
        return net

    return score


def oracle_scorer(df: pd.DataFrame) -> np.ndarray:
    """Net value from the simulator's true effects (an upper bound, not a model)."""
    net = sim.true_net_values(df)
    net[:, sim.NO_ACTION] = 0.0
    return net


@dataclass
class ConstrainedPolicy:
    """Greedy allocation of the highest net values subject to the constraints."""

    name: str
    scorer: Scorer
    constraints: Constraints = field(default_factory=Constraints)
    seed: int = 0

    def decide(self, df: pd.DataFrame) -> Decision:
        assert_no_protected_features()
        net = self.scorer(df)
        n, k = net.shape
        eligible = np.ones((n, k), bool)
        for rule in self.constraints.eligibility:
            eligible[:, rule.action] = rule.predicate(df)
        capped = df["offers_last_90d"].to_numpy() >= self.constraints.contact_cap
        candidate = eligible & (net >= self.constraints.min_net_value)
        candidate[:, sim.NO_ACTION] = False
        candidate[capped] = False
        actions, binding = self._allocate(net, candidate)
        self._label_blocked(binding, actions, net, eligible, capped, candidate)
        used = {a: int((actions == a).sum()) for a in range(1, k)}
        return Decision(actions, net, binding, used)

    def _allocate(self, net: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, k = net.shape
        rng = np.random.default_rng(self.seed)
        best = np.where(candidate, net, -np.inf).max(axis=1)
        order = rng.permutation(n)
        order = order[np.argsort(-best[order], kind="stable")]
        remaining = {a: self.constraints.budget.get(a, n) for a in range(1, k)}
        actions = np.full(n, sim.NO_ACTION)
        binding = np.full(n, "none", dtype=object)
        for i in order:
            if not candidate[i].any():
                continue
            ranked = [a for a in np.argsort(-net[i], kind="stable") if candidate[i, a]]
            for a in ranked:
                if remaining[a] > 0:
                    actions[i], remaining[a] = a, remaining[a] - 1
                    if a != ranked[0]:
                        binding[i] = "budget_fallback"
                    break
            else:
                binding[i] = "budget"
        return actions, binding

    def _label_blocked(self, binding, actions, net, eligible, capped, candidate) -> None:
        worth_it = net >= self.constraints.min_net_value
        worth_it[:, sim.NO_ACTION] = False
        idle = (actions == sim.NO_ACTION) & (binding == "none")  # "budget" is already labeled
        binding[idle & ~worth_it.any(axis=1)] = "below_threshold"
        binding[idle & worth_it.any(axis=1) & ~(worth_it & eligible).any(axis=1)] = "ineligible"
        binding[idle & capped & (worth_it & eligible).any(axis=1)] = "contact_cap"

    def actions(self, df: pd.DataFrame) -> np.ndarray:
        return self.decide(df).actions

    def action_probs(self, df: pd.DataFrame) -> np.ndarray:
        """Deterministic policy, so a one-hot matrix of shape (n, n_actions)."""
        return np.eye(sim.N_ACTIONS)[self.actions(df)]


@dataclass
class StochasticPolicy:
    """A policy defined directly by its action probabilities (logging, uniform)."""

    name: str
    probs_fn: Callable[[pd.DataFrame], np.ndarray]

    def action_probs(self, df: pd.DataFrame) -> np.ndarray:
        return self.probs_fn(df)


def logging_policy(epsilon: float = 0.3) -> StochasticPolicy:
    return StochasticPolicy("logging", lambda d: sim.logging_policy_probs(d, epsilon))


def heuristic_policy() -> StochasticPolicy:
    return StochasticPolicy("heuristic_rules", lambda d: np.eye(sim.N_ACTIONS)[sim.heuristic_actions(d)])


def budget_from_fractions(n: int, fractions: dict[int, float]) -> dict[int, int]:
    """Turn 'at most 8 percent of customers get action 1' into absolute caps."""
    return {a: int(np.floor(f * n)) for a, f in fractions.items()}


def check_constraints(decision: Decision, df: pd.DataFrame, constraints: Constraints) -> dict:
    """Independent audit of a decision; used by tests and the demo gate."""
    counts = decision.action_counts(sim.N_ACTIONS)
    budget_ok = all(counts[a] <= cap for a, cap in constraints.budget.items())
    capped = df["offers_last_90d"].to_numpy() >= constraints.contact_cap
    contact_ok = bool((decision.actions[capped] == sim.NO_ACTION).all())
    eligible_ok = True
    for rule in constraints.eligibility:
        taken = decision.actions == rule.action
        eligible_ok &= bool(rule.predicate(df)[taken].all())
    return {"budget_ok": bool(budget_ok), "contact_cap_ok": contact_ok,
            "eligibility_ok": bool(eligible_ok), "counts": counts.tolist()}
