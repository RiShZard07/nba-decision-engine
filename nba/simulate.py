"""Synthetic customer population with a known ground-truth response model.

Everything here is synthetic. The treatment effects are designed, not
discovered, which is what makes the simulator useful: it can answer
counterfactual questions ("what would this customer have done under action
3?") that real logged data never can, so estimators and policies can be
scored against the truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ACTIONS: tuple[str, ...] = (
    "no_action",
    "retention_discount",
    "device_upgrade_offer",
    "data_addon_offer",
    "proactive_support_call",
)
NO_ACTION = 0
N_ACTIONS = len(ACTIONS)
COSTS = np.array([0.0, 12.0, 8.0, 4.0, 6.0])
FEATURES: tuple[str, ...] = (
    "tenure_months",
    "monthly_spend",
    "plan_tier",
    "support_contacts_90d",
    "usage_trend",
    "device_age_months",
    "offers_last_90d",
    "churn_risk_score",
)
MARGIN_RATE = 0.45


@dataclass(frozen=True)
class SimulationConfig:
    """Knobs for one synthetic draw."""

    n_customers: int = 20_000
    seed: int = 7
    epsilon: float = 0.3


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_customers(n: int, seed: int) -> pd.DataFrame:
    """Draw a synthetic population with the features listed in FEATURES."""
    rng = np.random.default_rng(seed)
    tier = rng.choice(3, size=n, p=[0.45, 0.35, 0.20])
    tenure = np.clip(rng.gamma(shape=2.0, scale=18.0, size=n), 1, 120).round()
    spend = rng.lognormal(mean=np.log(50.0), sigma=0.45, size=n) * (1 + 0.3 * tier)
    spend = np.clip(spend, 15.0, 250.0).round(2)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "tenure_months": tenure.astype(int),
            "monthly_spend": spend,
            "plan_tier": tier,
            "support_contacts_90d": np.minimum(rng.poisson(0.9, size=n), 8),
            "usage_trend": np.clip(rng.normal(0.0, 0.5, size=n), -1, 1).round(3),
            "device_age_months": rng.integers(0, 49, size=n),
            "offers_last_90d": np.minimum(rng.poisson(0.6, size=n), 4),
        }
    )
    noise = rng.normal(0.0, 0.08, size=n)
    df["churn_risk_score"] = np.clip(true_churn_probability(df) + noise, 0, 1).round(3)
    return df


def true_churn_probability(df: pd.DataFrame) -> np.ndarray:
    """Ground-truth churn probability with no action taken."""
    z = (
        -1.0
        + 0.55 * df["support_contacts_90d"].to_numpy()
        - 1.3 * df["usage_trend"].to_numpy()
        - 0.012 * df["tenure_months"].to_numpy()
        + 0.015 * df["device_age_months"].to_numpy()
        - 0.35 * df["plan_tier"].to_numpy()
    )
    return _sigmoid(z)


def quarterly_margin(df: pd.DataFrame) -> np.ndarray:
    """Gross margin earned over the next quarter if the customer is retained."""
    return df["monthly_spend"].to_numpy() * 3.0 * MARGIN_RATE


def treatment_effects(df: pd.DataFrame) -> np.ndarray:
    """True incremental retention probability per action, shape (n, N_ACTIONS).

    Column 0 (no action) is zero. Each effect is heterogeneous and some
    segments respond negatively on purpose.
    """
    p_churn = true_churn_probability(df)
    age = df["device_age_months"].to_numpy()
    trend = df["usage_trend"].to_numpy()
    tier = df["plan_tier"].to_numpy()
    contacts = df["support_contacts_90d"].to_numpy()
    tau = np.zeros((len(df), N_ACTIONS))
    tau[:, 1] = np.clip(0.45 * p_churn - 0.06, -0.06, 0.36)
    tau[:, 2] = np.clip(0.007 * (age - 18), -0.12, 0.21)
    tau[:, 3] = np.clip(0.16 * trend + 0.04 * (tier < 2) - 0.03, -0.20, 0.20)
    tau[:, 4] = np.clip(0.08 * np.minimum(contacts, 4) - 0.05, -0.05, 0.27)
    return tau


def retention_probabilities(df: pd.DataFrame) -> np.ndarray:
    """P(retained next quarter | action), shape (n, N_ACTIONS)."""
    base = 1.0 - true_churn_probability(df)
    return np.clip(base[:, None] + treatment_effects(df), 0.01, 0.99)


def expected_rewards(df: pd.DataFrame) -> np.ndarray:
    """Expected net reward (margin if retained minus action cost) per action."""
    return retention_probabilities(df) * quarterly_margin(df)[:, None] - COSTS[None, :]


def true_net_values(df: pd.DataFrame) -> np.ndarray:
    """Expected incremental value of each action over no action."""
    return treatment_effects(df) * quarterly_margin(df)[:, None] - COSTS[None, :]


def oracle_actions(df: pd.DataFrame) -> np.ndarray:
    """Best action per customer when the truth is known (no constraints)."""
    return expected_rewards(df).argmax(axis=1)


def policy_value(df: pd.DataFrame, actions: np.ndarray) -> float:
    """True expected reward per customer of a deterministic action vector."""
    rewards = expected_rewards(df)
    return float(rewards[np.arange(len(df)), actions].mean())


def heuristic_actions(df: pd.DataFrame) -> np.ndarray:
    """The rule-based policy the business ran before any model existed.

    The rules point in the right direction but each one is a little too
    broad, so some customers in negatively responding segments get actioned.
    """
    actions = np.full(len(df), NO_ACTION)
    actions[df["usage_trend"].to_numpy() > 0.1] = 3
    actions[df["support_contacts_90d"].to_numpy() >= 1] = 4
    actions[df["device_age_months"].to_numpy() >= 27] = 2
    actions[df["churn_risk_score"].to_numpy() > 0.45] = 1
    return actions


def logging_policy_probs(df: pd.DataFrame, epsilon: float = 0.3) -> np.ndarray:
    """Epsilon-greedy over the heuristic: the propensities that get logged."""
    probs = np.full((len(df), N_ACTIONS), epsilon / N_ACTIONS)
    probs[np.arange(len(df)), heuristic_actions(df)] += 1.0 - epsilon
    return probs


def uniform_policy_probs(df: pd.DataFrame) -> np.ndarray:
    """Fully randomized assignment, used for honest Qini evaluation."""
    return np.full((len(df), N_ACTIONS), 1.0 / N_ACTIONS)


def simulate_logged(
    df: pd.DataFrame, seed: int, probs: np.ndarray | None = None, epsilon: float = 0.3
) -> pd.DataFrame:
    """Run a stochastic policy once and log action, propensity and outcome."""
    rng = np.random.default_rng(seed)
    if probs is None:
        probs = logging_policy_probs(df, epsilon)
    cum = probs.cumsum(axis=1)
    draws = rng.random(len(df))[:, None]
    actions = (draws > cum).sum(axis=1).clip(0, N_ACTIONS - 1)
    rows = np.arange(len(df))
    p_retain = retention_probabilities(df)[rows, actions]
    retained = (rng.random(len(df)) < p_retain).astype(int)
    margin = retained * quarterly_margin(df)
    out = df.copy()
    out["action"] = actions
    out["propensity"] = probs[rows, actions]
    out["retained"] = retained
    out["margin"] = margin.round(2)
    out["reward"] = (out["margin"] - COSTS[actions]).round(2)
    return out


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Model inputs as a float array in FEATURES order."""
    return df.loc[:, list(FEATURES)].to_numpy(dtype=float)
