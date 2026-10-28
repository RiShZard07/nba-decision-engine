"""Offline policy evaluation on logged bandit data.

Notation: for logged row i we have context x_i, action a_i, logging
propensity p_i = P_log(a_i | x_i) and reward r_i. The target policy gives
pi(a | x_i). With w_i = pi(a_i | x_i) / p_i:

    IPS    V = mean(w_i r_i)
    SNIPS  V = sum(w_i r_i) / sum(w_i)
    DM     V = mean(sum_a pi(a | x_i) q(x_i, a))
    DR     V = mean(sum_a pi(a | x_i) q(x_i, a) + w_i (r_i - q(x_i, a_i)))

where q is a regression model of reward on (x, a), fitted with cross-fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from nba import simulate as sim
from nba.uplift import default_regressor

ESTIMATORS = ("ips", "snips", "dm", "dr")


@dataclass(frozen=True)
class OPEResult:
    estimator: str
    estimate: float
    ci_low: float
    ci_high: float
    effective_sample_size: float

    def contains(self, value: float) -> bool:
        return self.ci_low <= value <= self.ci_high

    def to_dict(self) -> dict:
        return {k: round(float(v), 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


def importance_weights(target_probs: np.ndarray, actions: np.ndarray,
                       propensities: np.ndarray, clip: float = 20.0) -> np.ndarray:
    """w_i = pi(a_i | x_i) / p_i, clipped from above to limit variance."""
    w = target_probs[np.arange(len(actions)), actions] / propensities
    return np.minimum(w, clip)


def fit_outcome_model(X: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                      n_actions: int, seed: int = 0, n_folds: int = 2) -> np.ndarray:
    """Cross-fitted q(x, a) for every action, shape (n, n_actions).

    Each row's predictions come from models that never saw that row, which
    keeps the doubly robust correction term honest.
    """
    q_hat = np.zeros((len(X), n_actions))
    for train, held in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        for a in range(n_actions):
            rows = train[actions[train] == a]
            model = default_regressor(seed).fit(X[rows], rewards[rows])
            q_hat[held, a] = model.predict(X[held])
    return q_hat


def _bootstrap(terms: np.ndarray, n_boot: int, seed: int,
               denominator: np.ndarray | None = None) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(terms), size=(n_boot, len(terms)))
    num = terms[idx].sum(axis=1)
    den = len(terms) if denominator is None else denominator[idx].sum(axis=1)
    stats = num / den
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def evaluate_policy(logged: pd.DataFrame, target_probs: np.ndarray,
                    q_hat: np.ndarray | None = None, clip: float = 20.0,
                    n_boot: int = 200, seed: int = 0) -> dict[str, OPEResult]:
    """Point estimate and 95 percent bootstrap CI for each estimator."""
    actions = logged["action"].to_numpy(int)
    rewards = logged["reward"].to_numpy(float)
    w = importance_weights(target_probs, actions, logged["propensity"].to_numpy(float), clip)
    ess = float(w.sum() ** 2 / np.maximum((w**2).sum(), 1e-12))
    ips_terms = w * rewards
    out = {
        "ips": OPEResult("ips", float(ips_terms.mean()), *_bootstrap(ips_terms, n_boot, seed), ess),
        "snips": OPEResult("snips", float(ips_terms.sum() / w.sum()),
                           *_bootstrap(ips_terms, n_boot, seed, denominator=w), ess),
    }
    if q_hat is not None:
        dm_terms = (target_probs * q_hat).sum(axis=1)
        dr_terms = dm_terms + w * (rewards - q_hat[np.arange(len(actions)), actions])
        out["dm"] = OPEResult("dm", float(dm_terms.mean()), *_bootstrap(dm_terms, n_boot, seed), ess)
        out["dr"] = OPEResult("dr", float(dr_terms.mean()), *_bootstrap(dr_terms, n_boot, seed), ess)
    return out


def logging_policy_check(logged: pd.DataFrame, logging_probs: np.ndarray,
                         q_hat: np.ndarray | None = None) -> dict[str, float]:
    """Evaluating the logging policy on its own log must give the observed mean."""
    results = evaluate_policy(logged, logging_probs, q_hat=q_hat, n_boot=20)
    observed = float(logged["reward"].mean())
    out = {"observed_mean_reward": observed}
    out.update({f"{k}_estimate": r.estimate for k, r in results.items()})
    out.update({f"{k}_abs_error": abs(r.estimate - observed) for k, r in results.items()})
    return out


def true_policy_value(df: pd.DataFrame, target_probs: np.ndarray) -> float:
    """Ground truth from the simulator: expected reward under the target policy."""
    return float((sim.expected_rewards(df) * target_probs).sum(axis=1).mean())
