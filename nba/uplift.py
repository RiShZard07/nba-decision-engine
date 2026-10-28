"""Uplift (heterogeneous treatment effect) estimators built on scikit-learn.

Treatment is an integer array where 0 is the control arm (no action). Every
estimator returns uplift as an array of shape (n, n_actions): column j is the
estimated effect of action j relative to action 0, and column 0 is zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ModelFactory = Callable[[], object]


def default_regressor(seed: int = 0) -> HistGradientBoostingRegressor:
    """Small, well regularized boosted trees; fast enough for 20k rows."""
    return HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.06, max_depth=2, min_samples_leaf=200,
        l2_regularization=1.0, random_state=seed,
    )


def _as_array(X: np.ndarray | pd.DataFrame) -> np.ndarray:
    return X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)


@dataclass
class BaseUplift:
    """Shared plumbing: a model factory, a seed and the arm bookkeeping."""

    seed: int = 0
    model_factory: ModelFactory | None = None
    n_actions_: int = field(default=0, init=False)

    def _new_model(self):
        if self.model_factory is None:
            return default_regressor(self.seed)
        return clone(self.model_factory())

    def _check(self, X, treatment, outcome) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = _as_array(X)
        t = np.asarray(treatment, int)
        y = np.asarray(outcome, float)
        self.n_actions_ = int(t.max()) + 1
        if (t < 0).any() or (t == 0).sum() == 0:
            raise ValueError("treatment must contain control arm 0 and no negatives")
        return X, t, y


@dataclass
class TLearner(BaseUplift):
    """One outcome model per arm; uplift is the difference of predictions."""

    models_: list = field(default_factory=list, init=False)

    def fit(self, X, treatment, outcome) -> "TLearner":
        X, t, y = self._check(X, treatment, outcome)
        self.models_ = [self._new_model().fit(X[t == a], y[t == a]) for a in range(self.n_actions_)]
        return self

    def predict_outcome(self, X) -> np.ndarray:
        X = _as_array(X)
        return np.column_stack([m.predict(X) for m in self.models_])

    def predict_uplift(self, X) -> np.ndarray:
        mu = self.predict_outcome(X)
        return mu - mu[:, [0]]


@dataclass
class SLearner(BaseUplift):
    """One model on features plus a one-hot of the arm; uplift by substitution."""

    model_: object = field(default=None, init=False)

    def _design(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        onehot = np.eye(self.n_actions_)[t]
        return np.hstack([X, onehot])

    def fit(self, X, treatment, outcome) -> "SLearner":
        X, t, y = self._check(X, treatment, outcome)
        self.model_ = self._new_model().fit(self._design(X, t), y)
        return self

    def predict_outcome(self, X) -> np.ndarray:
        X = _as_array(X)
        cols = []
        for a in range(self.n_actions_):
            cols.append(self.model_.predict(self._design(X, np.full(len(X), a))))
        return np.column_stack(cols)

    def predict_uplift(self, X) -> np.ndarray:
        mu = self.predict_outcome(X)
        return mu - mu[:, [0]]


@dataclass
class XLearner(BaseUplift):
    """Two-stage estimator that imputes individual effects, then regresses them.

    For each action a: fit mu_0 on controls and mu_a on treated; impute
    D_treated = Y - mu_0(X) and D_control = mu_a(X) - Y; fit tau_treated on
    D_treated and tau_control on D_control; blend them with the propensity
    g(x) = P(T = a | X, T in {0, a}) as tau = g * tau_control + (1 - g) * tau_treated.
    """

    mu0_: object = field(default=None, init=False)
    stages_: dict = field(default_factory=dict, init=False)

    def fit(self, X, treatment, outcome) -> "XLearner":
        X, t, y = self._check(X, treatment, outcome)
        ctrl = t == 0
        self.mu0_ = self._new_model().fit(X[ctrl], y[ctrl])
        self.stages_ = {}
        for a in range(1, self.n_actions_):
            trt = t == a
            mu_a = self._new_model().fit(X[trt], y[trt])
            d_treated = y[trt] - self.mu0_.predict(X[trt])
            d_control = mu_a.predict(X[ctrl]) - y[ctrl]
            tau_t = self._new_model().fit(X[trt], d_treated)
            tau_c = self._new_model().fit(X[ctrl], d_control)
            both = ctrl | trt
            prop = HistGradientBoostingClassifier(
                max_iter=60, max_depth=3, random_state=self.seed
            ).fit(X[both], trt[both].astype(int))
            self.stages_[a] = (tau_t, tau_c, prop)
        return self

    def predict_uplift(self, X) -> np.ndarray:
        X = _as_array(X)
        out = np.zeros((len(X), self.n_actions_))
        for a, (tau_t, tau_c, prop) in self.stages_.items():
            g = prop.predict_proba(X)[:, 1]
            out[:, a] = g * tau_c.predict(X) + (1.0 - g) * tau_t.predict(X)
        return out


def make_learner(name: str, seed: int = 0) -> BaseUplift:
    """Factory used by the CLI and the experiment runner."""
    learners = {"t": TLearner, "s": SLearner, "x": XLearner}
    if name.lower() not in learners:
        raise ValueError(f"unknown learner {name!r}; choose from {sorted(learners)}")
    return learners[name.lower()](seed=seed)


def _ordered_counts(score, treatment, outcome):
    order = np.argsort(-np.asarray(score, float), kind="stable")
    t = np.asarray(treatment, float)[order]
    y = np.asarray(outcome, float)[order]
    n_t, n_c = np.cumsum(t), np.cumsum(1.0 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1.0 - t))
    return n_t, n_c, y_t, y_c


def _area(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))


def qini_curve(score, treatment, outcome) -> tuple[np.ndarray, np.ndarray]:
    """Qini curve: cumulative treated outcomes minus scaled control outcomes.

    At the top k rows (sorted by score), Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k).
    Returns x (fraction targeted, starting at 0) and Q (starting at 0).
    """
    n_t, n_c, y_t, y_c = _ordered_counts(score, treatment, outcome)
    q = y_t - y_c * n_t / np.maximum(n_c, 1.0)
    x = np.arange(len(q) + 1) / len(q)
    return x, np.concatenate([[0.0], q])


def uplift_curve(score, treatment, outcome) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative gain curve: (mean_t - mean_c) at the top k, scaled by k.

    The gain is zero wherever one of the two arms has no rows yet.
    """
    n_t, n_c, y_t, y_c = _ordered_counts(score, treatment, outcome)
    rate_t = np.divide(y_t, n_t, out=np.zeros_like(y_t), where=n_t > 0)
    rate_c = np.divide(y_c, n_c, out=np.zeros_like(y_c), where=n_c > 0)
    gain = np.where((n_t > 0) & (n_c > 0), (rate_t - rate_c) * (n_t + n_c), 0.0)
    x = np.arange(len(gain) + 1) / len(gain)
    return x, np.concatenate([[0.0], gain])


def qini_coefficient(score, treatment, outcome) -> float:
    """Area between the Qini curve and the straight line of random targeting."""
    x, q = qini_curve(score, treatment, outcome)
    return _area(x, q) - q[-1] / 2.0


def auuc(score, treatment, outcome) -> float:
    """Area under the cumulative gain curve, with the x axis as a fraction."""
    x, gain = uplift_curve(score, treatment, outcome)
    return _area(x, gain)


def qini_table(uplift_by_model: dict[str, np.ndarray], treatment, outcome,
               action_names: tuple[str, ...]) -> pd.DataFrame:
    """Qini and AUUC per model and action, each action against the control arm."""
    t = np.asarray(treatment, int)
    y = np.asarray(outcome, float)
    rows = []
    for name, uplift in uplift_by_model.items():
        for a in range(1, uplift.shape[1]):
            mask = (t == 0) | (t == a)
            score, trt, out = uplift[mask, a], (t[mask] == a).astype(int), y[mask]
            rows.append({"model": name, "action": action_names[a],
                         "qini": round(qini_coefficient(score, trt, out), 3),
                         "auuc": round(auuc(score, trt, out), 3)})
    return pd.DataFrame(rows)
