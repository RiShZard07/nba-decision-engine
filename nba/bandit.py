"""Contextual bandits with linear payoff models per action.

LinUCB picks the action with the highest optimistic score
theta_a . x + alpha * sqrt(x^T A_a^-1 x). Thompson sampling draws theta_a
from a Gaussian posterior and picks the best sampled score. Both keep one
ridge regression per action and update it with a Sherman-Morrison step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from nba import simulate as sim

RewardSampler = Callable[[int, int, np.random.Generator], float]


@dataclass
class BanditEnvironment:
    """Contexts plus the true expected reward of every action for each row."""

    contexts: np.ndarray
    expected: np.ndarray
    sampler: RewardSampler | None = None
    noise_std: float = 0.0

    @property
    def n_actions(self) -> int:
        return self.expected.shape[1]

    @property
    def dim(self) -> int:
        return self.contexts.shape[1]

    def sample_reward(self, idx: int, action: int, rng: np.random.Generator) -> float:
        if self.sampler is not None:
            return self.sampler(idx, action, rng)
        return float(self.expected[idx, action] + rng.normal(0.0, self.noise_std))

    def regret(self, idx: int, action: int) -> float:
        return float(self.expected[idx].max() - self.expected[idx, action])


def linear_environment(n: int = 2000, dim: int = 5, n_actions: int = 4, seed: int = 0,
                       noise_std: float = 0.3) -> BanditEnvironment:
    """A well specified environment: rewards are linear in the context."""
    rng = np.random.default_rng(seed)
    contexts = np.hstack([np.ones((n, 1)), rng.normal(size=(n, dim - 1))])
    theta = rng.normal(size=(n_actions, dim))
    return BanditEnvironment(contexts, contexts @ theta.T, noise_std=noise_std)


def bandit_context(df: pd.DataFrame) -> np.ndarray:
    """Standardized features, an intercept and spend interactions.

    The true reward differences between actions are products of spend and one
    other feature, so a linear model needs those products as inputs.
    """
    X = sim.feature_matrix(df)
    spend = df["monthly_spend"].to_numpy()[:, None]
    inter = spend * df[["churn_risk_score", "device_age_months", "usage_trend",
                        "support_contacts_90d", "plan_tier"]].to_numpy(float)
    X = np.hstack([X, inter])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    return np.hstack([np.ones((len(df), 1)), X])


def customer_environment(df: pd.DataFrame, scale: float = 100.0) -> BanditEnvironment:
    """Wrap the synthetic population; rewards are divided by `scale`."""
    contexts = bandit_context(df)
    probs = sim.retention_probabilities(df)
    margin = sim.quarterly_margin(df)

    def sampler(idx: int, action: int, rng: np.random.Generator) -> float:
        retained = float(rng.random() < probs[idx, action])
        return (retained * margin[idx] - sim.COSTS[action]) / scale

    return BanditEnvironment(contexts, sim.expected_rewards(df) / scale, sampler=sampler)


@dataclass
class _LinearArms:
    """Per-action ridge regression state shared by both bandits."""

    n_actions: int
    dim: int
    ridge: float = 1.0
    seed: int = 0
    rng: np.random.Generator = field(init=False)
    A_inv: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.A_inv = np.stack([np.eye(self.dim) / self.ridge for _ in range(self.n_actions)])
        self.b = np.zeros((self.n_actions, self.dim))

    def theta(self) -> np.ndarray:
        return np.einsum("aij,aj->ai", self.A_inv, self.b)

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        x = np.asarray(context, float)
        Ax = self.A_inv[action] @ x
        self.A_inv[action] -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.b[action] += reward * x

    def _argmax(self, scores: np.ndarray) -> int:
        best = np.flatnonzero(scores >= scores.max() - 1e-12)
        return int(self.rng.choice(best))


@dataclass
class LinUCB(_LinearArms):
    """Upper confidence bound with a linear model per action."""

    alpha: float = 1.0

    def select(self, context: np.ndarray) -> int:
        x = np.asarray(context, float)
        width = np.sqrt(np.einsum("i,aij,j->a", x, self.A_inv, x))
        return self._argmax(self.theta() @ x + self.alpha * width)


@dataclass
class LinearThompsonSampling(_LinearArms):
    """Gaussian posterior sampling: theta_a ~ N(A_a^-1 b_a, v^2 A_a^-1)."""

    v: float = 0.5

    def select(self, context: np.ndarray) -> int:
        x = np.asarray(context, float)
        scores = np.empty(self.n_actions)
        for a, mean in enumerate(self.theta()):
            cov = self.v**2 * self.A_inv[a]
            sample = self.rng.multivariate_normal(mean, cov, method="cholesky")
            scores[a] = sample @ x
        return self._argmax(scores)


@dataclass
class UniformRandom:
    """Baseline that ignores the context."""

    n_actions: int
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def select(self, context: np.ndarray) -> int:
        return int(self.rng.integers(self.n_actions))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        return None


@dataclass
class Oracle:
    """Knows the environment's expected rewards; the regret reference."""

    env: BanditEnvironment
    is_oracle: bool = True

    def select_index(self, idx: int) -> int:
        return int(self.env.expected[idx].argmax())

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        return None


@dataclass
class BanditResult:
    cumulative_reward: np.ndarray
    cumulative_regret: np.ndarray
    action_counts: np.ndarray

    @property
    def final_regret(self) -> float:
        return float(self.cumulative_regret[-1])


def simulate_online(policy, env: BanditEnvironment, rounds: int, seed: int = 0) -> BanditResult:
    """Play `rounds` steps against the environment and track reward and regret."""
    rng = np.random.default_rng(seed)
    rewards = np.zeros(rounds)
    regrets = np.zeros(rounds)
    counts = np.zeros(env.n_actions, int)
    for t in range(rounds):
        idx = int(rng.integers(len(env.contexts)))
        x = env.contexts[idx]
        if getattr(policy, "is_oracle", False):
            action = policy.select_index(idx)
        else:
            action = policy.select(x)
        reward = env.sample_reward(idx, action, rng)
        policy.update(x, action, reward)
        rewards[t], regrets[t] = reward, env.regret(idx, action)
        counts[action] += 1
    return BanditResult(np.cumsum(rewards), np.cumsum(regrets), counts)


def regret_summary(cumulative_regret: np.ndarray, ratio: float = 0.6) -> dict:
    """Documented sublinearity test: per-round regret in the final third of the
    run must be at most `ratio` times the per-round regret in the first third.
    A linear curve scores 1.0; a sqrt(T) curve scores about 0.32.
    """
    n = len(cumulative_regret)
    third = max(n // 3, 1)
    first_rate = float(cumulative_regret[third - 1]) / third
    last_rate = float(cumulative_regret[-1] - cumulative_regret[n - third - 1]) / third
    observed = last_rate / first_rate if first_rate > 0 else 0.0
    return {
        "final_regret": float(cumulative_regret[-1]),
        "first_third_rate": first_rate,
        "last_third_rate": last_rate,
        "ratio": observed,
        "threshold": ratio,
        "sublinear": bool(observed <= ratio),
    }
