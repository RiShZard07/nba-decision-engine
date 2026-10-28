"""Shared fixtures. Everything is synthetic and seeded; no network access."""

from __future__ import annotations

import numpy as np
import pytest

from nba import simulate as sim
from nba.uplift import TLearner


@pytest.fixture(scope="session")
def population():
    return sim.generate_customers(6000, seed=1)


@pytest.fixture(scope="session")
def logged(population):
    return sim.simulate_logged(population, seed=2)


@pytest.fixture(scope="session")
def fitted_t(logged):
    return TLearner(seed=0).fit(
        sim.feature_matrix(logged), logged["action"].to_numpy(), logged["retained"].to_numpy()
    )


@pytest.fixture(scope="session")
def constructed():
    """Continuous outcome with two treatments whose effects have known forms."""
    rng = np.random.default_rng(5)
    n = 6000
    X = rng.normal(size=(n, 4))
    t = rng.integers(0, 3, size=n)
    tau = np.column_stack([np.zeros(n), 2.0 * X[:, 0], 1.0 + X[:, 1]])
    y = 1.0 + X[:, 0] + 0.5 * X[:, 1] + tau[np.arange(n), t] + rng.normal(0.0, 0.5, size=n)
    return X, t, y, tau
