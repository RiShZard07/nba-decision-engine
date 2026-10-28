import numpy as np
import pytest

from nba import uplift as up


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


@pytest.mark.parametrize("learner, min_corr", [("t", 0.85), ("s", 0.6), ("x", 0.85)])
def test_learners_recover_sign_and_magnitude(constructed, learner, min_corr):
    X, t, y, tau = constructed
    model = up.make_learner(learner, seed=0).fit(X, t, y)
    est = model.predict_uplift(X)
    for a in (1, 2):
        assert _corr(est[:, a], tau[:, a]) > min_corr
        assert abs(est[:, a].mean() - tau[:, a].mean()) < 0.35
        # the sign of the effect flips across customers and the model must follow it
        agree = np.mean(np.sign(est[:, a]) == np.sign(tau[:, a]))
        assert agree > 0.75


@pytest.mark.parametrize("learner", ["t", "s", "x"])
def test_uplift_shape_and_control_column(constructed, learner):
    X, t, y, _ = constructed
    est = up.make_learner(learner, seed=0).fit(X[:2000], t[:2000], y[:2000]).predict_uplift(X[:50])
    assert est.shape == (50, 3)
    assert np.all(est[:, 0] == 0)


def test_make_learner_rejects_unknown():
    with pytest.raises(ValueError):
        up.make_learner("z")


def test_fit_requires_control_arm():
    with pytest.raises(ValueError):
        up.TLearner().fit(np.zeros((10, 2)), np.ones(10, int), np.zeros(10))


def test_qini_hand_computed_example():
    score = np.array([4.0, 3.0, 2.0, 1.0])
    treatment = np.array([1, 0, 1, 0])
    outcome = np.array([1.0, 0.0, 0.0, 1.0])
    x, q = up.qini_curve(score, treatment, outcome)
    assert np.allclose(x, [0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(q, [0, 1, 1, 1, 0])
    assert up.qini_coefficient(score, treatment, outcome) == pytest.approx(0.75)
    _, gain = up.uplift_curve(score, treatment, outcome)
    assert np.allclose(gain, [0, 0, 2, 1.5, 0])
    assert up.auuc(score, treatment, outcome) == pytest.approx(0.875)


def test_qini_and_auuc_increase_with_ranking_quality(constructed):
    X, t, y, tau = constructed
    mask = t <= 1
    trt, out, truth = t[mask], y[mask], tau[mask, 1]
    rng = np.random.default_rng(0)
    noisy = truth + rng.normal(0, 3 * truth.std(), size=len(truth))
    random = rng.normal(size=len(truth))
    for metric in (up.qini_coefficient, up.auuc):
        perfect, mid, rand = (metric(s, trt, out) for s in (truth, noisy, random))
        assert perfect > mid > rand
        assert abs(rand) < 0.25 * perfect


def test_qini_curve_endpoints_do_not_depend_on_ranking(constructed):
    X, t, y, tau = constructed
    mask = t <= 1
    trt, out = t[mask], y[mask]
    x1, q1 = up.qini_curve(tau[mask, 1], trt, out)
    x2, q2 = up.qini_curve(-tau[mask, 1], trt, out)
    assert q1[0] == 0 and q2[0] == 0
    assert q1[-1] == pytest.approx(q2[-1])
    assert len(x1) == mask.sum() + 1 and x1[-1] == 1.0


def test_qini_table_structure(constructed):
    X, t, y, tau = constructed
    table = up.qini_table({"truth": tau, "zeros": np.zeros_like(tau)}, t, y, ("c", "a1", "a2"))
    assert list(table.columns) == ["model", "action", "qini", "auuc"]
    assert len(table) == 4
    truth_rows = table[table.model == "truth"]
    assert (truth_rows.qini > table[table.model == "zeros"].qini.to_numpy()).all()
