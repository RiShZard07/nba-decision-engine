import json

import pytest

from nba import evaluate as ev


@pytest.fixture(scope="module")
def small_metrics(tmp_path_factory):
    out = tmp_path_factory.mktemp("exp")
    cfg = ev.ExperimentConfig(n_customers=2500, seed=3, bandit_rounds=600, n_boot=30, output_dir=str(out))
    return ev.run_experiment(cfg), out


def test_run_experiment_writes_outputs(small_metrics):
    metrics, out = small_metrics
    for name in ("metrics.json", "report.md", "model.pkl", "customers.csv", "logged.csv"):
        assert (out / name).exists()
    on_disk = json.loads((out / "metrics.json").read_text())
    assert on_disk["checks"] == metrics["checks"]
    assert {r["policy"] for r in metrics["policies"]} == {
        "logging", "heuristic_rules", "uplift_t", "uplift_s", "uplift_x", "oracle_truth"}
    assert len(metrics["qini_table"]) == 12
    assert metrics["n_train"] + metrics["n_eval"] == 2500
    assert "synthetic" in (out / "report.md").read_text().lower()


def test_policy_rows_carry_all_estimators_and_audits(small_metrics):
    metrics, _ = small_metrics
    for row in metrics["policies"]:
        assert {"ips", "snips", "dm", "dr"} <= set(row)
        assert row["dr"]["ci_low"] <= row["dr"]["estimate"] <= row["dr"]["ci_high"]
        if row["policy"].startswith(("uplift_", "oracle")):
            assert row["constraint_audit"]["budget_ok"] and row["constraint_audit"]["contact_cap_ok"]
    assert metrics["logging_policy_check"]["ips_abs_error"] == 0.0


def _rows(chosen_true, logging_true, ci, budget_ok=True):
    def row(name, true, est=None):
        est = true if est is None else est
        r = {"policy": name, "true_value": true,
             "dr": {"estimate": est, "ci_low": est - ci, "ci_high": est + ci}}
        if name != "logging":
            r["constraint_audit"] = {"budget_ok": budget_ok, "contact_cap_ok": True, "eligibility_ok": True}
        return r
    return [row("logging", logging_true), row("uplift_t", chosen_true), row("uplift_x", chosen_true - 1, chosen_true - 2)]


def _bandits(ok=True):
    return {"players": {"linucb": {"sublinear": ok}, "thompson": {"sublinear": True}}}


def test_gate_checks_pass_and_fail_logic():
    good = ev.gate_checks(_rows(65.0, 63.0, ci=1.0), _bandits())
    assert good["all_passed"] and good["chosen_policy"] == "uplift_t"
    assert not ev.gate_checks(_rows(62.0, 63.0, ci=1.0), _bandits())["chosen_beats_logging_truth"]
    assert not ev.gate_checks(_rows(65.0, 63.0, ci=0.0, budget_ok=False), _bandits())["budget_constraints_held"]
    assert not ev.gate_checks(_rows(65.0, 63.0, ci=1.0), _bandits(ok=False))["all_passed"]
    rows = _rows(65.0, 63.0, ci=0.5)
    rows[1]["dr"]["estimate"] = 63.0
    rows[1]["dr"]["ci_low"], rows[1]["dr"]["ci_high"] = 62.5, 63.5
    assert not ev.gate_checks(rows, _bandits())["dr_ci_contains_truth"]


def test_render_report_contains_tables(small_metrics):
    metrics, _ = small_metrics
    text = ev.render_report(metrics)
    for heading in ("## Policies", "## Uplift model quality", "## Online bandit", "## Regression gate"):
        assert heading in text
    assert "| uplift_x |" in text and chr(0x2014) not in text
