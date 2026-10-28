import pandas as pd
import pytest

from nba import __main__ as cli


def test_simulate_command_writes_files(tmp_path, capsys):
    assert cli.main(["simulate", "--n", "300", "--seed", "1", "--out", str(tmp_path)]) == 0
    customers = pd.read_csv(tmp_path / "customers.csv")
    logged = pd.read_csv(tmp_path / "logged.csv")
    assert len(customers) == 300 and len(logged) == 300
    assert {"action", "propensity", "retained", "reward"} <= set(logged.columns)
    assert "synthetic" in capsys.readouterr().out


def test_fit_command_saves_model(tmp_path, capsys):
    cli.main(["simulate", "--n", "1500", "--seed", "1", "--out", str(tmp_path)])
    assert cli.main(["fit", "--learner", "t", "--out", str(tmp_path), "--seed", "1"]) == 0
    assert (tmp_path / "model.pkl").exists()
    assert "Fitted t-learner" in capsys.readouterr().out


def test_evaluate_then_recommend(tmp_path, capsys):
    args = ["--n", "2000", "--seed", "5", "--rounds", "400", "--out", str(tmp_path)]
    assert cli.main(["evaluate", *args]) == 0
    out = capsys.readouterr().out
    assert "Synthetic data only" in out and "Checks:" in out
    assert cli.main(["recommend", "--customer-id", "3", "--out", str(tmp_path), "--json"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("Customer 3:") and '"customer_id": 3' in text


def test_recommend_without_model_fails_cleanly(tmp_path, capsys):
    assert cli.main(["recommend", "--customer-id", "1", "--out", str(tmp_path)]) == 2
    assert "run `python -m nba demo` first" in capsys.readouterr().err


@pytest.mark.parametrize("passed, code", [(True, 0), (False, 1)])
def test_demo_check_exit_code(monkeypatch, tmp_path, passed, code):
    fake = {"policies": [], "bandits": {"rounds": 0, "players": {}}, "checks": {"all_passed": passed},
            "runtime_seconds": 0.0, "config": {"output_dir": str(tmp_path)}}
    monkeypatch.setattr(cli, "run_experiment", lambda cfg: fake)
    assert cli.main(["demo", "--check", "--out", str(tmp_path)]) == code
    assert cli.main(["demo", "--out", str(tmp_path)]) == 0
