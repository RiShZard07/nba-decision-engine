"""End-to-end experiment: simulate, fit, decide, evaluate offline, gate, report.

All numbers produced here come from synthetic data with designed effects.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from nba import bandit as bd
from nba import ope
from nba import policy as pol
from nba import simulate as sim
from nba import uplift as up

LEARNERS = ("t", "s", "x")


@dataclass
class ExperimentConfig:
    n_customers: int = 20_000
    seed: int = 7
    epsilon: float = 0.3
    train_fraction: float = 0.6
    budget_fractions: dict[int, float] = field(
        default_factory=lambda: {1: 0.10, 2: 0.15, 3: 0.15, 4: 0.08}
    )
    contact_cap: int = 2
    min_net_value: float = 0.5
    bandit_rounds: int = 30_000
    n_boot: int = 200
    clip: float = 20.0
    output_dir: str = "output"


def _split(n: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    cut = int(n * fraction)
    return np.sort(order[:cut]), np.sort(order[cut:])


def fit_learners(train_log: pd.DataFrame, seed: int) -> dict[str, up.BaseUplift]:
    X = sim.feature_matrix(train_log)
    t, y = train_log["action"].to_numpy(), train_log["retained"].to_numpy()
    return {name: up.make_learner(name, seed).fit(X, t, y) for name in LEARNERS}


def uplift_quality(models: dict, df_eval: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Qini and AUUC on a fully randomized synthetic log, plus truth correlation."""
    rand_log = sim.simulate_logged(df_eval, seed, probs=sim.uniform_policy_probs(df_eval))
    X = sim.feature_matrix(df_eval)
    uplifts = {name: m.predict_uplift(X) for name, m in models.items()}
    table = up.qini_table(uplifts, rand_log["action"].to_numpy(), rand_log["retained"].to_numpy(), sim.ACTIONS)
    truth = sim.treatment_effects(df_eval)
    corr = []
    for row in table.itertuples():
        a = sim.ACTIONS.index(row.action)
        est = uplifts[row.model][:, a]
        corr.append(float(np.corrcoef(est, truth[:, a])[0, 1]) if est.std() > 0 else 0.0)
    table["corr_with_true_effect"] = np.round(corr, 3)
    return table


def build_policies(models: dict, constraints: pol.Constraints, epsilon: float, seed: int) -> list:
    policies = [pol.logging_policy(epsilon), pol.heuristic_policy()]
    for name, m in models.items():
        policies.append(pol.ConstrainedPolicy(f"uplift_{name}", pol.uplift_scorer(m), constraints, seed))
    policies.append(pol.ConstrainedPolicy("oracle_truth", pol.oracle_scorer, constraints, seed))
    return policies


def evaluate_policies(policies: list, df_eval: pd.DataFrame, eval_log: pd.DataFrame,
                      constraints: pol.Constraints, cfg: ExperimentConfig) -> list[dict]:
    X = sim.feature_matrix(eval_log)
    q_hat = ope.fit_outcome_model(X, eval_log["action"].to_numpy(), eval_log["reward"].to_numpy(),
                                  sim.N_ACTIONS, seed=cfg.seed)
    rows = []
    for p in policies:
        probs = p.action_probs(df_eval)
        truth = ope.true_policy_value(df_eval, probs)
        est = ope.evaluate_policy(eval_log, probs, q_hat, cfg.clip, cfg.n_boot, cfg.seed)
        row = {"policy": p.name, "true_value": round(truth, 4),
               "action_distribution": np.round(probs.mean(axis=0), 4).tolist()}
        for name, r in est.items():
            row[name] = r.to_dict()
            row[name]["abs_error"] = round(abs(r.estimate - truth), 4)
        if isinstance(p, pol.ConstrainedPolicy):
            decision = p.decide(df_eval)
            row["budget_used"] = {sim.ACTIONS[a]: c for a, c in decision.budget_used.items()}
            row["budget_caps"] = {sim.ACTIONS[a]: c for a, c in constraints.budget.items()}
            row["constraint_audit"] = pol.check_constraints(decision, df_eval, constraints)
            row["binding_counts"] = {k: int(v) for k, v in
                                     zip(*np.unique(decision.binding.astype(str), return_counts=True))}
        rows.append(row)
    return rows


def run_bandits(df: pd.DataFrame, rounds: int, seed: int) -> dict:
    env = bd.customer_environment(df)
    players = {
        "linucb": bd.LinUCB(env.n_actions, env.dim, alpha=2.0, seed=seed),
        "thompson": bd.LinearThompsonSampling(env.n_actions, env.dim, v=0.3, seed=seed),
        "uniform_random": bd.UniformRandom(env.n_actions, seed),
        "oracle": bd.Oracle(env),
    }
    checkpoints = np.linspace(0, rounds, 11, dtype=int)[1:] - 1
    out = {"rounds": rounds, "reward_scale": 100.0, "players": {}}
    for name, player in players.items():
        res = bd.simulate_online(player, env, rounds, seed)
        summary = bd.regret_summary(res.cumulative_regret)
        summary["cumulative_reward"] = round(float(res.cumulative_reward[-1]), 3)
        summary["action_counts"] = res.action_counts.tolist()
        summary["regret_curve"] = np.round(res.cumulative_regret[checkpoints], 3).tolist()
        out["players"][name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}
    return out


def gate_checks(policy_rows: list[dict], bandits: dict) -> dict:
    """The regression gate behind `python -m nba demo --check`."""
    by_name = {r["policy"]: r for r in policy_rows}
    learned = [r for r in policy_rows if r["policy"].startswith("uplift_")]
    chosen = max(learned, key=lambda r: r["dr"]["estimate"])
    dr = chosen["dr"]
    audits = [r["constraint_audit"] for r in policy_rows if "constraint_audit" in r]
    checks = {
        "chosen_policy": chosen["policy"],
        "chosen_beats_logging_truth": chosen["true_value"] > by_name["logging"]["true_value"],
        "dr_ci_contains_truth": dr["ci_low"] <= chosen["true_value"] <= dr["ci_high"],
        "budget_constraints_held": all(a["budget_ok"] and a["contact_cap_ok"] and a["eligibility_ok"]
                                       for a in audits),
        "bandit_regret_sublinear": all(bandits["players"][k]["sublinear"] for k in ("linucb", "thompson")),
    }
    checks["all_passed"] = all(v for k, v in checks.items() if k != "chosen_policy")
    return checks


def run_experiment(cfg: ExperimentConfig) -> dict:
    """Run everything and write metrics.json, report.md, model.pkl, customers.csv."""
    started = time.time()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = sim.generate_customers(cfg.n_customers, cfg.seed)
    logged = sim.simulate_logged(df, cfg.seed + 1, epsilon=cfg.epsilon)
    train_idx, eval_idx = _split(len(df), cfg.train_fraction, cfg.seed)
    df_eval, eval_log = df.iloc[eval_idx].reset_index(drop=True), logged.iloc[eval_idx].reset_index(drop=True)
    models = fit_learners(logged.iloc[train_idx], cfg.seed)
    qini = uplift_quality(models, df_eval, cfg.seed + 2)
    constraints = pol.Constraints(pol.budget_from_fractions(len(df_eval), cfg.budget_fractions),
                                  cfg.contact_cap, cfg.min_net_value)
    policies = build_policies(models, constraints, cfg.epsilon, cfg.seed)
    policy_rows = evaluate_policies(policies, df_eval, eval_log, constraints, cfg)
    log_check = ope.logging_policy_check(eval_log, sim.logging_policy_probs(df_eval, cfg.epsilon))
    bandits = run_bandits(df, cfg.bandit_rounds, cfg.seed)
    checks = gate_checks(policy_rows, bandits)
    metrics = {
        "synthetic_data_notice": "Every number here comes from a seeded synthetic simulator.",
        "config": asdict(cfg), "n_train": int(len(train_idx)), "n_eval": int(len(df_eval)),
        "qini_table": qini.to_dict(orient="records"), "policies": policy_rows,
        "logging_policy_check": {k: round(v, 4) for k, v in log_check.items()},
        "bandits": bandits, "checks": checks, "runtime_seconds": round(time.time() - started, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "report.md").write_text(render_report(metrics))
    chosen = checks["chosen_policy"].removeprefix("uplift_")
    bundle = {"learner": chosen, "model": models[chosen], "seed": cfg.seed,
              "budget_fractions": cfg.budget_fractions, "contact_cap": cfg.contact_cap,
              "min_net_value": cfg.min_net_value}
    (out_dir / "model.pkl").write_bytes(pickle.dumps(bundle))
    df.to_csv(out_dir / "customers.csv", index=False)
    logged.to_csv(out_dir / "logged.csv", index=False)
    return metrics


def _md_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def render_report(m: dict) -> str:
    """Markdown summary of one experiment run."""
    policy_rows = []
    for r in m["policies"]:
        used = r.get("budget_used")
        budget = "no caps" if used is None else "; ".join(
            f"{k.split('_')[0]} {v}/{r['budget_caps'][k]}" for k, v in used.items())
        policy_rows.append({
            "policy": r["policy"], "true value": f"{r['true_value']:.2f}",
            "DR estimate": f"{r['dr']['estimate']:.2f}",
            "DR 95% CI": f"[{r['dr']['ci_low']:.2f}, {r['dr']['ci_high']:.2f}]",
            "IPS": f"{r['ips']['estimate']:.2f}", "SNIPS": f"{r['snips']['estimate']:.2f}",
            "DM": f"{r['dm']['estimate']:.2f}", "DR abs error": f"{r['dr']['abs_error']:.2f}",
            "budget used/cap": budget,
            "action distribution": [round(v, 3) for v in r["action_distribution"]],
        })
    bandit_rows = [{"player": k, "final regret": v["final_regret"], "cumulative reward": v["cumulative_reward"],
                    "regret ratio (last third / first third)": v["ratio"], "sublinear": v["sublinear"]}
                   for k, v in m["bandits"]["players"].items()]
    checks = [{"check": k, "result": v} for k, v in m["checks"].items()]
    parts = [
        "# Experiment report (synthetic data)\n",
        "All customers, actions and outcomes are synthetic and generated by `nba/simulate.py` "
        f"with seed {m['config']['seed']}. Nothing here describes a real business.\n",
        f"Train rows: {m['n_train']}. Evaluation rows: {m['n_eval']}. Runtime: {m['runtime_seconds']} s.\n",
        "## Policies: offline estimates versus simulator truth\n",
        "Values are expected net reward per customer (margin if retained minus action cost). "
        "Action distribution order: " + ", ".join(sim.ACTIONS) + ".\n",
        _md_table(policy_rows, list(policy_rows[0].keys())), "",
        "## Uplift model quality per action\n",
        "Qini and AUUC are computed on a separate randomized synthetic log; the correlation uses "
        "the simulator's true effects, which real data never offers.\n",
        _md_table(m["qini_table"], ["model", "action", "qini", "auuc", "corr_with_true_effect"]), "",
        "## Logging policy self-check\n",
        _md_table([{"quantity": k, "value": v} for k, v in m["logging_policy_check"].items()],
                  ["quantity", "value"]), "",
        "## Online bandit simulation\n",
        f"{m['bandits']['rounds']} rounds, rewards divided by {m['bandits']['reward_scale']}. "
        "Sublinear means the per-round regret in the final third is at most 0.6 times that of the first third.\n",
        _md_table(bandit_rows, list(bandit_rows[0].keys())), "",
        "## Regression gate\n",
        _md_table(checks, ["check", "result"]), "",
    ]
    return "\n".join(parts)
