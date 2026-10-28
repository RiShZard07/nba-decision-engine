"""Command line entry point: python -m nba <command> [options]."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

from nba import simulate as sim
from nba.evaluate import ExperimentConfig, run_experiment
from nba.explain import explain_recommendation
from nba.policy import ConstrainedPolicy, Constraints, budget_from_fractions, uplift_scorer
from nba.uplift import make_learner


def _print_summary(m: dict) -> None:
    print("Synthetic data only. Expected net reward per customer (simulator truth, DR estimate, DR 95% CI):")
    for r in m["policies"]:
        dr = r["dr"]
        print(f"  {r['policy']:<16} true {r['true_value']:7.3f}   DR {dr['estimate']:7.3f}   "
              f"CI [{dr['ci_low']:.3f}, {dr['ci_high']:.3f}]   IPS err {r['ips']['abs_error']:.3f}   "
              f"DR err {dr['abs_error']:.3f}")
    print("Bandit regret after {} rounds:".format(m["bandits"]["rounds"]))
    for k, v in m["bandits"]["players"].items():
        print(f"  {k:<16} regret {v['final_regret']:9.2f}   ratio {v['ratio']:.3f}   sublinear {v['sublinear']}")
    print("Checks:", json.dumps(m["checks"]))
    print(f"Runtime {m['runtime_seconds']} s. Files written to {m['config']['output_dir']}/")


def cmd_simulate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = sim.generate_customers(args.n, args.seed)
    logged = sim.simulate_logged(df, args.seed + 1, epsilon=args.epsilon)
    df.to_csv(out / "customers.csv", index=False)
    logged.to_csv(out / "logged.csv", index=False)
    print(f"Wrote {len(df)} synthetic customers and a logged dataset to {out}/")
    print("Logged action shares:", logged["action"].value_counts(normalize=True).sort_index().round(3).to_dict())
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    out = Path(args.out)
    path = out / "logged.csv"
    if not path.exists():
        cmd_simulate(args)
    logged = pd.read_csv(path)
    model = make_learner(args.learner, args.seed).fit(
        sim.feature_matrix(logged), logged["action"].to_numpy(), logged["retained"].to_numpy()
    )
    bundle = {"learner": args.learner, "model": model, "seed": args.seed, "budget_fractions": {},
              "contact_cap": 2, "min_net_value": 0.5}
    (out / "model.pkl").write_bytes(pickle.dumps(bundle))
    print(f"Fitted {args.learner}-learner on {len(logged)} logged rows; saved to {out / 'model.pkl'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = ExperimentConfig(n_customers=args.n, seed=args.seed, epsilon=args.epsilon,
                           bandit_rounds=args.rounds, output_dir=args.out)
    metrics = run_experiment(cfg)
    _print_summary(metrics)
    if getattr(args, "check", False) and not metrics["checks"]["all_passed"]:
        print("CHECK FAILED", file=sys.stderr)
        return 1
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if not (out / "model.pkl").exists() or not (out / "customers.csv").exists():
        print("No fitted model found; run `python -m nba demo` first.", file=sys.stderr)
        return 2
    bundle = pickle.loads((out / "model.pkl").read_bytes())
    df = pd.read_csv(out / "customers.csv")
    constraints = Constraints(budget=budget_from_fractions(len(df), bundle["budget_fractions"]),
                              contact_cap=bundle["contact_cap"], min_net_value=bundle["min_net_value"])
    policy = ConstrainedPolicy(f"uplift_{bundle['learner']}", uplift_scorer(bundle["model"]),
                               constraints, bundle["seed"])
    decision = policy.decide(df)
    explanation = explain_recommendation(df, args.customer_id, bundle["model"], decision)
    print(explanation.to_text())
    if args.json:
        print(json.dumps(explanation.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nba", description="Next-best-action engine on synthetic data.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", default="output", help="output directory")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--n", type=int, default=20_000, help="number of synthetic customers")
        p.add_argument("--epsilon", type=float, default=0.3, help="logging policy exploration rate")

    p = sub.add_parser("simulate", help="write synthetic customers.csv and logged.csv")
    common(p)
    p.set_defaults(func=cmd_simulate)
    p = sub.add_parser("fit", help="fit one uplift learner on logged.csv")
    common(p)
    p.add_argument("--learner", choices=["t", "s", "x"], default="x")
    p.set_defaults(func=cmd_fit)
    for name, help_text in (("evaluate", "run the full experiment"), ("demo", "run everything into output/")):
        p = sub.add_parser(name, help=help_text)
        common(p)
        p.add_argument("--rounds", type=int, default=30_000, help="bandit rounds")
        p.add_argument("--check", action="store_true", help="exit nonzero unless the regression gate passes")
        p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("recommend", help="explain the recommendation for one customer")
    p.add_argument("--customer-id", type=int, required=True)
    p.add_argument("--out", default="output")
    p.add_argument("--json", action="store_true", help="also print the explanation as JSON")
    p.set_defaults(func=cmd_recommend)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
