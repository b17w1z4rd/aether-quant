"""Reproducible research runs, with an offline interactive report."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .data import demo_chain, prepare
from .dashboard import build
from .risk import default_portfolio, experiment, hedge_instruments
from .surface import fit


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run(args):
    synthetic = args.command == "demo"
    if synthetic:
        raw = demo_chain(args.seed)
        spot, rate, dividend = 100.0, 0.03, 0.0
    else:
        raw = pd.read_csv(args.chain)
        spot, rate, dividend = args.spot, args.rate, args.dividend
    quotes = prepare(raw, spot, rate, dividend)
    config = {"portfolio": default_portfolio(), "hedges": hedge_instruments()}
    if args.positions:
        config = json.loads(Path(args.positions).read_text(encoding="utf-8"))
    surface = fit(quotes, args.seed)
    risk = experiment(
        surface,
        spot,
        rate,
        dividend,
        config["portfolio"],
        seed=args.seed,
        budget=args.budget,
        hedge_legs=config["hedges"],
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = raw[["days", "strike", "bid", "ask"]].to_csv(index=False)
    (out / "chain.csv").write_text(csv, encoding="utf-8")
    quotes.to_csv(out / "prepared-quotes.csv", index=False)
    surface["test"].to_csv(out / "heldout-quotes.csv", index=False)
    pd.DataFrame(
        {k: risk[k] for k in ["moves", "vol_shocks", "unhedged", "hedged"]}
    ).to_csv(out / "evaluation-scenarios.csv", index=False)
    np.savez_compressed(
        out / "surface.npz",
        **{
            k: surface[k]
            for k in ["xs", "ts", "raw", "clean", "iv", "projected_iv", "low", "high"]
        },
    )
    np.savez_compressed(
        out / "risk.npz",
        **{
            k: risk[k]
            for k in [
                "moves",
                "vol_shocks",
                "unhedged",
                "hedged",
                "weights",
                "premiums",
                "base_premiums",
            ]
        },
    )
    write_json(out / "positions.json", config)
    write_json(out / "frontier.json", risk["frontier"])
    metrics = {
        "project": "Aether Quant",
        "version": "0.1.0",
        "synthetic_chain": synthetic,
        "seed": args.seed,
        "spot": spot,
        "rate": rate,
        "dividend": dividend,
        "input_csv_sha256": hashlib.sha256(csv.encode()).hexdigest(),
        "input_quotes": len(quotes),
        "training_quotes": len(surface["train"]),
        "heldout_quotes_evaluated": len(surface["test"]),
        "heldout_quotes_outside_common_domain": int(quotes.holdout.sum())
        - len(surface["test"]),
        "grid": {
            "expiries": len(surface["ts"]),
            "moneyness_points": len(surface["xs"]),
            "minimum_x": float(surface["xs"][0]),
            "maximum_x": float(surface["xs"][-1]),
        },
        "heldout_metrics": surface["metrics"],
        "projection": surface["audit"],
        "constraint_violations_before": surface["before"],
        "constraint_violations_after": surface["after"],
        "gp_kernel": surface["kernel"],
        "gp_warnings": surface["warnings"],
        "svi_parameters": surface["svi_parameters"],
        "svi_audit": surface["svi_audit"],
        "risk": risk["summary"],
        "hedge_weights": risk["weights"].tolist(),
        "runtime": {
            "python": platform.python_version(),
            **{
                p: importlib.metadata.version(p)
                for p in ["numpy", "pandas", "scipy", "scikit-learn", "plotly"]
            },
        },
        "scope": "Cross-sectional interpolation and assumed-distribution scenario risk; not historical investment performance.",
    }
    write_json(out / "metrics.json", metrics)
    build(
        quotes, surface, risk, spot, rate, dividend, out / "dashboard.html", synthetic
    )
    print(f"Report: {(out / 'dashboard.html').resolve()}")
    print(f"Heldout price RMSE: {surface['metrics']['projected_c']['rmse_price']:.6f}")
    print(
        f"Evaluation 97.5% ES: {risk['summary']['unhedged_es']:.2f} -> {risk['summary']['hedged_es']:.2f}"
    )
    print(f"Largest grid violation: {max(surface['after'].values()):.2e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ["demo", "run"]:
        p = commands.add_parser(command)
        p.add_argument(
            "--out",
            default=f"artifacts/{command}",
            help="Output directory (existing report files will be replaced)",
        )
        p.add_argument("--seed", type=int, default=42)
        p.add_argument(
            "--budget",
            type=float,
            default=600.0,
            help="Premium budget, including modeled execution friction",
        )
        p.add_argument(
            "--positions",
            help="JSON with portfolio and hedges arrays; see examples/positions.json",
        )
        if command == "run":
            p.add_argument(
                "--chain",
                required=True,
                help="CSV: days,strike,bid,ask; European calls only",
            )
            p.add_argument("--spot", required=True, type=float)
            p.add_argument(
                "--rate",
                type=float,
                default=0.03,
                help="Annual continuously compounded rate",
            )
            p.add_argument(
                "--dividend",
                type=float,
                default=0.0,
                help="Annual continuous dividend yield",
            )
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"aether-quant: {exc}\n")


if __name__ == "__main__":
    main()
