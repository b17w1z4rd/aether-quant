"""Render a static research board from a completed Aether run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from aether_quant.risk import reprice

BG = "#0d0b15"
CARD = "#181422"
TEXT = "#eeeaf6"
MUTED = "#a39caf"
VIOLET = "#b7a0ff"
TEAL = "#64e3d0"
CORAL = "#ff9f87"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="artifacts/demo")
    parser.add_argument("--out", default="docs/research-board.png")
    args = parser.parse_args()
    root = Path(args.run)
    s = dict(np.load(root / "surface.npz", allow_pickle=False))
    r = dict(np.load(root / "risk.npz", allow_pickle=False))
    m = json.loads((root / "metrics.json").read_text())
    positions = json.loads((root / "positions.json").read_text())
    frontier = json.loads((root / "frontier.json").read_text())
    quotes = pd.read_csv(root / "prepared-quotes.csv")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": "#3b304b",
            "axes.facecolor": CARD,
            "axes.titlecolor": TEXT,
            "savefig.facecolor": BG,
        }
    )
    fig = plt.figure(figsize=(16, 13), facecolor=BG)
    fig.text(0.045, 0.957, "AETHER  /  QUANT", fontsize=15, weight="bold", color=VIOLET)
    label = (
        "SYNTHETIC CHAIN + SIMULATED SCENARIOS"
        if m["synthetic_chain"]
        else "IMPORTED CHAIN + SIMULATED SCENARIOS"
    )
    fig.text(0.955, 0.958, label, fontsize=8, color=CORAL, ha="right")
    fig.text(0.043, 0.903, "The geometry of", fontsize=42, weight="light")
    fig.text(0.043, 0.855, "uncertainty.", fontsize=42, color=VIOLET, weight="light")
    fig.text(
        0.045,
        0.817,
        "Learn the surface. Repair the prices. Explore the tail.",
        color=MUTED,
        fontsize=13,
    )
    fig.text(
        0.67,
        0.908,
        "ONE EXPERIMENT. THREE LAYERS.",
        color=TEXT,
        fontsize=10,
        weight="bold",
    )
    fig.text(
        0.67,
        0.849,
        "Gaussian-process option surface\nFinite-grid price consistency\nScenario expected-shortfall optimization",
        color=MUTED,
        fontsize=11,
        linespacing=1.9,
    )
    fig.text(
        0.67,
        0.815,
        "15 interactive views in the offline dashboard",
        color=TEAL,
        fontsize=10,
    )

    def box(x, y, w, h):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            transform=fig.transFigure,
            facecolor=CARD,
            edgecolor="#372c48",
            linewidth=0.8,
            zorder=0,
        )
        fig.add_artist(patch)

    risk = m["risk"]
    reduction = (
        100 * (1 - risk["hedged_es"] / risk["unhedged_es"])
        if risk["unhedged_es"] > 0
        else 0
    )
    cards = [
        (
            "WITHHELD PRICE RMSE",
            f"{m['heldout_metrics']['projected_c']['rmse_price']:.4f}",
            "Projected surface / per share",
        ),
        (
            "INSIDE BID / ASK",
            f"{100*m['heldout_metrics']['projected_c']['inside_bid_ask_fraction']:.1f}%",
            "Withheld quote reconstruction",
        ),
        (
            "SIMULATED TAIL REDUCTION",
            f"{reduction:.1f}%",
            "97.5% expected shortfall / evaluation",
        ),
        (
            "HEDGE PREMIUM BUDGET",
            f"{risk['budget']:,.0f}",
            "Continuous quantities / multiplier 100",
        ),
    ]
    for i, (title, value, note) in enumerate(cards):
        x = 0.045 + i * 0.235
        box(x, 0.705, 0.215, 0.084)
        fig.text(x + 0.012, 0.768, title, fontsize=8, color=MUTED)
        fig.text(x + 0.012, 0.733, value, fontsize=24, color=VIOLET)
        fig.text(x + 0.012, 0.714, note, fontsize=7.8, color=MUTED)

    box(0.045, 0.384, 0.445, 0.292)
    box(0.51, 0.384, 0.445, 0.292)
    fig.text(0.06, 0.652, "01 / THE VOLATILITY LANDSCAPE", fontsize=10, weight="bold")
    fig.text(0.526, 0.652, "02 / A SMILE WITH UNCERTAINTY", fontsize=10, weight="bold")
    cmap = LinearSegmentedColormap.from_list(
        "aether", ["#242348", "#7263b1", "#c19cd8", "#ffd0b7"]
    )
    ax = fig.add_axes([0.07, 0.395, 0.40, 0.245], projection="3d", zorder=1)
    xx, tt = np.meshgrid(s["xs"], s["ts"] * 365)
    ax.plot_surface(
        xx,
        tt,
        s["projected_iv"] * 100,
        cmap=cmap,
        edgecolor="#675075",
        linewidth=0.35,
        antialiased=True,
    )
    ax.view_init(24, -53)
    ax.set_xlabel("Strike / forward", labelpad=5, fontsize=8)
    ax.set_ylabel("Days", labelpad=4, fontsize=8)
    ax.set_zlabel("IV (%)", labelpad=3, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(matplotlib.colors.to_rgba(CARD))
        axis._axinfo["grid"]["color"] = "#392e48"
    ax.set_box_aspect((1.45, 1, 0.7))
    fig.text(
        0.06,
        0.397,
        "Constraints apply at price-grid knots; no global guarantee.",
        fontsize=7.8,
        color=MUTED,
    )
    ax = fig.add_axes([0.559, 0.448, 0.372, 0.175], zorder=1)
    j = min(3, len(s["ts"]) - 1)
    ax.fill_between(
        s["xs"],
        s["low"][j] * 100,
        s["high"][j] * 100,
        color=VIOLET,
        alpha=0.19,
        label="Raw GP 95% band",
    )
    ax.plot(s["xs"], s["iv"][j] * 100, color=VIOLET, lw=1.5, label="GP")
    ax.plot(s["xs"], s["projected_iv"][j] * 100, color=TEAL, lw=1.5, label="Projected")
    held = quotes[(np.isclose(quotes.t, s["ts"][j])) & quotes.holdout]
    ax.scatter(
        held.x, held.iv * 100, color=CORAL, s=18, marker="D", zorder=4, label="Withheld"
    )
    ax.set_xlabel(f"Strike / forward  ·  {s['ts'][j]*365:.0f}-day expiry", fontsize=8)
    ax.set_ylabel("IV (%)", fontsize=8)
    ax.grid(color="#3a3048", alpha=0.5)
    ax.legend(loc="upper right", fontsize=6.7, frameon=False, labelcolor=MUTED)
    fig.text(
        0.526,
        0.397,
        "Raw predictive band; projection uncertainty is not propagated.",
        fontsize=7.8,
        color=MUTED,
    )

    for i in range(3):
        box(0.045 + i * 0.31, 0.08, 0.29, 0.266)
    for i, title in enumerate(
        [
            "03 / THE SHAPE OF THE TAIL",
            "04 / SPOT × IV STRESS",
            "05 / THE PRICE OF PROTECTION",
        ]
    ):
        fig.text(0.06 + i * 0.31, 0.323, title, fontsize=9.5, weight="bold")
    ax = fig.add_axes([0.082, 0.129, 0.229, 0.164], zorder=1)
    lo = min(r["unhedged"].min(), r["hedged"].min())
    hi = max(r["unhedged"].max(), r["hedged"].max())
    bins = np.linspace(lo, hi, 75)
    ax.hist(
        r["unhedged"], bins=bins, density=True, color=CORAL, alpha=0.7, label="Unhedged"
    )
    ax.hist(r["hedged"], bins=bins, density=True, color=TEAL, alpha=0.7, label="Hedged")
    ax.set_xlabel("Five-day P&L", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7, frameon=False, labelcolor=MUTED)
    ax.tick_params(labelsize=7)
    fig.text(
        0.06,
        0.091,
        f"{risk['test_scenarios']:,} independent, assumed-distribution scenarios.",
        fontsize=7.5,
        color=MUTED,
    )

    moves = np.linspace(-0.16, 0.16, 55)
    shocks = np.linspace(-0.06, 0.14, 45)
    mx, vy = np.meshgrid(moves, shocks)
    base, _, _ = reprice(
        s, positions["portfolio"], m["spot"], m["rate"], m["dividend"], mx, vy
    )
    hedge, _, _ = reprice(
        s, positions["hedges"], m["spot"], m["rate"], m["dividend"], mx, vy
    )
    pnl = (
        base.sum(axis=-1) + (hedge - risk["hedge_friction_per_contract"]) @ r["weights"]
    )
    ax = fig.add_axes([0.392, 0.129, 0.215, 0.164], zorder=1)
    heatmap = LinearSegmentedColormap.from_list("risk", [CORAL, "#2b203d", TEAL])
    norm = TwoSlopeNorm(
        vmin=min(-1, float(pnl.min())), vcenter=0, vmax=max(1, float(pnl.max()))
    )
    im = ax.pcolormesh(
        moves * 100, shocks * 100, pnl, shading="auto", cmap=heatmap, norm=norm
    )
    ax.contour(
        moves * 100, shocks * 100, pnl, levels=[0], colors=["#e5ddef"], linewidths=0.7
    )
    ax.set_xlabel("Spot move (%)", fontsize=8)
    ax.set_ylabel("IV shock (vol points)", fontsize=8)
    ax.tick_params(labelsize=7)
    cax = fig.add_axes([0.615, 0.129, 0.005, 0.164], zorder=1)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    fig.text(
        0.37,
        0.091,
        "Hedged P&L · full repricing under parallel IV shocks.",
        fontsize=7.5,
        color=MUTED,
    )

    ax = fig.add_axes([0.704, 0.129, 0.23, 0.164], zorder=1)
    for key, label, color in [
        ("training_es", "Optimization", VIOLET),
        ("test_es", "Evaluation", TEAL),
    ]:
        ax.plot(
            [x["spent"] for x in frontier],
            [x[key] for x in frontier],
            color=color,
            marker="o",
            markersize=4,
            lw=1.5,
            label=label,
        )
    ax.set_xlabel("Premium + friction", fontsize=8)
    ax.set_ylabel("97.5% expected shortfall", fontsize=8)
    ax.grid(color="#3a3048", alpha=0.5)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, labelcolor=MUTED)
    fig.text(
        0.68,
        0.091,
        "Budget sensitivity; evaluation does not select the budget.",
        fontsize=7.5,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.043,
        "AETHER QUANT  /  REPRODUCIBLE OPTIONS RESEARCH",
        fontsize=8,
        color=VIOLET,
    )
    fig.text(
        0.955,
        0.043,
        "Simulated diagnostics, not investment performance.  ·  Seed " + str(m["seed"]),
        fontsize=8,
        color=MUTED,
        ha="right",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out.resolve())


if __name__ == "__main__":
    main()
