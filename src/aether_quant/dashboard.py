import html
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from .risk import reprice, empirical_es

VIOLET = "#b7a0ff"
TEAL = "#64e3d0"
CORAL = "#ff9f87"
TEXT = "#e9e7f2"
MUTED = "#9b99b1"
BG = "#151321"
SCALE = [[0, "#171b35"], [0.35, "#6262b2"], [0.7, "#bb94df"], [1, "#ffcab3"]]


def build(q, s, r, spot, rate, dividend, path, synthetic=True):
    charts = []
    xs = s["xs"]
    days = s["ts"] * 365
    test = s["test"]
    summary = r["summary"]

    def panel(f, title, note, wide=False, height=370):
        f.update_layout(
            title=dict(text=title, font=dict(size=15)),
            height=height,
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(family="Arial, sans-serif", color=TEXT, size=11),
            margin=dict(l=60, r=35, t=75, b=50),
            colorway=[VIOLET, TEAL, CORAL],
            legend=dict(
                orientation="h", y=1.08, x=1, xanchor="right", font=dict(size=10)
            ),
        )
        f.update_xaxes(gridcolor="#28263b", zeroline=False)
        f.update_yaxes(gridcolor="#28263b", zeroline=False)
        charts.append((f, title, note, wide))

    f = go.Figure(
        go.Surface(
            x=xs,
            y=days,
            z=s["projected_iv"] * 100,
            colorscale=SCALE,
            colorbar=dict(title="IV %", len=0.65),
        )
    )
    panel(
        f,
        "01 / The learned volatility landscape",
        "Rotate the surface. Projection constrains normalized call prices on the calibration grid; it is not a global arbitrage-free volatility certificate.",
        True,
        560,
    )
    f.update_layout(
        scene=dict(
            xaxis_title="Strike / forward",
            yaxis_title="Days",
            zaxis_title="IV (%)",
            bgcolor=BG,
            camera=dict(eye=dict(x=1.55, y=-1.7, z=1.1)),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0,
                y=1.08,
                bgcolor="#29243f",
                font=dict(color=TEXT),
                buttons=[
                    dict(label=label, method="restyle", args=[{"z": [z]}])
                    for label, z in [
                        ("Constrained", s["projected_iv"] * 100),
                        ("Raw GP", s["iv"] * 100),
                        ("Predictive width", (s["high"] - s["low"]) * 100),
                    ]
                ],
            )
        ],
    )
    f = go.Figure()
    j = min(3, len(days) - 1)
    for d in range(len(days)):
        show = d == j
        rows = q[np.isclose(q.t, s["ts"][d])]
        f.add_trace(
            go.Scatter(
                x=xs,
                y=s["high"][d] * 100,
                line=dict(width=0),
                showlegend=False,
                visible=show,
            )
        )
        f.add_trace(
            go.Scatter(
                x=xs,
                y=s["low"][d] * 100,
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(183,160,255,.15)",
                name="GP 95% band",
                visible=show,
            )
        )
        f.add_trace(
            go.Scatter(
                x=xs,
                y=s["iv"][d] * 100,
                line=dict(color=VIOLET),
                name="Raw GP",
                visible=show,
            )
        )
        f.add_trace(
            go.Scatter(
                x=xs,
                y=s["projected_iv"][d] * 100,
                line=dict(color=TEAL),
                name="Projected",
                visible=show,
            )
        )
        f.add_trace(
            go.Scatter(
                x=rows.x,
                y=rows.iv * 100,
                mode="markers",
                marker=dict(
                    color=np.where(rows.holdout, CORAL, "#f0eafb"),
                    symbol=np.where(rows.holdout, "diamond", "circle"),
                    size=7,
                ),
                name="Quotes (orange=holdout)",
                visible=show,
            )
        )
    steps = [
        dict(
            label=f"{d:.0f}d",
            method="update",
            args=[{"visible": [i // 5 == k for i in range(5 * len(days))]}],
        )
        for k, d in enumerate(days)
    ]
    panel(
        f,
        "02 / The smile, with model uncertainty",
        "Use the expiry slider. Bands are from the raw GP model; uncertainty has not been propagated through the constrained projection.",
        True,
        450,
    )
    f.update_layout(
        sliders=[
            dict(
                active=j,
                steps=steps,
                currentvalue=dict(prefix="Expiry: "),
                pad=dict(t=25),
            )
        ]
    )
    f.update_xaxes(title="Strike / forward")
    f.update_yaxes(title="Implied vol (%)")
    f = go.Figure(
        go.Heatmap(
            x=xs,
            y=days,
            z=(s["clean"] - s["raw"]) * spot * np.exp(-dividend * s["ts"][:, None]),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Δ value"),
        )
    )
    panel(
        f,
        "03 / The cost of consistency",
        "Projected minus raw call price per share. Changes arise from joint strike and calendar constraints.",
    )
    f.update_xaxes(title="Strike / forward")
    f.update_yaxes(title="Days")
    labels = ["Call increasing", "Digital bounds", "Convexity", "Calendar"]
    keys = [
        "increasing_call_violation",
        "digital_bound_violation",
        "convexity_violation",
        "calendar_violation",
    ]
    f = go.Figure()
    for source, name, color in [("before", "Before", CORAL), ("after", "After", TEAL)]:
        f.add_trace(
            go.Bar(
                x=labels, y=[s[source][k] for k in keys], name=name, marker_color=color
            )
        )
    panel(
        f,
        "04 / Static checks on the finite grid",
        "Different diagnostic units share the panel. Numerical tolerance is 1e−6; all values and bounds are retained in metrics.json.",
    )
    f = go.Figure()
    for col, label, color in [
        ("gp_c", "GP", VIOLET),
        ("projected_c", "Projected", TEAL),
        ("svi_c", "SVI", CORAL),
    ]:
        f.add_trace(
            go.Scatter(
                x=test.mid,
                y=test[col] * test.scale,
                mode="markers",
                name=label,
                marker=dict(color=color, size=6, opacity=0.7),
            )
        )
    lim = float(test.mid.max())
    f.add_trace(
        go.Scatter(
            x=[0, lim],
            y=[0, lim],
            mode="lines",
            line=dict(color=MUTED, dash="dot"),
            showlegend=False,
        )
    )
    panel(
        f,
        "05 / Withheld quote reconstruction",
        "These strikes were excluded from model fitting. This measures cross-sectional interpolation, not tomorrow’s price prediction.",
    )
    f.update_xaxes(title="Observed midpoint")
    f.update_yaxes(title="Fitted midpoint")
    f = go.Figure(
        go.Bar(
            x=["GP", "Projected", "SVI"],
            y=[s["metrics"][k]["rmse_price"] for k in ["gp_c", "projected_c", "svi_c"]],
            marker_color=[VIOLET, TEAL, CORAL],
        )
    )
    panel(
        f,
        "06 / A conventional benchmark",
        "Price RMSE against withheld midquotes. SVI parameters use training quotes from each expiry, without an arbitrage constraint.",
    )
    f.update_yaxes(title="Price RMSE / share")
    slopes = np.diff(s["clean"], axis=1) / np.diff(xs)
    density = np.diff(slopes, axis=1) / ((xs[2:] - xs[:-2]) / 2)
    f = go.Figure(
        go.Heatmap(
            x=xs[1:-1],
            y=days,
            z=density,
            colorscale=SCALE,
            colorbar=dict(title="Density"),
        )
    )
    panel(
        f,
        "07 / Implied state-price geometry",
        "Finite-difference density proxy for normalized forward returns. The plotted strike window excludes tails and is not normalized to unit mass.",
    )
    f.update_xaxes(title="Strike / forward")
    f.update_yaxes(title="Days")
    f = go.Figure()
    for k in sorted({0, len(days) // 2, len(days) - 1}):
        f.add_trace(
            go.Scatter(
                x=(xs[:-1] + xs[1:]) / 2, y=-slopes[k], name=f"{days[k]:.0f} days"
            )
        )
    panel(
        f,
        "08 / Digital probability proxy",
        "Negative call-price slopes approximate risk-neutral tail probabilities on each cell; they are not physical return probabilities.",
    )
    f.update_xaxes(title="Strike / forward")
    f.update_yaxes(title="−∂ normalized call / ∂x")
    f = go.Figure()
    for key, label, color in [
        ("unhedged", "Before hedge", CORAL),
        ("hedged", "After hedge", TEAL),
    ]:
        f.add_trace(
            go.Histogram(
                x=r[key],
                nbinsx=85,
                name=label,
                marker_color=color,
                opacity=0.65,
                histnorm="probability density",
            )
        )
    panel(
        f,
        "09 / The shape of the tail",
        f"{summary['test_scenarios']:,} independent evaluation scenarios; {summary['test_clamped_scenarios']} require surface-domain clamping. Student-t spot shocks, occasional jumps and correlated IV shocks are explicit assumptions.",
        True,
        400,
    )
    f.update_layout(barmode="overlay")
    f.update_xaxes(title="Five-day portfolio P&L")
    f.update_yaxes(title="Density")
    alphas = [0.90, 0.95, 0.975, 0.99]
    f = go.Figure()
    for key, label, color in [
        ("unhedged", "Before hedge", CORAL),
        ("hedged", "After hedge", TEAL),
    ]:
        f.add_trace(
            go.Scatter(
                x=np.array(alphas) * 100,
                y=[empirical_es(-r[key], a) for a in alphas],
                name=label,
                mode="lines+markers",
                line=dict(color=color),
            )
        )
    panel(
        f,
        "10 / Expected shortfall by confidence",
        "Tail losses measured on evaluation scenarios. The hedge was optimized at 97.5%; these are model-dependent simulated losses.",
    )
    f.update_xaxes(title="Confidence (%)")
    f.update_yaxes(title="Expected shortfall")
    f = go.Figure(
        go.Bar(
            x=[h["label"] for h in r["hedge_legs"]],
            y=r["weights"],
            marker_color=[VIOLET, TEAL, CORAL],
        )
    )
    panel(
        f,
        "11 / What the optimizer chose",
        "Long-only quantities under the premium budget and per-leg cap. Continuous quantities are a research relaxation; contracts have not been rounded.",
    )
    f.update_yaxes(title="Hedge contracts")
    moves = np.linspace(-0.16, 0.16, 45)
    vol = np.linspace(-0.06, 0.14, 41)
    xx, yy = np.meshgrid(moves, vol)
    base, _, clamp = reprice(s, r["portfolio"], spot, rate, dividend, xx, yy)
    h, _, clamp_h = reprice(s, r["hedge_legs"], spot, rate, dividend, xx, yy)
    before = base.sum(axis=-1)
    after = before + (h - summary["hedge_friction_per_contract"]) @ r["weights"]
    f = go.Figure(
        go.Heatmap(
            x=moves * 100,
            y=vol * 100,
            z=after,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="P&L"),
        )
    )
    panel(
        f,
        "12 / Joint spot × volatility stress",
        f"Toggle the hedge to inspect the same stress grid. Full repricing uses sticky forward-moneyness IV and a parallel IV shock. Domain clamping affects {int((clamp|clamp_h).sum())} of {xx.size} stress points.",
        True,
        470,
    )
    f.update_xaxes(title="Spot move (%)")
    f.update_yaxes(title="IV shock (vol points)")
    f.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0,
                y=1.1,
                bgcolor="#29243f",
                font=dict(color=TEXT),
                buttons=[
                    dict(label="Hedged", method="restyle", args=[{"z": [after]}]),
                    dict(label="Unhedged", method="restyle", args=[{"z": [before]}]),
                ],
            )
        ]
    )
    f = go.Figure()
    for key, label, color in [
        ("training_es", "Optimization sample", VIOLET),
        ("test_es", "Evaluation sample", TEAL),
    ]:
        f.add_trace(
            go.Scatter(
                x=[p["spent"] for p in r["frontier"]],
                y=[p[key] for p in r["frontier"]],
                mode="lines+markers",
                name=label,
                line=dict(color=color),
            )
        )
    panel(
        f,
        "13 / Premium versus tail risk",
        "Each point reoptimizes on the training scenarios. The evaluation curve is a sensitivity analysis, not a rule for selecting a budget.",
    )
    f.update_xaxes(title="Premium + modeled friction")
    f.update_yaxes(title="97.5% expected shortfall")
    f = go.Figure(
        go.Scatter(
            x=r["moves"] * 100,
            y=r["vol_shocks"] * 100,
            mode="markers",
            marker=dict(
                size=4,
                color=r["hedged"],
                colorscale="RdBu",
                cmid=0,
                opacity=0.6,
                colorbar=dict(title="P&L"),
            ),
        )
    )
    panel(
        f,
        "14 / Scenario dependence",
        "A negative spot/vol relationship is imposed by the simulator. It is not estimated from historical observations.",
    )
    f.update_xaxes(title="Spot move (%)")
    f.update_yaxes(title="IV shock (vol points)")
    f = go.Figure()
    for held, label, color in [(False, "Training", VIOLET), (True, "Withheld", CORAL)]:
        rows = q[q.holdout == held]
        f.add_trace(
            go.Scatter(
                x=rows.x,
                y=rows.days,
                mode="markers",
                name=label,
                marker=dict(
                    color=color, size=8, symbol="diamond" if held else "circle"
                ),
            )
        )
    panel(
        f,
        "15 / Where the model has evidence",
        "Training and withheld quotes from one synchronized option-chain snapshot. No longitudinal validation is claimed.",
        True,
        310,
    )
    f.update_xaxes(title="Strike / forward")
    f.update_yaxes(title="Days to expiry")
    fragments = []
    for i, (fig, title, note, wide) in enumerate(charts):
        fragments.append(
            '<article class="panel '
            + ("wide" if wide else "")
            + '">'
            + fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id=f"chart-{i}",
                config={"displaylogo": False, "responsive": True},
            )
            + "<p>"
            + html.escape(note)
            + "</p></article>"
        )
    badge = "SYNTHETIC OPTION CHAIN" if synthetic else "IMPORTED OPTION CHAIN"
    reduced = (
        (1 - summary["hedged_es"] / summary["unhedged_es"]) * 100
        if summary["unhedged_es"] > 0
        else 0
    )
    page = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Aether Quant · The geometry of uncertainty</title><style>
*{box-sizing:border-box}html{color-scheme:dark}body{margin:0;background:#0d0b15;color:#e9e7f2;font-family:Arial,sans-serif}header,main,footer{max-width:1440px;margin:auto;padding:30px 36px}nav{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #342b45;padding-bottom:24px}.brand{font-weight:800;font-size:21px;letter-spacing:4px}.badge{font-size:10px;letter-spacing:1px;color:#ffb59e;border:1px solid #704c54;padding:9px 14px;border-radius:30px}.hero{padding:48px 0 24px;display:grid;grid-template-columns:1.8fr 1fr;gap:45px}.eyebrow{color:#b7a0ff;font-size:11px;letter-spacing:3px}h1{font-size:clamp(40px,5vw,75px);line-height:1.04;letter-spacing:-3px;font-weight:500;margin:22px 0}h1 em{color:#b7a0ff;font-style:normal}.lead{color:#aaa5bb;font-size:15px;line-height:1.8;max-width:660px}.aside{align-self:end;border:1px solid #413452;border-radius:17px;padding:25px;background:linear-gradient(145deg,#231930,#12111e);font-size:12px;color:#aaa5bb;line-height:1.9}.aside strong{color:#e9e7f2;display:block;letter-spacing:1px;font-size:11px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:22px}.metric{border-top:2px solid #b7a0ff;background:#191523;border-radius:0 0 13px 13px;padding:21px}.metric label{color:#a9a0b8;font-size:10px;letter-spacing:1.6px}.metric b{display:block;font-size:29px;font-weight:500;color:#d3c3ff;margin:14px 0 9px}.metric small{font-size:11px;color:#a29caf}.section{display:flex;justify-content:space-between;gap:20px;padding:18px 0 26px;font-size:12px;color:#a29caf}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:23px}.panel{border:1px solid #30273e;border-radius:17px;background:#151321;overflow:hidden;min-width:0}.wide{grid-column:1/-1}.panel p{margin:0;padding:0 25px 23px;font-size:12px;line-height:1.8;color:#a39caf}footer{font-size:12px;line-height:1.9;color:#aaa2b9;border-top:1px solid #33293e;margin-top:40px;padding-bottom:50px}@media(max-width:780px){header,main,footer{padding:22px 15px}.hero{grid-template-columns:1fr;padding-top:28px}.metrics{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.wide{grid-column:auto}.section{flex-direction:column}.badge{font-size:8px}h1{letter-spacing:-2px}}
</style><script>"""
        + get_plotlyjs()
        + """</script></head><body>"""
    )
    page += (
        f"""<header><nav><div class="brand">AETHER <span style="color:#b7a0ff">/</span> QUANT</div><div class="badge">{badge}</div></nav><div class="hero"><div><div class="eyebrow">PROBABILISTIC LEARNING × CONSTRAINED FINANCE</div><h1>The geometry<br>of <em>uncertainty.</em></h1><p class="lead">Learn an option surface. Repair its price geometry. Measure the tails. Then ask what a finite hedge budget can actually change.</p></div><div class="aside"><strong>ONE RESEARCH SYSTEM · THREE LAYERS</strong>Gaussian-process surface + SVI benchmark<br>Finite-grid static-consistency projection<br>Scenario expected-shortfall hedge optimization<br><br>{len(s['train'])} training quotes / {len(test)} held out<br>{summary['training_scenarios']:,} optimization / {summary['test_scenarios']:,} evaluation scenarios</div></div><div class="metrics">
<div class="metric"><label>WITHHELD PRICE RMSE</label><b>{s['metrics']['projected_c']['rmse_price']:.4f}</b><small>Projected surface · per share</small></div><div class="metric"><label>INSIDE BID / ASK</label><b>{s['metrics']['projected_c']['inside_bid_ask_fraction']*100:.1f}%</b><small>Withheld midpoint reconstruction</small></div><div class="metric"><label>SIMULATED TAIL REDUCTION</label><b>{reduced:.1f}%</b><small>97.5% expected shortfall · test scenarios</small></div><div class="metric"><label>HEDGE PREMIUM BUDGET</label><b>{summary['budget']:,.0f}</b><small>Continuous quantities · multiplier 100</small></div></div></header><main><div class="section"><span>THE RESEARCH ATLAS / 15 interactive views</span><span>Slider-controlled smiles · surface rotation · hedge toggles</span></div><div class="grid">"""
        + "".join(fragments)
        + """</div></main><footer><strong>Research scope.</strong> The demo is simulated and does not establish investment performance. Static constraints hold only on the reported normalized-price grid; IV interpolation, extrapolation and stressed repricing carry additional model risk. Gaussian-process intervals are model assumptions, not coverage guarantees. Scenario risk depends on an assumed joint return distribution. Hedging uses continuous long-only quantities with simplified friction; executable prices, integer contracts, liquidity, taxes and margin are not modeled. No orders are sent.</footer></body></html>"""
    )
    Path(path).write_text(page)
