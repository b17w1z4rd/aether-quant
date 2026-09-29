import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, eye
from .pricing import price


def default_portfolio():
    return [
        {
            "label": "Short 90d call",
            "kind": "call",
            "strike_ratio": 1.05,
            "days": 90,
            "quantity": -2,
        },
        {
            "label": "Short 90d put",
            "kind": "put",
            "strike_ratio": 0.95,
            "days": 90,
            "quantity": -2,
        },
        {
            "label": "Long 180d call",
            "kind": "call",
            "strike_ratio": 1.0,
            "days": 180,
            "quantity": 1,
        },
    ]


def hedge_instruments():
    return [
        {
            "label": "60d ATM put",
            "kind": "put",
            "strike_ratio": 1.0,
            "days": 60,
            "quantity": 1,
        },
        {
            "label": "60d ATM call",
            "kind": "call",
            "strike_ratio": 1.0,
            "days": 60,
            "quantity": 1,
        },
        {
            "label": "90d 95% put",
            "kind": "put",
            "strike_ratio": 0.95,
            "days": 90,
            "quantity": 1,
        },
    ]


def scenarios(n, seed, horizon=5):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(6, size=n) * np.sqrt(4 / 6)
    jump = np.where(rng.random(n) < 0.015, rng.normal(-0.04, 0.05, n), 0.0)
    moves = (
        np.exp(
            -0.5 * 0.25**2 * horizon / 365 + 0.25 * np.sqrt(horizon / 365) * z + jump
        )
        - 1
    )
    # Negative spot/vol correlation is an assumption, not a learned market estimate.
    vol_shock = -0.55 * moves + rng.normal(0, 0.018, n)
    return moves, vol_shock


def reprice(
    surface, legs, spot, rate, dividend, moves, vol_shocks, horizon=5, multiplier=100
):
    moves, vol_shocks = np.broadcast_arrays(
        np.asarray(moves, float), np.asarray(vol_shocks, float)
    )
    if (
        np.any(moves <= -1)
        or not np.isfinite(moves).all()
        or not np.isfinite(vol_shocks).all()
    ):
        raise ValueError("Invalid scenario")
    interp = RegularGridInterpolator(
        (surface["ts"], surface["xs"]), surface["projected_iv"], bounds_error=True
    )
    ts, xs = surface["ts"], surface["xs"]
    pnl = []
    premiums = []
    clamped = np.zeros(moves.shape, dtype=bool)
    h = horizon / 365
    for leg in legs:
        if (
            not np.isfinite([leg["days"], leg["strike_ratio"], leg["quantity"]]).all()
            or leg["strike_ratio"] <= 0
        ):
            raise ValueError(
                "Instrument fields must be finite, with positive strike_ratio"
            )
        if leg["kind"] not in ("call", "put") or leg["days"] <= horizon:
            raise ValueError("Invalid instrument/expiry")
        strike = spot * float(leg["strike_ratio"])
        t = leg["days"] / 365
        x0 = strike / (spot * np.exp((rate - dividend) * t))
        if not (ts[0] <= t <= ts[-1] and xs[0] <= x0 <= xs[-1]):
            raise ValueError("Portfolio entry outside calibrated surface domain")
        iv0 = float(interp([[t, x0]])[0])
        entry = (
            float(price(spot, strike, t, rate, iv0, leg["kind"], dividend)) * multiplier
        )
        remain = t - h
        newspot = spot * (1 + moves)
        newx = strike / (newspot * np.exp((rate - dividend) * remain))
        outside = (
            (newx < xs[0]) | (newx > xs[-1]) | (remain < ts[0]) | (remain > ts[-1])
        )
        clamped |= outside
        points = np.column_stack(
            [
                np.full(moves.size, np.clip(remain, ts[0], ts[-1])),
                np.clip(newx.ravel(), xs[0], xs[-1]),
            ]
        )
        iv = np.maximum(0.01, interp(points).reshape(moves.shape) + vol_shocks)
        future = (
            price(newspot, strike, remain, rate, iv, leg["kind"], dividend) * multiplier
        )
        pnl.append((future - entry * np.exp(rate * h)) * leg["quantity"])
        premiums.append(entry * leg["quantity"])
    return np.stack(pnl, axis=-1), np.array(premiums), clamped


def empirical_es(loss, alpha=0.975):
    loss = np.asarray(loss, float)
    if (
        loss.ndim != 1
        or len(loss) == 0
        or not np.isfinite(loss).all()
        or not 0 < alpha < 1
    ):
        raise ValueError("Invalid loss sample")
    eta = float(np.quantile(loss, alpha, method="inverted_cdf"))
    return float(eta + np.maximum(loss - eta, 0).mean() / (1 - alpha))


def optimize(base, hedges, premiums, budget, alpha=0.975, max_contracts=5.0):
    base = np.asarray(base)
    hedges = np.asarray(hedges)
    premiums = np.asarray(premiums)
    n, m = hedges.shape
    if (
        n == 0
        or n != len(base)
        or len(premiums) != m
        or not np.isfinite(budget)
        or budget < 0
        or not 0 < alpha < 1
        or max_contracts <= 0
        or np.any(premiums <= 0)
        or not all(np.isfinite(a).all() for a in (base, hedges, premiums))
    ):
        raise ValueError("Invalid hedge inputs")
    # x = [hedge quantities, VaR auxiliary eta, one positive-part slack per scenario]
    objective = np.r_[np.zeros(m), 1.0, np.full(n, 1 / ((1 - alpha) * n))]
    rows = hstack(
        [-csr_matrix(hedges), -csr_matrix(np.ones((n, 1))), -eye(n, format="csr")],
        format="csr",
    )
    cap = csr_matrix(np.r_[premiums, 0.0, np.zeros(n)][None, :])
    result = linprog(
        objective,
        A_ub=vstack([rows, cap], format="csr"),
        b_ub=np.r_[base, budget],
        bounds=[(0, max_contracts)] * m + [(None, None)] + [(0, None)] * n,
        method="highs",
    )
    if not result.success:
        raise RuntimeError("Hedge optimizer failed: " + result.message)
    return result.x[:m], {
        "success": True,
        "training_objective_es": float(result.fun),
        "premium_spent": float(premiums @ result.x[:m]),
        "budget": float(budget),
        "alpha": alpha,
        "max_contracts_per_leg": max_contracts,
    }


def experiment(
    surface,
    spot,
    rate,
    dividend,
    portfolio,
    seed=42,
    budget=600.0,
    ntrain=1500,
    ntest=4000,
    hedge_legs=None,
):
    hedge_legs = hedge_instruments() if hedge_legs is None else hedge_legs
    if not portfolio or not hedge_legs or any(h["quantity"] != 1 for h in hedge_legs):
        raise ValueError(
            "Need portfolio and hedge instruments; hedge quantity must be 1"
        )
    train_moves, train_vol = scenarios(ntrain, seed)
    base, _, clamp_train = reprice(
        surface, portfolio, spot, rate, dividend, train_moves, train_vol
    )
    hedge, premiums, clamp_h = reprice(
        surface, hedge_legs, spot, rate, dividend, train_moves, train_vol
    )
    # Flat execution friction per unit hedge quantity, charged against scenario P&L.
    friction = 1.0
    hedge = hedge - friction
    weights, audit = optimize(base.sum(axis=1), hedge, premiums + friction, budget)
    moves, vol = scenarios(ntest, seed + 1009)
    legpnl, base_premiums, clamp = reprice(
        surface, portfolio, spot, rate, dividend, moves, vol
    )
    hedgepnl, _, clamph = reprice(surface, hedge_legs, spot, rate, dividend, moves, vol)
    unhedged = legpnl.sum(axis=1)
    hedged = unhedged + (hedgepnl - friction) @ weights
    frontier = []
    for cap in np.linspace(0, budget, 5):
        fw, fa = optimize(base.sum(axis=1), hedge, premiums + friction, float(cap))
        fp = unhedged + (hedgepnl - friction) @ fw
        frontier.append(
            {
                "budget": float(cap),
                "spent": fa["premium_spent"],
                "training_es": fa["training_objective_es"],
                "test_es": empirical_es(-fp),
                "test_mean_pnl": float(fp.mean()),
            }
        )
    summary = {
        "unhedged_es": empirical_es(-unhedged),
        "hedged_es": empirical_es(-hedged),
        "unhedged_mean_pnl": float(unhedged.mean()),
        "hedged_mean_pnl": float(hedged.mean()),
        "training_scenarios": ntrain,
        "test_scenarios": ntest,
        "training_seed": seed,
        "test_seed": seed + 1009,
        "test_clamped_scenarios": int((clamp | clamph).sum()),
        "training_clamped_scenarios": int((clamp_train | clamp_h).sum()),
        "horizon_calendar_days": 5,
        "contract_multiplier": 100,
        "hedge_friction_per_contract": friction,
        **audit,
    }
    return {
        "frontier": frontier,
        "weights": weights,
        "hedge_legs": hedge_legs,
        "premiums": premiums,
        "base_premiums": base_premiums,
        "portfolio": portfolio,
        "moves": moves,
        "vol_shocks": vol,
        "unhedged": unhedged,
        "hedged": hedged,
        "summary": summary,
    }
