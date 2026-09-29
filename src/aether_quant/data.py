import numpy as np
import pandas as pd
from .pricing import price, implied_vol


def demo_chain(seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    spot = 100.0
    rate = 0.03
    for days in (21, 35, 60, 90, 150, 240, 365):
        t = days / 365
        f = spot * np.exp(rate * t)
        for x in np.linspace(0.72, 1.30, 25):
            k = np.log(x)
            iv = 0.20 + 0.065 * np.exp(-3 * t) - 0.17 * k + 0.27 * k * k
            true = float(price(spot, x * f, t, rate, iv))
            noise = rng.normal(0, 0.006 + 0.025 * np.exp(-10 * abs(k)))
            mid = max(max(spot - x * f * np.exp(-rate * t), 0) + 1e-5, true + noise)
            half = 0.015 + 0.035 * abs(k) + 0.015 * np.sqrt(t)
            bid = max(max(spot - x * f * np.exp(-rate * t), 0) + 1e-6, mid - half)
            ask = max(bid + 0.001, mid + half)
            rows.append(dict(days=days, strike=x * f, bid=bid, ask=ask))
    return pd.DataFrame(rows)


def prepare(chain, spot, rate, dividend=0.0):
    if not np.isfinite([spot, rate, dividend]).all() or spot <= 0:
        raise ValueError("Invalid spot/carry")
    columns = ["days", "strike", "bid", "ask"]
    if not set(columns).issubset(chain.columns):
        raise ValueError("CSV requires days,strike,bid,ask for European calls")
    q = chain[columns].apply(pd.to_numeric, errors="raise").copy()
    if (
        not np.isfinite(q.to_numpy()).all()
        or (q[["days", "strike", "bid"]] <= 0).any().any()
        or (q.ask < q.bid).any()
    ):
        raise ValueError("Quotes must be positive, finite and uncrossed")
    if q.duplicated(["days", "strike"]).any():
        raise ValueError("Duplicate strike/expiry quote")
    q = q.sort_values(["days", "strike"]).reset_index(drop=True)
    q["t"] = q.days / 365
    q["forward"] = spot * np.exp((rate - dividend) * q.t)
    q["scale"] = q.forward * np.exp(-rate * q.t)
    q["x"] = q.strike / q.forward
    q["mid"] = (q.bid + q.ask) / 2
    q["c"] = q.mid / q.scale
    q["iv"] = [implied_vol(x, t, c) for x, t, c in zip(q.x, q.t, q.c)]
    q["logw"] = np.log(np.maximum(q.iv, 1e-5) ** 2 * q.t)
    q["holdout"] = False
    for _, part in q.groupby("days"):
        if len(part) < 12:
            raise ValueError("At least 12 strikes per expiry are required")
        # Interior interleaved holdout: interpolation test, never a time-series forecast.
        ids = part.index.to_numpy()[2:-1:4]
        q.loc[ids, "holdout"] = True
    if q.days.nunique() < 3:
        raise ValueError("At least 3 expiries are required")
    return q
