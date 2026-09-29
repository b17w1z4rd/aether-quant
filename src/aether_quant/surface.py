import warnings
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds, least_squares
from scipy.interpolate import RegularGridInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from .pricing import call_normalized, implied_vol


def constraints(xs, ts):
    n, m = len(ts), len(xs)
    rows = []
    lower = []
    upper = []

    def add(entries, lo, hi):
        a = np.zeros(n * m)
        for index, value in entries:
            a[index] = value
        rows.append(a)
        lower.append(lo)
        upper.append(hi)

    dx = np.diff(xs)
    for j in range(n):
        for i in range(m - 1):
            add([(j * m + i, -1), (j * m + i + 1, 1)], -dx[i], 0)
        for i in range(m - 2):
            add(
                [
                    (j * m + i, 1 / dx[i]),
                    (j * m + i + 1, -1 / dx[i] - 1 / dx[i + 1]),
                    (j * m + i + 2, 1 / dx[i + 1]),
                ],
                0,
                np.inf,
            )
    for j in range(n - 1):
        for i in range(m):
            add([(j * m + i, -1), ((j + 1) * m + i, 1)], 0, np.inf)
    return np.stack(rows), np.array(lower), np.array(upper)


def diagnostics(c, xs):
    slopes = np.diff(c, axis=1) / np.diff(xs)[None, :]
    return {
        "lower_bound_violation": float(
            max(0, np.max(np.maximum(1 - xs, 0)[None, :] - c))
        ),
        "upper_bound_violation": float(max(0, np.max(c - 1))),
        "increasing_call_violation": float(max(0, np.max(slopes))),
        "digital_bound_violation": float(max(0, np.max(-1 - slopes))),
        "convexity_violation": float(max(0, -np.min(np.diff(slopes, axis=1)))),
        "calendar_violation": float(max(0, -np.min(np.diff(c, axis=0)))),
    }


def project(raw, xs, ts):
    raw = np.asarray(raw, float)
    if raw.shape != (len(ts), len(xs)) or not np.isfinite(raw).all():
        raise ValueError("Invalid price grid")
    matrix, lo, hi = constraints(xs, ts)
    lower = np.tile(np.maximum(1 - xs, 0), len(ts))
    upper = np.ones(raw.size)
    # Feasible initial surface at a constant volatility.
    x, t = np.meshgrid(xs, ts)
    start = call_normalized(x, t, 0.30).ravel()
    target = raw.ravel()
    result = minimize(
        lambda z: 0.5 * np.sum((z - target) ** 2),
        start,
        jac=lambda z: z - target,
        bounds=Bounds(lower, upper),
        constraints=[LinearConstraint(matrix, lo, hi)],
        method="SLSQP",
        options={"maxiter": 400, "ftol": 1e-11},
    )
    out = result.x.reshape(raw.shape)
    if not result.success or max(diagnostics(out, xs).values()) > 1e-6:
        raise RuntimeError(
            "Surface projection failed feasibility checks: " + result.message
        )
    return out, {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "squared_adjustment": float(np.sum((out - raw) ** 2)),
        "max_adjustment": float(np.max(abs(out - raw))),
    }


def svi(k, p):
    a, b, rho, m, s = p
    v = k - m
    return a + b * (rho * v + np.sqrt(v * v + s * s))


def fit(q, seed=42):
    train = q[~q.holdout].copy()
    test = q[q.holdout].copy()
    x = np.column_stack([np.log(train.x), np.log(train.t)])
    kernel = ConstantKernel(1.0, (0.01, 100)) * RBF(
        [0.3, 1.0], (0.03, 5.0)
    ) + WhiteKernel(0.01, (1e-5, 0.5))
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=0, random_state=seed
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp.fit(x, train.logw)
    # Common forward-moneyness range supported by all expiry slices.
    xmin = float(q.groupby("days").x.min().max())
    xmax = float(q.groupby("days").x.max().min())
    if xmax - xmin < 0.1:
        raise ValueError("Need a shared moneyness range of at least 0.1")
    xs = np.linspace(xmin, xmax, 31)
    ts = np.sort(q.t.unique())
    xx, tt = np.meshgrid(xs, ts)
    features = np.column_stack([np.log(xx.ravel()), np.log(tt.ravel())])
    mean, std = gp.predict(features, return_std=True)
    iv = np.sqrt(np.exp(mean.reshape(xx.shape)) / tt)
    low = np.sqrt(np.exp((mean - 1.96 * std).reshape(xx.shape)) / tt)
    high = np.sqrt(np.exp((mean + 1.96 * std).reshape(xx.shape)) / tt)
    raw = call_normalized(xx, tt, iv)
    clean, audit = project(raw, xs, ts)
    projected_iv = np.array(
        [implied_vol(x, t, c) for x, t, c in zip(xx.ravel(), tt.ravel(), clean.ravel())]
    ).reshape(raw.shape)
    interp = RegularGridInterpolator((ts, xs), clean, bounds_error=True)
    test = test[(test.x >= xmin) & (test.x <= xmax)].copy()
    if len(test) == 0:
        raise ValueError("No withheld quotes inside the shared calibration domain")
    mt, st = gp.predict(
        np.column_stack([np.log(test.x), np.log(test.t)]), return_std=True
    )
    test["gp_iv"] = np.sqrt(np.exp(mt) / test.t)
    test["gp_c"] = call_normalized(test.x, test.t, test.gp_iv)
    test["projected_c"] = interp(np.column_stack([test.t, test.x]))
    test["svi_c"] = np.nan
    svi_parameters = {}
    svi_audit = {}
    for t, rows in train.groupby("t"):
        k = np.log(rows.x.to_numpy())
        w = rows.iv.to_numpy() ** 2 * t

        # Parameterize the minimum variance, allowing a to be negative while
        # maintaining positive total variance everywhere. Choose restarts on
        # training residuals only, never held-out reconstruction scores.
        def raw_parameters(p):
            wmin, b, rho, m, s = p
            return np.array([wmin - b * s * np.sqrt(1 - rho * rho), b, rho, m, s])

        candidates = [
            least_squares(
                lambda p: svi(k, raw_parameters(p)) - w,
                [max(1e-7, float(w.min()) / 2), 0.1, -0.3, start, 0.15],
                bounds=([1e-8, 1e-8, -0.99, -0.5, 0.005], [1.0, 3.0, 0.99, 0.5, 2.0]),
                max_nfev=2000,
            )
            for start in (-0.1, 0.0, 0.1)
        ]
        result = min(candidates, key=lambda candidate: candidate.cost)
        parameters = raw_parameters(result.x)
        sel = np.isclose(test.t, t)
        predicted = np.sqrt(
            np.maximum(svi(np.log(test.loc[sel, "x"]), parameters), 1e-10) / t
        )
        test.loc[sel, "svi_c"] = call_normalized(test.loc[sel, "x"], t, predicted)
        svi_parameters[str(t)] = parameters.tolist()
        svi_audit[str(t)] = {
            "converged": bool(result.success),
            "cost": float(result.cost),
            "training_restarts": 3,
        }
    metrics = {}
    for column in ("gp_c", "projected_c", "svi_c"):
        error = (test[column] - test.c) * test.scale
        inside = (test[column] * test.scale >= test.bid) & (
            test[column] * test.scale <= test.ask
        )
        metrics[column] = {
            "rmse_price": float(np.sqrt(np.mean(error**2))),
            "mae_price": float(np.mean(abs(error))),
            "inside_bid_ask_fraction": float(inside.mean()),
        }
    return {
        "xs": xs,
        "ts": ts,
        "raw": raw,
        "clean": clean,
        "iv": iv,
        "projected_iv": projected_iv,
        "low": low,
        "high": high,
        "test": test,
        "train": train,
        "metrics": metrics,
        "audit": audit,
        "before": diagnostics(raw, xs),
        "after": diagnostics(clean, xs),
        "kernel": str(gp.kernel_),
        "warnings": [str(w.message) for w in caught],
        "svi_parameters": svi_parameters,
        "svi_audit": svi_audit,
    }
