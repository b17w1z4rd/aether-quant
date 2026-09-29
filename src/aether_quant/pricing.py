import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq


def call_normalized(x, t, vol):
    x, t, vol = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(t, float), np.asarray(vol, float)
    )
    if (
        not all(np.isfinite(a).all() for a in (x, t, vol))
        or np.any(x <= 0)
        or np.any(t <= 0)
        or np.any(vol <= 0)
    ):
        raise ValueError("Positive finite moneyness, time and volatility required")
    s = vol * np.sqrt(t)
    d1 = -np.log(x) / s + s / 2
    d2 = d1 - s
    return ndtr(d1) - x * ndtr(d2)


def implied_vol(x, t, c):
    if not np.isfinite([x, t, c]).all() or x <= 0 or t <= 0:
        raise ValueError("Invalid implied-volatility inputs")
    lower = max(1 - x, 0)
    if c < lower - 1e-10 or c >= 1:
        raise ValueError("Call price outside normalized bounds")
    if c <= lower + 1e-12:
        return 1e-5
    if call_normalized(x, t, 8.0) < c:
        raise ValueError("Implied volatility exceeds solver bracket")
    return brentq(lambda v: float(call_normalized(x, t, v)) - c, 1e-5, 8.0, xtol=1e-10)


def price(spot, strike, t, rate, vol, kind="call", dividend=0.0):
    if kind not in ("call", "put"):
        raise ValueError("kind must be call or put")
    forward = np.asarray(spot) * np.exp((rate - dividend) * np.asarray(t))
    discount = np.exp(-rate * np.asarray(t))
    call = forward * discount * call_normalized(np.asarray(strike) / forward, t, vol)
    return call if kind == "call" else call - discount * (forward - strike)
