import unittest
import numpy as np
from aether_quant.pricing import call_normalized, implied_vol, price
from aether_quant.surface import project, diagnostics, fit
from aether_quant.data import demo_chain, prepare
from aether_quant.risk import empirical_es, optimize, scenarios


class PricingTests(unittest.TestCase):
    def test_implied_vol_roundtrip(self):
        for x in (0.9, 1.0, 1.1):
            c = float(call_normalized(x, 0.5, 0.25))
            self.assertAlmostEqual(implied_vol(x, 0.5, c), 0.25, places=7)

    def test_put_call_parity(self):
        c = price(100, 105, 0.4, 0.03, 0.2, "call", 0.01)
        p = price(100, 105, 0.4, 0.03, 0.2, "put", 0.01)
        self.assertAlmostEqual(
            float(c - p), 100 * np.exp(-0.01 * 0.4) - 105 * np.exp(-0.03 * 0.4)
        )

    def test_bad_quote_rejected(self):
        with self.assertRaises(ValueError):
            implied_vol(1.0, 0.5, 1.2)


class SurfaceTests(unittest.TestCase):
    def test_projection_repairs_injected_violations(self):
        x = np.linspace(0.8, 1.2, 9)
        t = np.array([0.1, 0.3, 0.7])
        xx, tt = np.meshgrid(x, t)
        raw = call_normalized(xx, tt, 0.25)
        raw[0, 4] += 0.035
        raw[1, 5] = raw[0, 5] - 0.01
        self.assertGreater(max(diagnostics(raw, x).values()), 0.001)
        repaired, audit = project(raw, x, t)
        self.assertTrue(audit["success"])
        self.assertLess(max(diagnostics(repaired, x).values()), 1e-6)

    def test_feasible_surface_stays_close(self):
        x = np.linspace(0.8, 1.2, 9)
        t = np.array([0.1, 0.3, 0.7])
        xx, tt = np.meshgrid(x, t)
        raw = call_normalized(xx, tt, 0.30)
        clean, _ = project(raw, x, t)
        np.testing.assert_allclose(raw, clean, atol=1e-7)

    def test_holdout_is_interior_and_disjoint(self):
        q = prepare(demo_chain(), 100, 0.03)
        for _, part in q.groupby("days"):
            self.assertFalse(part.holdout.iloc[0])
            self.assertFalse(part.holdout.iloc[-1])
            self.assertTrue(part.holdout.any())

    def test_holdout_targets_do_not_change_fitted_surface(self):
        q = prepare(demo_chain(), 100, 0.03)
        a = fit(q)
        changed = q.copy()
        changed.loc[changed.holdout, "logw"] += 1
        changed.loc[changed.holdout, "c"] += 0.01
        b = fit(changed)
        np.testing.assert_allclose(a["raw"], b["raw"])
        np.testing.assert_allclose(a["clean"], b["clean"])

    def test_crossed_quotes_rejected(self):
        q = demo_chain()
        q.loc[0, "bid"] = q.loc[0, "ask"] + 1
        with self.assertRaises(ValueError):
            prepare(q, 100, 0.03)


class RiskTests(unittest.TestCase):
    def test_empirical_es_known_tail(self):
        self.assertAlmostEqual(empirical_es(np.arange(100), 0.95), 97.0)

    def test_empirical_es_fractional_tail_weight(self):
        self.assertAlmostEqual(empirical_es(np.arange(4), 0.2), 1.875)

    def test_zero_budget_has_no_hedges(self):
        b = np.linspace(-10, 10, 50)
        h = np.column_stack([-b, b])
        w, _ = optimize(b, h, np.array([2.0, 3.0]), 0)
        np.testing.assert_allclose(w, 0, atol=1e-8)

    def test_budget_and_tail_improvement_on_training_set(self):
        b = np.linspace(-100, 100, 200)
        h = (-b)[:, None]
        w, a = optimize(b, h, np.array([10.0]), 10)
        self.assertLessEqual(10 * w[0], 10 + 1e-7)
        self.assertLess(empirical_es(-(b + h @ w)), empirical_es(-b))

    def test_scenario_seeds_are_independent(self):
        a, _ = scenarios(100, 42)
        b, _ = scenarios(100, 1051)
        self.assertFalse(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
