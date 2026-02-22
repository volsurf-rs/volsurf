import math

from conftest import ESSVI_EQUITY
from volsurf import EssviSurface


class TestEssviAccessors:
    def test_eta(self):
        s = EssviSurface(**ESSVI_EQUITY)
        assert s.eta == 0.5

    def test_gamma(self):
        s = EssviSurface(**ESSVI_EQUITY)
        assert s.gamma == 0.5

    def test_theta_max(self):
        s = EssviSurface(**ESSVI_EQUITY)
        assert s.theta_max == max(ESSVI_EQUITY["thetas"])

    def test_tenors(self):
        s = EssviSurface(**ESSVI_EQUITY)
        assert s.tenors() == ESSVI_EQUITY["tenors"]

    def test_forwards(self):
        s = EssviSurface(**ESSVI_EQUITY)
        assert s.forwards() == ESSVI_EQUITY["forwards"]

    def test_thetas(self):
        s = EssviSurface(**ESSVI_EQUITY)
        assert s.thetas() == ESSVI_EQUITY["thetas"]

    def test_black_variance_consistent(self):
        s = EssviSurface(**ESSVI_EQUITY)
        t, k = 0.5, 100.0
        v = s.black_vol(t, k)
        var = s.black_variance(t, k)
        assert abs(var - v * v * t) < 1e-12

    def test_smile_at_queryable(self):
        s = EssviSurface(**ESSVI_EQUITY)
        smile = s.smile_at(0.5)
        v = smile.vol(100.0)
        assert v > 0 and math.isfinite(v)

    def test_smile_at_variance_consistent(self):
        s = EssviSurface(**ESSVI_EQUITY)
        smile = s.smile_at(0.5)
        v = smile.vol(100.0)
        var = smile.variance(100.0)
        assert abs(var - v * v * 0.5) < 1e-14

    def test_smile_at_density_positive(self):
        s = EssviSurface(**ESSVI_EQUITY)
        smile = s.smile_at(1.0)
        d = smile.density(100.0)
        assert d > 0 and math.isfinite(d)

    def test_smile_at_arb_report(self):
        s = EssviSurface(**ESSVI_EQUITY)
        smile = s.smile_at(1.0)
        report = smile.is_arbitrage_free()
        assert isinstance(report.is_free, bool)
        assert isinstance(report.butterfly_violations, list)
