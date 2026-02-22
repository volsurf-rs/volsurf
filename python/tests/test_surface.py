import math

import pytest
from volsurf import (
    DupireLocalVol,
    EssviSurface,
    SmileModel,
    SsviSurface,
    SurfaceBuilder,
)

SSVI_EQUITY = dict(
    rho=-0.3, eta=0.5, gamma=0.5,
    tenors=[0.25, 0.5, 1.0, 2.0],
    forwards=[100.0, 100.0, 100.0, 100.0],
    thetas=[0.04, 0.08, 0.16, 0.32],
)

ESSVI_EQUITY = dict(
    rho_0=-0.4, rho_m=-0.2, a=0.5, eta=0.5, gamma=0.5,
    tenors=[0.25, 0.5, 1.0, 2.0],
    forwards=[100.0, 100.0, 100.0, 100.0],
    thetas=[0.04, 0.08, 0.16, 0.32],
)

STRIKES = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
VOLS = [0.28, 0.24, 0.22, 0.20, 0.22, 0.24, 0.28]


def build_sabr_surface():
    b = SurfaceBuilder()
    b.model(SmileModel.sabr(0.5))
    b.spot(100.0)
    b.rate(0.05)
    b.add_tenor(0.25, STRIKES, VOLS)
    b.add_tenor(1.0, STRIKES, VOLS)
    return b.build()


class TestSurfaceBuilder:
    def test_build_single_tenor_svi(self):
        b = SurfaceBuilder()
        b.spot(100.0)
        b.rate(0.05)
        b.add_tenor(0.25, STRIKES, VOLS)
        surf = b.build()
        v = surf.black_vol(0.25, 100.0)
        assert v > 0 and math.isfinite(v)

    def test_build_multi_tenor_svi(self):
        b = SurfaceBuilder()
        b.spot(100.0)
        b.rate(0.05)
        b.add_tenor(0.25, STRIKES, VOLS)
        b.add_tenor(1.0, STRIKES, VOLS)
        surf = b.build()
        assert surf.black_vol(0.25, 100.0) > 0
        assert surf.black_vol(1.0, 100.0) > 0

    def test_build_with_sabr(self):
        surf = build_sabr_surface()
        v = surf.black_vol(0.25, 100.0)
        assert v > 0 and math.isfinite(v)

    def test_build_with_spline(self):
        b = SurfaceBuilder()
        b.model(SmileModel.cubic_spline())
        b.spot(100.0)
        b.rate(0.05)
        b.add_tenor(0.5, [80.0, 100.0, 120.0], [0.25, 0.20, 0.25])
        surf = b.build()
        assert surf.black_vol(0.5, 100.0) > 0

    def test_vol_variance_consistent(self):
        surf = build_sabr_surface()
        t, k = 0.5, 100.0
        v = surf.black_vol(t, k)
        var = surf.black_variance(t, k)
        assert abs(var - v * v * t) < 1e-12

    def test_smile_at_returns_queryable(self):
        surf = build_sabr_surface()
        smile = surf.smile_at(0.25)
        assert smile.vol(100.0) > 0
        assert smile.forward > 0
        assert smile.expiry > 0

    def test_query_between_tenors(self):
        surf = build_sabr_surface()
        v = surf.black_vol(0.5, 100.0)
        assert v > 0 and math.isfinite(v)

    def test_diagnostics(self):
        surf = build_sabr_surface()
        diag = surf.diagnostics()
        assert isinstance(diag.is_free, bool)
        assert isinstance(diag.smile_reports, list)
        assert isinstance(diag.calendar_violations, list)

    def test_missing_spot(self):
        b = SurfaceBuilder()
        b.rate(0.05)
        b.add_tenor(0.25, STRIKES, VOLS)
        with pytest.raises((ValueError, RuntimeError)):
            b.build()

    def test_missing_rate(self):
        b = SurfaceBuilder()
        b.spot(100.0)
        b.add_tenor(0.25, STRIKES, VOLS)
        with pytest.raises((ValueError, RuntimeError)):
            b.build()

    def test_no_tenors(self):
        b = SurfaceBuilder()
        b.spot(100.0)
        b.rate(0.05)
        with pytest.raises((ValueError, RuntimeError)):
            b.build()

    def test_add_tenor_with_forward(self):
        b = SurfaceBuilder()
        b.model(SmileModel.sabr(0.5))
        b.spot(100.0)
        b.rate(0.05)
        b.add_tenor_with_forward(0.25, STRIKES, VOLS, 101.25)
        surf = b.build()
        assert surf.black_vol(0.25, 100.0) > 0

    def test_dividend_yield(self):
        b = SurfaceBuilder()
        b.model(SmileModel.sabr(0.5))
        b.spot(100.0)
        b.rate(0.05)
        b.dividend_yield(0.02)
        b.add_tenor(0.25, STRIKES, VOLS)
        surf = b.build()
        assert surf.black_vol(0.25, 100.0) > 0


class TestSsviSurface:
    def test_construct(self):
        s = SsviSurface(**SSVI_EQUITY)
        assert s.rho == -0.3
        assert s.eta == 0.5
        assert s.gamma == 0.5

    def test_accessors(self):
        s = SsviSurface(**SSVI_EQUITY)
        assert s.tenors() == [0.25, 0.5, 1.0, 2.0]
        assert s.forwards() == [100.0, 100.0, 100.0, 100.0]
        assert s.thetas() == [0.04, 0.08, 0.16, 0.32]

    def test_black_vol_atm(self):
        s = SsviSurface(**SSVI_EQUITY)
        v = s.black_vol(1.0, 100.0)
        assert v > 0 and math.isfinite(v)

    def test_black_variance_consistent(self):
        s = SsviSurface(**SSVI_EQUITY)
        t, k = 0.5, 100.0
        v = s.black_vol(t, k)
        var = s.black_variance(t, k)
        assert abs(var - v * v * t) < 1e-12

    def test_smile_at(self):
        s = SsviSurface(**SSVI_EQUITY)
        smile = s.smile_at(0.5)
        assert smile.vol(100.0) > 0

    def test_diagnostics(self):
        s = SsviSurface(**SSVI_EQUITY)
        diag = s.diagnostics()
        assert isinstance(diag.is_free, bool)

    def test_rejects_rho_at_boundary(self):
        with pytest.raises(ValueError):
            SsviSurface(1.0, 0.5, 0.5, [0.25], [100.0], [0.04])

    def test_rejects_zero_eta(self):
        with pytest.raises(ValueError):
            SsviSurface(-0.3, 0.0, 0.5, [0.25], [100.0], [0.04])

    def test_rejects_gamma_out_of_range(self):
        with pytest.raises(ValueError):
            SsviSurface(-0.3, 0.5, 1.5, [0.25], [100.0], [0.04])

    def test_rejects_non_increasing_tenors(self):
        with pytest.raises(ValueError):
            SsviSurface(-0.3, 0.5, 0.5, [1.0, 0.5], [100.0, 100.0], [0.16, 0.08])

    def test_rejects_empty_tenors(self):
        with pytest.raises(ValueError):
            SsviSurface(-0.3, 0.5, 0.5, [], [], [])


class TestEssviSurface:
    def test_construct(self):
        s = EssviSurface(**ESSVI_EQUITY)
        assert s.rho_0 == -0.4
        assert s.rho_m == -0.2
        assert s.a == 0.5

    def test_black_vol_atm(self):
        s = EssviSurface(**ESSVI_EQUITY)
        v = s.black_vol(1.0, 100.0)
        assert v > 0 and math.isfinite(v)

    def test_diagnostics(self):
        s = EssviSurface(**ESSVI_EQUITY)
        diag = s.diagnostics()
        assert isinstance(diag.is_free, bool)

    def test_rejects_rho_0_at_boundary(self):
        with pytest.raises(ValueError):
            EssviSurface(1.0, -0.2, 0.5, 0.5, 0.5, [0.25], [100.0], [0.04])


class TestDupireLocalVol:
    def test_local_vol_positive(self):
        surf = build_sabr_surface()
        dupire = DupireLocalVol(surf)
        lv = dupire.local_vol(0.5, 100.0)
        assert lv > 0 and math.isfinite(lv)

    def test_rejects_zero_expiry(self):
        surf = build_sabr_surface()
        dupire = DupireLocalVol(surf)
        with pytest.raises((ValueError, RuntimeError)):
            dupire.local_vol(0.0, 100.0)

    def test_rejects_zero_strike(self):
        surf = build_sabr_surface()
        dupire = DupireLocalVol(surf)
        with pytest.raises((ValueError, RuntimeError)):
            dupire.local_vol(0.5, 0.0)
