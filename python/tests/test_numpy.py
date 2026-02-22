import numpy as np
import pytest
from conftest import ESSVI_EQUITY, SSVI_EQUITY
from volsurf import (
    EssviSurface,
    SabrSmile,
    SmileModel,
    SplineSmile,
    SsviSurface,
    SurfaceBuilder,
    SviSmile,
)

EXPIRIES = np.array([0.25, 0.5, 1.0])
STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0])


class TestSsviVolGrid:
    def test_shape(self):
        s = SsviSurface(**SSVI_EQUITY)
        vols = s.vol_grid(EXPIRIES, STRIKES)
        assert vols.shape == (3, 5)

    def test_dtype(self):
        s = SsviSurface(**SSVI_EQUITY)
        vols = s.vol_grid(EXPIRIES, STRIKES)
        assert vols.dtype == np.float64

    def test_scalar_consistency(self):
        s = SsviSurface(**SSVI_EQUITY)
        vols = s.vol_grid(EXPIRIES, STRIKES)
        for i, t in enumerate(EXPIRIES):
            for j, k in enumerate(STRIKES):
                assert abs(vols[i, j] - s.black_vol(float(t), float(k))) < 1e-14

    def test_all_positive_finite(self):
        s = SsviSurface(**SSVI_EQUITY)
        vols = s.vol_grid(EXPIRIES, STRIKES)
        assert np.all(vols > 0)
        assert np.all(np.isfinite(vols))

    def test_single_expiry(self):
        s = SsviSurface(**SSVI_EQUITY)
        vols = s.vol_grid(np.array([0.5]), STRIKES)
        assert vols.shape == (1, 5)

    def test_single_strike(self):
        s = SsviSurface(**SSVI_EQUITY)
        vols = s.vol_grid(EXPIRIES, np.array([100.0]))
        assert vols.shape == (3, 1)


class TestEssviVolGrid:
    def test_shape_and_consistency(self):
        s = EssviSurface(**ESSVI_EQUITY)
        vols = s.vol_grid(EXPIRIES, STRIKES)
        assert vols.shape == (3, 5)
        for i, t in enumerate(EXPIRIES):
            for j, k in enumerate(STRIKES):
                assert abs(vols[i, j] - s.black_vol(float(t), float(k))) < 1e-14


class TestSurfaceVolGrid:
    def test_builder_surface(self):
        b = SurfaceBuilder()
        b.model(SmileModel.sabr(0.5))
        b.spot(100.0)
        b.rate(0.05)
        b.add_tenor(0.25, [80.0, 90.0, 100.0, 110.0, 120.0],
                     [0.28, 0.24, 0.20, 0.22, 0.26])
        b.add_tenor(1.0, [80.0, 90.0, 100.0, 110.0, 120.0],
                     [0.28, 0.24, 0.20, 0.22, 0.26])
        surf = b.build()
        vols = surf.vol_grid(np.array([0.25, 1.0]), np.array([90.0, 100.0, 110.0]))
        assert vols.shape == (2, 3)
        for i, t in enumerate([0.25, 1.0]):
            for j, k in enumerate([90.0, 100.0, 110.0]):
                assert abs(vols[i, j] - surf.black_vol(t, k)) < 1e-14


class TestSmileVolArray:
    def test_opaque_smile(self):
        s = SsviSurface(**SSVI_EQUITY)
        smile = s.smile_at(0.5)
        va = smile.vol_array(STRIKES)
        assert va.shape == (5,)
        assert va.dtype == np.float64
        for j, k in enumerate(STRIKES):
            assert abs(va[j] - smile.vol(float(k))) < 1e-14

    def test_svi_smile(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1)
        va = smile.vol_array(STRIKES)
        assert va.shape == (5,)
        for j, k in enumerate(STRIKES):
            assert abs(va[j] - smile.vol(float(k))) < 1e-14

    def test_sabr_smile(self):
        smile = SabrSmile(100.0, 1.0, 0.2, 0.5, -0.3, 0.3)
        va = smile.vol_array(STRIKES)
        assert va.shape == (5,)
        for j, k in enumerate(STRIKES):
            assert abs(va[j] - smile.vol(float(k))) < 1e-14

    def test_spline_smile(self):
        smile = SplineSmile(
            100.0, 1.0,
            [80.0, 90.0, 100.0, 110.0, 120.0],
            [0.065, 0.045, 0.04, 0.045, 0.065],
        )
        va = smile.vol_array(STRIKES)
        assert va.shape == (5,)
        for j, k in enumerate(STRIKES):
            assert abs(va[j] - smile.vol(float(k))) < 1e-14

    def test_all_positive_finite(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1)
        va = smile.vol_array(STRIKES)
        assert np.all(va > 0)
        assert np.all(np.isfinite(va))
