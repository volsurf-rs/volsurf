import math

import numpy as np
import pytest
from conftest import SSVI_EQUITY
from volsurf import SsviSurface


class TestOpaqueSmile:
    """Tests for PySmile returned by surface.smile_at() (Box<dyn SmileSection>)."""

    def test_variance_consistent(self):
        s = SsviSurface(**SSVI_EQUITY)
        smile = s.smile_at(0.5)
        v = smile.vol(100.0)
        var = smile.variance(100.0)
        assert abs(var - v * v * 0.5) < 1e-14

    def test_density_atm_positive(self):
        s = SsviSurface(**SSVI_EQUITY)
        smile = s.smile_at(1.0)
        d = smile.density(100.0)
        assert d > 0 and math.isfinite(d)

    def test_is_arbitrage_free(self):
        s = SsviSurface(**SSVI_EQUITY)
        smile = s.smile_at(1.0)
        report = smile.is_arbitrage_free()
        assert hasattr(report, "is_free")
        assert isinstance(report.is_free, bool)
        assert hasattr(report, "butterfly_violations")

    def test_vol_array_consistency(self):
        s = SsviSurface(**SSVI_EQUITY)
        smile = s.smile_at(0.5)
        strikes = np.array([80.0, 100.0, 120.0])
        va = smile.vol_array(strikes)
        assert va.shape == (3,)
        for j, k in enumerate(strikes):
            assert abs(va[j] - smile.vol(float(k))) < 1e-14

    def test_vol_array_invalid_strike(self):
        s = SsviSurface(**SSVI_EQUITY)
        smile = s.smile_at(0.5)
        with pytest.raises(ValueError):
            smile.vol_array(np.array([0.0, 100.0]))

    def test_vol_array_negative_strike(self):
        s = SsviSurface(**SSVI_EQUITY)
        smile = s.smile_at(0.5)
        with pytest.raises(ValueError):
            smile.vol_array(np.array([-10.0, 100.0]))
