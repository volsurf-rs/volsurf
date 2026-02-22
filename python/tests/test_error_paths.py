import json
import math

import numpy as np
import pytest
from conftest import ESSVI_EQUITY, SSVI_EQUITY
from volsurf import (
    DisplacedImpliedVol,
    EssviSurface,
    NormalImpliedVol,
    OptionType,
    SabrSmile,
    SmileModel,
    SplineSmile,
    SsviSurface,
    SurfaceBuilder,
    SviSmile,
    displaced_price,
    moneyness,
    normal_price,
)


class TestNormalPriceErrors:
    def test_rejects_negative_vol(self):
        with pytest.raises(ValueError):
            normal_price(100.0, 100.0, -1.0, 1.0, OptionType.Call)

    def test_rejects_negative_expiry(self):
        with pytest.raises(ValueError):
            normal_price(100.0, 100.0, 20.0, -1.0, OptionType.Call)


class TestDisplacedPriceErrors:
    def test_rejects_beta_above_one(self):
        with pytest.raises(ValueError):
            displaced_price(100.0, 100.0, 0.2, 1.0, 1.1, OptionType.Call)

    def test_rejects_beta_below_zero(self):
        with pytest.raises(ValueError):
            displaced_price(100.0, 100.0, 0.2, 1.0, -0.1, OptionType.Call)

    def test_rejects_zero_forward(self):
        with pytest.raises(ValueError):
            displaced_price(0.0, 100.0, 0.2, 1.0, 0.5, OptionType.Call)


class TestNormalImpliedVolErrors:
    def test_rejects_negative_price(self):
        with pytest.raises((ValueError, RuntimeError)):
            NormalImpliedVol.compute(-1.0, 100.0, 100.0, 1.0, OptionType.Call)

    def test_rejects_zero_expiry(self):
        with pytest.raises((ValueError, RuntimeError)):
            NormalImpliedVol.compute(5.0, 100.0, 100.0, 0.0, OptionType.Call)


class TestDisplacedImpliedVolErrors:
    def test_rejects_negative_price(self):
        calc = DisplacedImpliedVol(0.5)
        with pytest.raises((ValueError, RuntimeError)):
            calc.compute(-1.0, 100.0, 100.0, 1.0, OptionType.Call)

    def test_beta_zero_round_trip(self):
        f, k, t, sigma = 100.0, 100.0, 1.0, 0.2
        price = displaced_price(f, k, sigma, t, 0.0, OptionType.Call)
        calc = DisplacedImpliedVol(0.0)
        iv = calc.compute(price, f, k, t, OptionType.Call)
        assert abs(iv - sigma) < 1e-10


class TestMoneynessErrors:
    def test_zero_strike(self):
        with pytest.raises(ValueError):
            moneyness(0.0, 100.0)

    def test_zero_forward(self):
        with pytest.raises(ValueError):
            moneyness(100.0, 0.0)


class TestSurfaceVolGridError:
    def test_invalid_strike(self):
        s = SsviSurface(**SSVI_EQUITY)
        with pytest.raises(ValueError):
            s.vol_grid(np.array([0.5]), np.array([0.0, 100.0]))


class TestSerdeErrors:
    def test_sabr_from_json_invalid_params(self):
        bad = json.dumps({
            "forward": 100.0, "expiry": 1.0,
            "alpha": -0.1, "beta": 0.5, "rho": -0.3, "nu": 0.3,
        })
        with pytest.raises(ValueError):
            SabrSmile.from_json(bad)

    def test_spline_from_json_invalid_params(self):
        bad = json.dumps({
            "forward": 100.0, "expiry": 1.0,
            "strikes": [80.0, 100.0, 120.0], "variances": [0.04, -0.01, 0.04],
        })
        with pytest.raises(ValueError):
            SplineSmile.from_json(bad)

    def test_ssvi_from_json_invalid_params(self):
        bad = json.dumps({
            "rho": 1.0, "eta": 0.5, "gamma": 0.5,
            "tenors": [0.25], "forwards": [100.0], "thetas": [0.04],
        })
        with pytest.raises(ValueError):
            SsviSurface.from_json(bad)

    def test_essvi_from_json_invalid_params(self):
        bad = json.dumps({
            "rho_0": 1.0, "rho_m": -0.2, "a": 0.5, "eta": 0.5, "gamma": 0.5,
            "theta_max": 0.32, "tenors": [0.25], "forwards": [100.0], "thetas": [0.04],
        })
        with pytest.raises(ValueError):
            EssviSurface.from_json(bad)

    def test_malformed_json_all_types(self):
        for cls in [SviSmile, SabrSmile, SplineSmile, SsviSurface, EssviSurface]:
            with pytest.raises(ValueError):
                cls.from_json("not json at all")

    def test_empty_json_object(self):
        for cls in [SviSmile, SabrSmile, SplineSmile, SsviSurface, EssviSurface]:
            with pytest.raises(ValueError):
                cls.from_json("{}")
