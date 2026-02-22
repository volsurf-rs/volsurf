import math

import pytest
from volsurf import (
    BlackImpliedVol,
    DisplacedImpliedVol,
    NormalImpliedVol,
    OptionType,
    black_price,
    displaced_price,
    forward_price,
    log_moneyness,
    moneyness,
    normal_price,
)


class TestBlackPrice:
    def test_atm_call_positive(self):
        p = black_price(100.0, 100.0, 0.20, 1.0, OptionType.Call)
        assert p > 0

    def test_put_call_parity(self):
        f, k, vol, t = 100.0, 90.0, 0.20, 1.0
        call = black_price(f, k, vol, t, OptionType.Call)
        put = black_price(f, k, vol, t, OptionType.Put)
        assert abs((call - put) - (f - k)) < 1e-12

    def test_zero_vol_call_intrinsic(self):
        p = black_price(100.0, 80.0, 0.0, 1.0, OptionType.Call)
        assert abs(p - 20.0) < 1e-12

    def test_zero_expiry_call_intrinsic(self):
        p = black_price(100.0, 80.0, 0.20, 0.0, OptionType.Call)
        assert abs(p - 20.0) < 1e-12

    def test_zero_vol_otm_call(self):
        p = black_price(100.0, 120.0, 0.0, 1.0, OptionType.Call)
        assert abs(p) < 1e-12

    def test_rejects_negative_vol(self):
        with pytest.raises(ValueError):
            black_price(100.0, 100.0, -0.01, 1.0, OptionType.Call)

    def test_rejects_zero_forward(self):
        with pytest.raises(ValueError):
            black_price(0.0, 100.0, 0.20, 1.0, OptionType.Call)

    def test_rejects_zero_strike(self):
        with pytest.raises(ValueError):
            black_price(100.0, 0.0, 0.20, 1.0, OptionType.Call)

    def test_rejects_negative_expiry(self):
        with pytest.raises(ValueError):
            black_price(100.0, 100.0, 0.20, -1.0, OptionType.Call)


class TestBlackImpliedVol:
    def test_round_trip_atm_call(self):
        f, k, t, sigma = 100.0, 100.0, 1.0, 0.20
        price = black_price(f, k, sigma, t, OptionType.Call)
        iv = BlackImpliedVol.compute(price, f, k, t, OptionType.Call)
        assert abs(iv - sigma) < 1e-12

    def test_round_trip_atm_put(self):
        f, k, t, sigma = 100.0, 100.0, 1.0, 0.20
        price = black_price(f, k, sigma, t, OptionType.Put)
        iv = BlackImpliedVol.compute(price, f, k, t, OptionType.Put)
        assert abs(iv - sigma) < 1e-12

    def test_round_trip_otm_call(self):
        f, k, t, sigma = 100.0, 120.0, 1.0, 0.25
        price = black_price(f, k, sigma, t, OptionType.Call)
        iv = BlackImpliedVol.compute(price, f, k, t, OptionType.Call)
        assert abs(iv - sigma) < 1e-12

    def test_round_trip_itm_call(self):
        f, k, t, sigma = 100.0, 80.0, 1.0, 0.25
        price = black_price(f, k, sigma, t, OptionType.Call)
        iv = BlackImpliedVol.compute(price, f, k, t, OptionType.Call)
        assert abs(iv - sigma) < 1e-12

    def test_rejects_negative_price(self):
        with pytest.raises(ValueError):
            BlackImpliedVol.compute(-1.0, 100.0, 100.0, 1.0, OptionType.Call)

    def test_rejects_zero_forward(self):
        with pytest.raises(ValueError):
            BlackImpliedVol.compute(5.0, 0.0, 100.0, 1.0, OptionType.Call)

    def test_rejects_zero_strike(self):
        with pytest.raises(ValueError):
            BlackImpliedVol.compute(5.0, 100.0, 0.0, 1.0, OptionType.Call)

    def test_rejects_zero_expiry(self):
        with pytest.raises(ValueError):
            BlackImpliedVol.compute(5.0, 100.0, 100.0, 0.0, OptionType.Call)

    def test_rejects_price_above_forward(self):
        with pytest.raises(RuntimeError):
            BlackImpliedVol.compute(150.0, 100.0, 100.0, 1.0, OptionType.Call)


class TestNormalPrice:
    def test_put_call_parity(self):
        f, k, vol, t = 100.0, 90.0, 20.0, 1.0
        call = normal_price(f, k, vol, t, OptionType.Call)
        put = normal_price(f, k, vol, t, OptionType.Put)
        assert abs((call - put) - (f - k)) < 1e-12

    def test_atm_formula(self):
        f, k, sigma, t = 100.0, 100.0, 20.0, 1.0
        expected = sigma * math.sqrt(t) / math.sqrt(2.0 * math.pi)
        actual = normal_price(f, k, sigma, t, OptionType.Call)
        assert abs(actual - expected) < 1e-10

    def test_zero_vol_intrinsic(self):
        p = normal_price(100.0, 80.0, 0.0, 1.0, OptionType.Call)
        assert abs(p - 20.0) < 1e-12

    def test_negative_forward_allowed(self):
        call = normal_price(-1.0, -0.5, 0.5, 1.0, OptionType.Call)
        put = normal_price(-1.0, -0.5, 0.5, 1.0, OptionType.Put)
        assert abs((call - put) - (-1.0 - (-0.5))) < 1e-12


class TestNormalImpliedVol:
    def test_round_trip_atm_call(self):
        f, k, t, sigma = 100.0, 100.0, 1.0, 20.0
        price = normal_price(f, k, sigma, t, OptionType.Call)
        iv = NormalImpliedVol.compute(price, f, k, t, OptionType.Call)
        assert abs(iv - sigma) < 1e-10

    def test_round_trip_atm_put(self):
        f, k, t, sigma = 100.0, 100.0, 1.0, 20.0
        price = normal_price(f, k, sigma, t, OptionType.Put)
        iv = NormalImpliedVol.compute(price, f, k, t, OptionType.Put)
        assert abs(iv - sigma) < 1e-10


class TestDisplacedPrice:
    def test_beta_one_equals_black(self):
        f, k, vol, t = 100.0, 110.0, 0.20, 1.0
        bp = black_price(f, k, vol, t, OptionType.Call)
        dp = displaced_price(f, k, vol, t, 1.0, OptionType.Call)
        assert abs(bp - dp) < 1e-12

    def test_put_call_parity(self):
        f, k, vol, t, beta = 100.0, 90.0, 0.20, 1.0, 0.5
        call = displaced_price(f, k, vol, t, beta, OptionType.Call)
        put = displaced_price(f, k, vol, t, beta, OptionType.Put)
        assert abs((call - put) - (f - k)) < 1e-10


class TestDisplacedImpliedVol:
    def test_construct_valid(self):
        calc = DisplacedImpliedVol(0.5)
        assert calc.beta == 0.5

    def test_round_trip_beta_half(self):
        f, k, t, sigma, beta = 100.0, 100.0, 1.0, 0.20, 0.5
        price = displaced_price(f, k, sigma, t, beta, OptionType.Call)
        calc = DisplacedImpliedVol(beta)
        iv = calc.compute(price, f, k, t, OptionType.Call)
        assert abs(iv - sigma) < 1e-10

    def test_round_trip_beta_one(self):
        f, k, t, sigma = 100.0, 110.0, 1.0, 0.25
        price = black_price(f, k, sigma, t, OptionType.Call)
        calc = DisplacedImpliedVol(1.0)
        iv = calc.compute(price, f, k, t, OptionType.Call)
        assert abs(iv - sigma) < 1e-12

    def test_rejects_beta_above_one(self):
        with pytest.raises(ValueError):
            DisplacedImpliedVol(1.1)

    def test_rejects_beta_below_zero(self):
        with pytest.raises(ValueError):
            DisplacedImpliedVol(-0.1)


class TestUtilities:
    def test_log_moneyness_atm(self):
        assert abs(log_moneyness(100.0, 100.0)) < 1e-14

    def test_log_moneyness_otm(self):
        k = log_moneyness(110.0, 100.0)
        assert abs(k - math.log(1.1)) < 1e-14

    def test_moneyness_atm(self):
        assert abs(moneyness(100.0, 100.0) - 1.0) < 1e-14

    def test_moneyness_ratio(self):
        assert abs(moneyness(80.0, 100.0) - 0.8) < 1e-14

    def test_forward_price_known_value(self):
        f = forward_price(100.0, 0.05, 0.0, 1.0)
        assert abs(f - 100.0 * math.exp(0.05)) < 1e-10

    def test_forward_price_with_dividend(self):
        f = forward_price(100.0, 0.05, 0.02, 1.0)
        assert abs(f - 100.0 * math.exp(0.03)) < 1e-10

    def test_forward_price_zero_rate(self):
        f = forward_price(100.0, 0.0, 0.0, 1.0)
        assert abs(f - 100.0) < 1e-14

    def test_log_moneyness_zero_forward(self):
        with pytest.raises(ValueError):
            log_moneyness(100.0, 0.0)

    def test_log_moneyness_zero_strike(self):
        with pytest.raises(ValueError):
            log_moneyness(0.0, 100.0)

    def test_forward_price_zero_spot(self):
        with pytest.raises(ValueError):
            forward_price(0.0, 0.05, 0.0, 1.0)

    def test_forward_price_negative_expiry(self):
        with pytest.raises(ValueError):
            forward_price(100.0, 0.05, 0.0, -1.0)
