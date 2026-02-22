import math

import pytest
from volsurf import SviSmile, SabrSmile, SplineSmile


class TestSviSmile:
    def test_construct_valid(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1)
        assert smile.forward == 100.0
        assert smile.expiry == 1.0

    def test_vol_atm(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1)
        # w_atm = a + b*sigma = 0.04 + 0.4*0.1 = 0.08
        expected = math.sqrt(0.08)
        assert abs(smile.vol(100.0) - expected) < 1e-14

    def test_negative_rho_skew(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1)
        assert smile.vol(80.0) > smile.vol(120.0)

    def test_variance_consistent(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1)
        v = smile.vol(90.0)
        var = smile.variance(90.0)
        assert abs(var - v * v * 1.0) < 1e-14

    def test_density_atm_positive(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1)
        assert smile.density(100.0) > 0

    def test_arb_free_clean(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.2)
        report = smile.is_arbitrage_free()
        assert report.is_free is True
        assert len(report.butterfly_violations) == 0

    def test_arb_free_violated(self):
        smile = SviSmile(100.0, 1.0, 0.001, 0.8, -0.7, 0.0, 0.05)
        report = smile.is_arbitrage_free()
        assert report.is_free is False
        assert len(report.butterfly_violations) > 0
        worst = report.worst_violation()
        assert worst is not None
        assert worst.density < 0

    def test_rejects_zero_forward(self):
        with pytest.raises(ValueError):
            SviSmile(0.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1)

    def test_rejects_negative_forward(self):
        with pytest.raises(ValueError):
            SviSmile(-100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1)

    def test_rejects_zero_expiry(self):
        with pytest.raises(ValueError):
            SviSmile(100.0, 0.0, 0.04, 0.4, -0.4, 0.0, 0.1)

    def test_rejects_rho_at_one(self):
        with pytest.raises(ValueError):
            SviSmile(100.0, 1.0, 0.04, 0.4, 1.0, 0.0, 0.1)

    def test_rejects_rho_at_neg_one(self):
        with pytest.raises(ValueError):
            SviSmile(100.0, 1.0, 0.04, 0.4, -1.0, 0.0, 0.1)

    def test_rejects_negative_b(self):
        with pytest.raises(ValueError):
            SviSmile(100.0, 1.0, 0.04, -0.1, -0.4, 0.0, 0.1)

    def test_rejects_zero_sigma(self):
        with pytest.raises(ValueError):
            SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.0)

    def test_rejects_zero_strike_in_vol(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1)
        with pytest.raises(ValueError):
            smile.vol(0.0)

    def test_rejects_negative_strike_in_vol(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1)
        with pytest.raises(ValueError):
            smile.vol(-10.0)

    def test_flat_smile_b_zero(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.0, 0.0, 0.0, 0.1)
        expected = math.sqrt(0.04)
        assert abs(smile.vol(80.0) - expected) < 1e-14
        assert abs(smile.vol(120.0) - expected) < 1e-14


class TestSabrSmile:
    def test_construct_valid(self):
        smile = SabrSmile(100.0, 1.0, 0.2, 0.5, -0.3, 0.3)
        assert smile.forward == 100.0
        assert smile.expiry == 1.0

    def test_vol_atm_positive(self):
        smile = SabrSmile(100.0, 1.0, 0.2, 0.5, -0.3, 0.3)
        v = smile.vol(100.0)
        assert v > 0
        assert math.isfinite(v)

    def test_variance_consistent(self):
        smile = SabrSmile(100.0, 1.0, 0.2, 0.5, -0.3, 0.3)
        v = smile.vol(90.0)
        var = smile.variance(90.0)
        assert abs(var - v * v * 1.0) < 1e-14

    def test_density_atm_positive(self):
        smile = SabrSmile(100.0, 1.0, 0.2, 0.5, -0.3, 0.3)
        d = smile.density(100.0)
        assert d > 0 and math.isfinite(d)

    def test_arb_free_conservative(self):
        smile = SabrSmile(100.0, 1.0, 0.2, 0.5, -0.3, 0.3)
        report = smile.is_arbitrage_free()
        assert report.is_free is True

    def test_arb_free_extreme_nu(self):
        smile = SabrSmile(100.0, 1.0, 0.3, 0.5, -0.5, 2.0)
        report = smile.is_arbitrage_free()
        assert report.is_free is False

    def test_rejects_zero_alpha(self):
        with pytest.raises(ValueError):
            SabrSmile(100.0, 1.0, 0.0, 0.5, -0.3, 0.3)

    def test_rejects_negative_alpha(self):
        with pytest.raises(ValueError):
            SabrSmile(100.0, 1.0, -0.1, 0.5, -0.3, 0.3)

    def test_rejects_beta_below_zero(self):
        with pytest.raises(ValueError):
            SabrSmile(100.0, 1.0, 0.2, -0.1, -0.3, 0.3)

    def test_rejects_beta_above_one(self):
        with pytest.raises(ValueError):
            SabrSmile(100.0, 1.0, 0.2, 1.1, -0.3, 0.3)

    def test_rejects_rho_at_boundary(self):
        with pytest.raises(ValueError):
            SabrSmile(100.0, 1.0, 0.2, 0.5, 1.0, 0.3)
        with pytest.raises(ValueError):
            SabrSmile(100.0, 1.0, 0.2, 0.5, -1.0, 0.3)

    def test_rejects_negative_nu(self):
        with pytest.raises(ValueError):
            SabrSmile(100.0, 1.0, 0.2, 0.5, -0.3, -0.1)

    def test_nu_zero_valid(self):
        smile = SabrSmile(100.0, 1.0, 0.2, 0.5, -0.3, 0.0)
        assert smile.vol(100.0) > 0


class TestSplineSmile:
    def test_construct_valid_3_points(self):
        smile = SplineSmile(100.0, 1.0, [80.0, 100.0, 120.0], [0.04, 0.04, 0.04])
        assert smile.forward == 100.0
        assert smile.expiry == 1.0

    def test_construct_valid_5_points(self):
        smile = SplineSmile(
            100.0, 1.0,
            [80.0, 90.0, 100.0, 110.0, 120.0],
            [0.065, 0.045, 0.04, 0.045, 0.065],
        )
        assert smile.vol(100.0) > 0

    def test_vol_at_knot_points(self):
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        variances = [0.065, 0.045, 0.04, 0.045, 0.065]
        smile = SplineSmile(100.0, 1.0, strikes, variances)
        for k, var in zip(strikes, variances):
            assert abs(smile.variance(k) - var) < 1e-12

    def test_flat_smile_constant_vol(self):
        smile = SplineSmile(100.0, 1.0, [80.0, 100.0, 120.0], [0.04, 0.04, 0.04])
        expected = math.sqrt(0.04)
        assert abs(smile.vol(90.0) - expected) < 1e-12
        assert abs(smile.vol(110.0) - expected) < 1e-12

    def test_extrapolation_left(self):
        variances = [0.065, 0.045, 0.04, 0.045, 0.065]
        smile = SplineSmile(
            100.0, 1.0,
            [80.0, 90.0, 100.0, 110.0, 120.0],
            variances,
        )
        assert abs(smile.variance(50.0) - 0.065) < 1e-12

    def test_extrapolation_right(self):
        variances = [0.065, 0.045, 0.04, 0.045, 0.065]
        smile = SplineSmile(
            100.0, 1.0,
            [80.0, 90.0, 100.0, 110.0, 120.0],
            variances,
        )
        assert abs(smile.variance(200.0) - 0.065) < 1e-12

    def test_density_atm_positive(self):
        smile = SplineSmile(
            100.0, 1.0,
            [80.0, 90.0, 100.0, 110.0, 120.0],
            [0.065, 0.045, 0.04, 0.045, 0.065],
        )
        d = smile.density(100.0)
        assert d > 0 and math.isfinite(d)

    def test_arb_free_convex(self):
        smile = SplineSmile(
            100.0, 1.0,
            [80.0, 90.0, 100.0, 110.0, 120.0],
            [0.065, 0.045, 0.04, 0.045, 0.065],
        )
        report = smile.is_arbitrage_free()
        assert report.is_free is True

    def test_rejects_fewer_than_3_points(self):
        with pytest.raises(ValueError):
            SplineSmile(100.0, 1.0, [80.0, 120.0], [0.04, 0.04])

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            SplineSmile(100.0, 1.0, [80.0, 100.0, 120.0], [0.04, 0.04])

    def test_rejects_unsorted_strikes(self):
        with pytest.raises(ValueError):
            SplineSmile(100.0, 1.0, [120.0, 100.0, 80.0], [0.04, 0.04, 0.04])

    def test_rejects_duplicate_strikes(self):
        with pytest.raises(ValueError):
            SplineSmile(100.0, 1.0, [80.0, 80.0, 120.0], [0.04, 0.04, 0.04])

    def test_rejects_negative_variance(self):
        with pytest.raises(ValueError):
            SplineSmile(100.0, 1.0, [80.0, 100.0, 120.0], [0.04, -0.01, 0.04])
