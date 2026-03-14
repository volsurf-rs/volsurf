"""Tests for DataFilter, WeightingScheme, and calibrate_with_config."""

from volsurf import (
    DataFilter,
    WeightingScheme,
    SmileModel,
    SurfaceBuilder,
    SviSmile,
    SabrSmile,
)


class TestDataFilter:
    def test_default_construction(self):
        f = DataFilter()
        assert f.max_log_moneyness is None
        assert f.min_vol is None
        assert f.vol_cliff_filter is None

    def test_kwargs_construction(self):
        f = DataFilter(max_log_moneyness=0.5, min_vol=0.01, vol_cliff_filter=False)
        assert f.max_log_moneyness == 0.5
        assert f.min_vol == 0.01
        assert f.vol_cliff_filter is False

    def test_partial_kwargs(self):
        f = DataFilter(max_log_moneyness=0.3)
        assert f.max_log_moneyness == 0.3
        assert f.min_vol is None

    def test_repr(self):
        f = DataFilter(max_log_moneyness=0.5)
        assert "0.5" in repr(f)

    def test_equality(self):
        a = DataFilter(max_log_moneyness=0.5)
        b = DataFilter(max_log_moneyness=0.5)
        assert a == b

    def test_inequality(self):
        a = DataFilter(max_log_moneyness=0.5)
        b = DataFilter(max_log_moneyness=0.3)
        assert a != b


class TestWeightingScheme:
    def test_model_default(self):
        w = WeightingScheme.model_default()
        assert repr(w) == "ModelDefault"

    def test_vega(self):
        w = WeightingScheme.vega()
        assert repr(w) == "Vega"

    def test_uniform(self):
        w = WeightingScheme.uniform()
        assert repr(w) == "Uniform"

    def test_equality(self):
        assert WeightingScheme.vega() == WeightingScheme.vega()
        assert WeightingScheme.vega() != WeightingScheme.uniform()


class TestSviCalibrateWithConfig:
    def _market_data(self):
        svi = SviSmile(100.0, 0.25, 0.04, 0.4, -0.3, 0.02, 0.15)
        strikes = list(range(80, 121))
        vols = [svi.vol(k) for k in strikes]
        return list(zip(strikes, vols))

    def test_defaults_match_calibrate(self):
        market = self._market_data()
        a = SviSmile.calibrate(100.0, 0.25, market)
        b = SviSmile.calibrate_with_config(100.0, 0.25, market)
        assert abs(a.vol(100.0) - b.vol(100.0)) < 1e-12

    def test_with_filter(self):
        market = self._market_data()
        f = DataFilter(max_log_moneyness=0.25)
        result = SviSmile.calibrate_with_config(100.0, 0.25, market, filter=f)
        assert result.vol(100.0) > 0

    def test_with_weighting(self):
        market = self._market_data()
        result = SviSmile.calibrate_with_config(
            100.0, 0.25, market, weighting=WeightingScheme.uniform()
        )
        assert result.vol(100.0) > 0

    def test_with_seed(self):
        market = self._market_data()
        first = SviSmile.calibrate(100.0, 0.25, market)
        second = SviSmile.calibrate_with_config(100.0, 0.25, market, seed=first)
        assert abs(first.vol(100.0) - second.vol(100.0)) < 0.001


class TestSabrCalibrateWithConfig:
    def _market_data(self):
        sabr = SabrSmile(100.0, 0.5, 0.3, 0.5, -0.3, 0.4)
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        vols = [sabr.vol(k) for k in strikes]
        return list(zip(strikes, vols))

    def test_defaults_match_calibrate(self):
        market = self._market_data()
        a = SabrSmile.calibrate(100.0, 0.5, 0.5, market)
        b = SabrSmile.calibrate_with_config(100.0, 0.5, 0.5, market)
        assert abs(a.vol(100.0) - b.vol(100.0)) < 1e-12

    def test_with_seed(self):
        market = self._market_data()
        first = SabrSmile.calibrate(100.0, 0.5, 0.5, market)
        second = SabrSmile.calibrate_with_config(100.0, 0.5, 0.5, market, seed=first)
        assert abs(first.vol(100.0) - second.vol(100.0)) < 0.01

    def test_with_vega_weighting(self):
        market = self._market_data()
        result = SabrSmile.calibrate_with_config(
            100.0, 0.5, 0.5, market, weighting=WeightingScheme.vega()
        )
        assert result.vol(100.0) > 0


class TestSurfaceBuilderConfig:
    def test_build_with_data_filter(self):
        svi = SviSmile(100.0, 0.25, 0.04, 0.4, -0.3, 0.02, 0.15)
        strikes = list(range(80, 121))
        vols = [svi.vol(k) for k in strikes]

        builder = SurfaceBuilder()
        builder.spot(100.0)
        builder.rate(0.05)
        builder.model(SmileModel.svi())
        builder.data_filter(DataFilter(max_log_moneyness=0.5))
        builder.add_tenor(0.25, strikes, vols)
        surface = builder.build()
        assert surface.black_vol(0.25, 100.0) > 0

    def test_build_with_weighting(self):
        svi = SviSmile(100.0, 0.25, 0.04, 0.4, -0.3, 0.02, 0.15)
        strikes = list(range(80, 121))
        vols = [svi.vol(k) for k in strikes]

        builder = SurfaceBuilder()
        builder.spot(100.0)
        builder.rate(0.05)
        builder.model(SmileModel.svi())
        builder.weighting(WeightingScheme.uniform())
        builder.add_tenor(0.25, strikes, vols)
        surface = builder.build()
        assert surface.black_vol(0.25, 100.0) > 0
