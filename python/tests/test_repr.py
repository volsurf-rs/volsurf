from volsurf import SabrSmile, SmileModel, SsviSurface, SurfaceBuilder, SviSmile
from conftest import SSVI_EQUITY

STRIKES = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
VOLS = [0.28, 0.24, 0.22, 0.20, 0.22, 0.24, 0.28]


class TestSmileModelRepr:
    def test_svi_repr(self):
        assert "Svi" in repr(SmileModel.svi())

    def test_sabr_repr(self):
        r = repr(SmileModel.sabr(0.5))
        assert "Sabr" in r
        assert "0.5" in r

    def test_cubic_spline_repr(self):
        assert "CubicSpline" in repr(SmileModel.cubic_spline())

    def test_equality_same(self):
        assert SmileModel.svi() == SmileModel.svi()
        assert SmileModel.sabr(0.5) == SmileModel.sabr(0.5)

    def test_equality_different(self):
        assert not (SmileModel.svi() == SmileModel.cubic_spline())
        assert not (SmileModel.sabr(0.5) == SmileModel.sabr(0.7))


class TestArbitrageReportRepr:
    def test_clean_report(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.2)
        report = smile.is_arbitrage_free()
        r = repr(report)
        assert "ArbitrageReport" in r
        assert "is_free=true" in r

    def test_violated_report(self):
        smile = SviSmile(100.0, 1.0, 0.001, 0.8, -0.7, 0.0, 0.05)
        report = smile.is_arbitrage_free()
        r = repr(report)
        assert "ArbitrageReport" in r
        assert "is_free=false" in r

    def test_worst_violation_none_when_clean(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.2)
        report = smile.is_arbitrage_free()
        assert report.worst_violation() is None


class TestButterflyViolationRepr:
    def test_repr_format(self):
        smile = SviSmile(100.0, 1.0, 0.001, 0.8, -0.7, 0.0, 0.05)
        report = smile.is_arbitrage_free()
        for v in report.butterfly_violations:
            r = repr(v)
            assert "ButterflyViolation" in r
            assert "strike=" in r
            assert "density=" in r
            break


class TestSurfaceDiagnosticsRepr:
    def test_repr_format(self):
        b = SurfaceBuilder()
        b.model(SmileModel.sabr(0.5))
        b.spot(100.0)
        b.rate(0.05)
        b.add_tenor(0.25, [80.0, 90.0, 100.0, 110.0, 120.0],
                     [0.28, 0.24, 0.20, 0.22, 0.26])
        surf = b.build()
        diag = surf.diagnostics()
        r = repr(diag)
        assert "SurfaceDiagnostics" in r
        assert "is_free=" in r
        assert "smiles=" in r

    def test_smile_reports_have_content(self):
        s = SsviSurface(**SSVI_EQUITY)
        diag = s.diagnostics()
        for report in diag.smile_reports:
            assert hasattr(report, "is_free")
            assert hasattr(report, "butterfly_violations")


class TestButterflyViolationFields:
    def test_field_access(self):
        smile = SviSmile(100.0, 1.0, 0.001, 0.8, -0.7, 0.0, 0.05)
        report = smile.is_arbitrage_free()
        worst = report.worst_violation()
        assert worst is not None
        assert isinstance(worst.strike, float) and worst.strike > 0
        assert isinstance(worst.density, float) and worst.density < 0
        assert isinstance(worst.magnitude, float) and worst.magnitude > 0


class TestCalendarViolationRepr:
    def _build_inverted_surface(self):
        """Build surface with inverted total variance to trigger calendar violations."""
        b = SurfaceBuilder()
        b.model(SmileModel.sabr(0.5))
        b.spot(100.0)
        b.rate(0.05)
        # high vols on short tenor, low vols on long → inverted total variance
        b.add_tenor(0.25, STRIKES, [v * 1.8 for v in VOLS])
        b.add_tenor(1.0, STRIKES, [v * 0.5 for v in VOLS])
        return b.build()

    def test_repr_format(self):
        surf = self._build_inverted_surface()
        diag = surf.diagnostics()
        if diag.calendar_violations:
            v = diag.calendar_violations[0]
            r = repr(v)
            assert "CalendarViolation" in r
            assert "strike=" in r
            assert "short=" in r
            assert "long=" in r

    def test_field_access(self):
        surf = self._build_inverted_surface()
        diag = surf.diagnostics()
        if diag.calendar_violations:
            v = diag.calendar_violations[0]
            assert isinstance(v.strike, float) and v.strike > 0
            assert isinstance(v.tenor_short, float)
            assert isinstance(v.tenor_long, float)
            assert v.tenor_short < v.tenor_long
            assert isinstance(v.variance_short, float)
            assert isinstance(v.variance_long, float)
