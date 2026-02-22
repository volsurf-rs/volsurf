import json

import pytest
from volsurf import (
    EssviSurface,
    SabrSmile,
    SplineSmile,
    SsviSurface,
    SviSmile,
)


class TestSviSerde:
    def test_round_trip_vol_consistent(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1)
        s = smile.to_json()
        smile2 = SviSmile.from_json(s)
        assert abs(smile.vol(100.0) - smile2.vol(100.0)) < 1e-14
        assert abs(smile.vol(80.0) - smile2.vol(80.0)) < 1e-14

    def test_json_structure(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1)
        d = json.loads(smile.to_json())
        assert set(d.keys()) == {"forward", "expiry", "a", "b", "rho", "m", "sigma"}

    def test_idempotent(self):
        smile = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1)
        j1 = smile.to_json()
        j2 = SviSmile.from_json(j1).to_json()
        assert j1 == j2

    def test_from_json_invalid_params(self):
        bad = json.dumps({
            "forward": -100.0, "expiry": 1.0,
            "a": 0.04, "b": 0.4, "rho": -0.4, "m": 0.0, "sigma": 0.1,
        })
        with pytest.raises(ValueError):
            SviSmile.from_json(bad)


class TestSabrSerde:
    def test_round_trip_vol_consistent(self):
        smile = SabrSmile(100.0, 1.0, 0.2, 0.5, -0.3, 0.3)
        s = smile.to_json()
        smile2 = SabrSmile.from_json(s)
        assert abs(smile.vol(100.0) - smile2.vol(100.0)) < 1e-14

    def test_json_structure(self):
        smile = SabrSmile(100.0, 1.0, 0.2, 0.5, -0.3, 0.3)
        d = json.loads(smile.to_json())
        assert set(d.keys()) == {"forward", "expiry", "alpha", "beta", "rho", "nu"}

    def test_idempotent(self):
        smile = SabrSmile(100.0, 1.0, 0.2, 0.5, -0.3, 0.3)
        j1 = smile.to_json()
        j2 = SabrSmile.from_json(j1).to_json()
        assert j1 == j2


class TestSplineSerde:
    def test_round_trip_vol_consistent(self):
        smile = SplineSmile(
            100.0, 1.0,
            [80.0, 90.0, 100.0, 110.0, 120.0],
            [0.065, 0.045, 0.04, 0.045, 0.065],
        )
        s = smile.to_json()
        smile2 = SplineSmile.from_json(s)
        assert abs(smile.vol(100.0) - smile2.vol(100.0)) < 1e-14
        assert abs(smile.vol(90.0) - smile2.vol(90.0)) < 1e-14

    def test_no_coeffs_in_json(self):
        smile = SplineSmile(100.0, 1.0, [80.0, 100.0, 120.0], [0.04, 0.04, 0.04])
        s = smile.to_json()
        assert "coeffs" not in s

    def test_json_structure(self):
        smile = SplineSmile(100.0, 1.0, [80.0, 100.0, 120.0], [0.04, 0.04, 0.04])
        d = json.loads(smile.to_json())
        assert set(d.keys()) == {"forward", "expiry", "strikes", "variances"}


class TestSsviSerde:
    def test_round_trip_vol_consistent(self):
        s = SsviSurface(-0.3, 0.5, 0.5, [0.25, 0.5, 1.0], [100.0]*3, [0.04, 0.08, 0.16])
        j = s.to_json()
        s2 = SsviSurface.from_json(j)
        assert abs(s.black_vol(0.5, 100.0) - s2.black_vol(0.5, 100.0)) < 1e-14

    def test_json_structure(self):
        s = SsviSurface(-0.3, 0.5, 0.5, [0.25], [100.0], [0.04])
        d = json.loads(s.to_json())
        assert set(d.keys()) == {"rho", "eta", "gamma", "tenors", "forwards", "thetas"}

    def test_idempotent(self):
        s = SsviSurface(-0.3, 0.5, 0.5, [0.25, 0.5], [100.0, 100.0], [0.04, 0.08])
        j1 = s.to_json()
        j2 = SsviSurface.from_json(j1).to_json()
        assert j1 == j2


class TestEssviSerde:
    def test_round_trip_vol_consistent(self):
        s = EssviSurface(
            -0.4, -0.2, 0.5, 0.5, 0.5,
            [0.25, 0.5, 1.0], [100.0]*3, [0.04, 0.08, 0.16],
        )
        j = s.to_json()
        s2 = EssviSurface.from_json(j)
        assert abs(s.black_vol(0.5, 100.0) - s2.black_vol(0.5, 100.0)) < 1e-14

    def test_idempotent(self):
        s = EssviSurface(
            -0.4, -0.2, 0.5, 0.5, 0.5,
            [0.25, 0.5], [100.0, 100.0], [0.04, 0.08],
        )
        j1 = s.to_json()
        j2 = EssviSurface.from_json(j1).to_json()
        assert j1 == j2
