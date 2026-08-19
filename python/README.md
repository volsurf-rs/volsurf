# volsurf (Python)

Python bindings for [volsurf](https://github.com/volsurf-rs/volsurf), a volatility
surface library for equity and FX derivatives. Built with PyO3.

## Install

```bash
pip install volsurf
```

Requires Python 3.9 or later. NumPy is the only runtime dependency.

To build from source:

```bash
maturin develop --release -m python/Cargo.toml
```

## Build a surface

`SurfaceBuilder` sets its options through method calls rather than chaining, so
call each setter on its own line.

```python
from volsurf import SurfaceBuilder, SmileModel

strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
vols = [0.28, 0.24, 0.22, 0.20, 0.22, 0.24, 0.28]

b = SurfaceBuilder()
b.spot(100.0)
b.rate(0.05)
b.add_tenor(0.25, strikes, vols)
b.add_tenor(1.00, strikes, vols)
surface = b.build()

surface.black_vol(0.5, 100.0)      # vol at any (expiry, strike)
surface.black_variance(0.5, 100.0) # total variance
surface.tenors()                   # [0.25, 1.0]
```

Pick a smile model with `b.model(...)`. `SmileModel.svi()` is the default and
needs 5 strikes per tenor, `SmileModel.cubic_spline()` needs 3, and
`SmileModel.sabr(beta)` needs 4.

```python
b.model(SmileModel.sabr(0.5))
b.dividend_yield(0.02)
b.add_tenor_with_forward(0.25, strikes, vols, 101.2)
```

## Query a smile

`smile_at()` returns the calibrated section for one tenor.

```python
smile = surface.smile_at(0.5)
smile.vol(105.0)
smile.density(105.0)   # risk-neutral density (Breeden-Litzenberger)
smile.forward          # a property, as are smile.expiry and smile.model_name
```

Sections taken from a `SurfaceBuilder` surface report `model_name == "CubicSpline"`.
The builder fits your chosen model per tenor, then hands back a spline through the
fitted variances. Construct a model directly if you need its own parameters.

Construct smiles directly when you already have parameters, or calibrate one
tenor at a time:

```python
from volsurf import SviSmile, SabrSmile, SplineSmile

svi = SviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.2)  # forward, expiry, a, b, rho, m, sigma
sabr = SabrSmile(100.0, 1.0, 0.20, 0.5, -0.3, 0.4)     # forward, expiry, alpha, beta, rho, nu

market = [(80.0, 0.28), (90.0, 0.24), (100.0, 0.20), (110.0, 0.24), (120.0, 0.28)]
fitted = SviSmile.calibrate(100.0, 1.0, market)
```

## Global surfaces

`SsviSurface` and `EssviSurface` parameterize the whole surface at once. Both
take parameters directly or calibrate from per-tenor market data.

```python
from volsurf import SsviSurface, EssviSurface

ssvi = SsviSurface(
    rho=-0.3, eta=0.5, gamma=0.5,
    tenors=[0.25, 0.5, 1.0],
    forwards=[100.0, 100.0, 100.0],
    thetas=[0.04, 0.08, 0.16],   # ATM total variance
)

essvi = EssviSurface.calibrate(
    [market_3m, market_1y],   # per-tenor [(strike, vol), ...]
    [0.25, 1.0],              # tenors
    [100.0, 100.0],           # forwards
)
```

To see how well each tenor fit, run the first stage on its own.
`EssviSurface.fit_per_tenor(market_data, tenors, forwards)` returns a list of
`PerTenorFit` — each with `rms_error`, `theta`, and the fitted `svi` slice — and
`EssviSurface.from_per_tenor(fits)` turns that list into the surface.

```python
fits = EssviSurface.fit_per_tenor([market_3m, market_1y], [0.25, 1.0], [100.0, 100.0])
print([f.rms_error for f in fits])
surface = EssviSurface.from_per_tenor(fits)
```

## NumPy grids

Every surface exposes `vol_grid(expiries, strikes)`, which returns a
`(len(expiries), len(strikes))` array of `float64`. Smiles expose
`vol_array(strikes)`.

```python
import numpy as np

vols = surface.vol_grid(np.array([0.25, 0.5, 1.0]), np.array([90.0, 100.0, 110.0]))
vols.shape   # (3, 3)
```

Grid values match the scalar calls to within 1e-14, so use whichever fits the
calling code.

## Arbitrage checks

```python
report = smile.is_arbitrage_free()
report.is_free                 # property
report.butterfly_violations    # property, a list
report.worst_violation()       # method

diag = surface.diagnostics()
for v in diag.calendar_violations:
    print(v)
```

`is_arbitrage_free_with(config)` and `diagnostics_with(config)` take an
`ArbitrageScanConfig` when you want a denser or sparser scan grid than the
default.

## Calibration control

`DataFilter` drops points before the fit. `WeightingScheme` sets how the
remaining points are weighted.

```python
from volsurf import DataFilter, WeightingScheme

f = DataFilter(max_log_moneyness=0.5, min_vol=0.01)  # drop far wings and sub-floor quotes

b.data_filter(f)
b.weighting(WeightingScheme.vega())
```

The same options reach single-tenor fits through
`SviSmile.calibrate_with_config(forward, expiry, market_vols, filter, weighting, seed)`.
Pass `seed` to warm-start from prior parameters.

## Implied volatility

Prices are undiscounted and quoted on the forward.

```python
from volsurf import BlackImpliedVol, NormalImpliedVol, DisplacedImpliedVol
from volsurf import black_price, normal_price, displaced_price, OptionType

price = black_price(100.0, 100.0, 0.20, 1.0, OptionType.Call)
vol = BlackImpliedVol.compute(price, 100.0, 100.0, 1.0, OptionType.Call)
```

Use `NormalImpliedVol` (Bachelier) when forwards can go negative, and
`DisplacedImpliedVol(beta)` to interpolate between normal (`beta=0`) and Black
(`beta=1`).

Conventions helpers: `log_moneyness(strike, forward)`, `moneyness(strike, forward)`,
and `forward_price(spot, rate, dividend_yield, expiry)`.

## Local volatility

`DupireLocalVol` wraps any surface and extracts local vol by finite differences.

```python
from volsurf import DupireLocalVol

lv = DupireLocalVol(surface)          # or DupireLocalVol(surface, bump_size=1e-3)
lv.local_vol(0.5, 100.0)
```

## Serialization

Model structs round-trip through JSON with `to_json()` and `from_json()`.

```python
s = ssvi.to_json()
restored = SsviSurface.from_json(s)
```

## Errors

Invalid inputs raise `ValueError` with the underlying message: negative vols,
zero or negative forwards, non-finite values, too few strikes for the chosen
model, and calibration failures.

## Tests

```bash
pytest python/tests
```
