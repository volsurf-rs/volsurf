# volsurf

[![CI](https://github.com/volsurf-rs/volsurf/actions/workflows/ci.yml/badge.svg)](https://github.com/volsurf-rs/volsurf/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/volsurf.svg)](https://crates.io/crates/volsurf)
[![docs.rs](https://docs.rs/volsurf/badge.svg)](https://docs.rs/volsurf)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Volatility surface construction for equity and FX derivatives.

`volsurf` builds implied volatility surfaces from market data, calibrates parametric smile models (SVI, SABR, SSVI), and detects butterfly and calendar arbitrage. Vol queries take 20 ns or less, fast enough for real-time pricing engines.

## Features

**Smile Models**
- **SVI** (Gatheral 2006) -- 5-parameter with quasi-explicit calibration (Zeliade 2009), analytical g-function density, butterfly arbitrage detection
- **SABR** (Hagan 2002) -- 4-parameter with Hagan closed-form, analytic alpha + Nelder-Mead calibration, 12-digit accuracy vs reference values
- **Cubic spline** -- non-parametric on total variance, Thomas algorithm O(n), flat extrapolation

**Surface Construction**
- **SSVI** global parameterization (Gatheral-Jacquier 2014) with two-stage calibration
- **eSSVI** extended SSVI (Hendriks-Martini 2019) with tenor-dependent rho, calendar-spread no-arb guarantees, 3-stage calibration from market data
- **Piecewise** per-tenor surfaces with linear variance interpolation
- **Builder API** with SVI/SABR/spline model selection, dividend yield, per-tenor forward override
- Ragged strike grids -- different strikes per tenor, no rectangular matrix assumption

**Arbitrage Detection**
- Butterfly arbitrage via analytical g-function (SVI) and numerical density scan (SABR)
- Calendar spread arbitrage via cross-tenor variance monotonicity
- Analytical calendar arbitrage for SSVI surfaces
- Combined `SurfaceDiagnostics` report
- Configurable scan grids via `ArbitrageScanConfig` on `is_arbitrage_free_with()` and `diagnostics_with()`

**Calibration Control**
- `DataFilter` drops wing strikes, sub-floor vols, and vol cliffs before fitting
- `WeightingScheme` weights the least-squares fit by vega or uniformly (Zeliade 2009, Hagan 2002)
- Warm-starting from prior parameters; SVI falls back to grid search when a seeded fit diverges

**Implied Volatility**
- **Black** (lognormal) implied vol via Jackel rational approximation (near-machine-precision)
- **Normal** (Bachelier) implied vol and pricing (Jackel 2017) -- supports negative forwards
- **Displaced diffusion** -- interpolates between normal (beta=0) and Black (beta=1)

**Local Volatility**
- **Dupire** local vol extraction (Dupire 1994) from any `VolSurface` via finite differences
- **Boundary adapter** -- `with_boundary()` evaluates short expiries at a floor, avoiding the Dupire denominator blow-up as total variance goes to zero

**Design**
- No global state -- evaluation date is a parameter, not a singleton
- Immutable surfaces -- no observer pattern
- Thread-safe -- all types are `Send + Sync`
- Zero-alloc vol queries after construction
- Newtypes for type safety (`Vol`, `NormalVol`, `Variance`, `Strike`, `Tenor`)
- Serde serialization on all model structs and value types

## Installation

```toml
[dependencies]
volsurf = "3.0"
```

Optional features:

```toml
volsurf = { version = "3.0", features = ["parallel", "logging"] }
```

| Feature | Description |
|---------|-------------|
| `parallel` | `rayon` support for parallel surface construction |
| `logging` | `tracing` instrumentation in calibration and build paths |

Requires Rust edition 2024 (rustc 1.85+).

## Quick Start

### Build a surface from market data

```rust
use volsurf::surface::{SurfaceBuilder, VolSurface};
use volsurf::{Strike, Tenor};

let strikes = vec![80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
let vols = vec![0.28, 0.24, 0.22, 0.20, 0.22, 0.24, 0.28];

let surface = SurfaceBuilder::new()
    .spot(100.0)
    .rate(0.05)
    .add_tenor(0.25, &strikes, &vols)
    .add_tenor(1.00, &strikes, &vols)
    .build()?;

// Query vol at any (expiry, strike) point
let vol = surface.black_vol(Tenor(0.5), Strike(100.0))?;
```

### Choose a smile model

```rust
use volsurf::surface::{SurfaceBuilder, SmileModel, VolSurface};
use volsurf::{Strike, Tenor};

let surface = SurfaceBuilder::new()
    .spot(100.0)
    .rate(0.05)
    .model(SmileModel::Sabr { beta: 0.5 })
    .add_tenor(0.25, &strikes, &vols)
    .add_tenor(1.00, &strikes, &vols)
    .build()?;
```

Available models: `SmileModel::Svi` (default, 5+ strikes), `SmileModel::CubicSpline` (3+ strikes), `SmileModel::Sabr { beta }` (4+ strikes).

### SSVI global surface

```rust
use volsurf::surface::{SsviSurface, VolSurface};
use volsurf::{Strike, Tenor};

let surface = SsviSurface::new(
    -0.3, 0.5, 0.5,                    // rho, eta, gamma
    vec![0.25, 0.5, 1.0],              // tenors
    vec![100.0, 100.0, 100.0],         // forwards
    vec![0.04, 0.08, 0.16],            // thetas (ATM total variance)
)?;

let vol = surface.black_vol(Tenor(0.5), Strike(100.0))?;
let smile = surface.smile_at(Tenor(0.5))?;
```

### Calibrate eSSVI from market data

```rust
use volsurf::surface::{EssviSurface, VolSurface};
use volsurf::{Strike, Tenor};

// Per-tenor (strike, implied_vol) pairs
let data_3m: Vec<(f64, f64)> = (0..10)
    .map(|i| (80.0 + 4.0 * i as f64, 0.20 + 0.01 * (i as f64 - 5.0).abs()))
    .collect();
let data_1y: Vec<(f64, f64)> = (0..10)
    .map(|i| (80.0 + 4.0 * i as f64, 0.18 + 0.008 * (i as f64 - 5.0).abs()))
    .collect();

let surface = EssviSurface::calibrate(
    &[data_3m, data_1y],
    &[0.25, 1.0],       // tenors
    &[100.0, 100.0],    // forwards
)?;

let vol = surface.black_vol(Tenor(0.5), Strike(95.0))?;
```

### Check for arbitrage

```rust
use volsurf::smile::{SviSmile, SmileSection};

let smile = SviSmile::new(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.2)?;
let report = smile.is_arbitrage_free()?;
assert!(report.is_free());

// Surface-level diagnostics (butterfly + calendar)
let diagnostics = surface.diagnostics()?;
if !diagnostics.is_free() {
    for cal in &diagnostics.calendar_violations {
        println!("Calendar violation at K={}", cal.strike);
    }
}
```

### Extract implied vol from option prices

```rust
use volsurf::implied::BlackImpliedVol;
use volsurf::OptionType;

let vol = BlackImpliedVol::compute(10.45, 100.0, 100.0, 1.0, OptionType::Call)?;
```

Runnable programs for every layer live in [`examples/`](examples/). Run one with `cargo run --example local_vol`.

## Architecture

Five-layer pipeline following the natural domain flow:

```
Option Prices -> Implied Vol -> Smile -> Surface -> Local Vol
                  (Layer 1)    (Layer 2) (Layer 3)   (Layer 5)
                                      Arbitrage Detection
                                         (Layer 4)
```

```
volsurf
├── calibration    DataFilter, WeightingScheme, apply_filter
├── conventions    log_moneyness, moneyness, forward_price
├── error          VolSurfError, Result<T>
├── implied
│   ├── black      BlackImpliedVol, black_price
│   ├── normal     NormalImpliedVol, normal_price
│   └── displaced  DisplacedImpliedVol, displaced_price
├── smile
│   ├── svi        SviSmile (Gatheral 2006)
│   ├── sabr       SabrSmile (Hagan 2002)
│   ├── spline     SplineSmile (cubic on variance)
│   └── arbitrage  ArbitrageReport, ButterflyViolation, ArbitrageScanConfig
├── surface
│   ├── ssvi       SsviSurface (Gatheral-Jacquier 2014)
│   ├── essvi      EssviSurface, EssviSlice (Hendriks-Martini 2019)
│   ├── piecewise  PiecewiseSurface (per-tenor interpolation)
│   ├── builder    SurfaceBuilder, SmileModel
│   └── arbitrage  SurfaceDiagnostics, CalendarViolation
├── local_vol      LocalVol trait, DupireLocalVol (Dupire 1994), BoundaryLocalVol
└── types          Strike, Tenor, Vol, NormalVol, Variance, OptionType
```

## Benchmarks

Measured with Criterion.rs on Apple Silicon. All performance targets exceeded.

### Vol Queries (single point evaluation)

| Operation | Time |
|-----------|------|
| SVI vol | 4.7 ns |
| Spline vol | 4.6 ns |
| SSVI slice vol | 11 ns |
| SABR vol | 17 ns |
| Piecewise surface vol | 18 ns |
| SSVI surface vol | 20 ns |

### Calibration

| Operation | Time |
|-----------|------|
| SABR calibration | 74 us |
| SVI calibration | 107 us |
| SSVI calibration | 266 us |

### Surface Construction

| Operation | Time |
|-----------|------|
| SABR surface (5 tenors) | 381 us |
| SVI surface (5 tenors) | 553 us |
| SVI surface (20 tenors) | 2.6 ms |

**Targets**: vol query < 100 ns, SABR calibration < 1 ms, 20-tenor surface < 10 ms.

## Bindings

### Python

```bash
pip install volsurf
```

Built with PyO3. See [`python/README.md`](python/README.md) for the API and usage examples.

### WebAssembly

```bash
wasm-pack build wasm/ --target web
```

Built with wasm-bindgen. v2.3 added implied vol, conventions, and local vol, so the WASM bindings now cover all five layers. See [`wasm/README.md`](wasm/README.md) for the JavaScript API and usage examples.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full history. Recent releases:

| Version | Name | Key Features |
|---------|------|--------------|
| **v3.0** | **Strict Contracts** | **Silent calibration and arbitrage-scan fallbacks are now errors; `StickyKind` removed, `NormalVol` newtype** |
| v2.4 | Published | Internal consolidation, Python/WASM CI, first crates.io release since 2.1 |
| v2.3 | WASM Parity | Implied vol, conventions, and local vol in the WASM bindings |
| v2.2 | Local Vol Boundary | `BoundaryLocalVol` small-time adapter, `with_boundary()` |
| v2.1 | API Polish | `model_name()`, `tenors()`, configurable arbitrage scans and calibration |
| v2.0 | Type-Safe Inputs | `Strike`/`Tenor` newtypes for all API inputs |
| v1.0 | Stable | API stability, PyO3 bindings, WASM target |

## References

- Gatheral, J. *The Volatility Surface: A Practitioner's Guide* (2006)
- Gatheral, J. & Jacquier, A. "Arbitrage-free SVI Volatility Surfaces" (2014)
- Hagan, P. et al. "Managing Smile Risk", *Wilmott* (2002)
- Jackel, P. "Let's Be Rational" (2013)
- Zeliade Systems, "Quasi-Explicit Calibration of Gatheral's SVI Model" (2009)
- Hendriks, S. & Martini, C. "The Extended SSVI Volatility Surface" (2019)
- Dupire, B. "Pricing with a Smile", *Risk* (1994)
- Jackel, P. "Implied Normal Volatility" (2017)
- Breeden, D.T. & Litzenberger, R.H. "Prices of State-Contingent Claims Implicit in Option Prices" (1978)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
