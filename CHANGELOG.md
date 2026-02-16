# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] "Market Ready" - 2026-02-16

### Added

- `SabrSmile` — Hagan (2002) SABR implied vol with unified code path, Taylor expansion for small z, and 12-digit accuracy against reference values
- `SabrSmile::calibrate()` — analytic alpha via Newton on ATM cubic, rho/nu optimization via Nelder-Mead in transformed parameter space with 15x15 grid initialization
- `SsviSurface` — Gatheral-Jacquier (2014) global SSVI parameterization with power-law phi function, theta interpolation, and flat-vol extrapolation
- `SsviSlice` — lightweight single-tenor SSVI evaluator with analytical first and second derivatives for g-function butterfly detection
- `SsviSurface::calibrate()` — two-stage calibration: per-tenor SVI to extract theta/rho, then global (eta, gamma) optimization
- `SsviSurface::calendar_arb_analytical()` — analytical calendar arbitrage detection via dw/dtheta derivative
- `ArbitrageReport::merge()` and `worst_violation()` for combining and summarizing multi-tenor diagnostic results
- `SmileModel::Sabr { beta }` variant for `SurfaceBuilder` integration (minimum 4 strikes per tenor)
- `examples/sabr_smile.rs` — SABR calibration and smile evaluation
- `examples/ssvi_surface.rs` — SSVI surface construction and querying
- Runnable doc examples on 8 core public API items

## [0.1.0] "First Light" - 2026-02-15

### Added

- Domain newtypes: `Strike`, `Tenor`, `Vol`, `Variance`, `OptionType` with `Copy`, `Debug`, `Serde` support
- `VolSurfError` enum with `thiserror`, `#[non_exhaustive]`, and 4 structured variants: `CalibrationError`, `InvalidInput`, `NumericalError`, `ArbitrageViolation`
- `SmileSection` trait (`Send + Sync + Debug`) for single-tenor smile evaluation with `vol()`, `variance()`, `density()`, `forward()`, `expiry()`, `is_arbitrage_free()`
- `VolSurface` trait (`Send + Sync + Debug`) for multi-tenor surfaces with `black_vol()`, `black_variance()`, `smile_at()`, `diagnostics()`
- `LocalVol` trait for future Dupire local vol extraction
- `BlackImpliedVol` — Black-Scholes implied vol extraction via Jackel rational approximation with round-trip accuracy < 1e-12
- `black_price()` — undiscounted Black-Scholes call/put pricing
- `SviSmile` — SVI parameterization (a, b, rho, m, sigma) with Gatheral-Jacquier validation, analytical density via g-function, and butterfly arbitrage detection
- `SviSmile::calibrate()` — Zeliade (2009) quasi-explicit method with linear least-squares, 15x15 grid search, and Nelder-Mead refinement
- `SplineSmile` — natural cubic spline on variance with Thomas algorithm, binary search, and flat extrapolation
- `PiecewiseSurface` — per-tenor `SmileSection` storage with linear variance interpolation, arbitrary-tenor `smile_at()`, and calendar + butterfly diagnostics
- `SurfaceBuilder` — fluent API for surface construction: `.spot()`, `.rate()`, `.tenor()`, `.model()`, `.build()` with forward price computation and auto-sorting by expiry
- `SmileModel` enum — selector for `SurfaceBuilder`: `Svi` (default, 5+ strikes) and `CubicSpline` (3+ strikes)
- Default `density()` on `SmileSection` via numerical Breeden-Litzenberger
- `StickyKind` enum, `log_moneyness()`, `moneyness()`, `forward_price()` utilities
- `ArbitrageReport`, `ButterflyViolation`, `SurfaceDiagnostics`, `CalendarViolation` diagnostic types
- `parallel` Cargo feature for optional rayon support
- `logging` Cargo feature for optional tracing instrumentation
- Examples: `basic_surface`, `smile_models`, `implied_vol`

[Unreleased]: https://github.com/volsurf-rs/volsurf/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/volsurf-rs/volsurf/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/volsurf-rs/volsurf/releases/tag/v0.1.0
