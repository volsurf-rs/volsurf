# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.3.0] - 2026-06-06

### Added

- WASM binding parity with the Python crate (PAN-28):
  - Implied vol: `blackPrice`/`normalPrice`/`displacedPrice` undiscounted pricing fns, `WasmBlackImpliedVol`/`WasmNormalImpliedVol` (static `compute`) and `WasmDisplacedImpliedVol` (instance, with `beta`) for Black/Normal/displaced-diffusion IV extraction, and a `WasmOptionType` (`Call`/`Put`) enum
  - Conventions: `logMoneyness`, `moneyness`, `forwardPrice` helpers
  - Local vol: `WasmDupireLocalVol` and `WasmBoundaryLocalVol` (the v2.2 PAN-25 small-time boundary adapter), reachable from any surface via `dupireLocalVol(bumpSize?)` / `dupireLocalVolWithBoundary(bumpSize?)` on `WasmSsviSurface`, `WasmEssviSurface`, and `WasmPiecewiseSurface`
  - WASM smoke tests covering price→IV round-trips, convention known-values, flat-surface `σ_loc ≡ σ`, and the `t = 0` boundary rescue
- Lockstep version bump of all three crates (core, `volsurf-python`, `volsurf-wasm`) to 2.3.0; core `src/` is unchanged in this release

## [2.2.0] - 2026-06-06

### Added

- `BoundaryLocalVol<L: LocalVol>` — opt-in adapter that smooths the Dupire small-time boundary: a query at `t ≤ floor` evaluates the inner local vol at `t = floor` (keeping total variance `w = σ²·T` away from the singular `1/w`, `k²/w²` terms in the Gatheral denominator as `T → 0`); `t > floor` delegates exactly to the strict path (PAN-25)
- `DupireLocalVol::with_boundary()` — wraps a `DupireLocalVol` in a `BoundaryLocalVol` whose `floor` defaults to the finite-difference `bump_size`
- `BoundaryLocalVol` and `DupireLocalVol` re-exported at the crate root

## [2.1.0] "API Polish" - 2026-03-25

### Added

- `SmileSection::model_name()` — returns the model identifier (`"SVI"`, `"SABR"`, `"CubicSpline"`, `"SSVI"`, `"eSSVI"`)
- `VolSurface::tenors()` — accessor returning the surface's tenors as `&[f64]`
- `expiry` field on `ArbitrageReport` for per-tenor attribution
- Configurable arbitrage scanning via `ArbitrageScanConfig`: `SmileSection::is_arbitrage_free_with(config)` and `VolSurface::diagnostics_with(config)`
- Configurable calibration: `DataFilter`, `WeightingScheme`, and warm-starting via `calibrate_with_config` on smile models
- `Clone` + `PartialEq` on `VolSurfError`

### Changed

- **BREAKING**: `is_arbitrage_free()` is now computed on demand from the smile/surface rather than stored at construction
- SVI warm-start falls back to grid search when the seeded optimization diverges

## [2.0.0] "Type-Safe Inputs" - 2026-03-13

### Changed

- **BREAKING**: All trait method inputs now use `Strike`/`Tenor` newtypes instead of bare `f64` — `SmileSection::vol(Strike)`, `VolSurface::black_vol(Tenor, Strike)`, `LocalVol::local_vol(Tenor, Strike)`, etc.
- Python and WASM bindings unchanged — FFI boundary wraps `f64 → Strike`/`Tenor` internally
- Updated `types.rs` module docs to reflect input newtypes strategy
- Extracted `impl_wasm_smile_methods!` macro to DRY SmileSection wrappers in WASM crate

### Added

- NaN/Infinity rejection tests for all smile models, slices, and surfaces

## [1.0.0] "Stable" - 2026-02-25

### Added

- `volsurf-wasm` crate — WebAssembly bindings via `wasm-bindgen` for SVI, SABR, SSVI, eSSVI, and SurfaceBuilder with 27 smoke tests
- `volsurf-python` crate — PyO3 bindings with NumPy integration, serde round-trip support, and 207 tests
- WASM CI job (build + clippy) in GitHub Actions

### Changed

- eSSVI Stage 2/3 calibration optimizations: precomputed `ln(xs)/ln(theta_ratio)` for `exp` instead of `powf`, adaptive rho grid, 21-point quadratic `a`-scan
- API stability review: sealed internal modules, documented all public types, ensured `Send + Sync + Debug` on all traits

### Fixed

- README updated with v1.0 version numbers, bindings section, and changelog

## [0.4.0] - 2026-02-22

### Changed

- Tracing fields in calibration diagnostics: `rms` renamed to `rms_implied_vol` (SABR) and `rms_total_variance` (SSVI/eSSVI) to clarify the metric space

### Removed

- `VolSurfError::ArbitrageViolation` variant — was unused; `VolSurfError` is `#[non_exhaustive]` so downstream wildcard matches are unaffected, but code referencing this variant by name will need updating

### Added

- Non-uniform strike calibration round-trip tests for SVI and SABR
- 12 coverage gap tests across SVI, SABR, SSVI, and arbitrage modules

## [0.3.0] "Production Grade" - 2026-02-22

### Added

- `EssviSurface` — Hendriks-Martini (2019) extended SSVI with tenor-dependent rho for calendar arbitrage freedom
- `EssviSlice` — zero-cost newtype over `SsviSlice` with baked-in rho(theta)
- `EssviSurface::calibrate()` — 3-stage calibration: per-tenor SVI, rho(theta) regression, global (eta, gamma) optimization with Eq. 5.7 constraint enforcement
- `SurfaceBuilder::dividend_yield()` for forward calculation via F = S*exp((r-q)*T)
- `SurfaceBuilder::add_tenor_with_forward()` to bypass forward computation with market-observed forwards
- `log_moneyness()`, `moneyness()`, `forward_price()` now return `Result<f64>` with input validation
- Parallel surface construction via `rayon` feature in `SurfaceBuilder::build()`
- Dupire local vol benchmarks validating NFR performance targets
- SECURITY.md with private vulnerability reporting via GitHub Security Advisories

### Fixed

- Better error messages when calibration produces non-monotone ATM total variances
- Integration tests use non-constant forwards for realistic DJX scenarios

## [0.2.1] - 2026-02-17

### Added

- `NormalImpliedVol` — Bachelier implied vol extraction via Jäckel (2017) rational approximation with `normal_price()` standalone pricing function
- `DisplacedImpliedVol` — displaced diffusion model with beta-blended Black/Normal pricing and IV extraction; delegates to pure Black (β=1) or Normal (β=0) at boundaries
- `DupireLocalVol` — local volatility extraction from any `VolSurface` via Gatheral (2006) Eq. 1.10 using finite differences on total implied variance, with forward-adjusted time derivatives at constant log-moneyness
- GitHub Actions CI workflow (test, clippy, fmt, doc)
- Apache-2.0 LICENSE file
- README with badges, quick-start guide, benchmarks, and architecture overview
- crates.io publish metadata (keywords, categories, repository, homepage)

### Fixed

- Serde deserialization now validates all smile/surface types via `#[serde(try_from)]` — `SsviSurface`, `SsviSlice`, `SabrSmile`, `SviSmile`, `SplineSmile`
- Black IV accuracy claim corrected from "3 ULP" to "near-machine-precision" in module docs
- Normal IV accuracy claim corrected from "2 ULP" to "near-machine-precision" in module docs
- 14 edge case tests added from implied vol paper audits (5 black, 5 normal, 4 displaced)

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
- `VolSurfError` enum with `thiserror`, `#[non_exhaustive]`, and 4 structured variants: `CalibrationError`, `InvalidInput`, `NumericalError`, `ArbitrageViolation` (removed in v0.4.0)
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

[Unreleased]: https://github.com/volsurf-rs/volsurf/compare/v2.3.0...HEAD
[2.3.0]: https://github.com/volsurf-rs/volsurf/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/volsurf-rs/volsurf/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/volsurf-rs/volsurf/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/volsurf-rs/volsurf/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/volsurf-rs/volsurf/compare/v0.4.0...v1.0.0
[0.4.0]: https://github.com/volsurf-rs/volsurf/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/volsurf-rs/volsurf/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/volsurf-rs/volsurf/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/volsurf-rs/volsurf/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/volsurf-rs/volsurf/releases/tag/v0.1.0
