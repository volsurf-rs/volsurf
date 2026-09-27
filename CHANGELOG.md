# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.1] - 2026-09-27

### Fixed

- `SviSmile`'s identifiability docs, added in 4.0.0, covered only `m` outside
  the quoted range and gave the linear limit's sign for `m` above the ladder
  alone. `SviSmile` now documents when its five parameters stop being
  separately identified, and every accessor points at that section. The
  total-variance curve goes linear where `|k − m| ≫ σ` across the quoted
  ladder — pinning `b(1 − ρ)` when `m` sits above the ladder and `b(1 + ρ)`
  when it sits below, with `b` and `ρ` trading off along the curve
  `b(1 ∓ ρ) = const` and `σ` dropping out entirely — and flattens the same way
  when `σ` is wide relative to the ladder, leaving only two pinned numbers, the
  slope `b(ρ − m/σ)` and the line's intercept. Comparing `m` against the quoted
  strike range is no longer offered as a check: it both over- and under-flags.

## [4.0.0] - 2026-08-23

### Added

- `types::DisplacedVol` — the displaced-diffusion vol parameter, which is a
  Black vol only at β = 1 and was being returned as one at every β.
- `SmileSection::default_scan_config()` — the grid `is_arbitrage_free()` scans.
  A model whose approximation stops short of the wings states its own domain
  once instead of overriding `is_arbitrage_free()` to pass a grid, and an
  implementor outside this crate no longer silently inherits SVI's.
- `ArbitrageScanConfig::default()`, so
  `ArbitrageScanConfig { n_points: 500, ..Default::default() }` compiles.
- `SplineSmile::calibrate` and `calibrate_with_config` — fit a spline from
  `(strike, vol)` quotes like SVI and SABR do. `new()` takes sorted strikes and
  total variances, and every caller was writing that conversion itself.
  `calibrate_with_config` takes no `WeightingScheme`: a spline passes through
  every surviving quote, so no residual bears a weight. Both reach the Python
  bindings as `SplineSmile.calibrate` and `SplineSmile.calibrate_with_config`;
  WASM exposes no spline smile type, so there is no JS counterpart.
- Accessors for parameters that could previously only be read back through
  serde: `SviSmile::a/b/m/sigma` and `SplineSmile::strikes/variances`.
- `VolSurface::forward(expiry)` — reads the forward directly instead of
  building a whole smile section for it. `DupireLocalVol` needs three forwards
  and no vols per query; on a `PiecewiseSurface` that used to cost ~150
  variance evaluations, three spline solves, and three allocations. The
  `local_vol/dupire_piecewise_single_query` benchmark goes from 6.75 µs to
  180 ns, a 37× speedup.
- `VolSurface::calendar_violations()` — calendar spread checks reachable
  through `&dyn VolSurface`, with a grid-scanning default. `SsviSurface`
  overrides it with the exact `∂w/∂θ` test.
- `smile::SmileCalibrator` — the per-tenor calibration contract the models
  already shared informally. `SurfaceBuilder::calibrator()` accepts any
  implementation, so a model defined outside this crate can be built into a
  surface on the same footing as the `SmileModel` variants. Its
  `validate()` method (default `Ok(())`) checks parameters fixed at
  construction; `SurfaceBuilder::build()` calls it before reading any tenor
  data, so a misconfigured model reports its own error rather than whatever
  the first tenor trips over.
- `validate_in_range` and `validate_open_unit_interval` behind the ρ, β, and γ
  checks, so those messages are uniform across models.

### Changed

- **BREAKING**: `black_price`, `normal_price` and `displaced_price` take the vol
  newtype their extractor returns — `Vol`, `NormalVol` and `DisplacedVol`
  respectively — rather than a bare `f64`. A Bachelier vol of `20.0` and a Black
  vol of `0.20` were interchangeable at the one call site where the two units
  meet. Wrap the argument: `black_price(f, k, Vol(0.2), t, ty)`, or pass an
  extractor's output straight through. The Python and WASM bindings still take
  and return bare floats.
- **BREAKING**: `DisplacedImpliedVol::compute` returns `DisplacedVol` rather
  than `Vol`. Both are tuple newtypes, so `.0` still reaches the `f64`.
- **BREAKING**: `ArbitrageScanConfig::svi_default()` and `sabr_default()` are
  `wide()` and `narrow()`. They describe grid width, not models — `svi_default()`
  was also the SSVI default and the trait-wide fallback. `Default` returns
  `wide()`. Renamed in the Python and WASM bindings too.
- **BREAKING**: `DataFilter`, `WeightingScheme` and `ArbitrageScanConfig` are
  passed by value, not by reference, on every `calibrate_with_config`,
  `is_arbitrage_free_with`, `diagnostics_with`, `apply_filter` and
  `SmileCalibrator::calibrate`. All three are `Copy`, and
  `SurfaceBuilder::data_filter` already took one by value. Drop the `&`.
- **BREAKING**: surface calibration takes `(tenors, forwards, market_data)`
  rather than `(market_data, tenors, forwards)`, matching the smile layer's
  coordinates-then-quotes order. Affects `SsviSurface::calibrate*`,
  `EssviSurface::calibrate*` and `EssviSurface::fit_per_tenor*`, in the Python
  and WASM bindings as well. `market_data` has a distinct type, so a call left
  in the old order fails to compile rather than mis-binding.
- **BREAKING**: `EssviSurface::rho(theta)` is `rho_at(theta)` and
  `EssviSurface::a()` is `rho_exponent()`. `rho()` everywhere else in the crate
  reads a stored parameter; on `EssviSurface` alone it evaluated a function.
  `a` is the exponent in ρ(θ) = ρ₀ + (ρₘ − ρ₀)(θ/θ_max)^a — still the name of
  the constructor argument, which follows the paper. The `a` half lands in the
  bindings too: the Python getter `EssviSurface.a` and the WASM getter
  `WasmEssviSurface.a` are `rho_exponent`, and unlike the Rust rename those
  fail at runtime — `essvi.a` raises `AttributeError` in Python and is
  `undefined` in JS. (`rho_at` is not exposed in either binding.) Every
  `EssviSurface` accessor now carries a doc comment.
- **BREAKING**: `EssviSurface::calendar_check_structural()` is
  `calendar_violations_structural()`, matching the `VolSurface::calendar_violations()`
  it sits beside.
- `PiecewiseSurface::smile_at()` returns the stored smile on an exact tenor
  match rather than a cubic-spline resampling of it. The section now keeps its
  model identity (`model_name()` reports `"SVI"`, not `"CubicSpline"`), its
  analytic density, and its wing behaviour — previously `smile_at(T).vol(K)`
  and `black_vol(T, K)` disagreed outside `[0.5F, 2F]` on the same surface.
  Off-grid expiries are still resampled onto a spline.
- On `SmileModel::CubicSpline`, `SurfaceBuilder::build()` now validates each
  quote before applying the `DataFilter`, so a non-finite or non-positive vol
  errors with `InvalidInput` instead of being filtered out and fitted around.
  The spline arm routes through `SplineSmile::calibrate_with_config`, which
  aligns it with the SVI and SABR arms. A build with a `min_vol` filter and one
  zero or NaN vol in a chain that previously succeeded on the surviving quotes
  now fails that tenor.
- `ArbitrageScanConfig` is re-exported at the crate root, alongside `DataFilter`.
- SVI's calibration is split into named stages (weighting, vol-cliff filter,
  ATM interpolation, multi-start search, ATM sanity check) that are unit-tested
  directly. The fit itself is unchanged.
- Error message wording, on the same `VolSurfError` variants as before. Match
  on the variant, not the string:
  - Bad tenors or forwards passed to `SsviSurface::calibrate*` or
    `EssviSurface::fit_per_tenor*` now read
    `"tenors must be positive and finite, got tenors[0]=0"` rather than
    `"tenors[0] must be positive and finite, got 0"`. Both surfaces route
    through the shared slice validator, so they no longer disagree.
  - An out-of-range SABR `beta` reads `"SABR beta must be in [0, 1], got NaN"`
    rather than `"SABR beta must be in [0, 1] and finite, got NaN"`; `NaN` and
    `inf` are still rejected. `SabrSmile::new`/`calibrate_with_config` name the
    parameter the same way (was `"beta must be in [0, 1]"`), and the Python and
    WASM bindings raise the core message instead of their own copy of it.
  - `SurfaceBuilder::build()`'s min-strikes error names the model by
    `model_name()` — `"(model: SABR)"` — instead of debug-formatting the
    `SmileModel`, so the message no longer carries `beta`.
- WASM errors from `InvalidInput` and `NumericalError` now carry the bare
  message rather than the `Display` form, matching the Python bindings: a JS
  caller sees `"tenors must be positive and finite, got tenors[0]=0"`, not
  `"invalid input: tenors must be positive and finite, got tenors[0]=0"`. This
  applies to every message on those two variants, not just SABR `beta`. It also
  makes `InvalidInput` and `NumericalError` indistinguishable to a JS caller;
  `CalibrationError` still carries its `"{model}: {message}"` prefix. These
  bindings throw plain strings rather than a typed value, so they offer no
  equivalent of the Python bindings' `ValueError`/`RuntimeError` split.
- `SplineSmile` now overrides `is_arbitrage_free_with` instead of
  `is_arbitrage_free`, so both entry points scan the same domain. They
  previously disagreed: `is_arbitrage_free()` ran a hand-rolled scan over the
  knot range with a hardcoded point count and tolerance and ignored
  `ArbitrageScanConfig` entirely, while `is_arbitrage_free_with(config)` fell
  through to the trait default and scanned log-moneyness `[k_min, k_max]`.
  The configured range is now honoured but clipped to the knot range, stepping
  in from the boundary knots. Outside `[K₀, Kₙ]` the spline flat-extrapolates,
  where the finite-difference density is cancellation noise, and `w` is only C⁰
  at the boundary itself — on a convex 5-knot smile the unclipped scan reported
  six spurious violations, two of them at magnitude ~13. A `config` that does
  not overlap the knot range now returns `InvalidInput` rather than silently
  reporting a clean scan it never performed.

- `SviSmile::b` and `SviSmile::rho` document that neither is identified when
  `m` falls outside the quoted log-moneyness range. The total-variance curve
  goes linear there and the two trade off along a ray, so a low RMSE does not
  imply a determined `b`, and `rho`'s sign stops tracking which wing is
  steeper.

### Removed

Breaking. Each has a drop-in replacement on the `VolSurface` trait — bring it
into scope with `use volsurf::surface::VolSurface`:

- `SsviSurface::calendar_arb_analytical()` → `SsviSurface::calendar_violations()`,
  which now returns `Result<Vec<CalendarViolation>>`.
- The inherent `SsviSurface::tenors()` and `EssviSurface::tenors()`, which
  shadowed the identical trait method.

Breaking, Python bindings only:

- Python 3.9 support. `requires-python` is now `>=3.10` and CI tests 3.10
  and 3.14. 3.9 reached end of life in October 2025, and supporting it held
  the test matrix on a pytest release predating the CVE-2025-71176 tmpdir
  fix, which needs 3.10+. The Rust crate is unaffected.

## [3.0.0] - 2026-08-22

A major bump for the API contract, not for new capability. Three public items
changed shape and several calls that used to degrade silently now return an
error instead (PAN-135).

Migrating:

- `conventions::StickyKind` is gone. Nothing consumed it, so there is no
  replacement to adopt — remove the import.
- `NormalImpliedVol::compute` returns `types::NormalVol` rather than `Vol`. Both
  are tuple newtypes, so `.0` still reaches the `f64`.
- `ButterflyViolation::magnitude` is a method: `v.magnitude` becomes
  `v.magnitude()`.
- Calibrations that relied on a `DataFilter` quietly falling back to the
  unfiltered data now fail with a `CalibrationError`. Widen the filter, or lower
  `min_points`, to get the previous fit.
- Butterfly scans that could not evaluate every grid point returned a clean
  report; they now return a `NumericalError`. A report means the whole grid was
  covered.

The Python and WASM bindings keep their existing signatures — both already
returned bare floats, and both wrappers absorb the `magnitude` change.

### Removed

- **BREAKING**: `conventions::StickyKind`. The enum had no consumer anywhere in the
  crate — nothing accepted it and nothing returned it — so the sticky-strike /
  sticky-delta choice its docs described could not actually be applied. It will
  return alongside an API that honours it.

### Changed

- **BREAKING**: `NormalImpliedVol::compute` returns the new `types::NormalVol` rather
  than `Vol`. Bachelier volatility is quoted in price units per √year while `Vol` is an
  annualized proportion, so the newtype that exists to stop unit mixing was itself
  mixing units. The Python and WASM bindings are unaffected — both return a bare float.
- **BREAKING**: `ButterflyViolation::magnitude` is now a method rather than a public
  field, and is computed as `density.abs()`. As two public fields they could disagree.
  Serialized violations lose the redundant `magnitude` key as a result.

### Fixed

- **BREAKING**: a `DataFilter` that leaves fewer points than the model needs is now a
  `CalibrationError` instead of a silent fallback to the unfiltered data. Calibrating on
  the full set gave a fit the caller never asked for and had no way to detect. Affects
  SVI, SABR, SSVI, eSSVI and the builder's cubic-spline path; the remedy is to widen the
  filter, and the error reports how many points survived out of how many. The same now
  holds for `DataFilter::vol_cliff_filter`, which is on by default for SVI and for the
  per-tenor SVI stage of SSVI/eSSVI: a cliff that leaves fewer than five points on the
  retained side is a `CalibrationError` rather than a fit across the cliff.
- **BREAKING**: `PiecewiseSurface::new` now rejects a smile whose `expiry()` disagrees
  with the tenor it is paired with, including a non-finite `expiry()`. Queries locate
  smiles by the tenor grid, so a mismatched pair was evaluated at the wrong maturity.
- `SurfaceBuilder::build` no longer requires `spot` and `rate` when every tenor was added
  through `add_tenor_with_forward`. Futures-options surfaces, where the forward is the
  futures price and there is no spot to quote, previously had to pass placeholder values —
  and `spot` had to be positive, so even that was awkward.
- **BREAKING**: butterfly arbitrage scans no longer skip grid points whose density
  cannot be evaluated. `is_arbitrage_free()` and `is_arbitrage_free_with()` now
  return `NumericalError` naming the offending strike, so a returned report always
  covers the whole configured grid. Previously a smile with non-positive total
  variance reported as arbitrage-free — and in the g-function path (SVI, SSVI,
  eSSVI) the skipped points were violations that had already been detected.
  Behaviour is unchanged for models built through their validated constructors.
- The release workflow now skips `cargo publish` when the tagged version is
  already on crates.io, so tagging a version that was published by hand no
  longer fails the run. It also selects the workspace package by name rather
  than by position when checking the tag against the manifest.

### Security

- PyO3 0.28 → 0.29 and `rust-numpy` 0.28 → 0.29, clearing RUSTSEC advisories for an
  out-of-bounds read in `PyList`/`PyTuple` `nth`/`nth_back` and a missing `Sync`
  bound on `PyCFunction::new_closure`. The two move together because `rust-numpy`
  pins PyO3 and PyO3 sets `links = "python"`, so the graph admits one version.
  Neither advisory affects the published `volsurf` crate, which has no PyO3
  dependency; only the `volsurf-python` binding is impacted.
- Require pytest ≥ 9.0.3 on Python 3.10+ for the tmpdir advisory

## [2.4.0] - 2026-08-19

First release published to crates.io since 2.1.0. Versions 2.2.0 and 2.3.0
were tagged and changelogged but never uploaded, so `volsurf = "2.3"` did not
resolve; they remain git-only releases and their contents ship here (PAN-134).

### Added

- Python `DupireLocalVol` accepts `SsviSurface` and `EssviSurface` directly, not
  only the opaque surface handle
- CI job running the Python binding's pytest suite against Python 3.9 and 3.14
  via `uv sync --locked` + `uv run pytest`, with pytest declared in a `dev`
  dependency group (PAN-129)
- CI job running the WASM tests via `wasm-pack test --node` (PAN-41)

### Changed

- Consolidated volatility model internals across smile, surface and implied-vol
  modules — shared validation, optimizer and arbitrage-scan helpers, with no
  change to the public Rust API (−77 net lines across 18 files)

### Removed

- `homepage` metadata in `Cargo.toml` — the site it pointed at no longer exists;
  crates.io falls back to `repository`

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

[Unreleased]: https://github.com/volsurf-rs/volsurf/compare/v4.0.1...HEAD
[4.0.1]: https://github.com/volsurf-rs/volsurf/compare/v4.0.0...v4.0.1
[4.0.0]: https://github.com/volsurf-rs/volsurf/compare/v3.0.0...v4.0.0
[3.0.0]: https://github.com/volsurf-rs/volsurf/compare/v2.4.0...v3.0.0
[2.4.0]: https://github.com/volsurf-rs/volsurf/compare/v2.3.0...v2.4.0
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
