# volsurf-wasm

WebAssembly bindings for the [volsurf](https://crates.io/crates/volsurf) volatility surface library.

## Install

```bash
npm install volsurf-wasm
```

Built with `wasm-pack --target web`: an ES module whose default export,
`init()`, loads the `.wasm` binary and must be awaited before any other call.

## Quick Start

```typescript
import init, { WasmSviSmile, WasmSurfaceBuilder } from "volsurf-wasm";

await init();

// Construct an SVI smile directly from parameters
const smile = new WasmSviSmile(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1);
console.log(smile.vol(100.0));  // ATM implied vol

// Calibrate from market data (flattened strike/vol pairs)
const calibrated = WasmSviSmile.calibrate(100.0, 1.0, [
  80, 0.28,  90, 0.24,  95, 0.22,  100, 0.20,  105, 0.22,  110, 0.24,  120, 0.28,
]);
console.log(calibrated.vol(100.0));
```

## Building from Source

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

wasm-pack build wasm/ --target web
```

Output in `wasm/pkg/`: `.wasm` binary, `.js` loader, `.d.ts` type definitions, `package.json`.

## API

### Implied Vol

Undiscounted pricing and implied-vol extraction (Jäckel rational approximations).
`WasmOptionType` is a `{ Call, Put }` enum passed into every pricer.

```typescript
import { WasmOptionType, black_price, WasmBlackImpliedVol,
         normal_price, WasmNormalImpliedVol,
         displaced_price, WasmDisplacedImpliedVol } from "volsurf-wasm";

// Black (lognormal)
const price = black_price(100, 100, 0.20, 1.0, WasmOptionType.Call);
const iv = WasmBlackImpliedVol.compute(price, 100, 100, 1.0, WasmOptionType.Call);  // ~0.20

// Normal (Bachelier) — vol is in price units
const np = normal_price(100, 100, 20.0, 1.0, WasmOptionType.Put);
const niv = WasmNormalImpliedVol.compute(np, 100, 100, 1.0, WasmOptionType.Put);   // ~20.0

// Displaced diffusion (interpolates normal ↔ Black); instance carries beta ∈ [0, 1]
const dp = displaced_price(100, 100, 0.20, 1.0, 0.5, WasmOptionType.Call);
const calc = new WasmDisplacedImpliedVol(0.5);
calc.beta;                                              // 0.5
const div = calc.compute(dp, 100, 100, 1.0, WasmOptionType.Call);  // ~0.20
```

### Conventions

```typescript
import { log_moneyness, moneyness, forward_price } from "volsurf-wasm";

log_moneyness(100, 100);        // ~0      (k = ln(K / F))
moneyness(120, 100);           // ~1.2    (m = K / F)
forward_price(100, 0.05, 0, 1); // ~105.127 (F = S·exp((r − q)·T))
```

### Smiles

**WasmSviSmile** — SVI parametric model (Gatheral 2004)

```typescript
// From parameters
const svi = new WasmSviSmile(forward, expiry, a, b, rho, m, sigma);

// From market data: [strike1, vol1, strike2, vol2, ...] (min 5 pairs)
const svi = WasmSviSmile.calibrate(forward, expiry, marketVolsFlat);

svi.vol(strike)       // implied vol at strike
svi.variance(strike)  // total variance (sigma^2 * T)
svi.density(strike)   // risk-neutral density (Breeden-Litzenberger)
svi.forward           // forward price
svi.expiry            // time to expiry
svi.to_json()          // serialize to JSON string
WasmSviSmile.from_json(s)  // deserialize
```

**WasmSabrSmile** — SABR stochastic vol model (Hagan 2002)

```typescript
const sabr = new WasmSabrSmile(forward, expiry, alpha, beta, rho, nu);
const sabr = WasmSabrSmile.calibrate(forward, expiry, beta, marketVolsFlat);
// Same query methods as SVI
```

### Surfaces

**WasmSsviSurface** — Global SSVI parameterization (Gatheral-Jacquier 2014)

```typescript
const ssvi = new WasmSsviSurface(rho, eta, gamma, tenors, forwards, thetas);
ssvi.black_vol(expiry, strike)
ssvi.black_variance(expiry, strike)
ssvi.rho    // getter
ssvi.eta    // getter
ssvi.gamma  // getter
ssvi.tenors()
ssvi.forwards()
ssvi.thetas()
ssvi.to_json() / WasmSsviSurface.from_json(s)
```

**WasmEssviSurface** — Extended SSVI with maturity-dependent correlation (Hendriks-Martini 2019)

```typescript
const essvi = new WasmEssviSurface(rho_0, rho_m, a, eta, gamma, tenors, forwards, thetas);
// Same query methods as SSVI, plus:
essvi.rho_0
essvi.rho_m
essvi.rho_exponent
essvi.theta_max
```

**WasmSurfaceBuilder** — Piecewise surface from market data

```typescript
const builder = new WasmSurfaceBuilder();
builder.spot(100.0);
builder.rate(0.05);
builder.model_sabr(0.5);  // or model_svi(), model_cubic_spline()
builder.add_tenor(0.25, strikes, vols);
builder.add_tenor(1.0, strikes, vols);
const surface = builder.build();  // returns WasmPiecewiseSurface

surface.black_vol(0.5, 100.0)
surface.black_variance(0.5, 100.0)
```

### Local Vol

Dupire local volatility (Gatheral 2006, Eq. 1.10) composed over any surface.
Because WASM has no unified surface type, obtain a local-vol object by calling a
method on the surface — available on `WasmSsviSurface`, `WasmEssviSurface`, and
`WasmPiecewiseSurface`. `bump_size` is optional (defaults to 0.01).

```typescript
const lv = surface.dupire_local_vol();        // WasmDupireLocalVol (optional bump_size)
lv.local_vol(0.5, 100.0);                     // σ_loc at (expiry, strike)

// Boundary adapter (v2.2 / PAN-25): a query at t ≤ floor (the bump size) is
// evaluated at t = floor, rescuing the t → 0 singularity of the strict path.
const bdy = surface.dupire_local_vol_with_boundary();  // WasmBoundaryLocalVol
bdy.local_vol(0.0, 100.0);                    // succeeds where dupire_local_vol throws at t = 0
```

### Error Handling

All constructors and query methods throw on invalid input or calibration failure:

```typescript
try {
  const smile = new WasmSviSmile(100, 1, 0.04, 0.4, -0.4, 0, 0.1);
  const vol = smile.vol(100);
} catch (e) {
  console.error(e);  // string describing the error
}
```

The thrown value is a plain string carrying the core error message, with no
variant tag: an invalid-input rejection and a numerical failure are
indistinguishable to a caller. (The Python bindings do separate them, by
raising `ValueError` and `RuntimeError` respectively.)

### Serialization

All model types support JSON round-trip:

```typescript
const json = smile.to_json();
const restored = WasmSviSmile.from_json(json);
```

## Browser Usage

```html
<script type="module">
  import init, { WasmSviSmile } from './pkg/volsurf_wasm.js';
  await init();
  const smile = new WasmSviSmile(100, 1, 0.04, 0.4, -0.4, 0, 0.1);
  console.log(smile.vol(100));
</script>
```

## Demo

`wasm/demo.html` is a self-contained page that calibrates SVI and SABR smiles in the
browser and plots the smile grid, term structure, risk-neutral density, and delta smile.

It imports `./pkg/volsurf_wasm.js`, which is gitignored, so build first — and serve over
HTTP, since ES modules and `fetch` of the `.wasm` do not work from `file://`:

```bash
wasm-pack build wasm/ --target web
python3 -m http.server --directory wasm 8000
```

Then open <http://localhost:8000/demo.html>.

## License

Apache-2.0
