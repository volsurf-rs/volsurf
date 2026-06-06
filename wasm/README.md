# volsurf-wasm

WebAssembly bindings for the [volsurf](https://crates.io/crates/volsurf) volatility surface library.

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
import { WasmOptionType, blackPrice, WasmBlackImpliedVol,
         normalPrice, WasmNormalImpliedVol,
         displacedPrice, WasmDisplacedImpliedVol } from "volsurf-wasm";

// Black (lognormal)
const price = blackPrice(100, 100, 0.20, 1.0, WasmOptionType.Call);
const iv = WasmBlackImpliedVol.compute(price, 100, 100, 1.0, WasmOptionType.Call);  // ~0.20

// Normal (Bachelier) — vol is in price units
const np = normalPrice(100, 100, 20.0, 1.0, WasmOptionType.Put);
const niv = WasmNormalImpliedVol.compute(np, 100, 100, 1.0, WasmOptionType.Put);   // ~20.0

// Displaced diffusion (interpolates normal ↔ Black); instance carries beta ∈ [0, 1]
const dp = displacedPrice(100, 100, 0.20, 1.0, 0.5, WasmOptionType.Call);
const calc = new WasmDisplacedImpliedVol(0.5);
calc.beta;                                              // 0.5
const div = calc.compute(dp, 100, 100, 1.0, WasmOptionType.Call);  // ~0.20
```

### Conventions

```typescript
import { logMoneyness, moneyness, forwardPrice } from "volsurf-wasm";

logMoneyness(100, 100);        // ~0      (k = ln(K / F))
moneyness(120, 100);           // ~1.2    (m = K / F)
forwardPrice(100, 0.05, 0, 1); // ~105.127 (F = S·exp((r − q)·T))
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
svi.toJson()          // serialize to JSON string
WasmSviSmile.fromJson(s)  // deserialize
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
ssvi.blackVol(expiry, strike)
ssvi.blackVariance(expiry, strike)
ssvi.rho    // getter
ssvi.eta    // getter
ssvi.gamma  // getter
ssvi.tenors()
ssvi.forwards()
ssvi.thetas()
ssvi.toJson() / WasmSsviSurface.fromJson(s)
```

**WasmEssviSurface** — Extended SSVI with maturity-dependent correlation (Hendriks-Martini 2019)

```typescript
const essvi = new WasmEssviSurface(rho0, rhoM, a, eta, gamma, tenors, forwards, thetas);
// Same query methods as SSVI, plus:
essvi.rho0
essvi.rhoM
essvi.a
essvi.thetaMax
```

**WasmSurfaceBuilder** — Piecewise surface from market data

```typescript
const builder = new WasmSurfaceBuilder();
builder.spot(100.0);
builder.rate(0.05);
builder.modelSabr(0.5);  // or modelSvi(), modelCubicSpline()
builder.addTenor(0.25, strikes, vols);
builder.addTenor(1.0, strikes, vols);
const surface = builder.build();  // returns WasmPiecewiseSurface

surface.blackVol(0.5, 100.0)
surface.blackVariance(0.5, 100.0)
```

### Local Vol

Dupire local volatility (Gatheral 2006, Eq. 1.10) composed over any surface.
Because WASM has no unified surface type, obtain a local-vol object by calling a
method on the surface — available on `WasmSsviSurface`, `WasmEssviSurface`, and
`WasmPiecewiseSurface`. `bumpSize` is optional (defaults to 0.01).

```typescript
const lv = surface.dupireLocalVol();        // WasmDupireLocalVol (optional bumpSize)
lv.localVol(0.5, 100.0);                     // σ_loc at (expiry, strike)

// Boundary adapter (v2.2 / PAN-25): a query at t ≤ floor (the bump size) is
// evaluated at t = floor, rescuing the t → 0 singularity of the strict path.
const bdy = surface.dupireLocalVolWithBoundary();  // WasmBoundaryLocalVol
bdy.localVol(0.0, 100.0);                    // succeeds where dupireLocalVol throws at t = 0
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

### Serialization

All model types support JSON round-trip:

```typescript
const json = smile.toJson();
const restored = WasmSviSmile.fromJson(json);
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

## License

Apache-2.0
