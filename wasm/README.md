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
