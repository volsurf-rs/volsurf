use wasm_bindgen_test::*;

use volsurf_wasm::*;

// ── SVI Smile ──

#[wasm_bindgen_test]
fn svi_construct_and_query() {
    let smile = WasmSviSmile::new(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1).unwrap();
    let vol = smile.vol(100.0).unwrap();
    assert!(vol > 0.0 && vol < 1.0, "ATM vol={vol}");
    assert!((smile.forward() - 100.0).abs() < 1e-12);
    assert!((smile.expiry() - 1.0).abs() < 1e-12);
}

#[wasm_bindgen_test]
fn svi_variance_and_density() {
    let smile = WasmSviSmile::new(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1).unwrap();
    let var = smile.variance(100.0).unwrap();
    let vol = smile.vol(100.0).unwrap();
    assert!(
        (var - vol * vol).abs() < 1e-10,
        "variance={var}, vol^2={}",
        vol * vol
    );

    let d = smile.density(100.0).unwrap();
    assert!(d >= 0.0, "density must be non-negative, got {d}");
}

#[wasm_bindgen_test]
fn svi_calibrate() {
    let market: Vec<f64> = vec![
        80.0, 0.30, 85.0, 0.27, 90.0, 0.24, 95.0, 0.22, 100.0, 0.20, 105.0, 0.22, 110.0, 0.25,
        115.0, 0.28, 120.0, 0.32,
    ];
    let smile = WasmSviSmile::calibrate(100.0, 1.0, market).unwrap();
    let atm = smile.vol(100.0).unwrap();
    assert!((atm - 0.20).abs() < 0.02, "ATM vol={atm}, expected ~0.20");
}

#[wasm_bindgen_test]
fn svi_invalid_params_rejected() {
    assert!(WasmSviSmile::new(100.0, 1.0, 0.04, -0.1, -0.4, 0.0, 0.1).is_err());
    assert!(WasmSviSmile::new(-100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1).is_err());
    assert!(WasmSviSmile::new(100.0, -1.0, 0.04, 0.4, -0.4, 0.0, 0.1).is_err());
    assert!(WasmSviSmile::new(100.0, 1.0, 0.04, 0.4, 1.0, 0.0, 0.1).is_err());
    assert!(WasmSviSmile::new(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, -0.1).is_err());
}

#[wasm_bindgen_test]
fn svi_json_roundtrip() {
    let smile = WasmSviSmile::new(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1).unwrap();
    let json = smile.to_json().unwrap();
    let restored = WasmSviSmile::from_json(&json).unwrap();
    let v1 = smile.vol(95.0).unwrap();
    let v2 = restored.vol(95.0).unwrap();
    assert!(
        (v1 - v2).abs() < 1e-14,
        "json roundtrip mismatch: {v1} vs {v2}"
    );
}

#[wasm_bindgen_test]
fn svi_from_json_invalid() {
    assert!(WasmSviSmile::from_json("not json").is_err());
    assert!(WasmSviSmile::from_json("{}").is_err());
}

// ── SABR Smile ──

#[wasm_bindgen_test]
fn sabr_construct_and_query() {
    let smile = WasmSabrSmile::new(100.0, 1.0, 0.2, 0.5, -0.3, 0.4).unwrap();
    let vol = smile.vol(100.0).unwrap();
    assert!(vol > 0.0 && vol < 1.0, "ATM vol={vol}");
    assert!((smile.forward() - 100.0).abs() < 1e-12);
    assert!((smile.expiry() - 1.0).abs() < 1e-12);
}

#[wasm_bindgen_test]
fn sabr_variance_and_density() {
    let smile = WasmSabrSmile::new(100.0, 1.0, 0.2, 0.5, -0.3, 0.4).unwrap();
    let var = smile.variance(100.0).unwrap();
    let vol = smile.vol(100.0).unwrap();
    assert!((var - vol * vol).abs() < 1e-10);

    let d = smile.density(100.0).unwrap();
    assert!(d >= 0.0, "density={d}");
}

#[wasm_bindgen_test]
fn sabr_calibrate() {
    let market: Vec<f64> = vec![
        85.0, 0.25, 90.0, 0.23, 95.0, 0.21, 100.0, 0.20, 105.0, 0.21, 110.0, 0.23, 115.0, 0.26,
    ];
    let smile = WasmSabrSmile::calibrate(100.0, 1.0, 0.5, market).unwrap();
    let atm = smile.vol(100.0).unwrap();
    assert!((atm - 0.20).abs() < 0.02, "ATM vol={atm}");
}

#[wasm_bindgen_test]
fn sabr_invalid_params_rejected() {
    assert!(WasmSabrSmile::new(-100.0, 1.0, 0.2, 0.5, -0.3, 0.4).is_err());
    assert!(WasmSabrSmile::new(100.0, 1.0, -0.2, 0.5, -0.3, 0.4).is_err());
    assert!(WasmSabrSmile::new(100.0, 1.0, 0.2, 1.5, -0.3, 0.4).is_err());
    assert!(WasmSabrSmile::new(100.0, 1.0, 0.2, 0.5, 1.0, 0.4).is_err());
    assert!(WasmSabrSmile::new(100.0, 1.0, 0.2, 0.5, -0.3, -0.4).is_err());
}

#[wasm_bindgen_test]
fn sabr_json_roundtrip() {
    let smile = WasmSabrSmile::new(100.0, 1.0, 0.2, 0.5, -0.3, 0.4).unwrap();
    let json = smile.to_json().unwrap();
    let restored = WasmSabrSmile::from_json(&json).unwrap();
    let v1 = smile.vol(110.0).unwrap();
    let v2 = restored.vol(110.0).unwrap();
    assert!((v1 - v2).abs() < 1e-14);
}

// ── SSVI Surface ──

#[wasm_bindgen_test]
fn ssvi_construct_and_query() {
    let tenors = vec![0.25, 0.5, 1.0];
    let forwards = vec![100.0, 100.0, 100.0];
    let thetas = vec![0.01, 0.025, 0.06];
    let surf = WasmSsviSurface::new(-0.3, 1.5, 0.5, tenors, forwards, thetas).unwrap();

    let vol = surf.black_vol(0.5, 100.0).unwrap();
    assert!(vol > 0.0 && vol < 1.0, "vol={vol}");

    let var = surf.black_variance(0.5, 100.0).unwrap();
    assert!((var - vol * vol * 0.5).abs() < 1e-10, "var={var}");

    assert!((surf.rho() - -0.3).abs() < 1e-12);
    assert!((surf.eta() - 1.5).abs() < 1e-12);
    assert!((surf.gamma() - 0.5).abs() < 1e-12);
}

#[wasm_bindgen_test]
fn ssvi_vector_accessors() {
    let tenors = vec![0.25, 1.0];
    let forwards = vec![100.0, 102.0];
    let thetas = vec![0.01, 0.04];
    let surf = WasmSsviSurface::new(-0.3, 1.5, 0.5, tenors, forwards, thetas).unwrap();

    assert_eq!(surf.tenors(), vec![0.25, 1.0]);
    assert_eq!(surf.forwards(), vec![100.0, 102.0]);
    assert_eq!(surf.thetas(), vec![0.01, 0.04]);
}

#[wasm_bindgen_test]
fn ssvi_invalid_params_rejected() {
    let t = vec![0.25, 1.0];
    let f = vec![100.0, 100.0];
    let th = vec![0.01, 0.04];
    // rho out of range
    assert!(WasmSsviSurface::new(1.0, 1.5, 0.5, t.clone(), f.clone(), th.clone()).is_err());
    // eta <= 0
    assert!(WasmSsviSurface::new(-0.3, 0.0, 0.5, t.clone(), f.clone(), th.clone()).is_err());
    // mismatched lengths
    assert!(WasmSsviSurface::new(-0.3, 1.5, 0.5, vec![0.25], f.clone(), th.clone()).is_err());
}

#[wasm_bindgen_test]
fn ssvi_json_roundtrip() {
    let tenors = vec![0.25, 1.0];
    let forwards = vec![100.0, 100.0];
    let thetas = vec![0.01, 0.04];
    let surf = WasmSsviSurface::new(-0.3, 1.5, 0.5, tenors, forwards, thetas).unwrap();
    let json = surf.to_json().unwrap();
    let restored = WasmSsviSurface::from_json(&json).unwrap();
    let v1 = surf.black_vol(0.5, 95.0).unwrap();
    let v2 = restored.black_vol(0.5, 95.0).unwrap();
    assert!((v1 - v2).abs() < 1e-14);
}

// ── eSSVI Surface ──

#[wasm_bindgen_test]
fn essvi_construct_and_query() {
    let tenors = vec![0.25, 0.5, 1.0];
    let forwards = vec![100.0, 100.0, 100.0];
    let thetas = vec![0.01, 0.025, 0.06];
    let surf = WasmEssviSurface::new(-0.4, -0.2, 0.3, 1.5, 0.5, tenors, forwards, thetas).unwrap();

    let vol = surf.black_vol(0.5, 100.0).unwrap();
    assert!(vol > 0.0 && vol < 1.0, "vol={vol}");

    let var = surf.black_variance(0.5, 100.0).unwrap();
    assert!((var - vol * vol * 0.5).abs() < 1e-10);

    assert!((surf.rho_0() - -0.4).abs() < 1e-12);
    assert!((surf.rho_m() - -0.2).abs() < 1e-12);
    assert!((surf.a() - 0.3).abs() < 1e-12);
    assert!((surf.eta() - 1.5).abs() < 1e-12);
    assert!((surf.gamma() - 0.5).abs() < 1e-12);
    assert!(surf.theta_max() > 0.0);
}

#[wasm_bindgen_test]
fn essvi_vector_accessors() {
    let tenors = vec![0.25, 1.0];
    let forwards = vec![100.0, 102.0];
    let thetas = vec![0.01, 0.04];
    let surf = WasmEssviSurface::new(-0.4, -0.2, 0.3, 1.5, 0.5, tenors, forwards, thetas).unwrap();
    assert_eq!(surf.tenors(), vec![0.25, 1.0]);
    assert_eq!(surf.forwards(), vec![100.0, 102.0]);
    assert_eq!(surf.thetas(), vec![0.01, 0.04]);
}

#[wasm_bindgen_test]
fn essvi_json_roundtrip() {
    let tenors = vec![0.25, 1.0];
    let forwards = vec![100.0, 100.0];
    let thetas = vec![0.01, 0.04];
    let surf = WasmEssviSurface::new(-0.4, -0.2, 0.3, 1.5, 0.5, tenors, forwards, thetas).unwrap();
    let json = surf.to_json().unwrap();
    let restored = WasmEssviSurface::from_json(&json).unwrap();
    let v1 = surf.black_vol(0.5, 95.0).unwrap();
    let v2 = restored.black_vol(0.5, 95.0).unwrap();
    assert!((v1 - v2).abs() < 1e-14);
}

// ── SurfaceBuilder ──

fn nine_strike_smile() -> (Vec<f64>, Vec<f64>) {
    (
        vec![80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0],
        vec![0.30, 0.27, 0.24, 0.22, 0.20, 0.22, 0.24, 0.27, 0.30],
    )
}

#[wasm_bindgen_test]
fn builder_svi_surface() {
    let (strikes, vols) = nine_strike_smile();
    let mut builder = WasmSurfaceBuilder::new();
    builder.spot(100.0).unwrap();
    builder.rate(0.05).unwrap();
    builder
        .add_tenor(0.25, strikes.clone(), vols.clone())
        .unwrap();
    builder.add_tenor(1.0, strikes, vols).unwrap();
    let surface = builder.build().unwrap();
    let vol = surface.black_vol(0.5, 100.0).unwrap();
    assert!(vol > 0.0 && vol < 1.0, "vol={vol}");

    let var = surface.black_variance(0.5, 100.0).unwrap();
    assert!((var - vol * vol * 0.5).abs() < 1e-10);
}

#[wasm_bindgen_test]
fn builder_sabr_surface() {
    let mut builder = WasmSurfaceBuilder::new();
    builder.spot(100.0).unwrap();
    builder.rate(0.05).unwrap();
    builder.model_sabr(0.5).unwrap();
    builder
        .add_tenor(
            0.5,
            vec![85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0],
            vec![0.26, 0.23, 0.21, 0.20, 0.21, 0.23, 0.26],
        )
        .unwrap();
    let surface = builder.build().unwrap();
    let vol = surface.black_vol(0.5, 100.0).unwrap();
    assert!(vol > 0.0 && vol < 1.0, "vol={vol}");
}

#[wasm_bindgen_test]
fn builder_cubic_spline_surface() {
    let mut builder = WasmSurfaceBuilder::new();
    builder.spot(100.0).unwrap();
    builder.rate(0.05).unwrap();
    builder.model_cubic_spline().unwrap();
    builder
        .add_tenor(1.0, vec![90.0, 100.0, 110.0], vec![0.24, 0.20, 0.24])
        .unwrap();
    let surface = builder.build().unwrap();
    let vol = surface.black_vol(1.0, 100.0).unwrap();
    assert!(vol > 0.0 && vol < 1.0, "vol={vol}");
}

#[wasm_bindgen_test]
fn builder_with_forward_and_dividend() {
    let mut builder = WasmSurfaceBuilder::new();
    builder.spot(100.0).unwrap();
    builder.rate(0.05).unwrap();
    builder.dividend_yield(0.02).unwrap();
    builder
        .add_tenor_with_forward(
            1.0,
            vec![80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0],
            vec![0.28, 0.25, 0.22, 0.20, 0.18, 0.20, 0.22, 0.25, 0.28],
            103.0,
        )
        .unwrap();
    let surface = builder.build().unwrap();
    let vol = surface.black_vol(1.0, 100.0).unwrap();
    assert!(vol > 0.0 && vol < 1.0, "vol={vol}");
}

#[wasm_bindgen_test]
fn builder_consumed_after_build() {
    let (strikes, vols) = nine_strike_smile();
    let mut builder = WasmSurfaceBuilder::new();
    builder.spot(100.0).unwrap();
    builder.rate(0.05).unwrap();
    builder.add_tenor(1.0, strikes, vols).unwrap();
    let _surface = builder.build().unwrap();

    assert!(builder.spot(200.0).is_err());
    assert!(builder.rate(0.0).is_err());
    assert!(builder.model_svi().is_err());
    assert!(builder.model_sabr(0.5).is_err());
    assert!(builder.model_cubic_spline().is_err());
    assert!(builder.dividend_yield(0.0).is_err());
    assert!(builder.add_tenor(1.0, vec![], vec![]).is_err());
    assert!(builder.build().is_err());
}

// ── Edge cases ──

#[wasm_bindgen_test]
fn odd_length_market_data_rejected() {
    assert!(WasmSviSmile::calibrate(100.0, 1.0, vec![80.0, 0.28, 90.0]).is_err());
}

#[wasm_bindgen_test]
fn empty_market_data_rejected() {
    assert!(WasmSviSmile::calibrate(100.0, 1.0, vec![]).is_err());
    assert!(WasmSabrSmile::calibrate(100.0, 1.0, 0.5, vec![]).is_err());
}

#[wasm_bindgen_test]
fn too_few_points_rejected() {
    // SVI needs >= 5 pairs
    let four_pairs: Vec<f64> = vec![90.0, 0.24, 95.0, 0.22, 100.0, 0.20, 105.0, 0.22];
    assert!(WasmSviSmile::calibrate(100.0, 1.0, four_pairs).is_err());
    // SABR needs >= 4 pairs
    let three_pairs: Vec<f64> = vec![90.0, 0.24, 100.0, 0.20, 110.0, 0.24];
    assert!(WasmSabrSmile::calibrate(100.0, 1.0, 0.5, three_pairs).is_err());
}

#[wasm_bindgen_test]
fn version_returns_string() {
    assert_eq!(version(), "0.1.0");
}
