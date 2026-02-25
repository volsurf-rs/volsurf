use wasm_bindgen_test::*;

use volsurf_wasm::*;

#[wasm_bindgen_test]
fn svi_construct_and_query() {
    let smile = WasmSviSmile::new(100.0, 1.0, 0.04, 0.4, -0.4, 0.0, 0.1).unwrap();
    let vol = smile.vol(100.0).unwrap();
    assert!(vol > 0.0 && vol < 1.0, "ATM vol={vol}");
    assert!((smile.forward() - 100.0).abs() < 1e-12);
    assert!((smile.expiry() - 1.0).abs() < 1e-12);
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
    let result = WasmSviSmile::new(100.0, 1.0, 0.04, -0.1, -0.4, 0.0, 0.1);
    assert!(result.is_err(), "negative b should fail");
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
fn sabr_construct_and_query() {
    let smile = WasmSabrSmile::new(100.0, 1.0, 0.2, 0.5, -0.3, 0.4).unwrap();
    let vol = smile.vol(100.0).unwrap();
    assert!(vol > 0.0 && vol < 1.0, "ATM vol={vol}");
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
fn sabr_json_roundtrip() {
    let smile = WasmSabrSmile::new(100.0, 1.0, 0.2, 0.5, -0.3, 0.4).unwrap();
    let json = smile.to_json().unwrap();
    let restored = WasmSabrSmile::from_json(&json).unwrap();
    let v1 = smile.vol(110.0).unwrap();
    let v2 = restored.vol(110.0).unwrap();
    assert!((v1 - v2).abs() < 1e-14);
}

#[wasm_bindgen_test]
fn ssvi_surface_construct_and_query() {
    let tenors = vec![0.25, 0.5, 1.0];
    let forwards = vec![100.0, 100.0, 100.0];
    let thetas = vec![0.01, 0.025, 0.06];
    let surf = WasmSsviSurface::new(-0.3, 1.5, 0.5, tenors, forwards, thetas).unwrap();
    let vol = surf.black_vol(0.5, 100.0).unwrap();
    assert!(vol > 0.0 && vol < 1.0, "vol={vol}");
    assert!((surf.rho() - -0.3).abs() < 1e-12);
    assert!((surf.eta() - 1.5).abs() < 1e-12);
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

#[wasm_bindgen_test]
fn essvi_surface_construct_and_query() {
    let tenors = vec![0.25, 0.5, 1.0];
    let forwards = vec![100.0, 100.0, 100.0];
    let thetas = vec![0.01, 0.025, 0.06];
    let surf = WasmEssviSurface::new(-0.4, -0.2, 0.3, 1.5, 0.5, tenors, forwards, thetas).unwrap();
    let vol = surf.black_vol(0.5, 100.0).unwrap();
    assert!(vol > 0.0 && vol < 1.0, "vol={vol}");
    assert!((surf.rho_0() - -0.4).abs() < 1e-12);
    assert!((surf.rho_m() - -0.2).abs() < 1e-12);
}

#[wasm_bindgen_test]
fn builder_svi_surface() {
    let mut builder = WasmSurfaceBuilder::new();
    builder.spot(100.0).unwrap();
    builder.rate(0.05).unwrap();
    builder
        .add_tenor(
            0.25,
            vec![80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0],
            vec![0.30, 0.27, 0.24, 0.22, 0.20, 0.22, 0.24, 0.27, 0.30],
        )
        .unwrap();
    builder
        .add_tenor(
            1.0,
            vec![80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0],
            vec![0.28, 0.25, 0.22, 0.20, 0.18, 0.20, 0.22, 0.25, 0.28],
        )
        .unwrap();
    let surface = builder.build().unwrap();
    let vol = surface.black_vol(0.5, 100.0).unwrap();
    assert!(vol > 0.0 && vol < 1.0, "vol={vol}");
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
fn builder_consumed_after_build() {
    let mut builder = WasmSurfaceBuilder::new();
    builder.spot(100.0).unwrap();
    builder.rate(0.05).unwrap();
    builder
        .add_tenor(
            1.0,
            vec![80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0],
            vec![0.28, 0.25, 0.22, 0.20, 0.18, 0.20, 0.22, 0.25, 0.28],
        )
        .unwrap();
    let _surface = builder.build().unwrap();
    let err = builder.spot(200.0);
    assert!(err.is_err(), "builder should be consumed after build()");
}

#[wasm_bindgen_test]
fn odd_length_market_data_rejected() {
    let result = WasmSviSmile::calibrate(100.0, 1.0, vec![80.0, 0.28, 90.0]);
    assert!(result.is_err(), "odd-length flat array should fail");
}

#[wasm_bindgen_test]
fn version_returns_string() {
    let v = version();
    assert_eq!(v, "0.1.0");
}
