use serde::Deserialize;
use volsurf::smile::SviSmile;
use volsurf::{SmileSection, VolSurfError};

#[derive(Deserialize)]
struct Fixture {
    forward: f64,
    expiry_years: f64,
    expected_atm_iv: f64,
    market_vols: Vec<[f64; 2]>,
}

fn load_fixture(name: &str) -> Fixture {
    let path = format!(
        "{}/tests/fixtures/svi_failures/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    let data = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&data).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn filter_by_moneyness(data: &[[f64; 2]], forward: f64, pct: f64) -> Vec<(f64, f64)> {
    let lo = forward * (1.0 - pct);
    let hi = forward * (1.0 + pct);
    data.iter()
        .filter(|p| p[0] >= lo && p[0] <= hi)
        .map(|p| (p[0], p[1]))
        .collect()
}

fn assert_calibration_sane(
    forward: f64,
    expiry: f64,
    data: &[(f64, f64)],
    expected_atm: f64,
    label: &str,
) {
    if data.len() < 5 {
        println!("{label}: skipped (only {} points)", data.len());
        return;
    }
    match SviSmile::calibrate(forward, expiry, data) {
        Ok(smile) => {
            let atm = SmileSection::vol(&smile, forward).unwrap().0;
            let ratio = atm / expected_atm;
            println!(
                "{label}: OK  n={:3}  ATM={:.4} expected={:.4} ratio={:.3}",
                data.len(),
                atm,
                expected_atm,
                ratio
            );
            assert!(
                atm > expected_atm * 0.5 && atm < expected_atm * 2.0,
                "{label}: ATM vol {atm:.4} outside [0.5x, 2x] of expected {expected_atm:.4}"
            );
        }
        Err(VolSurfError::CalibrationError { message, .. }) => {
            println!("{label}: REJECTED  n={:3}  reason={message}", data.len());
            assert!(
                message.contains("Roger Lee")
                    || message.contains("ATM total variance")
                    || message.contains("grid search"),
                "{label}: unexpected rejection reason: {message}"
            );
        }
        Err(e) => panic!("{label}: unexpected error variant: {e}"),
    }
}

// Per-fixture tests at multiple moneyness ranges

#[test]
fn jan13_feb21_39dte_5pct() {
    let f = load_fixture("jan13_feb21_39dte.json");
    let subset = filter_by_moneyness(&f.market_vols, f.forward, 0.05);
    assert_calibration_sane(
        f.forward,
        f.expiry_years,
        &subset,
        f.expected_atm_iv,
        "jan13 ±5%",
    );
}

#[test]
fn jan13_feb21_39dte_10pct() {
    let f = load_fixture("jan13_feb21_39dte.json");
    let subset = filter_by_moneyness(&f.market_vols, f.forward, 0.10);
    assert_calibration_sane(
        f.forward,
        f.expiry_years,
        &subset,
        f.expected_atm_iv,
        "jan13 ±10%",
    );
}

#[test]
fn jan13_feb21_39dte_15pct() {
    let f = load_fixture("jan13_feb21_39dte.json");
    let subset = filter_by_moneyness(&f.market_vols, f.forward, 0.15);
    assert_calibration_sane(
        f.forward,
        f.expiry_years,
        &subset,
        f.expected_atm_iv,
        "jan13 ±15%",
    );
}

#[test]
fn jan24_feb21_28dte_stable_5pct() {
    let f = load_fixture("jan24_feb21_28dte_stable.json");
    let subset = filter_by_moneyness(&f.market_vols, f.forward, 0.05);
    assert_calibration_sane(
        f.forward,
        f.expiry_years,
        &subset,
        f.expected_atm_iv,
        "jan24 ±5%",
    );
}

#[test]
fn jan24_feb21_28dte_stable_10pct() {
    let f = load_fixture("jan24_feb21_28dte_stable.json");
    let subset = filter_by_moneyness(&f.market_vols, f.forward, 0.10);
    assert_calibration_sane(
        f.forward,
        f.expiry_years,
        &subset,
        f.expected_atm_iv,
        "jan24 ±10%",
    );
}

#[test]
fn jan24_feb21_28dte_stable_15pct() {
    let f = load_fixture("jan24_feb21_28dte_stable.json");
    let subset = filter_by_moneyness(&f.market_vols, f.forward, 0.15);
    assert_calibration_sane(
        f.forward,
        f.expiry_years,
        &subset,
        f.expected_atm_iv,
        "jan24 ±15%",
    );
}

#[test]
fn mar03_apr17_45dte_5pct() {
    let f = load_fixture("mar03_apr17_45dte.json");
    let subset = filter_by_moneyness(&f.market_vols, f.forward, 0.05);
    assert_calibration_sane(
        f.forward,
        f.expiry_years,
        &subset,
        f.expected_atm_iv,
        "mar03 ±5%",
    );
}

#[test]
fn mar03_apr17_45dte_10pct() {
    let f = load_fixture("mar03_apr17_45dte.json");
    let subset = filter_by_moneyness(&f.market_vols, f.forward, 0.10);
    assert_calibration_sane(
        f.forward,
        f.expiry_years,
        &subset,
        f.expected_atm_iv,
        "mar03 ±10%",
    );
}

#[test]
fn mar03_apr17_45dte_15pct() {
    let f = load_fixture("mar03_apr17_45dte.json");
    let subset = filter_by_moneyness(&f.market_vols, f.forward, 0.15);
    assert_calibration_sane(
        f.forward,
        f.expiry_years,
        &subset,
        f.expected_atm_iv,
        "mar03 ±15%",
    );
}

#[test]
fn mar10_apr17_38dte_5pct() {
    let f = load_fixture("mar10_apr17_38dte.json");
    let subset = filter_by_moneyness(&f.market_vols, f.forward, 0.05);
    assert_calibration_sane(
        f.forward,
        f.expiry_years,
        &subset,
        f.expected_atm_iv,
        "mar10 ±5%",
    );
}

#[test]
fn mar10_apr17_38dte_10pct() {
    let f = load_fixture("mar10_apr17_38dte.json");
    let subset = filter_by_moneyness(&f.market_vols, f.forward, 0.10);
    assert_calibration_sane(
        f.forward,
        f.expiry_years,
        &subset,
        f.expected_atm_iv,
        "mar10 ±10%",
    );
}

#[test]
fn mar10_apr17_38dte_15pct() {
    let f = load_fixture("mar10_apr17_38dte.json");
    let subset = filter_by_moneyness(&f.market_vols, f.forward, 0.15);
    assert_calibration_sane(
        f.forward,
        f.expiry_years,
        &subset,
        f.expected_atm_iv,
        "mar10 ±15%",
    );
}

// Full-range calibration: all strikes from each fixture.
// With 300+ options, this exercises the vol-cliff filter and wide-range behavior.

#[test]
fn jan13_feb21_39dte_full_range() {
    let f = load_fixture("jan13_feb21_39dte.json");
    let data: Vec<(f64, f64)> = f.market_vols.iter().map(|p| (p[0], p[1])).collect();
    assert_calibration_sane(
        f.forward,
        f.expiry_years,
        &data,
        f.expected_atm_iv,
        "jan13 full",
    );
}

#[test]
fn mar10_apr17_38dte_full_range() {
    let f = load_fixture("mar10_apr17_38dte.json");
    let data: Vec<(f64, f64)> = f.market_vols.iter().map(|p| (p[0], p[1])).collect();
    assert_calibration_sane(
        f.forward,
        f.expiry_years,
        &data,
        f.expected_atm_iv,
        "mar10 full",
    );
}

// Regression guard: the stable fixture at ±5% must always calibrate successfully.
// This catches regressions where valid, well-behaved data starts failing.
#[test]
fn stable_fixture_must_succeed() {
    let f = load_fixture("jan24_feb21_28dte_stable.json");
    for pct in [0.05, 0.10, 0.15] {
        let subset = filter_by_moneyness(&f.market_vols, f.forward, pct);
        if subset.len() < 5 {
            continue;
        }
        let smile = SviSmile::calibrate(f.forward, f.expiry_years, &subset)
            .unwrap_or_else(|e| panic!("stable ±{:.0}% must succeed: {e}", pct * 100.0));
        let atm = SmileSection::vol(&smile, f.forward).unwrap().0;
        let ratio = atm / f.expected_atm_iv;
        assert!(
            ratio > 0.8 && ratio < 1.2,
            "stable ±{:.0}%: ATM ratio {ratio:.3} outside [0.8, 1.2]",
            pct * 100.0
        );
    }
}

// Roger Lee bound tests live in src/smile/svi.rs (new_rejects_roger_lee_bound,
// new_accepts_roger_lee_boundary) — not duplicated here.
