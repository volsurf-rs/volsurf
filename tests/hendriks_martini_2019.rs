//! Validation against Hendriks & Martini (2019), "The Extended SSVI Volatility Surface".
//!
//! Verifies eSSVI implementation reproduces the paper's formulas and satisfies
//! theoretical results: Theorem 4.1, Proposition 3.5, Eq. 5.7.
//!
//! Paper parameters from DJX calibration (Section 6.1):
//!   φ(θ) = 0.98·θ^{−0.42}  →  η=0.98, γ=0.42
//!   ρ(θ) = −0.76 − 0.11·(θ/θ_max)^{0.51}  →  ρ₀=−0.76, ρₘ=−0.87, a=0.51

use approx::assert_abs_diff_eq;
use volsurf::surface::{EssviSlice, EssviSurface, SsviSlice, SsviSurface, VolSurface};
use volsurf::SmileSection;

const ETA: f64 = 0.98;
const GAMMA: f64 = 0.42;
const RHO_0: f64 = -0.76;
const RHO_M: f64 = -0.87;
const A_PARAM: f64 = 0.51;

// DJX maturities from Figure 3
fn djx_tenors() -> Vec<f64> {
    vec![0.0192, 0.0575, 0.1342, 0.2110, 0.3836, 0.6329, 0.8822, 1.3808, 1.8795, 2.8959]
}

// θ = σ²·T estimated from Figure 3 ATM vols (~16-22%)
fn djx_thetas() -> Vec<f64> {
    let atm_vols = [0.20, 0.20, 0.22, 0.20, 0.18, 0.17, 0.17, 0.17, 0.17, 0.16];
    let tenors = djx_tenors();
    atm_vols
        .iter()
        .zip(tenors.iter())
        .map(|(v, t)| v * v * t)
        .collect()
}

fn djx_forwards() -> Vec<f64> {
    vec![160.0; 10]
}

fn paper_surface() -> EssviSurface {
    EssviSurface::new(
        RHO_0,
        RHO_M,
        A_PARAM,
        ETA,
        GAMMA,
        djx_tenors(),
        djx_forwards(),
        djx_thetas(),
    )
    .unwrap()
}

// ── Eq 2.2: total variance formula ──────────────────────────────

// Hand-compute w(k, θ) = (θ/2)[1 + ρφk + √((φk + ρ)² + 1 − ρ²)]
fn w_formula(theta: f64, k: f64, rho: f64, eta: f64, gamma: f64) -> f64 {
    let phi = eta / theta.powf(gamma);
    let phi_k = phi * k;
    let one_minus_rho_sq = 1.0 - rho * rho;
    (theta / 2.0) * (1.0 + rho * phi_k + ((phi_k + rho).powi(2) + one_minus_rho_sq).sqrt())
}

#[test]
fn eq_2_2_atm_identity() {
    // At k=0: w(0,θ) = θ for any θ, ρ, η, γ (Eq. 2.1 with φk=0)
    for &theta in &[0.001, 0.01, 0.04, 0.16, 0.5, 1.0] {
        for &rho in &[-0.9, -0.5, 0.0, 0.5, 0.9] {
            let w = w_formula(theta, 0.0, rho, 0.5, 0.5);
            assert_abs_diff_eq!(w, theta, epsilon = 1e-14);
        }
    }
}

#[test]
fn eq_2_2_hand_computed_values() {
    // θ=0.04, η=0.5, γ=0.5, ρ=-0.3, F=100
    // φ = 0.5/√0.04 = 2.5
    let theta = 0.04;
    let rho = -0.3;
    let eta = 0.5;
    let gamma = 0.5;

    let slice = EssviSlice::new(100.0, 1.0, rho, eta, gamma, theta).unwrap();

    // k=0.2 (K=100·e^0.2≈122.14): φk=0.5
    // w = 0.02[1 - 0.15 + √(0.04 + 0.91)] = 0.02[0.85 + √0.95] = 0.02·1.82468...
    let k = 0.2;
    let expected_pos = w_formula(theta, k, rho, eta, gamma);
    let strike_pos = 100.0 * k.exp();
    let var_pos = slice.variance(strike_pos).unwrap().0;
    assert_abs_diff_eq!(var_pos, expected_pos, epsilon = 1e-14);
    assert_abs_diff_eq!(
        expected_pos,
        0.02 * (0.85 + 0.95_f64.sqrt()),
        epsilon = 1e-14
    );

    // k=-0.2 (K≈81.87): φk=-0.5
    // w = 0.02[1 + 0.15 + √(0.64 + 0.91)] = 0.02[1.15 + √1.55]
    let expected_neg = w_formula(theta, -k, rho, eta, gamma);
    let strike_neg = 100.0 * (-k).exp();
    let var_neg = slice.variance(strike_neg).unwrap().0;
    assert_abs_diff_eq!(var_neg, expected_neg, epsilon = 1e-14);
    assert_abs_diff_eq!(
        expected_neg,
        0.02 * (1.15 + 1.55_f64.sqrt()),
        epsilon = 1e-14
    );

    // Skew: w(-k) > w(k) when ρ < 0 (put side more expensive)
    assert!(var_neg > var_pos, "negative rho should produce put skew");
}

#[test]
fn eq_2_2_code_matches_formula_across_strikes() {
    // Sweep k from -1.5 to 1.5 and verify EssviSlice output matches Eq 2.2
    let theta = 0.09;
    let rho = -0.5;
    let eta = 0.8;
    let gamma = 0.4;
    let forward = 100.0;
    let expiry = 1.0;

    let slice = EssviSlice::new(forward, expiry, rho, eta, gamma, theta).unwrap();

    for i in -15..=15 {
        let k = i as f64 * 0.1;
        let strike = forward * k.exp();
        let var_code = slice.variance(strike).unwrap().0;
        let var_hand = w_formula(theta, k, rho, eta, gamma);
        assert!(
            (var_code - var_hand).abs() < 1e-13,
            "mismatch at k={k}: code={var_code}, hand={var_hand}"
        );
    }
}

// ── Eq 5.6: ρ(θ) parametric family ─────────────────────────────

#[test]
fn eq_5_6_rho_boundaries_and_interpolation() {
    let s = paper_surface();
    let thetas = djx_thetas();
    let theta_max = *thetas.last().unwrap();

    // ρ(0) = ρ₀
    assert_abs_diff_eq!(s.rho(0.0), RHO_0, epsilon = 1e-14);

    // ρ(θ_max) = ρₘ
    assert_abs_diff_eq!(s.rho(theta_max), RHO_M, epsilon = 1e-14);

    // Intermediate values match closed-form
    for &theta in &thetas {
        let t = (theta / theta_max).powf(A_PARAM);
        let expected = RHO_0 + (RHO_M - RHO_0) * t;
        assert_abs_diff_eq!(s.rho(theta), expected, epsilon = 1e-14);
    }
}

#[test]
fn eq_5_6_rho_monotone_with_paper_params() {
    // ρₘ < ρ₀ (−0.87 < −0.76), so ρ(θ) is decreasing
    let s = paper_surface();
    let thetas = djx_thetas();

    for w in thetas.windows(2) {
        let r_lo = s.rho(w[0]);
        let r_hi = s.rho(w[1]);
        assert!(
            r_hi < r_lo,
            "ρ should decrease: ρ({})={r_lo} > ρ({})={r_hi}",
            w[0],
            w[1]
        );
    }
}

// ── Eq 5.7: constraint on a ────────────────────────────────────

#[test]
fn eq_5_7_paper_params_satisfy_constraint() {
    // ρ₀ > ρₘ, so a_max = (1−γ)(1+ρₘ)/(ρ₀−ρₘ)
    let gamma_thm = 1.0 - GAMMA;
    let a_max = gamma_thm * (1.0 + RHO_M) / (RHO_0 - RHO_M);

    // a_max = 0.58 * 0.13 / 0.11 ≈ 0.6855
    assert_abs_diff_eq!(a_max, 0.58 * 0.13 / 0.11, epsilon = 1e-10);
    assert!(
        A_PARAM < a_max,
        "paper's a={A_PARAM} should be below a_max={a_max}"
    );

    // Surface construction succeeds
    assert!(paper_surface().calendar_check_structural().is_empty());
}

#[test]
fn eq_5_7_boundary_acceptance_and_rejection() {
    let gamma_thm = 1.0 - GAMMA;
    let a_max = gamma_thm * (1.0 + RHO_M) / (RHO_0 - RHO_M);

    let thetas = djx_thetas();
    let tenors = djx_tenors();
    let fwds = djx_forwards();

    // Exactly at a_max: should succeed
    assert!(EssviSurface::new(
        RHO_0,
        RHO_M,
        a_max,
        ETA,
        GAMMA,
        tenors.clone(),
        fwds.clone(),
        thetas.clone(),
    )
    .is_ok());

    // Above a_max: should fail
    assert!(EssviSurface::new(
        RHO_0,
        RHO_M,
        a_max + 0.01,
        ETA,
        GAMMA,
        tenors,
        fwds,
        thetas,
    )
    .is_err());
}

// ── Theorem 4.1: calendar no-arb structural check ──────────────

#[test]
fn thm_4_1_structural_check_paper_params() {
    let s = paper_surface();
    let gamma_thm = 1.0 - GAMMA;
    let thetas = djx_thetas();
    let theta_max = *thetas.last().unwrap();

    // For power-law φ with γ≤1: γ_thm = 1−γ ∈ [0,1], always first branch of Eq 4.10
    // Condition: (δ + ρ·γ_thm)² ≤ γ_thm²
    for &theta in &thetas {
        let rho = s.rho(theta);
        let t = (theta / theta_max).clamp(0.0, 1.0);
        let delta = A_PARAM * (RHO_M - RHO_0) * t.powf(A_PARAM);

        let lhs = (delta + rho * gamma_thm).powi(2);
        let rhs = gamma_thm.powi(2);
        assert!(
            lhs <= rhs + 1e-10,
            "Thm 4.1 violated at θ={theta}: (δ+ρ·γ)²={lhs:.6} > γ²={rhs:.6}"
        );
    }

    // Full structural check via our code
    assert!(s.calendar_check_structural().is_empty());
}

#[test]
fn thm_4_1_full_diagnostics_paper_params() {
    let s = paper_surface();
    let diag = s.diagnostics().unwrap();

    assert!(
        diag.calendar_violations.is_empty(),
        "paper params should have no calendar violations, found {}",
        diag.calendar_violations.len()
    );
    assert!(
        diag.is_free,
        "paper params should be fully arb-free"
    );
}

// ── Proposition 3.5: two-slice discrete no-arb ──────────────────

#[test]
fn prop_3_5_necessary_conditions() {
    // Two SSVI slices: θ₁=0.04, θ₂=0.16, η=0.5, γ=0.5, ρ₁=-0.4, ρ₂=-0.3
    let theta_1: f64 = 0.04;
    let theta_2: f64 = 0.16;
    let eta: f64 = 0.5;
    let gamma: f64 = 0.5;
    let rho_1: f64 = -0.4;
    let rho_2: f64 = -0.3;

    let phi_1 = eta / theta_1.powf(gamma); // 2.5
    let phi_2 = eta / theta_2.powf(gamma); // 1.25

    let theta_ratio = theta_2 / theta_1; // 4.0
    let phi_ratio = phi_2 / phi_1; // 0.5

    // Condition 1: θ₂/θ₁ ≥ 1
    assert!(theta_ratio >= 1.0);

    // Condition 2: θφ ≥ max((1+ρ₁)/(1+ρ₂), (1−ρ₁)/(1−ρ₂))
    let theta_phi = theta_ratio * phi_ratio; // 2.0
    let bound_pos: f64 = (1.0 + rho_1) / (1.0 + rho_2); // 0.6/0.7 = 0.857
    let bound_neg: f64 = (1.0 - rho_1) / (1.0 - rho_2); // 1.4/1.3 = 1.077
    let bound = bound_pos.max(bound_neg);
    assert!(theta_phi >= bound, "θφ={theta_phi} < max bound={bound}");

    // Condition 3 (sufficient): φ ≤ 1
    assert!(phi_ratio <= 1.0, "for power-law φ with γ∈[0,1], φ₂/φ₁ ≤ 1 when θ₂ ≥ θ₁");
}

#[test]
fn prop_3_5_total_variance_monotonicity() {
    // The two slices from Prop 3.5 should satisfy w₂(k) ≥ w₁(k) for all k
    let theta_1 = 0.04;
    let theta_2 = 0.16;
    let rho_1 = -0.4;
    let rho_2 = -0.3;
    let eta = 0.5;
    let gamma = 0.5;

    // Sweep k from −2 to 2
    for i in -200..=200 {
        let k = i as f64 * 0.01;
        let w1 = w_formula(theta_1, k, rho_1, eta, gamma);
        let w2 = w_formula(theta_2, k, rho_2, eta, gamma);
        assert!(
            w2 >= w1 - 1e-14,
            "calendar arb at k={k:.2}: w1={w1:.8} > w2={w2:.8}"
        );
    }
}

// ── SSVI degeneracy: eSSVI with constant ρ = SSVI ───────────────

#[test]
fn ssvi_degeneracy_slice() {
    // EssviSlice with any ρ should be bit-identical to SsviSlice with same params
    let rho = -0.4;
    let eta = 0.6;
    let gamma = 0.45;
    let theta = 0.09;
    let forward = 100.0;
    let expiry = 1.0;

    let essvi = EssviSlice::new(forward, expiry, rho, eta, gamma, theta).unwrap();
    let ssvi = SsviSlice::new(forward, expiry, rho, eta, gamma, theta).unwrap();

    for &strike in &[60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0] {
        let v_essvi = essvi.vol(strike).unwrap().0;
        let v_ssvi = ssvi.vol(strike).unwrap().0;
        assert_eq!(
            v_essvi.to_bits(),
            v_ssvi.to_bits(),
            "bit-mismatch at K={strike}"
        );
    }
}

#[test]
fn ssvi_degeneracy_surface() {
    // EssviSurface with ρ₀ = ρₘ should produce identical vols to SsviSurface
    let rho = -0.4;
    let eta = 0.6;
    let gamma = 0.45;
    let tenors = vec![0.25, 0.5, 1.0, 2.0];
    let forwards = vec![100.0; 4];
    let thetas = vec![0.01, 0.025, 0.05, 0.10];

    let essvi = EssviSurface::new(
        rho,
        rho,
        0.5, // a is irrelevant when ρ₀ = ρₘ
        eta,
        gamma,
        tenors.clone(),
        forwards.clone(),
        thetas.clone(),
    )
    .unwrap();

    let ssvi = SsviSurface::new(rho, eta, gamma, tenors.clone(), forwards.clone(), thetas).unwrap();

    for &t in &tenors {
        for &k in &[70.0, 85.0, 100.0, 115.0, 140.0] {
            let v_essvi = essvi.black_vol(t, k).unwrap().0;
            let v_ssvi = ssvi.black_vol(t, k).unwrap().0;
            assert_eq!(
                v_essvi.to_bits(),
                v_ssvi.to_bits(),
                "bit-mismatch at T={t}, K={k}"
            );
        }
    }
}

// ── Full surface: paper params + DJX-like data ──────────────────

#[test]
fn paper_surface_total_variance_monotone_in_theta() {
    // Calendar no-arb: at any fixed k, w(k, θ₂) ≥ w(k, θ₁) when θ₂ > θ₁
    let s = paper_surface();
    let tenors = djx_tenors();
    let fwd = 160.0;

    for &k_offset in &[-0.3_f64, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3] {
        let strike = fwd * k_offset.exp();
        let mut prev_var = 0.0;

        for (i, &t) in tenors.iter().enumerate() {
            let var = s.black_variance(t, strike).unwrap().0;
            assert!(
                var >= prev_var - 1e-12,
                "calendar arb at k={k_offset}, tenor[{i}]={t}: var={var:.8} < prev={prev_var:.8}"
            );
            prev_var = var;
        }
    }
}

#[test]
fn paper_surface_atm_variance_matches_thetas() {
    let s = paper_surface();
    let thetas = djx_thetas();
    let tenors = djx_tenors();

    for (i, (&t, &theta)) in tenors.iter().zip(thetas.iter()).enumerate() {
        let var = s.black_variance(t, 160.0).unwrap().0;
        assert!(
            (var - theta).abs() < 1e-12,
            "ATM variance mismatch at tenor[{i}]={t}: var={var}, theta={theta}"
        );
    }
}
