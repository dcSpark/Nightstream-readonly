//! Stage 5: EV minimal (commit‑evo + Ajtai) — binding switch

use std::sync::Arc;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn build_pp_and_vectors() -> (neo_ajtai::PP<neo_math::Rq>, Vec<i64>, Vec<F>, Vec<F>) {
    let d = neo_math::ring::D; let m = 1usize; let kappa = 2usize;
    let mut z_digits: Vec<i64> = Vec::with_capacity(d*m);
    for i in 0..(d*m) { z_digits.push(match i % 3 { 0 => -1, 1 => 0, _ => 1 }); }
    let to_f = |zi: i64| if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
    let z_f: Vec<F> = z_digits.iter().copied().map(to_f).collect();
    use rand::SeedableRng; use rand::rngs::StdRng;
    let mut rng = StdRng::from_seed([99u8; 32]);
    let pp = neo_ajtai::setup(&mut rng, d, kappa, m).expect("setup");
    let c_step = neo_ajtai::commit(&pp, &z_f).data; // L·Z
    (pp, z_digits, z_f, c_step)
}

#[test]
fn ev_case_a_bind_step_vectors() {
    let (pp, z_digits, _z_f, c_step) = build_pp_and_vectors();
    let n = c_step.len();
    let c_prev = vec![F::ZERO; n];
    let rho = F::ONE; // simplify so c_next = c_step
    let c_next: Vec<F> = c_step.iter().zip(c_prev.iter()).map(|(s, p)| *p + rho * *s).collect();

    // me.c_coords should be c_next; Ajtai binds to acc_c_step
    #[allow(deprecated)]
    let me = neo_ccs::MEInstance { c_coords: c_next.clone(), y_outputs: vec![], r_point: vec![], base_b: 2,
        header_digest: [0u8; 32], c_step_coords: vec![], u_offset: 0, u_len: 0 };
    #[allow(deprecated)]
    let wit = neo_ccs::MEWitness { z_digits: z_digits.clone(), weight_vectors: vec![], ajtai_rows: None };

    let ev = neo_spartan_bridge::me_to_r1cs::IvcEvEmbed {
        rho,
        y_prev: vec![],
        y_next: vec![],
        y_step_public: None,
        fold_chain_digest: None,
        acc_c_prev: Some(c_prev.clone()),
        acc_c_step: Some(c_step.clone()),
        acc_c_next: Some(c_next.clone()),
        rho_eff: None,
    };

    let proof = neo_spartan_bridge::compress_ivc_verifier_to_lean_proof_with_linkage_and_pi_ccs(
        &me, &wit, Some(Arc::new(pp)), Some(ev), None, None, None
    ).expect("prove");
    let ok = neo_spartan_bridge::verify_lean_proof(&proof).expect("verify path");
    assert!(ok, "EV Case A (bind to step) should verify");
}

#[test]
fn ev_case_b_bind_c_next_and_parity() {
    let (pp, z_digits, _z_f, c_step) = build_pp_and_vectors();
    let n = c_step.len();
    let c_prev = vec![F::ZERO; n];
    let rho = F::ONE; // c_next = c_step so Ajtai==c_next
    let c_next: Vec<F> = c_step.iter().zip(c_prev.iter()).map(|(s, p)| *p + rho * *s).collect();

    #[allow(deprecated)]
    let me = neo_ccs::MEInstance { c_coords: c_next.clone(), y_outputs: vec![], r_point: vec![], base_b: 2,
        header_digest: [0u8; 32], c_step_coords: vec![], u_offset: 0, u_len: 0 };
    #[allow(deprecated)]
    let wit = neo_ccs::MEWitness { z_digits: z_digits.clone(), weight_vectors: vec![], ajtai_rows: None };

    // Provide c_next but no acc_c_step; parity guard should hold and Ajtai binds to c_next (== PP·Z)
    let ev = neo_spartan_bridge::me_to_r1cs::IvcEvEmbed {
        rho,
        y_prev: vec![], y_next: vec![], y_step_public: None, fold_chain_digest: None,
        acc_c_prev: None, acc_c_step: None, acc_c_next: Some(c_next.clone()), rho_eff: None,
    };

    let proof = neo_spartan_bridge::compress_ivc_verifier_to_lean_proof_with_linkage_and_pi_ccs(
        &me, &wit, Some(Arc::new(pp)), Some(ev), None, None, None
    ).expect("prove");
    let ok = neo_spartan_bridge::verify_lean_proof(&proof).expect("verify path");
    assert!(ok, "EV Case B (bind to c_next) should verify when c_next==PP·Z");
}

#[test]
fn ev_tamper_acc_c_step_fails() {
    let (pp, z_digits, _z_f, mut c_step) = build_pp_and_vectors();
    let n = c_step.len();
    let c_prev = vec![F::ZERO; n];
    let rho = F::ONE;
    let c_next: Vec<F> = c_step.iter().zip(c_prev.iter()).map(|(s, p)| *p + rho * *s).collect();

    // Tamper one coordinate in acc_c_step
    c_step[0] = c_step[0] + F::ONE;
    #[allow(deprecated)]
    let me = neo_ccs::MEInstance { c_coords: c_next.clone(), y_outputs: vec![], r_point: vec![], base_b: 2,
        header_digest: [0u8; 32], c_step_coords: vec![], u_offset: 0, u_len: 0 };
    #[allow(deprecated)]
    let wit = neo_ccs::MEWitness { z_digits, weight_vectors: vec![], ajtai_rows: None };
    let ev = neo_spartan_bridge::me_to_r1cs::IvcEvEmbed {
        rho,
        y_prev: vec![], y_next: vec![], y_step_public: None, fold_chain_digest: None,
        acc_c_prev: Some(c_prev), acc_c_step: Some(c_step), acc_c_next: Some(c_next), rho_eff: None,
    };
    let proof = neo_spartan_bridge::compress_ivc_verifier_to_lean_proof_with_linkage_and_pi_ccs(
        &me, &wit, Some(Arc::new(pp)), Some(ev), None, None, None
    ).expect("prove");
    // With tampered acc_c_step, verification must fail
    let verified = neo_spartan_bridge::verify_lean_proof(&proof).unwrap_or(false);
    assert!(!verified, "Tampered acc_c_step should cause verification to fail");
}
