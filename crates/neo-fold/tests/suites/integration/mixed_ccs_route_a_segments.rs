#![allow(non_snake_case)]

#[path = "../../common/fixtures.rs"]
mod fixtures;

use fixtures::build_twist_shout_2step_fixture;
use neo_ccs::relations::{CcsClaim, CcsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify, ShardSegmentKind};
use neo_math::{F, K};
use neo_memory::witness::{StepInstanceBundle, StepWitnessBundle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn build_ccs_only_step(fx: &fixtures::ShardFixture, salt: u64) -> StepWitnessBundle<neo_ajtai::Commitment, F, K> {
    let m = fx.ccs.m;
    let m_in = fx.steps_witness[0].mcs.0.m_in;
    // Keep CCS-only synthetic steps in a bounded coefficient range for SuperNeo NC validity.
    let z: Vec<F> = (0..m)
        .map(|i| match (salt.wrapping_add(i as u64)) % 3 {
            0 => -F::ONE,
            1 => F::ZERO,
            _ => F::ONE,
        })
        .collect();
    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();
    let Z = neo_memory::ajtai::encode_vector_for_ccs_m(&fx.params, z.len(), &z).expect("encode witness for CCS width");
    let c = fx.l.commit(&Z);
    StepWitnessBundle::from((CcsClaim { c, x, m_in }, CcsWitness { w, Z }))
}

#[test]
fn mixed_ccs_only_and_route_a_segments_prove_verify() {
    let fx = build_twist_shout_2step_fixture(123);

    let mut steps_witness: Vec<StepWitnessBundle<neo_ajtai::Commitment, F, K>> =
        vec![build_ccs_only_step(&fx, 100), build_ccs_only_step(&fx, 200)];
    steps_witness.extend(fx.steps_witness.iter().cloned());
    steps_witness.push(build_ccs_only_step(&fx, 300));

    let steps_instance: Vec<StepInstanceBundle<neo_ajtai::Commitment, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_p = Poseidon2Transcript::new(b"mixed-ccs-route-a/segment-test");
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &fx.params,
        &fx.ccs,
        &steps_witness,
        &fx.acc_init,
        &fx.acc_wit_init,
        &fx.l,
        fx.mixers,
    )
    .expect("prove");

    assert!(
        proof
            .steps
            .iter()
            .any(|step| !step.mem.proofs.is_empty() || !step.batched_time.claimed_sums.is_empty()),
        "expected at least one Route-A step proof in mixed segmentation",
    );
    assert!(
        proof
            .steps
            .iter()
            .any(|step| step.mem.proofs.is_empty() && step.batched_time.claimed_sums.is_empty()),
        "expected at least one ccs-only batched step proof in mixed segmentation",
    );
    let seg = proof.segment_meta.as_ref().expect("segment_meta");
    assert_eq!(seg.len(), 3, "expected [CCS-only, Route-A, CCS-only] segments");
    assert_eq!(seg[0].kind, ShardSegmentKind::CcsOnly);
    assert_eq!(seg[0].public_steps, 2);
    assert!(seg[0].proof_steps >= 1);
    assert_eq!(seg[1].kind, ShardSegmentKind::RouteA);
    assert_eq!(seg[1].public_steps, 2);
    assert_eq!(seg[1].proof_steps, 1);
    assert_eq!(seg[2].kind, ShardSegmentKind::CcsOnly);
    assert_eq!(seg[2].public_steps, 1);
    assert_eq!(seg[2].proof_steps, 1);

    let mut tr_v = Poseidon2Transcript::new(b"mixed-ccs-route-a/segment-test");
    let outputs = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &fx.params,
        &fx.ccs,
        &steps_instance,
        &fx.acc_init,
        &proof,
        fx.mixers,
    )
    .expect("verify");

    assert_eq!(outputs.obligations.main.len(), fx.params.k_rho as usize);
}

#[test]
fn mixed_verify_rejects_missing_segment_meta() {
    let fx = build_twist_shout_2step_fixture(456);

    let mut steps_witness: Vec<StepWitnessBundle<neo_ajtai::Commitment, F, K>> =
        vec![build_ccs_only_step(&fx, 11), build_ccs_only_step(&fx, 22)];
    steps_witness.extend(fx.steps_witness.iter().cloned());
    let steps_instance: Vec<StepInstanceBundle<neo_ajtai::Commitment, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_p = Poseidon2Transcript::new(b"mixed-ccs-route-a/segment-meta-required");
    let mut proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &fx.params,
        &fx.ccs,
        &steps_witness,
        &fx.acc_init,
        &fx.acc_wit_init,
        &fx.l,
        fx.mixers,
    )
    .expect("prove");
    proof.segment_meta = None;

    let mut tr_v = Poseidon2Transcript::new(b"mixed-ccs-route-a/segment-meta-required");
    let err = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &fx.params,
        &fx.ccs,
        &steps_instance,
        &fx.acc_init,
        &proof,
        fx.mixers,
    )
    .expect_err("missing segment_meta must fail");

    assert!(
        err.to_string().contains("requires segment_meta"),
        "unexpected error: {err}"
    );
}

#[test]
fn mixed_verify_rejects_tampered_route_a_kind_in_segment_meta() {
    let fx = build_twist_shout_2step_fixture(789);

    let mut steps_witness: Vec<StepWitnessBundle<neo_ajtai::Commitment, F, K>> =
        vec![build_ccs_only_step(&fx, 31), build_ccs_only_step(&fx, 32)];
    steps_witness.extend(fx.steps_witness.iter().cloned());
    let steps_instance: Vec<StepInstanceBundle<neo_ajtai::Commitment, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_p = Poseidon2Transcript::new(b"mixed-ccs-route-a/tamper-route-kind");
    let mut proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &fx.params,
        &fx.ccs,
        &steps_witness,
        &fx.acc_init,
        &fx.acc_wit_init,
        &fx.l,
        fx.mixers,
    )
    .expect("prove");

    let seg = proof.segment_meta.as_mut().expect("segment_meta");
    let route_idx = seg
        .iter()
        .position(|e| e.kind == ShardSegmentKind::RouteA)
        .expect("route-a segment metadata entry");
    seg[route_idx].kind = ShardSegmentKind::CcsOnly;

    let mut tr_v = Poseidon2Transcript::new(b"mixed-ccs-route-a/tamper-route-kind");
    let err = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &fx.params,
        &fx.ccs,
        &steps_instance,
        &fx.acc_init,
        &proof,
        fx.mixers,
    )
    .expect_err("tampered route-a kind must fail");

    assert!(err.to_string().contains("marked CcsOnly"), "unexpected error: {err}");
}

#[test]
fn mixed_verify_rejects_truncated_segment_meta() {
    let fx = build_twist_shout_2step_fixture(790);

    let mut steps_witness: Vec<StepWitnessBundle<neo_ajtai::Commitment, F, K>> =
        vec![build_ccs_only_step(&fx, 41), build_ccs_only_step(&fx, 42)];
    steps_witness.extend(fx.steps_witness.iter().cloned());
    let steps_instance: Vec<StepInstanceBundle<neo_ajtai::Commitment, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_p = Poseidon2Transcript::new(b"mixed-ccs-route-a/truncated-segment-meta");
    let mut proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &fx.params,
        &fx.ccs,
        &steps_witness,
        &fx.acc_init,
        &fx.acc_wit_init,
        &fx.l,
        fx.mixers,
    )
    .expect("prove");

    let seg = proof.segment_meta.as_mut().expect("segment_meta");
    seg.pop();

    let mut tr_v = Poseidon2Transcript::new(b"mixed-ccs-route-a/truncated-segment-meta");
    let err = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &fx.params,
        &fx.ccs,
        &steps_instance,
        &fx.acc_init,
        &proof,
        fx.mixers,
    )
    .expect_err("truncated segment_meta must fail");

    assert!(err.to_string().contains("consumed"), "unexpected error: {err}");
}

#[test]
fn mixed_verify_rejects_tampered_route_a_proof_steps() {
    let fx = build_twist_shout_2step_fixture(791);

    let mut steps_witness: Vec<StepWitnessBundle<neo_ajtai::Commitment, F, K>> =
        vec![build_ccs_only_step(&fx, 51), build_ccs_only_step(&fx, 52)];
    steps_witness.extend(fx.steps_witness.iter().cloned());
    let steps_instance: Vec<StepInstanceBundle<neo_ajtai::Commitment, F, K>> =
        steps_witness.iter().map(StepInstanceBundle::from).collect();

    let mut tr_p = Poseidon2Transcript::new(b"mixed-ccs-route-a/tamper-route-proof-steps");
    let mut proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &fx.params,
        &fx.ccs,
        &steps_witness,
        &fx.acc_init,
        &fx.acc_wit_init,
        &fx.l,
        fx.mixers,
    )
    .expect("prove");

    let seg = proof.segment_meta.as_mut().expect("segment_meta");
    let route_idx = seg
        .iter()
        .position(|e| e.kind == ShardSegmentKind::RouteA)
        .expect("route-a segment metadata entry");
    seg[route_idx].proof_steps += 1;

    let mut tr_v = Poseidon2Transcript::new(b"mixed-ccs-route-a/tamper-route-proof-steps");
    let err = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &fx.params,
        &fx.ccs,
        &steps_instance,
        &fx.acc_init,
        &proof,
        fx.mixers,
    )
    .expect_err("tampered route-a proof_steps must fail");

    assert!(
        err.to_string()
            .contains("proof too short for segment_meta entry"),
        "unexpected error: {err}"
    );
}
