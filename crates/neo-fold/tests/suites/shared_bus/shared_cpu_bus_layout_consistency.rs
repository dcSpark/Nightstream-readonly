#![allow(non_snake_case)]

#[path = "../../common/fixtures.rs"]
mod fixtures;

use fixtures::{build_twist_shout_2step_fixture, prove};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{ShardProof, StepProof};
use neo_math::K;
use neo_memory::cpu::build_bus_layout_for_instances;
use neo_memory::mle::chi_at_index;

fn first_materialized_step(proof: &ShardProof) -> &StepProof {
    let step0 = proof
        .steps
        .first()
        .expect("expected at least one proof step");
    if step0
        .compressed_substeps
        .as_ref()
        .is_some_and(|sub| !sub.is_empty())
    {
        return step0
            .compressed_substeps
            .as_ref()
            .and_then(|sub| sub.first())
            .expect("expected at least one compressed materialized proof step");
    }
    step0
}

fn opening_eval_and_point_for_col(step: &StepProof, logical_col: usize) -> (K, Vec<K>) {
    let opening = step
        .fold
        .openings
        .iter()
        .find(|opening| opening.col_ids.iter().any(|&c| c == logical_col))
        .expect("expected named opening carrying requested logical bus col");
    assert!(!opening.evals.is_empty(), "named opening evals must be non-empty");
    let idx = opening
        .col_ids
        .iter()
        .position(|&c| c == logical_col)
        .unwrap_or(0);
    assert!(idx < opening.evals.len(), "opening index must be in-bounds");
    (opening.evals[idx], opening.point.clone())
}

#[test]
fn shared_cpu_bus_copyout_indices_match_bus_layout() {
    let fx = build_twist_shout_2step_fixture(123);
    let proof = prove(FoldingMode::Optimized, &fx);

    let step0_wit = &fx.steps_witness[0];
    let step0_inst = &fx.steps_instance[0];
    let first_proof_step = first_materialized_step(&proof);

    let s0 = fx.ccs.clone();
    let bus = build_bus_layout_for_instances(
        s0.m,
        step0_inst.mcs_inst.m_in,
        1,
        step0_inst.lut_insts.iter().map(|inst| inst.d * inst.ell),
        step0_inst.mem_insts.iter().map(|inst| inst.d * inst.ell),
    )
    .expect("bus layout");

    let z = neo_memory::ajtai::decode_vector_for_ccs_m(&fx.params, s0.m, &step0_wit.mcs.1.Z)
        .expect("decode logical witness from packed Z");

    let time_row = bus.time_index(0);

    let shout0 = &bus.shout_cols[0].lanes[0];
    let twist0 = &bus.twist_cols[0].lanes[0];
    let col_ids = [
        shout0.has_lookup,
        shout0.primary_val(),
        twist0.has_write,
        twist0.wv,
        twist0.inc,
    ];

    for col_id in col_ids {
        let z_idx = bus.bus_cell(col_id, 0);
        let expected_base: K = z[z_idx].into();
        let cpu_cols_len = first_proof_step.fold.time_cpu_commitments.len();
        let logical_col = *first_proof_step
            .fold
            .time_col_ids
            .get(cpu_cols_len + col_id)
            .expect("expected logical id for shared-bus column");
        let (actual, point) = opening_eval_and_point_for_col(first_proof_step, logical_col);
        let chi = chi_at_index(point.as_slice(), time_row);
        let expected = expected_base * chi;
        assert_eq!(actual, expected, "copyout mismatch at col_id={col_id}");
    }
}
