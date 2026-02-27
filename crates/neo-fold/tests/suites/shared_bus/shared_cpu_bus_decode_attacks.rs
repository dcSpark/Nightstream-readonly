use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::shard::{ShardProof, StepProof};
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use neo_memory::riscv::trace::{rv32_decode_lookup_transport_cols, Rv32DecodeSidecarLayout};
use p3_field::PrimeCharacteristicRing;

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

fn first_materialized_step_mut(proof: &mut ShardProof) -> &mut StepProof {
    let step0 = proof
        .steps
        .first_mut()
        .expect("expected at least one proof step");
    if step0
        .compressed_substeps
        .as_ref()
        .is_some_and(|sub| !sub.is_empty())
    {
        return step0
            .compressed_substeps
            .as_mut()
            .and_then(|sub| sub.first_mut())
            .expect("expected at least one compressed materialized proof step");
    }
    step0
}

fn prove_decode_trace_program() -> (Rv32TraceWiringRun, ShardProof) {
    // Program exercises both ALU-imm and ALU-reg decode/linkage paths.
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");

    let proof = run.proof().clone();
    (run, proof)
}

fn tamper_decode_opening_scalar(proof: &mut ShardProof, decode_col: usize) {
    let layout = Rv32DecodeSidecarLayout::new();
    let decode_open_cols = rv32_decode_lookup_transport_cols(&layout);
    assert!(
        decode_open_cols.contains(&decode_col),
        "decode col must be present in WP decode opening set"
    );
    let wp_point = {
        let step = first_materialized_step(proof);
        assert_eq!(
            step.mem.wp_me_claims.len(),
            1,
            "expected one WP ME claim carrying decode openings"
        );
        step.mem.wp_me_claims[0].r.clone()
    };
    let step = first_materialized_step_mut(proof);
    let wp_open_idx = step
        .fold
        .openings
        .iter()
        .position(|opening| opening.point == wp_point && opening.col_ids.iter().any(|&c| c == decode_col))
        .or_else(|| {
            step.fold
                .openings
                .iter()
                .position(|opening| opening.point == wp_point)
        })
        .expect("decode openings must be present in WP named openings");
    let wp_open = &mut step.fold.openings[wp_open_idx];
    assert!(!wp_open.evals.is_empty(), "WP named opening evals must be non-empty");
    let open_idx = wp_open
        .col_ids
        .iter()
        .position(|&c| c == decode_col)
        .unwrap_or(0);
    assert!(open_idx < wp_open.evals.len(), "decode opening index must be in-bounds");
    wp_open.evals[open_idx] += K::ONE;
}

#[test]
fn decode_write_gate_tamper_is_rejected() {
    let (run, mut proof) = prove_decode_trace_program();
    let layout = Rv32DecodeSidecarLayout::new();
    tamper_decode_opening_scalar(&mut proof, layout.ram_has_read);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered decode-stage RAM read selector opening must fail verification"
    );
}

#[test]
fn decode_alu_table_delta_tamper_is_rejected() {
    let (run, mut proof) = prove_decode_trace_program();
    let layout = Rv32DecodeSidecarLayout::new();
    tamper_decode_opening_scalar(&mut proof, layout.ram_has_write);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered decode-stage RAM write selector opening must fail verification"
    );
}
