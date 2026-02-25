use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::shard::{ShardProof, StepProof};
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, BranchCondition, RiscvInstruction, RiscvOpcode};
use neo_memory::riscv::trace::{rv32_decode_lookup_backed_cols, Rv32DecodeSidecarLayout, Rv32TraceLayout};
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

fn prove_control_trace_program(program: Vec<RiscvInstruction>) -> (Rv32TraceWiringRun, ShardProof) {
    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");
    let proof = run.proof().clone();
    (run, proof)
}

fn rv32_wp_opening_cols(layout: &Rv32TraceLayout) -> Vec<usize> {
    vec![
        layout.active,
        layout.is_virtual,
        layout.virtual_sequence_remaining,
        layout.instr_word,
        layout.rs1_addr,
        layout.rs1_val,
        layout.rs2_addr,
        layout.rs2_val,
        layout.rd_addr,
        layout.rd_val,
        layout.rd_has_write,
        layout.ram_addr,
        layout.ram_rv,
        layout.ram_wv,
        layout.shout_has_lookup,
        layout.shout_table_id,
        layout.shout_val,
        layout.shout_lhs,
        layout.shout_rhs,
        layout.shout_add_sub_key,
        layout.jalr_drop_bit,
        layout.pc_before,
        layout.pc_after,
    ]
}

fn tamper_named_wp_opening_scalar(proof: &mut ShardProof, target_col: usize) {
    let wp_point = {
        let step = first_materialized_step(proof);
        assert_eq!(
            step.mem.wp_me_claims.len(),
            1,
            "expected one WP ME claim reused by control stage checks"
        );
        step.mem.wp_me_claims[0].r.clone()
    };
    let step = first_materialized_step_mut(proof);
    let wp_open_idx = step
        .fold
        .openings
        .iter()
        .position(|opening| opening.point == wp_point && opening.col_ids.iter().any(|&c| c == target_col))
        .or_else(|| {
            step.fold
                .openings
                .iter()
                .position(|opening| opening.point == wp_point)
        })
        .expect("control stage openings must be present in WP named openings");
    let wp_open = &mut step.fold.openings[wp_open_idx];
    assert!(!wp_open.evals.is_empty(), "WP named opening evals must be non-empty");
    let open_idx = wp_open
        .col_ids
        .iter()
        .position(|&c| c == target_col)
        .unwrap_or(0);
    assert!(
        open_idx < wp_open.evals.len(),
        "control stage opening index must be in-bounds"
    );
    wp_open.evals[open_idx] += K::ONE;
}

fn tamper_control_decode_opening_scalar(proof: &mut ShardProof, decode_col: usize) {
    let layout = Rv32DecodeSidecarLayout::new();
    let decode_open_cols = rv32_decode_lookup_backed_cols(&layout);
    assert!(
        decode_open_cols.contains(&decode_col),
        "decode col must be present in control stage decode opening set"
    );
    tamper_named_wp_opening_scalar(proof, decode_col);
}

fn tamper_control_wp_opening_scalar(proof: &mut ShardProof, trace_col: usize) {
    let layout = Rv32TraceLayout::new();
    let open_cols = rv32_wp_opening_cols(&layout);
    assert!(
        open_cols.contains(&trace_col),
        "trace col must be present in control stage WP opening set"
    );
    tamper_named_wp_opening_scalar(proof, trace_col);
}

#[test]
fn control_jal_target_tamper_is_rejected() {
    let program = vec![
        RiscvInstruction::Jal { rd: 1, imm: 8 },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let (run, mut proof) = prove_control_trace_program(program);
    let decode = Rv32DecodeSidecarLayout::new();
    tamper_control_decode_opening_scalar(&mut proof, decode.imm_j);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered control stage JAL target opening must fail verification"
    );
}

#[test]
fn control_jalr_target_tamper_is_rejected() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 8,
        },
        RiscvInstruction::Jalr { rd: 2, rs1: 1, imm: 0 },
        RiscvInstruction::Halt,
    ];
    let (run, mut proof) = prove_control_trace_program(program);
    let decode = Rv32DecodeSidecarLayout::new();
    tamper_control_decode_opening_scalar(&mut proof, decode.imm_i);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered control stage JALR target opening must fail verification"
    );
}

#[test]
fn control_branch_decision_target_tamper_is_rejected() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 0,
            rs2: 0,
            imm: 8,
        },
        RiscvInstruction::Halt,
    ];
    let (run, mut proof) = prove_control_trace_program(program);
    let decode = Rv32DecodeSidecarLayout::new();
    tamper_control_decode_opening_scalar(&mut proof, decode.funct3_bit[0]);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered control stage branch decision/target opening must fail verification"
    );
}

#[test]
fn control_control_writeback_tamper_is_rejected() {
    let program = vec![RiscvInstruction::Auipc { rd: 1, imm: 1 }, RiscvInstruction::Halt];
    let (run, mut proof) = prove_control_trace_program(program);
    let trace = Rv32TraceLayout::new();
    tamper_control_wp_opening_scalar(&mut proof, trace.rd_val);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered control stage control-writeback opening must fail verification"
    );
}
