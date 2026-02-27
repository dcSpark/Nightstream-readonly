use neo_fold::pi_ccs::FoldingMode;
use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::shard::{ShardProof, StepProof};
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode, REG_ID};
use neo_memory::riscv::trace::Rv32TraceLayout;
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

fn prove_mulh_decomposition_trace_program() -> (Rv32TraceWiringRun, ShardProof) {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .mode(FoldingMode::Optimized)
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");

    let proof = run.proof().clone();
    (run, proof)
}

fn tamper_reg_domain_bit_opening(run: &Rv32TraceWiringRun, proof: &mut ShardProof, lane_idx: usize, write_bit: bool) {
    let steps = run.steps_public();
    assert_eq!(steps.len(), 1, "expected one trace step");
    let step0 = &steps[0];

    let reg_inst_idx = step0
        .mem_insts
        .iter()
        .position(|inst| inst.mem_id == REG_ID.0)
        .expect("expected REG mem instance");
    let reg_layout = step0.mem_insts[reg_inst_idx].twist_layout();
    let reg_lane = reg_layout.lanes.get(lane_idx).expect("expected REG lane");

    let bit_local_col = if write_bit {
        let len = reg_lane.wa_bits.end - reg_lane.wa_bits.start;
        assert!(len > 5, "expected REG write address width to include bit[5]");
        reg_lane.wa_bits.start + 5
    } else {
        let len = reg_lane.ra_bits.end - reg_lane.ra_bits.start;
        assert!(len > 5, "expected REG read address width to include bit[5]");
        reg_lane.ra_bits.start + 5
    };
    let cpu_cols_len = step0.time_columns.cpu_cols.len();
    let bit_col = *step0
        .time_columns
        .col_ids
        .get(cpu_cols_len + bit_local_col)
        .expect("REG bit local column must map to a logical time column id");

    let step = first_materialized_step_mut(proof);
    let open_idx = step
        .fold
        .openings
        .iter()
        .position(|opening| opening.col_ids.iter().any(|&c| c == bit_col))
        .expect("shared-bus openings must be present in named openings");
    let opening = &mut step.fold.openings[open_idx];
    assert!(
        !opening.evals.is_empty(),
        "shared-bus named opening evals must be non-empty"
    );
    let eval_idx = opening
        .col_ids
        .iter()
        .position(|&c| c == bit_col)
        .unwrap_or(0);
    assert!(
        eval_idx < opening.evals.len(),
        "shared-bus opening index must be in-bounds"
    );
    opening.evals[eval_idx] += K::ONE;
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
    ]
}

fn tamper_trace_wp_opening_scalar(proof: &mut ShardProof, trace_col: usize) {
    let layout = Rv32TraceLayout::new();
    let wp_cols = rv32_wp_opening_cols(&layout);
    let open_idx = wp_cols
        .iter()
        .position(|&c| c == trace_col)
        .expect("trace col must be present in WP opening set");
    let wp_point = {
        let step = first_materialized_step(proof);
        assert_eq!(step.mem.wp_me_claims.len(), 1, "expected one WP ME claim");
        step.mem.wp_me_claims[0].r.clone()
    };
    let step = first_materialized_step_mut(proof);
    let wp_open_idx = step
        .fold
        .openings
        .iter()
        .position(|opening| opening.point == wp_point && opening.col_ids.iter().any(|&c| c == trace_col))
        .or_else(|| {
            step.fold
                .openings
                .iter()
                .position(|opening| opening.point == wp_point)
        })
        .expect("WP openings must be present in named openings");
    let wp_open = &mut step.fold.openings[wp_open_idx];
    assert!(!wp_open.evals.is_empty(), "WP named opening evals must be non-empty");
    let eval_idx = wp_open
        .col_ids
        .iter()
        .position(|&c| c == trace_col)
        .unwrap_or(open_idx);
    assert!(eval_idx < wp_open.evals.len(), "WP opening index must be in-bounds");
    wp_open.evals[eval_idx] += K::ONE;
}

fn force_trace_wp_opening_scalar_zero(proof: &mut ShardProof, trace_col: usize) {
    let layout = Rv32TraceLayout::new();
    let wp_cols = rv32_wp_opening_cols(&layout);
    let open_idx = wp_cols
        .iter()
        .position(|&c| c == trace_col)
        .expect("trace col must be present in WP opening set");
    let wp_point = {
        let step = first_materialized_step(proof);
        assert_eq!(step.mem.wp_me_claims.len(), 1, "expected one WP ME claim");
        step.mem.wp_me_claims[0].r.clone()
    };
    let step = first_materialized_step_mut(proof);
    let wp_open_idx = step
        .fold
        .openings
        .iter()
        .position(|opening| opening.point == wp_point && opening.col_ids.iter().any(|&c| c == trace_col))
        .or_else(|| {
            step.fold
                .openings
                .iter()
                .position(|opening| opening.point == wp_point)
        })
        .expect("WP openings must be present in named openings");
    let wp_open = &mut step.fold.openings[wp_open_idx];
    assert!(!wp_open.evals.is_empty(), "WP named opening evals must be non-empty");
    let eval_idx = wp_open
        .col_ids
        .iter()
        .position(|&c| c == trace_col)
        .unwrap_or(open_idx);
    assert!(eval_idx < wp_open.evals.len(), "WP opening index must be in-bounds");
    wp_open.evals[eval_idx] = K::ZERO;
}

fn prove_nonvirtual_trace_program() -> (Rv32TraceWiringRun, ShardProof) {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .mode(FoldingMode::Optimized)
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");
    let proof = run.proof().clone();
    (run, proof)
}

fn prove_mulhu_trace_program() -> (Rv32TraceWiringRun, ShardProof) {
    let program = vec![
        RiscvInstruction::Lui { rd: 1, imm: 16 },
        RiscvInstruction::Lui { rd: 2, imm: 16 },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhu,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .mode(FoldingMode::Optimized)
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");
    let proof = run.proof().clone();
    (run, proof)
}

fn prove_mul_trace_program() -> (Rv32TraceWiringRun, ShardProof) {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .mode(FoldingMode::Optimized)
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");
    let proof = run.proof().clone();
    (run, proof)
}

fn prove_divrem_decomposition_trace_program(op: RiscvOpcode) -> (Rv32TraceWiringRun, ShardProof) {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 73,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 9,
        },
        RiscvInstruction::RAlu {
            op,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .mode(FoldingMode::Optimized)
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");
    let proof = run.proof().clone();
    (run, proof)
}

#[test]
fn virtual_domain_reg_write_bit_tamper_is_rejected() {
    let (run, mut proof) = prove_mulh_decomposition_trace_program();
    // REG lane0 is the write lane. Tamper write address bit[5] opening.
    tamper_reg_domain_bit_opening(&run, &mut proof, /*lane_idx=*/ 0, /*write_bit=*/ true);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered REG write-domain opening must fail verification"
    );
}

#[test]
fn nonvirtual_domain_reg_read_bit_tamper_is_rejected() {
    let (run, mut proof) = prove_mulh_decomposition_trace_program();
    // REG lane1 is read-only lane (rs2). Tamper read address bit[5] opening.
    tamper_reg_domain_bit_opening(&run, &mut proof, /*lane_idx=*/ 1, /*write_bit=*/ false);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered REG read-domain opening must fail verification"
    );
}

#[test]
fn virtual_flag_tamper_requires_virtual_write_domain() {
    let (run, mut proof) = prove_nonvirtual_trace_program();
    let trace = Rv32TraceLayout::new();
    tamper_trace_wp_opening_scalar(&mut proof, trace.is_virtual);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered is_virtual opening must fail virtual write-domain verification"
    );
}

#[test]
fn virtual_flag_zero_tamper_breaks_nonvirtual_arch_domain() {
    let (run, mut proof) = prove_mulh_decomposition_trace_program();
    let trace = Rv32TraceLayout::new();
    force_trace_wp_opening_scalar_zero(&mut proof, trace.is_virtual);
    assert!(
        run.verify_proof(&proof).is_err(),
        "forcing is_virtual opening to zero must fail non-virtual architectural-domain verification"
    );
}

#[test]
fn virtual_sequence_remaining_tamper_is_rejected() {
    let (run, mut proof) = prove_mulh_decomposition_trace_program();
    let trace = Rv32TraceLayout::new();
    tamper_trace_wp_opening_scalar(&mut proof, trace.virtual_sequence_remaining);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered virtual_sequence_remaining opening must fail verification"
    );
}

#[test]
fn mulhu_add_sub_key_opening_tamper_is_rejected() {
    let (run, mut proof) = prove_mulhu_trace_program();
    let trace = Rv32TraceLayout::new();
    tamper_trace_wp_opening_scalar(&mut proof, trace.shout_add_sub_key);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered MULHU shout_add_sub_key opening must fail verification"
    );
}

#[test]
fn mul_add_sub_key_opening_tamper_is_rejected() {
    let (run, mut proof) = prove_mul_trace_program();
    let trace = Rv32TraceLayout::new();
    tamper_trace_wp_opening_scalar(&mut proof, trace.shout_add_sub_key);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered MUL shout_add_sub_key opening must fail verification"
    );
}

#[test]
fn div_shout_table_id_tamper_is_rejected() {
    let (run, mut proof) = prove_divrem_decomposition_trace_program(RiscvOpcode::Div);
    let trace = Rv32TraceLayout::new();
    tamper_trace_wp_opening_scalar(&mut proof, trace.shout_table_id);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered DIV shout_table_id opening must fail verification"
    );
}

#[test]
fn divu_shout_table_id_tamper_is_rejected() {
    let (run, mut proof) = prove_divrem_decomposition_trace_program(RiscvOpcode::Divu);
    let trace = Rv32TraceLayout::new();
    tamper_trace_wp_opening_scalar(&mut proof, trace.shout_table_id);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered DIVU shout_table_id opening must fail verification"
    );
}

#[test]
fn rem_shout_table_id_tamper_is_rejected() {
    let (run, mut proof) = prove_divrem_decomposition_trace_program(RiscvOpcode::Rem);
    let trace = Rv32TraceLayout::new();
    tamper_trace_wp_opening_scalar(&mut proof, trace.shout_table_id);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered REM shout_table_id opening must fail verification"
    );
}

#[test]
fn remu_shout_table_id_tamper_is_rejected() {
    let (run, mut proof) = prove_divrem_decomposition_trace_program(RiscvOpcode::Remu);
    let trace = Rv32TraceLayout::new();
    tamper_trace_wp_opening_scalar(&mut proof, trace.shout_table_id);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered REMU shout_table_id opening must fail verification"
    );
}
