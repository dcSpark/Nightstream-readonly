use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_math::K;
use neo_memory::riscv::ccs::TraceShoutBusSpec;
use neo_memory::riscv::lookups::{
    encode_program, BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode, RiscvShoutTables, PROG_ID, RAM_ID,
    REG_ID,
};
use neo_memory::riscv::trace::{
    rv32_decode_lookup_backed_cols, rv32_is_decode_lookup_table_id, rv32_width_lookup_backed_cols,
    Rv32DecodeSidecarLayout, Rv32WidthSidecarLayout,
};
use p3_field::PrimeCharacteristicRing;

#[test]
fn rv32_trace_wiring_runner_prove_verify() {
    // Program: ADDI x1, x0, 1; ADDI x2, x1, 2; HALT
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
            rs1: 1,
            imm: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .min_trace_len(1)
        .prove()
        .expect("trace wiring prove");

    run.verify().expect("trace wiring verify");

    assert_eq!(run.fold_count(), 1, "trace runner should produce one folding step");
    assert_eq!(run.trace_len(), 3, "active trace length mismatch");
    assert_eq!(
        run.exec_table().rows.len(),
        3,
        "exec table should not be padded to next power-of-two"
    );
    assert_eq!(
        run.layout().t,
        run.exec_table().rows.len(),
        "layout.t should match exec rows"
    );

    let steps_public = run.steps_public();
    assert_eq!(steps_public.len(), 1, "trace runner should expose one step instance");
    let mut mem_ids: Vec<u32> = steps_public[0]
        .mem_insts
        .iter()
        .map(|inst| inst.mem_id)
        .collect();
    mem_ids.sort_unstable();
    let mut expected_mem_ids = vec![PROG_ID.0, RAM_ID.0, REG_ID.0];
    expected_mem_ids.retain(|&id| id != RAM_ID.0);
    expected_mem_ids.sort_unstable();
    assert_eq!(
        mem_ids, expected_mem_ids,
        "trace runner should default to used-sidecar instantiation (no RAM sidecar when unused)"
    );
    assert_eq!(
        run.used_memory_ids(),
        expected_mem_ids.as_slice(),
        "run artifact should record auto-derived S_memory"
    );
    let add_table_id = RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Add).0;
    let used_lookup_ids = run.used_shout_table_ids();
    assert!(
        used_lookup_ids.contains(&add_table_id),
        "run artifact should include opcode-backed S_lookup tables"
    );
    let decode_lookup_count = used_lookup_ids
        .iter()
        .copied()
        .filter(|table_id| rv32_is_decode_lookup_table_id(*table_id))
        .count();
    assert_eq!(
        decode_lookup_count,
        rv32_decode_lookup_backed_cols(&Rv32DecodeSidecarLayout::new()).len(),
        "run artifact should include decode lookup families in S_lookup"
    );
}

#[test]
fn rv32_trace_wiring_runner_reg_output_binding_prove_verify() {
    // Program: ADDI x2, x0, 3; HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .reg_output_claim(/*reg=*/ 2, /*expected=*/ neo_math::F::from_u64(3))
        .prove()
        .expect("trace wiring prove with reg output binding");

    run.verify()
        .expect("trace wiring verify with reg output binding");
}

#[test]
fn rv32_trace_wiring_runner_allows_without_insecure_ack() {
    let program = vec![RiscvInstruction::Halt];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .prove()
        .expect("trace wiring should no longer require insecure benchmark-only ack");
    run.verify()
        .expect("trace wiring proof should verify without insecure benchmark-only ack");
}

#[test]
fn rv32_trace_wiring_runner_prove_verify_without_insecure_ack() {
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
        .min_trace_len(1)
        .prove()
        .expect("trace wiring should prove without insecure benchmark-only ack");

    run.verify()
        .expect("trace wiring proof should verify without insecure benchmark-only ack");
}

#[test]
fn rv32_trace_wiring_runner_shared_bus_default_and_legacy_fallback_differ() {
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

    let run_shared = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .min_trace_len(1)
        .prove()
        .expect("trace wiring prove");

    let legacy_err = match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .shared_cpu_bus(false)
        .min_trace_len(1)
        .prove()
    {
        Ok(_) => panic!("legacy no-shared fallback must be rejected"),
        Err(e) => e,
    };

    let msg = legacy_err.to_string();
    assert!(
        msg.contains("no-shared fallback is removed"),
        "unexpected no-shared rejection error: {msg}"
    );
    assert_eq!(
        run_shared.ccs_num_variables(),
        run_shared.layout().m,
        "shared-bus trace layout width must match CCS width"
    );
}

#[test]
fn rv32_trace_wiring_runner_shout_override_must_superset_inferred_set() {
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

    let err = match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .shout_ops([RiscvOpcode::Xor])
        .prove()
    {
        Ok(_) => panic!("shout override that misses required tables must fail"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("superset") && msg.contains("Add"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn rv32_trace_wiring_runner_rejects_extra_shout_spec_without_table_spec() {
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

    let err = match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .extra_shout_bus_specs([TraceShoutBusSpec {
            table_id: 1000,
            ell_addr: 13,
            n_vals: 1usize,
}])
        .prove()
    {
        Ok(_) => panic!("extra shout geometry without table spec must fail"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("extra_shout_bus_specs includes table_id=1000 without a table spec"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn rv32_trace_wiring_runner_accepts_extra_shout_spec_with_matching_table_spec() {
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
        .extra_lut_table_spec(1000, neo_memory::witness::LutTableSpec::IdentityU32)
        .extra_shout_bus_specs([TraceShoutBusSpec {
            table_id: 1000,
            ell_addr: 32,
            n_vals: 1usize,
}])
        .prove()
        .expect("trace wiring prove with extra table/spec");
    run.verify()
        .expect("trace wiring verify with extra table/spec");

    assert!(
        run.used_shout_table_ids().contains(&1000),
        "run should record extra table_id in used shout set"
    );
}

#[test]
fn rv32_trace_wiring_runner_rejects_extra_table_spec_colliding_with_opcode_table() {
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

    let add_table_id = RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Add).0;
    let err = match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .extra_lut_table_spec(add_table_id, neo_memory::witness::LutTableSpec::IdentityU32)
        .prove()
    {
        Ok(_) => panic!("extra_lut_table_spec collision with inferred opcode table must fail"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("extra_lut_table_spec collides with existing table_id"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn rv32_trace_wiring_runner_rejects_max_steps_above_trace_cap() {
    let program = vec![RiscvInstruction::Halt];
    let program_bytes = encode_program(&program);

    let err = match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .max_steps((1usize << 20) + 1)
        .prove()
    {
        Ok(_) => panic!("max_steps above trace cap must be rejected"),
        Err(e) => e,
    };

    let msg = err.to_string();
    assert!(
        msg.contains("max_steps=") && msg.contains("trace-mode hard cap"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn rv32_trace_wiring_runner_rejects_min_trace_len_above_trace_cap() {
    let program = vec![RiscvInstruction::Halt];
    let program_bytes = encode_program(&program);

    let err = match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .min_trace_len((1usize << 20) + 1)
        .prove()
    {
        Ok(_) => panic!("min_trace_len above trace cap must be rejected"),
        Err(e) => e,
    };

    let msg = err.to_string();
    assert!(
        msg.contains("min_trace_len=") && msg.contains("trace-mode hard cap"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn rv32_trace_wiring_runner_chunked_ivc_step_linking() {
    // Program: ADDI x1, x0, 1; ADDI x2, x1, 2; HALT
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
            rs1: 1,
            imm: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(2)
        .prove()
        .expect("trace wiring prove with chunked ivc");

    run.verify().expect("trace wiring verify with chunked ivc");

    assert_eq!(
        run.fold_count(),
        2,
        "chunk_rows=2 over 3 rows should produce two fold steps"
    );
    let steps = run.steps_public();
    assert_eq!(steps.len(), 2, "expected two public steps");

    let layout = run.layout();
    let prev = &steps[0].mcs_inst.x;
    let cur = &steps[1].mcs_inst.x;
    assert_eq!(
        prev[layout.pc_final], cur[layout.pc0],
        "trace step linking must enforce pc_final -> pc0 across steps"
    );
}

#[test]
fn rv32_trace_wiring_runner_chunked_ivc_batches_no_shared_val_lanes_per_mem() {
    // Program: ADDI x1, x0, 1; ADDI x2, x1, 2; HALT
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
            rs1: 1,
            imm: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(2)
        .prove()
        .expect("trace wiring prove with chunked ivc");
    run.verify().expect("trace wiring verify with chunked ivc");

    let steps_public = run.steps_public();
    let shard_proof = run.proof();
    assert_eq!(steps_public.len(), 2, "expected two public steps");
    assert_eq!(shard_proof.steps.len(), 2, "expected two proof steps");

    // Step 0 (shared-bus): one current CPU val claim.
    let proof_step0 = &shard_proof.steps[0];
    assert_eq!(
        proof_step0.mem.val_me_claims.len(),
        1,
        "step0(shared) must emit one current CPU val claim"
    );
    assert_eq!(
        proof_step0.val_fold.len(),
        1,
        "step0(shared) must emit one val-fold proof"
    );

    // Step 1 (shared-bus): val claims are [current_cpu, previous_cpu], each with its own fold proof.
    let proof_step1 = &shard_proof.steps[1];
    assert_eq!(
        proof_step1.mem.val_me_claims.len(),
        2,
        "step1(shared) must emit current+previous CPU val claims"
    );
    assert_eq!(
        proof_step1.val_fold.len(),
        2,
        "step1(shared) must emit one val-fold proof per claim"
    );
}

#[test]
fn rv32_trace_wiring_runner_wb_wp_folds_are_emitted_and_required() {
    // Program: ADDI x1, x0, 1; HALT
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
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");

    let proof = run.proof().clone();
    assert_eq!(proof.steps.len(), 1, "expected one step proof");
    assert!(
        !proof.steps[0].mem.wb_me_claims.is_empty(),
        "expected WB ME claims for RV32 trace route-A"
    );
    assert!(
        !proof.steps[0].mem.wp_me_claims.is_empty(),
        "expected WP ME claims for RV32 trace route-A"
    );
    assert!(
        !proof.steps[0].wb_fold.is_empty(),
        "expected wb_fold proofs for RV32 trace route-A"
    );
    assert!(
        !proof.steps[0].wp_fold.is_empty(),
        "expected wp_fold proofs for RV32 trace route-A"
    );

    let mut proof_missing_wb = proof.clone();
    proof_missing_wb.steps[0].wb_fold.clear();
    assert!(
        run.verify_proof(&proof_missing_wb).is_err(),
        "missing wb_fold must fail verification"
    );

    let mut proof_missing_wp = proof.clone();
    proof_missing_wp.steps[0].wp_fold.clear();
    assert!(
        run.verify_proof(&proof_missing_wp).is_err(),
        "missing wp_fold must fail verification"
    );
}

#[test]
fn rv32_trace_wiring_runner_decode_openings_are_embedded_in_wp_and_required() {
    // Program: ADDI x1, x0, 1; HALT
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
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");

    let proof = run.proof().clone();
    assert_eq!(proof.steps.len(), 1, "expected one step proof");
    assert_eq!(proof.steps[0].mem.wp_me_claims.len(), 1, "expected one WP ME claim");
    let mut proof_missing_decode_me = proof.clone();
    let decode_layout = Rv32DecodeSidecarLayout::new();
    let decode_open_cols = rv32_decode_lookup_backed_cols(&decode_layout);
    let me = &mut proof_missing_decode_me.steps[0].mem.wp_me_claims[0];
    let decode_start = me
        .y_scalars
        .len()
        .checked_sub(decode_open_cols.len())
        .expect("decode openings must be appended to WP ME tail");
    let decode_idx = decode_open_cols
        .iter()
        .position(|&c| c == decode_layout.op_alu_imm)
        .expect("decode opening column must be present");
    me.y_scalars[decode_start + decode_idx] += K::ONE;
    assert!(
        run.verify_proof(&proof_missing_decode_me).is_err(),
        "tampered decode lookup opening embedded in WP ME must fail verification"
    );
}

#[test]
fn rv32_trace_wiring_runner_width_openings_on_wp_are_required() {
    // Program: ADDI x1, x0, 1; HALT
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
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");

    let proof = run.proof().clone();
    assert_eq!(proof.steps.len(), 1, "expected one step proof");
    assert_eq!(proof.steps[0].mem.wp_me_claims.len(), 1, "expected one WP ME claim");

    let mut proof_tampered_width_open = proof.clone();
    let width_layout = Rv32WidthSidecarLayout::new();
    let width_open_cols = rv32_width_lookup_backed_cols(&width_layout);
    let wp_me = &mut proof_tampered_width_open.steps[0].mem.wp_me_claims[0];
    let width_open_start = wp_me
        .y_scalars
        .len()
        .checked_sub(width_open_cols.len())
        .expect("width openings must be appended to WP ME tail");
    let width_idx = width_open_cols
        .iter()
        .position(|&c| c == width_layout.rs2_low_bit[0])
        .expect("width opening column must be present");
    wp_me.y_scalars[width_open_start + width_idx] += K::ONE;
    assert!(
        run.verify_proof(&proof_tampered_width_open).is_err(),
        "tampered width lookup opening embedded in WP ME must fail verification"
    );
}

#[test]
fn rv32_trace_wiring_runner_control_claims_are_emitted_and_required() {
    // Program: ADDI x1, x0, 1; HALT
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
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");

    let proof = run.proof().clone();
    assert_eq!(proof.steps.len(), 1, "expected one step proof");

    let labels = &proof.steps[0].batched_time.labels;
    let find_w4 = |label: &'static [u8]| -> usize {
        labels
            .iter()
            .position(|l| *l == label)
            .expect("missing required control stage claim label in batched_time")
    };
    let control_linear_idx = find_w4(b"control/next_pc_linear");
    let control_control_idx = find_w4(b"control/next_pc_control");
    let control_branch_idx = find_w4(b"control/branch_semantics");
    let _control_writeback_idx = find_w4(b"control/writeback");
    assert!(
        control_linear_idx < labels.len() && control_control_idx < labels.len() && control_branch_idx < labels.len(),
        "control stage labels must be present in batched_time"
    );

    let mut proof_missing_control_claim = proof.clone();
    let _ = proof_missing_control_claim.steps[0]
        .batched_time
        .claimed_sums
        .remove(control_control_idx);
    let _ = proof_missing_control_claim.steps[0]
        .batched_time
        .degree_bounds
        .remove(control_control_idx);
    let _ = proof_missing_control_claim.steps[0]
        .batched_time
        .labels
        .remove(control_control_idx);
    let _ = proof_missing_control_claim.steps[0]
        .batched_time
        .round_polys
        .remove(control_control_idx);
    assert!(
        run.verify_proof(&proof_missing_control_claim).is_err(),
        "missing control/next_pc_control claim artifact must fail verification"
    );

    let mut proof_tampered_control_round = proof.clone();
    let coeff = proof_tampered_control_round.steps[0].batched_time.round_polys[control_control_idx]
        .get_mut(0)
        .and_then(|round| round.get_mut(0))
        .expect("control/next_pc_control first-round coeff must exist");
    *coeff += K::ONE;
    assert!(
        run.verify_proof(&proof_tampered_control_round).is_err(),
        "tampered control/next_pc_control round polynomial must fail verification"
    );
}

#[test]
fn rv32_trace_wiring_runner_rejects_zero_chunk_rows() {
    let program = vec![RiscvInstruction::Halt];
    let program_bytes = encode_program(&program);

    let err = match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(0)
        .prove()
    {
        Ok(_) => panic!("chunk_rows=0 must be rejected"),
        Err(e) => e,
    };

    let msg = err.to_string();
    assert!(msg.contains("chunk_rows"), "unexpected error message: {msg}");
}

#[test]
fn rv32_trace_wiring_runner_rejects_amo_via_wb_decode_scope_lock() {
    // Program includes one AMO row. In Tier 2.1 trace mode this is rejected by WB/decode stage
    // decode residuals (scope lock), not by the N0 main-trace CCS.
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoaddW,
            rd: 2,
            rs1: 0,
            rs2: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    assert!(
        Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
            .prove()
            .is_err(),
        "AMO must be rejected in Tier 2.1 trace mode via WB/decode stage scope lock"
    );
}

fn prove_verify_trace_program(program: Vec<RiscvInstruction>) {
    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .min_trace_len(program.len())
        .max_steps(program.len())
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");
}

#[test]
fn rv32_trace_wiring_runner_accepts_mixed_addi_andi_halt() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::And,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    prove_verify_trace_program(program);
}

#[test]
fn rv32_trace_wiring_runner_accepts_mixed_addi_ori_halt() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Or,
            rd: 3,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    prove_verify_trace_program(program);
}

#[test]
fn rv32_trace_wiring_runner_accepts_mixed_with_srai_halt() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sra,
            rd: 2,
            rs1: 1,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    prove_verify_trace_program(program);
}

#[test]
fn rv32_trace_wiring_runner_accepts_full_mixed_sequence_halt() {
    let mut program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::And,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Or,
            rd: 3,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Xor,
            rd: 4,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Slt,
            rd: 6,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sltu,
            rd: 7,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sll,
            rd: 8,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Srl,
            rd: 9,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sra,
            rd: 10,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 0,
            rs2: 0,
            imm: 8,
        },
    ];
    program.push(RiscvInstruction::Halt);
    prove_verify_trace_program(program.clone());
}
