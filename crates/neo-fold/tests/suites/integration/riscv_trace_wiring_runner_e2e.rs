use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::lookups::{
    encode_program, BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode, RiscvShoutTables, PROG_ID, RAM_ID,
    REG_ID,
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
    assert_eq!(
        run.used_shout_table_ids(),
        [add_table_id].as_slice(),
        "run artifact should record auto-derived S_lookup"
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

    let run_legacy = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .shared_cpu_bus(false)
        .min_trace_len(1)
        .prove()
        .expect("trace wiring prove (legacy no-shared fallback)");

    assert!(
        run_shared.ccs_num_variables() > run_legacy.ccs_num_variables(),
        "shared-bus trace path must reserve bus-tail columns in the main CCS"
    );
    assert_eq!(
        run_shared.ccs_num_variables(),
        run_shared.layout().m,
        "shared-bus trace layout width must match CCS width"
    );
    assert_eq!(
        run_legacy.ccs_num_variables(),
        run_legacy.layout().m,
        "legacy trace layout width must match CCS width"
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
        .shared_cpu_bus(false)
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
        .shared_cpu_bus(false)
        .chunk_rows(2)
        .prove()
        .expect("trace wiring prove with chunked ivc");
    run.verify().expect("trace wiring verify with chunked ivc");

    let steps_public = run.steps_public();
    let shard_proof = run.proof();
    assert_eq!(steps_public.len(), 2, "expected two public steps");
    assert_eq!(shard_proof.steps.len(), 2, "expected two proof steps");

    // Step 0: no previous step, so there is one val claim per mem instance.
    let mem_count_step0 = steps_public[0].mem_insts.len();
    let proof_step0 = &shard_proof.steps[0];
    assert_eq!(
        proof_step0.mem.val_me_claims.len(),
        mem_count_step0,
        "step0 must emit one current val claim per mem instance"
    );
    assert_eq!(
        proof_step0.val_fold.len(),
        mem_count_step0,
        "step0 must emit one val-fold proof per mem instance"
    );

    // Step 1: has previous step, so val claims are [current..., previous...], but
    // proof lanes are batched per mem instance.
    let mem_count_step1 = steps_public[1].mem_insts.len();
    let proof_step1 = &shard_proof.steps[1];
    assert_eq!(
        proof_step1.mem.val_me_claims.len(),
        mem_count_step1 * 2,
        "step1 must emit current+previous val claims per mem instance"
    );
    assert_eq!(
        proof_step1.val_fold.len(),
        mem_count_step1,
        "step1 must batch val-fold proofs per mem instance"
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
fn rv32_trace_wiring_runner_rejects_amo_via_wb_w2_scope_lock() {
    // Program includes one AMO row. In Tier 2.1 trace mode this is rejected by WB/W2
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
        "AMO must be rejected in Tier 2.1 trace mode via WB/W2 scope lock"
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

fn prove_verify_trace_program_legacy_no_shared(program: Vec<RiscvInstruction>) {
    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .shared_cpu_bus(false)
        .min_trace_len(program.len())
        .max_steps(program.len())
        .prove()
        .expect("trace wiring prove (legacy no-shared)");
    run.verify()
        .expect("trace wiring verify (legacy no-shared)");
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
    prove_verify_trace_program_legacy_no_shared(program);
}
