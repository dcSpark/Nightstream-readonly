use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode, PROG_ID, RAM_ID, REG_ID};
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
    expected_mem_ids.sort_unstable();
    assert_eq!(
        mem_ids, expected_mem_ids,
        "trace runner should include PROG/REG/RAM sidecar instances even without output binding"
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
fn rv32_trace_wiring_runner_main_ccs_has_no_bus_tail() {
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

    let run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .min_trace_len(1)
        .prove()
        .expect("trace wiring prove");

    assert_eq!(
        run.ccs_num_variables(),
        run.layout().m,
        "main trace CCS still appears to include extra width (bus tail)"
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
