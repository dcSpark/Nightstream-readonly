#![allow(non_snake_case)]

use neo_fold::riscv_shard::Rv32B1;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;

fn trace_mode_program_bytes() -> Vec<u8> {
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
    encode_program(&program)
}

#[test]
fn rv32_b1_trace_wiring_mode_prove_verify() {
    let program_bytes = trace_mode_program_bytes();

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(8) // ignored by trace-wiring mode
        .ram_bytes(0x100) // ignored by trace-wiring mode
        .reg_init_u32(/*reg=*/ 3, /*value=*/ 9)
        .ram_init_u32(/*addr=*/ 16, /*value=*/ 7)
        .trace_min_len(8)
        .prove_trace_wiring()
        .expect("trace wiring prove via Rv32B1");

    run.verify().expect("trace wiring verify");

    assert_eq!(run.fold_count(), 1, "trace-wiring mode should produce one folding step");
    assert_eq!(run.trace_len(), 3, "active trace length mismatch");
    assert_eq!(
        run.exec_table().rows.len(),
        8,
        "trace_min_len should set padded trace length"
    );
    assert_eq!(run.layout().t, 8, "layout t should match padded trace length");
}

#[test]
fn rv32_b1_trace_wiring_mode_does_not_force_pow2_padding() {
    let program_bytes = trace_mode_program_bytes();

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .trace_min_len(1)
        .prove_trace_wiring()
        .expect("trace wiring prove via Rv32B1");

    run.verify().expect("trace wiring verify");

    assert_eq!(run.trace_len(), 3, "active trace length mismatch");
    assert_eq!(
        run.exec_table().rows.len(),
        3,
        "trace-wiring mode should keep unpadded trace length when min bound is smaller"
    );
    assert_eq!(run.layout().t, 3, "layout t should match unpadded trace length");
}

#[test]
fn rv32_b1_trace_wiring_mode_ram_output_binding_prove_verify() {
    // Program: ADDI x1, x0, 7; SW x1, 16(x0); HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::Store {
            op: neo_memory::riscv::lookups::RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 16,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .output_claim(/*addr=*/ 16, /*value=*/ neo_math::F::from_u64(7))
        .prove_trace_wiring()
        .expect("trace wiring prove with RAM output binding");

    run.verify()
        .expect("trace wiring verify with RAM output binding");
}

#[test]
fn rv32_b1_trace_wiring_mode_reg_output_binding_prove_verify() {
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

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .reg_output_claim(/*reg=*/ 2, /*value=*/ neo_math::F::from_u64(3))
        .prove_trace_wiring()
        .expect("trace wiring prove with REG output binding");

    run.verify()
        .expect("trace wiring verify with REG output binding");
}

#[test]
fn rv32_b1_trace_wiring_mode_wrong_reg_output_claim_fails_verify() {
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

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .reg_output_claim(/*reg=*/ 2, /*value=*/ neo_math::F::from_u64(4))
        .prove_trace_wiring()
        .expect("trace wiring prove with wrong REG claim still produces a proof");

    let err = run
        .verify()
        .expect_err("trace wiring verify should fail for wrong REG output claim");
    let msg = format!("{err}");
    assert!(msg.contains("output sumcheck failed"), "unexpected verify error: {msg}");
}

#[test]
fn rv32_b1_trace_wiring_mode_allows_without_insecure_ack() {
    let program_bytes = trace_mode_program_bytes();

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .prove_trace_wiring()
        .expect("trace-wiring mode should not require insecure benchmark-only ack");
    run.verify()
        .expect("trace-wiring proof should verify without insecure benchmark-only ack");
}
