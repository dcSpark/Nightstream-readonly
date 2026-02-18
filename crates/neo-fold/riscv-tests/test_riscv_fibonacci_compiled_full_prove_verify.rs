//! End-to-end prove+verify for a small Fibonacci-style RV32 program under trace wiring.

#![allow(non_snake_case)]

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_math::F;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_riscv_fibonacci_compiled_full_prove_verify() {
    // Straight-line "fib-style" program:
    // - x1 = 34
    // - x2 = 21
    // - x3 = x1 + x2 = 55
    // - mem[0x100] = x3
    // - HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 34,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 21,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 3,
            imm: 0x100,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let expected = F::from_u64(55);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(32)
        .chunk_rows(program.len())
        .min_trace_len(program.len())
        .max_steps(program.len())
        .shout_auto_minimal()
        .output(/*output_addr=*/ 0x100, /*expected_output=*/ expected)
        .prove()
        .expect("prove");

    run.verify().expect("verify");

    match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(32)
        .chunk_rows(program.len())
        .min_trace_len(program.len())
        .max_steps(program.len())
        .shout_auto_minimal()
        .output(/*output_addr=*/ 0x100, /*expected_output=*/ F::from_u64(56))
        .prove()
    {
        Ok(mut bad_run) => assert!(
            bad_run.verify().is_err(),
            "wrong output claim must fail verification"
        ),
        Err(_) => {}
    }
}
