#![allow(non_snake_case)]
#![cfg(feature = "paper-exact")]

//! Quick RV32 shared-bus integration test using crosscheck folding mode.
//!
//! This intentionally runs only a tiny trace (3 instructions, chunk_size=1) to keep the
//! paper-exact crosschecks from dominating CI time.

use neo_fold::pi_ccs::FoldingMode;
use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_math::F;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use neo_reductions::engines::CrosscheckCfg;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_riscv_program_crosscheck_tiny_trace() {
    let program = vec![
        // x1 = 7
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        // x1 = x1 + 5 = 12
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 1,
            imm: 5,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let crosscheck_cfg = CrosscheckCfg {
        fail_fast: true,
        initial_sum: false,
        per_round: false,
        terminal: false,
        outputs: true,
    };

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(32)
        .chunk_rows(1)
        .min_trace_len(3)
        .max_steps(3)
        .mode(FoldingMode::OptimizedWithCrosscheck(crosscheck_cfg))
        .reg_output_claim(/*reg=*/ 1, /*expected=*/ F::from_u64(12))
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
#[ignore = "paper-exact crosschecks are exponential; run manually when debugging"]
fn test_riscv_program_crosscheck_full_flags_one_step() {
    let program = vec![
        // x1 = 7
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        // x1 = x1 + 5 = 12
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 1,
            imm: 5,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let crosscheck_cfg = CrosscheckCfg {
        fail_fast: true,
        initial_sum: true,
        per_round: true,
        terminal: true,
        outputs: true,
    };

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(32)
        .chunk_rows(1)
        .min_trace_len(1)
        .max_steps(1)
        .mode(FoldingMode::OptimizedWithCrosscheck(crosscheck_cfg))
        .reg_output_claim(/*reg=*/ 1, /*expected=*/ F::from_u64(7))
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}

#[test]
#[ignore = "paper-exact crosschecks are exponential; run manually when debugging. Takes around 30 min to run."]
fn test_riscv_program_crosscheck_full_flags_two_steps() {
    let program = vec![
        // x1 = 7
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        // x1 = x1 + 5 = 12
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 1,
            imm: 5,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let crosscheck_cfg = CrosscheckCfg {
        fail_fast: true,
        initial_sum: true,
        per_round: true,
        terminal: true,
        outputs: true,
    };

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(32)
        .chunk_rows(1)
        .min_trace_len(2)
        .max_steps(2)
        .mode(FoldingMode::OptimizedWithCrosscheck(crosscheck_cfg))
        .reg_output_claim(/*reg=*/ 1, /*expected=*/ F::from_u64(12))
        .prove()
        .expect("prove");

    run.verify().expect("verify");
}
