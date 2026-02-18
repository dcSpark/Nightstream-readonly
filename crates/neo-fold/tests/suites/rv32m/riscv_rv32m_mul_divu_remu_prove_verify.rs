use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[test]
fn rv32_trace_prove_verify_add_sub_sequence() {
    let program = vec![
        // x1 = 7
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        // x2 = 13
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 13,
        },
        // x3 = x1 + x2 = 20
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        // x4 = x2 - x1 = 6
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sub,
            rd: 4,
            rs1: 2,
            rs2: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(program.len())
        .min_trace_len(program.len())
        .max_steps(program.len())
        .reg_output_claim(/*reg=*/ 3, /*expected=*/ F::from_u64(20))
        .reg_output_claim(/*reg=*/ 4, /*expected=*/ F::from_u64(6))
        .prove()
        .expect("prove");
    run.verify().expect("verify");
}

#[test]
fn rv32_trace_prove_verify_sltu_and_zero_flag_path() {
    let program = vec![
        // x1 = 5
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 5,
        },
        // x2 = 5
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 5,
        },
        // x3 = (x1 < x2) ? 1 : 0   => 0
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sltu,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        // x4 = x3 + 1 => 1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 3,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(program.len())
        .min_trace_len(program.len())
        .max_steps(program.len())
        .reg_output_claim(/*reg=*/ 3, /*expected=*/ F::from_u64(0))
        .reg_output_claim(/*reg=*/ 4, /*expected=*/ F::from_u64(1))
        .prove()
        .expect("prove");
    run.verify().expect("verify");
}

#[test]
fn rv32_trace_prove_verify_signed_compare_path() {
    let program = vec![
        // x1 = -7
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: -7,
        },
        // x2 = 3
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        },
        // x3 = (x1 < x2) signed => 1
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Slt,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(program.len())
        .min_trace_len(program.len())
        .max_steps(program.len())
        .reg_output_claim(/*reg=*/ 3, /*expected=*/ F::from_u64(1))
        .prove()
        .expect("prove");
    run.verify().expect("verify");
}
