use neo_fold::pi_ccs::FoldingMode;
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

#[test]
fn rv32_trace_prove_verify_mulhu_div_rem_paths() {
    let program = vec![
        // x1 = 65536
        RiscvInstruction::Lui { rd: 1, imm: 16 },
        // x2 = 65536
        RiscvInstruction::Lui { rd: 2, imm: 16 },
        // x3 = MULHU(x1, x2) = 1
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhu,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        // x4 = -7
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 0,
            imm: -7,
        },
        // x5 = 3
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 5,
            rs1: 0,
            imm: 3,
        },
        // x6 = DIV(x4, x5) = -2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 6,
            rs1: 4,
            rs2: 5,
        },
        // x7 = REM(x4, x5) = -1
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 7,
            rs1: 4,
            rs2: 5,
        },
        // x8 = 13
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 8,
            rs1: 0,
            imm: 13,
        },
        // x9 = 0
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 9,
            rs1: 0,
            imm: 0,
        },
        // x10 = DIV(x8, x9) => all ones
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 10,
            rs1: 8,
            rs2: 9,
        },
        // x11 = REM(x8, x9) => dividend
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 11,
            rs1: 8,
            rs2: 9,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .mode(FoldingMode::Optimized)
        .chunk_rows(program.len())
        .min_trace_len(program.len())
        .max_steps(program.len())
        .reg_output_claim(/*reg=*/ 3, /*expected=*/ F::from_u64(1))
        .reg_output_claim(/*reg=*/ 6, /*expected=*/ F::from_u64(0xFFFF_FFFEu64))
        .reg_output_claim(/*reg=*/ 7, /*expected=*/ F::from_u64(0xFFFF_FFFFu64))
        .reg_output_claim(/*reg=*/ 10, /*expected=*/ F::from_u64(0xFFFF_FFFFu64))
        .reg_output_claim(/*reg=*/ 11, /*expected=*/ F::from_u64(13))
        .prove()
        .expect("prove");
    run.verify().expect("verify");
}

#[test]
fn rv32_trace_prove_verify_mulhu_only() {
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
        .chunk_rows(program.len())
        .min_trace_len(program.len())
        .max_steps(program.len())
        .reg_output_claim(/*reg=*/ 3, /*expected=*/ F::from_u64(1))
        .prove()
        .expect("prove");
    run.verify().expect("verify");
}

#[test]
fn rv32_trace_prove_verify_div_only() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: -7,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .mode(FoldingMode::Optimized)
        .chunk_rows(program.len())
        .min_trace_len(program.len())
        .max_steps(program.len())
        .reg_output_claim(/*reg=*/ 3, /*expected=*/ F::from_u64(0xFFFF_FFFEu64))
        .prove()
        .expect("prove");
    run.verify().expect("verify");
}

#[test]
fn rv32_trace_prove_verify_rem_only() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: -7,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .mode(FoldingMode::Optimized)
        .chunk_rows(program.len())
        .min_trace_len(program.len())
        .max_steps(program.len())
        .reg_output_claim(/*reg=*/ 3, /*expected=*/ F::from_u64(0xFFFF_FFFFu64))
        .prove()
        .expect("prove");
    run.verify().expect("verify");
}

#[test]
fn rv32_trace_prove_verify_mulh_divu_paths() {
    let program = vec![
        // x1 = -2
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: -2,
        },
        // x2 = 3
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        },
        // x3 = MULH(x1, x2) = high32(signed(-2) * signed(3)) = 0xffffffff
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        // x4 = 13
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 4,
            rs1: 0,
            imm: 13,
        },
        // x5 = 5
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 5,
            rs1: 0,
            imm: 5,
        },
        // x6 = DIVU(x4, x5) = 2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 6,
            rs1: 4,
            rs2: 5,
        },
        // x7 = 0
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 7,
            rs1: 0,
            imm: 0,
        },
        // x8 = DIVU(x4, x7) = 0xffffffff (RISC-V div-by-zero behavior)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 8,
            rs1: 4,
            rs2: 7,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .mode(FoldingMode::Optimized)
        .chunk_rows(program.len())
        .min_trace_len(program.len())
        .max_steps(program.len())
        .reg_output_claim(/*reg=*/ 3, /*expected=*/ F::from_u64(0xFFFF_FFFFu64))
        .reg_output_claim(/*reg=*/ 6, /*expected=*/ F::from_u64(2))
        .reg_output_claim(/*reg=*/ 8, /*expected=*/ F::from_u64(0xFFFF_FFFFu64))
        .prove()
        .expect("prove");
    run.verify().expect("verify");
}
