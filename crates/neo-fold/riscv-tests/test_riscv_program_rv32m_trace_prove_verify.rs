//! End-to-end prove+verify for RV32M (M-extension) operations under the trace-mode runner.
//!
//! This validates that M-extension ops (MUL, DIV, etc.) are correctly handled
//! via Shout lookups in trace mode (table IDs 12-19).

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};

#[test]
#[ignore = "M-ext Shout tables need closed-form MLE or a packed-key proof path for xlen=32"]
fn test_riscv_program_rv32m_trace_prove_verify() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: -6,
        }, // x1 = -6 (0xFFFFFFFA)
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        }, // x2 = 3
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = (-6)*3 = -18
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 4,
            rs1: 1,
            rs2: 2,
        }, // x4 = (-6)/3 = -2
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(32)
        .min_trace_len(1)
        .prove()
        .expect("trace-mode prove with RV32M ops");

    run.verify().expect("trace-mode verify with RV32M ops");
}

#[test]
#[ignore = "M-ext Shout tables need closed-form MLE or a packed-key proof path for xlen=32"]
fn test_riscv_program_rv32m_all_ops_trace_prove_verify() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
        }, // x1 = 7
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        }, // x2 = 3
        // MUL: x3 = 7*3 = 21
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        // MULH: x4 = high bits of signed(7)*signed(3) = 0
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        // MULHSU: x5 = high bits of signed(7)*unsigned(3) = 0
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhsu,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        // MULHU: x6 = high bits of unsigned(7)*unsigned(3) = 0
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhu,
            rd: 6,
            rs1: 1,
            rs2: 2,
        },
        // DIV: x7 = 7/3 = 2
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 7,
            rs1: 1,
            rs2: 2,
        },
        // DIVU: x8 = 7/3 = 2 (unsigned)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 8,
            rs1: 1,
            rs2: 2,
        },
        // REM: x9 = 7%3 = 1
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 9,
            rs1: 1,
            rs2: 2,
        },
        // REMU: x10 = 7%3 = 1 (unsigned)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 10,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(32)
        .min_trace_len(1)
        .prove()
        .expect("trace-mode prove with all RV32M ops");

    run.verify()
        .expect("trace-mode verify with all RV32M ops");
}

#[test]
#[ignore = "M-ext Shout tables need closed-form MLE or a packed-key proof path for xlen=32"]
fn test_riscv_program_rv32m_signed_edge_cases_trace_prove_verify() {
    let program = vec![
        // x1 = -1 (0xFFFFFFFF)
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: -1,
        },
        // x2 = -1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: -1,
        },
        // MULH(-1, -1): high bits of 1 = 0
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        // MULHSU(-1, 0xFFFFFFFF): high bits of signed(-1)*unsigned(0xFFFFFFFF) = -1
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhsu,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        // DIV(-1, -1) = 1
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        // x6 = 0 (divisor = 0 case)
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 6,
            rs1: 0,
            imm: 0,
        },
        // DIV(-1, 0) = -1 (RISC-V spec: division by zero returns -1)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 7,
            rs1: 1,
            rs2: 6,
        },
        // DIVU(-1, 0) = 0xFFFFFFFF (RISC-V spec)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 8,
            rs1: 1,
            rs2: 6,
        },
        // REM(-1, 0) = -1 (RISC-V spec: remainder with divisor 0 = dividend)
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 9,
            rs1: 1,
            rs2: 6,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(32)
        .min_trace_len(1)
        .prove()
        .expect("trace-mode prove with signed M-ext edge cases");

    run.verify()
        .expect("trace-mode verify with signed M-ext edge cases");
}
