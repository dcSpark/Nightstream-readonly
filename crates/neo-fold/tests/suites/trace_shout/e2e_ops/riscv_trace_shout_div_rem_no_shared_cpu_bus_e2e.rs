#![allow(non_snake_case)]

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_math::F;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;

#[test]
fn riscv_trace_wiring_ccs_no_shared_cpu_bus_shout_div_rem_packed_prove_verify() {
    // Program:
    // - x1 = -7*4096, x2 = 3*4096  (DIV=-2, REM=-4096)
    // - x1 = -1*4096, x2 = 3*4096  (DIV=0,  REM=-4096)
    // - x1 = INT_MIN, x2 = -1      (DIV overflow case; REM=0)
    // - x1 = INT_MIN, x2 = 0       (DIV by zero => -1; REM by zero => lhs)
    // - HALT
    let program = vec![
        RiscvInstruction::Lui { rd: 1, imm: -7 },
        RiscvInstruction::Lui { rd: 2, imm: 3 },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 4,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Lui { rd: 1, imm: -1 },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 6,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Lui { rd: 1, imm: -524_288 },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: -1,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 7,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 8,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Div,
            rd: 9,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Rem,
            rd: 10,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(1)
        .max_steps(program.len())
        .reg_output_claim(/*reg=*/ 9, F::from_u64(0xffff_ffff))
        .reg_output_claim(/*reg=*/ 10, F::from_u64(0x8000_0000))
        .prove()
        .expect("rv32 trace prove (WB/WP route, DIV/REM)");
    run.verify().expect("rv32 trace verify (WB/WP route, DIV/REM)");
}
