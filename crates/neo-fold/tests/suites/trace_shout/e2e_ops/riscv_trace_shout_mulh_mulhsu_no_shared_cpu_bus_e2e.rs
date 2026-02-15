#![allow(non_snake_case)]

use neo_fold::riscv_shard::Rv32B1;
use neo_math::F;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;

#[test]
fn riscv_trace_wiring_ccs_no_shared_cpu_bus_shout_mulh_mulhsu_packed_prove_verify() {
    // Program:
    // - x1 = -7
    // - x2 = -3
    // - MULH   x3, x1, x2      (0)
    // - x5 = 13
    // - MULHSU x6, x1, x5      (0xffffffff)
    // - HALT
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
            imm: -3,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 5,
            rs1: 0,
            imm: 13,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulhsu,
            rd: 6,
            rs1: 1,
            rs2: 5,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .max_steps(program.len())
        .reg_output_claim(/*reg=*/ 3, F::from_u64(0))
        .reg_output_claim(/*reg=*/ 6, F::from_u64(0xffff_ffff))
        .prove()
        .expect("rv32_b1 prove (WB/WP route, MULH/MULHSU)");
    run.verify()
        .expect("rv32_b1 verify (WB/WP route, MULH/MULHSU)");
}
