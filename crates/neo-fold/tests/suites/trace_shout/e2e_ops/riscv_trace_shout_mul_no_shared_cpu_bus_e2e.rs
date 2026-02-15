#![allow(non_snake_case)]

use neo_fold::riscv_shard::Rv32B1;
use neo_math::F;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;

#[test]
fn riscv_trace_wiring_ccs_no_shared_cpu_bus_shout_mul_prove_verify() {
    // Program:
    // - LUI x1, 16        (x1 = 65536)
    // - LUI x2, 16        (x2 = 65536)
    // - MUL x3, x1, x2    (lo = 0)
    // - MUL x4, x2, x1    (lo = 0)
    // - HALT
    let program = vec![
        RiscvInstruction::Lui { rd: 1, imm: 16 },
        RiscvInstruction::Lui { rd: 2, imm: 16 },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 4,
            rs1: 2,
            rs2: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .max_steps(program.len())
        .reg_output_claim(/*reg=*/ 3, F::from_u64(0))
        .reg_output_claim(/*reg=*/ 4, F::from_u64(0))
        .prove()
        .expect("rv32_b1 prove (WB/WP route, MUL)");
    run.verify().expect("rv32_b1 verify (WB/WP route, MUL)");
}
