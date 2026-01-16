use neo_fold::riscv_shard::Rv32B1;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[test]
fn rv32_b1_prove_verify_mul_divu_remu() {
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
        // x3 = x1 * x2 = 91
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        // x4 = x3 / x1 = 13
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 4,
            rs1: 3,
            rs2: 1,
        },
        // x5 = x3 % x1 = 0
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 5,
            rs1: 3,
            rs2: 1,
        },
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .max_steps(program.len())
        .prove()
        .expect("prove");
    run.verify().expect("verify");

    let boundary = run.final_boundary_state().expect("final boundary state");
    assert_eq!(boundary.regs_final[3], F::from_u64(91));
    assert_eq!(boundary.regs_final[4], F::from_u64(13));
    assert_eq!(boundary.regs_final[5], F::from_u64(0));
}
