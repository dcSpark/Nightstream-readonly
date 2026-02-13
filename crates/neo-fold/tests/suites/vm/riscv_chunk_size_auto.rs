#![allow(non_snake_case)]

use neo_fold::riscv_shard::Rv32B1;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};

#[test]
fn rv32_b1_chunk_size_auto_prove_verify() {
    // Small halting program (length > 8 so the tuner has multiple candidates).
    let program: Vec<RiscvInstruction> = (0..9)
        .map(|i| RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 1,
            imm: (i + 1) as i32,
        })
        .chain(std::iter::once(RiscvInstruction::Halt))
        .collect();

    let program_bytes = encode_program(&program);

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size_auto()
        .ram_bytes(4)
        .max_steps(program.len())
        .prove()
        .expect("prove");

    run.verify().expect("verify");
    assert!(run.chunk_size() > 0);
    assert!(run.chunk_size() <= 256);
    assert!(run.fold_count() > 0);
}
