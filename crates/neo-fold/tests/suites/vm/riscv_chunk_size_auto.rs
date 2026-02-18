#![allow(non_snake_case)]

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};

#[test]
fn rv32_trace_chunk_rows_auto_prove_verify() {
    // Small halting program.
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

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .max_steps(program.len())
        .prove()
        .expect("prove");

    run.verify().expect("verify");
    assert!(run.trace_len() > 0);
    assert!(run.fold_count() > 0);
}
