use neo_fold::pi_ccs::FoldingMode;
use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};

#[test]
fn shared_cpu_bus_mul_large_operands_regression_proves_with_interleaved_keying() {
    // Initialize two large RV32 operands in the register file:
    //   x1 = 0x8000_0000
    //   x2 = 0x8000_0000
    // Then execute MUL in row 0. If MUL keying regresses back to a full
    // 64-bit product key (`0x4000_0000_0000_0000`), proving will fail at
    // witness encoding for b=2, D=54. This test protects against that.
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .mode(FoldingMode::Optimized)
        .reg_init_u32(1, 0x8000_0000)
        .reg_init_u32(2, 0x8000_0000)
        .chunk_rows(program.len())
        .min_trace_len(program.len())
        .max_steps(program.len())
        .prove()
        .expect("regression: MUL proving should succeed with interleaved keying");
    run.verify()
        .expect("regression: verification should succeed after proving");
}
