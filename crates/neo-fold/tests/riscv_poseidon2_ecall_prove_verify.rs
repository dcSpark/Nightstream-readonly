use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::lookups::{
    encode_program, RiscvInstruction, RiscvOpcode, POSEIDON2_ECALL_NUM, POSEIDON2_READ_ECALL_NUM,
};

fn load_u32_imm(rd: u8, value: u32) -> Vec<RiscvInstruction> {
    let upper = ((value as i64 + 0x800) >> 12) as i32;
    let lower = (value as i32) - (upper << 12);
    vec![
        RiscvInstruction::Lui { rd, imm: upper },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd,
            rs1: rd,
            imm: lower,
        },
    ]
}

fn poseidon2_ecall_program() -> Vec<RiscvInstruction> {
    let mut program = Vec::new();

    // a1 = 0 (n_elements = 0 for empty-input Poseidon2 hash).
    program.push(RiscvInstruction::IAlu {
        op: RiscvOpcode::Add,
        rd: 11,
        rs1: 0,
        imm: 0,
    });

    // a0 = POSEIDON2_ECALL_NUM -> compute ECALL.
    program.extend(load_u32_imm(10, POSEIDON2_ECALL_NUM));
    program.push(RiscvInstruction::Halt);

    // Read all 8 digest words via read ECALLs.
    for _ in 0..8 {
        program.extend(load_u32_imm(10, POSEIDON2_READ_ECALL_NUM));
        program.push(RiscvInstruction::Halt);
    }

    // Clear a0 -> final halt.
    program.push(RiscvInstruction::IAlu {
        op: RiscvOpcode::Add,
        rd: 10,
        rs1: 0,
        imm: 0,
    });
    program.push(RiscvInstruction::Halt);

    program
}

#[test]
fn rv32_trace_prove_verify_poseidon2_ecall_chunk1() {
    let program = poseidon2_ecall_program();
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .chunk_rows(1)
        .max_steps(program.len() + 64)
        .prove()
        .expect("prove should succeed");

    run.verify().expect("verify should succeed for Poseidon2 ECALL with chunk_size=1");
}

#[test]
fn rv32_trace_prove_verify_poseidon2_ecall_chunk4() {
    let program = poseidon2_ecall_program();
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .chunk_rows(4)
        .max_steps(program.len() + 64)
        .prove()
        .expect("prove should succeed");

    run.verify().expect("verify should succeed for Poseidon2 ECALL with chunk_size=4");
}

#[test]
fn rv32_trace_prove_verify_poseidon2_ecall_chunk32() {
    let program = poseidon2_ecall_program();
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .chunk_rows(32)
        .max_steps(program.len() + 64)
        .prove()
        .expect("prove should succeed");

    run.verify().expect("verify should succeed for Poseidon2 ECALL with chunk_size=32");
}
