use neo_fold::riscv_shard::Rv32B1;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};

#[test]
fn rv32m_sidecar_is_skipped_for_non_m_programs() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .ram_bytes(4)
        .max_steps(2)
        .prove()
        .expect("prove");
    run.verify().expect("verify");

    assert!(
        run.proof().rv32m.is_none(),
        "expected no RV32M sidecar proof for a non-M program"
    );
}

#[test]
fn rv32m_sidecar_is_sparse_over_time() {
    // Program: MULH once, then HALT.
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 1,
            rs1: 0,
            rs2: 0,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .ram_bytes(4)
        .max_steps(2)
        .prove()
        .expect("prove");
    run.verify().expect("verify");

    let rv32m = run
        .proof()
        .rv32m
        .as_ref()
        .expect("rv32m sidecar proof present");
    assert_eq!(
        rv32m.len(),
        1,
        "expected RV32M sidecar to be proven only for the single MULH step (one chunk)"
    );
    assert_eq!(rv32m[0].chunk_idx, 0, "expected RV32M proof for chunk 0");
    assert_eq!(rv32m[0].lanes, vec![0], "expected RV32M lane 0 only");
}
