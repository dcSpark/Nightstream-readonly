use neo_fold::riscv_shard::Rv32B1;
use neo_fold::PiCcsError;
use neo_math::F;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;

fn program_bytes_with_seed(seed: i32) -> Vec<u8> {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: seed,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 1,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    encode_program(&program)
}

fn assert_m_in_mismatch_rejected(err: PiCcsError) {
    match err {
        PiCcsError::InvalidInput(msg) => {
            assert!(
                msg.contains("all steps must share the same m_in"),
                "unexpected invalid-input error: {msg}"
            );
        }
        other => panic!("unexpected error kind: {other:?}"),
    }
}

#[test]
fn rv32_b1_build_verifier_binds_statement_memory_to_chunk0_mem_init() {
    let program_a = program_bytes_with_seed(7);
    let program_b = program_bytes_with_seed(9);

    let mut run_a = Rv32B1::from_rom(/*program_base=*/ 0, &program_a)
        .chunk_size(1)
        .prove()
        .expect("prove statement A");
    run_a.verify().expect("self-verify statement A");

    let proof_a = run_a.proof().clone();
    let steps_public_a = run_a.steps_public();

    let verifier_a = Rv32B1::from_rom(/*program_base=*/ 0, &program_a)
        .chunk_size(1)
        .build_verifier()
        .expect("build verifier for statement A");
    let ok = verifier_a
        .verify(&proof_a, &steps_public_a)
        .expect("matching statement should verify");
    assert!(ok, "matching statement should verify");

    let verifier_b = Rv32B1::from_rom(/*program_base=*/ 0, &program_b)
        .chunk_size(1)
        .build_verifier()
        .expect("build verifier for statement B");
    let err = verifier_b
        .verify(&proof_a, &steps_public_a)
        .expect_err("different statement memory must be rejected");

    match err {
        PiCcsError::InvalidInput(msg) => {
            assert!(
                msg.contains("chunk0 MemInstance.init mismatch"),
                "unexpected invalid-input error: {msg}"
            );
        }
        other => panic!("unexpected error kind: {other:?}"),
    }
}

#[test]
fn rv32_b1_build_verifier_rejects_external_steps_with_non_uniform_m_in() {
    let program = program_bytes_with_seed(11);

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program)
        .chunk_size(1)
        .prove()
        .expect("prove statement");
    run.verify().expect("self-verify statement");

    let proof = run.proof().clone();
    let mut steps_public = run.steps_public();
    assert!(
        steps_public.len() > 1,
        "test needs at least two steps to create m_in mismatch"
    );
    steps_public[1].mcs_inst.m_in += 1;

    let verifier = Rv32B1::from_rom(/*program_base=*/ 0, &program)
        .chunk_size(1)
        .build_verifier()
        .expect("build verifier");
    let err = verifier
        .verify(&proof, &steps_public)
        .expect_err("non-uniform m_in must be rejected");
    assert_m_in_mismatch_rejected(err);
}

#[test]
fn rv32_b1_build_verifier_output_binding_rejects_external_steps_with_non_uniform_m_in() {
    let program = program_bytes_with_seed(13);

    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program)
        .chunk_size(1)
        .output(/*output_addr=*/ 0, /*expected_output=*/ F::ZERO)
        .prove()
        .expect("prove statement with output binding");
    run.verify()
        .expect("self-verify statement with output binding");

    let proof = run.proof().clone();
    let mut steps_public = run.steps_public();
    assert!(
        steps_public.len() > 1,
        "test needs at least two steps to create m_in mismatch"
    );
    steps_public[1].mcs_inst.m_in += 1;

    let verifier = Rv32B1::from_rom(/*program_base=*/ 0, &program)
        .chunk_size(1)
        .output(/*output_addr=*/ 0, /*expected_output=*/ F::ZERO)
        .build_verifier()
        .expect("build verifier with output binding");
    let err = verifier
        .verify(&proof, &steps_public)
        .expect_err("non-uniform m_in must be rejected");
    assert_m_in_mismatch_rejected(err);
}
