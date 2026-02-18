use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;

#[test]
fn rv32_trace_claims_are_bound_to_main_commitment() {
    // Program: ADDI x1, x0, 1; HALT
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
    let steps = 2usize;

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(steps)
        .min_trace_len(steps)
        .max_steps(steps)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");

    let mut bad_proof = run.proof().clone();
    let mut tampered = false;
    for step in &mut bad_proof.steps {
        for claim in &mut step.mem.val_me_claims {
            if let Some(first) = claim.y_scalars.first_mut() {
                *first += K::ONE;
                tampered = true;
                break;
            }
        }
        if tampered {
            break;
        }
    }
    assert!(tampered, "expected at least one claim scalar to tamper");
    assert!(
        run.verify_proof(&bad_proof).is_err(),
        "tampered trace claims must not verify"
    );
}
