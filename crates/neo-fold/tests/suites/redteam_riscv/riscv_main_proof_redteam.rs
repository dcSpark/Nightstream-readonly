use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::shard::ShardProof;
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;

fn addi_halt_program_bytes(imm: i32) -> Vec<u8> {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm,
        },
        RiscvInstruction::Halt,
    ];
    encode_program(&program)
}

fn prove_run(program_bytes: &[u8], max_steps: usize) -> Rv32TraceWiringRun {
    let steps = max_steps;
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, program_bytes)
        .chunk_rows(steps)
        .min_trace_len(steps)
        .max_steps(steps)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");
    run
}

fn tamper_any_claim_scalar(proof: &mut ShardProof) {
    for step in &mut proof.steps {
        for claims in [
            &mut step.fold.ccs_out,
            &mut step.mem.val_me_claims,
            &mut step.mem.wb_me_claims,
            &mut step.mem.wp_me_claims,
        ] {
            for claim in claims.iter_mut() {
                if let Some(first) = claim.y_scalars.first_mut() {
                    *first += K::ONE;
                    return;
                }
            }
        }
    }
    panic!("expected at least one claim scalar to tamper");
}

#[test]
fn rv32_trace_main_proof_truncated_steps_must_fail() {
    let program_bytes = addi_halt_program_bytes(/*imm=*/ 1);
    let run = prove_run(&program_bytes, /*max_steps=*/ 2);

    let mut bad_proof = run.proof().clone();
    bad_proof.steps.clear();
    assert!(
        run.verify_proof(&bad_proof).is_err(),
        "truncated main proof must not verify"
    );
}

#[test]
fn rv32_trace_main_proof_tamper_claim_must_fail() {
    let program_bytes = addi_halt_program_bytes(/*imm=*/ 1);
    let run = prove_run(&program_bytes, /*max_steps=*/ 2);

    let mut bad_proof = run.proof().clone();
    tamper_any_claim_scalar(&mut bad_proof);
    assert!(
        run.verify_proof(&bad_proof).is_err(),
        "tampered main proof must not verify"
    );
}

#[test]
fn rv32_trace_main_proof_step_reordering_must_fail() {
    let program_bytes = addi_halt_program_bytes(/*imm=*/ 1);
    let run = prove_run(&program_bytes, /*max_steps=*/ 2);

    let mut bad_proof = run.proof().clone();
    if bad_proof.steps.len() >= 2 {
        bad_proof.steps.swap(0, 1);
    } else {
        tamper_any_claim_scalar(&mut bad_proof);
    }
    assert!(
        run.verify_proof(&bad_proof).is_err(),
        "reordered proof steps must not verify"
    );
}

#[test]
fn rv32_trace_main_proof_splicing_across_runs_must_fail() {
    let program_bytes_a = addi_halt_program_bytes(/*imm=*/ 1);
    let run_a = prove_run(&program_bytes_a, /*max_steps=*/ 2);

    let program_bytes_b = addi_halt_program_bytes(/*imm=*/ 2);
    let run_b = prove_run(&program_bytes_b, /*max_steps=*/ 2);

    assert!(
        run_a.verify_proof(run_b.proof()).is_err(),
        "splicing proof across runs must not verify"
    );
}
