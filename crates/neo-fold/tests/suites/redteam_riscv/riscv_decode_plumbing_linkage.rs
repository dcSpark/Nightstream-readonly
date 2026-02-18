use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::shard::ShardProof;
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;

fn prove_run_addi_halt(imm: i32) -> Rv32TraceWiringRun {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm,
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
    run
}

fn tamper_decode_related_scalar(proof: &mut ShardProof) {
    for step in &mut proof.steps {
        for claim in &mut step.mem.wp_me_claims {
            if let Some(first) = claim.y_scalars.first_mut() {
                *first += K::ONE;
                return;
            }
        }
    }
    panic!("expected at least one decode-related scalar in wp claims");
}

#[test]
fn rv32_trace_decode_plumbing_tampered_scalar_must_not_verify() {
    let run = prove_run_addi_halt(/*imm=*/ 1);
    let mut bad_proof = run.proof().clone();
    tamper_decode_related_scalar(&mut bad_proof);
    assert!(
        run.verify_proof(&bad_proof).is_err(),
        "decode-related tamper must not verify"
    );
}

#[test]
fn rv32_trace_decode_plumbing_splicing_across_runs_must_fail() {
    let run_a = prove_run_addi_halt(/*imm=*/ 1);
    let run_b = prove_run_addi_halt(/*imm=*/ 2);
    assert!(
        run_a.verify_proof(run_b.proof()).is_err(),
        "spliced decode commitments must not verify"
    );
}
