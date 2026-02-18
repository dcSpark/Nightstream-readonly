use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
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

fn tamper_wp_scalar(run: &Rv32TraceWiringRun) {
    let mut bad_proof = run.proof().clone();
    let mut tampered = false;
    for step in &mut bad_proof.steps {
        for claim in &mut step.mem.wp_me_claims {
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
    assert!(tampered, "expected at least one wp claim scalar to tamper");
    assert!(
        run.verify_proof(&bad_proof).is_err(),
        "decode-related malicious tamper must not verify"
    );
}

#[test]
fn rv32_trace_decode_malicious_imm_i_must_fail() {
    let run = prove_run_addi_halt(/*imm=*/ 1);
    tamper_wp_scalar(&run);
}

#[test]
fn rv32_trace_decode_malicious_rd_field_must_fail() {
    let run = prove_run_addi_halt(/*imm=*/ 2);
    tamper_wp_scalar(&run);
}
