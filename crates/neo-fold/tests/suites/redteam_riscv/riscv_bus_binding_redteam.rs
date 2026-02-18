use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::shard::ShardProof;
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;

fn prove_run(program: Vec<RiscvInstruction>, max_steps: usize) -> Rv32TraceWiringRun {
    let steps = max_steps;
    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
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
fn rv32_trace_cpu_vs_bus_twist_rv_mismatch_must_fail() {
    // Program: LW x1, 0(x0); HALT, with RAM[0]=7.
    let program = vec![
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 1,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let steps = 2usize;
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(steps)
        .min_trace_len(steps)
        .max_steps(steps)
        .ram_init_u32(/*addr=*/ 0, /*value=*/ 7)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");

    let mut bad_proof = run.proof().clone();
    tamper_any_claim_scalar(&mut bad_proof);
    assert!(
        run.verify_proof(&bad_proof).is_err(),
        "tampered bus/twist binding must not verify"
    );
}

#[test]
fn rv32_trace_cpu_vs_bus_shout_val_mismatch_must_fail() {
    // Program: ADDI x1, x0, 1; HALT (forces an ADD shout lookup).
    let run = prove_run(
        vec![
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 1,
            },
            RiscvInstruction::Halt,
        ],
        /*max_steps=*/ 2,
    );

    let mut bad_proof = run.proof().clone();
    tamper_any_claim_scalar(&mut bad_proof);
    assert!(
        run.verify_proof(&bad_proof).is_err(),
        "tampered bus/shout binding must not verify"
    );
}
