use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_math::K;
use neo_memory::riscv::lookups::{encode_program, BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode};
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

fn tamper_val_scalar(run: &Rv32TraceWiringRun) {
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
    assert!(tampered, "expected at least one val claim scalar to tamper");
    assert!(
        run.verify_proof(&bad_proof).is_err(),
        "semantics-related malicious tamper must not verify"
    );
}

#[test]
fn rv32_trace_semantics_malicious_alu_out_must_fail() {
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
    tamper_val_scalar(&run);
}

#[test]
fn rv32_trace_semantics_malicious_eff_addr_must_fail() {
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
    tamper_val_scalar(&run);
}

#[test]
fn rv32_trace_semantics_malicious_ram_wv_must_fail() {
    let run = prove_run(
        vec![
            RiscvInstruction::Store {
                op: RiscvMemOp::Sw,
                rs1: 0,
                rs2: 0,
                imm: 0,
            },
            RiscvInstruction::Halt,
        ],
        /*max_steps=*/ 2,
    );
    tamper_val_scalar(&run);
}

#[test]
fn rv32_trace_semantics_malicious_br_taken_must_fail() {
    let run = prove_run(
        vec![
            RiscvInstruction::Branch {
                cond: BranchCondition::Eq,
                rs1: 0,
                rs2: 0,
                imm: 8,
            },
            RiscvInstruction::Nop,
            RiscvInstruction::Halt,
        ],
        /*max_steps=*/ 2,
    );
    tamper_val_scalar(&run);
}
