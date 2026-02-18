use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_math::{F, K};
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode};
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

fn addi_sw_halt_program_bytes(value: i32, addr: i32) -> Vec<u8> {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: value,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: addr,
        },
        RiscvInstruction::Halt,
    ];
    encode_program(&program)
}

#[test]
fn redteam_output_claim_path_rejects_tampered_proof() {
    let program_bytes = addi_sw_halt_program_bytes(/*value=*/ 42, /*addr=*/ 0x100);
    let steps = 4usize;
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(steps)
        .min_trace_len(steps)
        .max_steps(steps)
        .output(/*output_addr=*/ 0x100, /*expected_output=*/ F::from_u64(42))
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
    assert!(tampered, "expected at least one scalar to tamper");
    assert!(
        run.verify_proof(&bad_proof).is_err(),
        "tampered proof should fail full verification"
    );
}

#[test]
fn redteam_verifier_rejects_spliced_proofs_across_runs() {
    let program_bytes_a = addi_halt_program_bytes(/*imm=*/ 7);
    let steps = 4usize;
    let mut run_a = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes_a)
        .chunk_rows(steps)
        .min_trace_len(steps)
        .max_steps(steps)
        .prove()
        .expect("prove a");
    run_a.verify().expect("verify a");

    let program_bytes_b = addi_halt_program_bytes(/*imm=*/ 8);
    let mut run_b = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes_b)
        .chunk_rows(steps)
        .min_trace_len(steps)
        .max_steps(steps)
        .prove()
        .expect("prove b");
    run_b.verify().expect("verify b");

    assert!(
        run_a.verify_proof(run_b.proof()).is_err(),
        "spliced proof across runs must not verify"
    );
}
