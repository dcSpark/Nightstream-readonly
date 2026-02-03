use neo_ajtai::Commitment as Cmt;
use neo_fold::{pi_ccs_prove_simple, pi_ccs_verify};
use neo_fold::riscv_shard::{Rv32B1, Rv32B1Run};
use neo_memory::riscv::ccs::build_rv32_b1_decode_plumbing_sidecar_ccs;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use super::helpers::{assert_prove_or_verify_fails, collect_mcs, mcs_recommit_step_after_private_tamper};

fn prove_run_addi_halt(imm: i32) -> Rv32B1Run {
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
    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .max_steps(2)
        .ram_bytes(0x200)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");
    run
}

fn prove_decode_sidecar_or_verify_fails(
    run: &Rv32B1Run,
    mcs_insts: &[neo_ccs::McsInstance<Cmt, F>],
    mcs_wits: &[neo_ccs::McsWitness<F>],
) {
    let decode_ccs = build_rv32_b1_decode_plumbing_sidecar_ccs(run.layout()).expect("decode sidecar ccs");

    let num_steps = mcs_insts.len();
    let mut tr = Poseidon2Transcript::new(b"neo.fold/rv32_b1/decode_plumbing_sidecar_batch");
    tr.append_message(b"decode_plumbing_sidecar/num_steps", &(num_steps as u64).to_le_bytes());
    let Ok((me_out, proof)) =
        pi_ccs_prove_simple(&mut tr, run.params(), &decode_ccs, mcs_insts, mcs_wits, run.committer())
    else {
        return;
    };

    let mut tr = Poseidon2Transcript::new(b"neo.fold/rv32_b1/decode_plumbing_sidecar_batch");
    tr.append_message(b"decode_plumbing_sidecar/num_steps", &(num_steps as u64).to_le_bytes());
    let res = pi_ccs_verify(&mut tr, run.params(), &decode_ccs, mcs_insts, &[], &me_out, &proof);
    assert_prove_or_verify_fails(res, "decode sidecar (malicious witness)");
}

#[test]
fn rv32_b1_decode_sidecar_malicious_imm_i_must_fail() {
    let run = prove_run_addi_halt(/*imm=*/ 1);

    let (mut mcs_insts, mut mcs_wits) = collect_mcs(run.steps_witness());
    let idx = run.layout().imm_i(0);
    mcs_recommit_step_after_private_tamper(
        run.params(),
        run.committer(),
        &mut mcs_insts[0],
        &mut mcs_wits[0],
        idx,
        F::ONE,
    );
    prove_decode_sidecar_or_verify_fails(&run, &mcs_insts, &mcs_wits);
}

#[test]
fn rv32_b1_decode_sidecar_malicious_rd_field_must_fail() {
    let run = prove_run_addi_halt(/*imm=*/ 1);

    let (mut mcs_insts, mut mcs_wits) = collect_mcs(run.steps_witness());
    let idx = run.layout().rd_field(0);
    mcs_recommit_step_after_private_tamper(
        run.params(),
        run.committer(),
        &mut mcs_insts[0],
        &mut mcs_wits[0],
        idx,
        F::ONE,
    );
    prove_decode_sidecar_or_verify_fails(&run, &mcs_insts, &mcs_wits);
}

