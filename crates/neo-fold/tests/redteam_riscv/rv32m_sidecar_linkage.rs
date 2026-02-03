use neo_fold::{pi_ccs_prove_simple, pi_ccs_verify};
use neo_memory::ajtai::encode_vector_balanced_to_mat;
use neo_memory::riscv::ccs::build_rv32_b1_rv32m_sidecar_ccs;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use neo_fold::riscv_shard::Rv32B1;

#[test]
fn rv32m_sidecar_is_bound_to_main_witness_commitment() {
    // Program: MUL x1, x0, x0; HALT
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
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
    run.verify().expect("baseline verify");

    // Build the RV32M sidecar CCS and collect the per-step MCS instances/witnesses.
    let rv32m_ccs = build_rv32_b1_rv32m_sidecar_ccs(run.layout()).expect("build rv32m sidecar ccs");

    let mut mcs_insts = Vec::with_capacity(run.steps_witness().len());
    let mut mcs_wits = Vec::with_capacity(run.steps_witness().len());
    for step in run.steps_witness() {
        let (inst, wit) = &step.mcs;
        mcs_insts.push(inst.clone());
        mcs_wits.push(wit.clone());
    }

    // Tamper with one RV32M-relevant witness coordinate (mul_hi at j=0),
    // while keeping the *original* MCS instances (commitments) fixed.
    let idx = run.layout().mul_hi(0);
    let m_in = mcs_insts[0].m_in;
    assert!(
        idx >= m_in,
        "expected mul_hi to be in the private witness region (idx={idx}, m_in={m_in})"
    );

    let mut z0 = Vec::with_capacity(mcs_insts[0].m_in + mcs_wits[0].w.len());
    z0.extend_from_slice(&mcs_insts[0].x);
    z0.extend_from_slice(&mcs_wits[0].w);
    assert_eq!(z0.len(), rv32m_ccs.m, "unexpected step witness width");

    z0[idx] += F::ONE;
    let z0_tampered = encode_vector_balanced_to_mat(run.params(), &z0);

    mcs_wits[0].w = z0[m_in..].to_vec();
    mcs_wits[0].Z = z0_tampered;

    let num_steps = mcs_insts.len();
    let mut tr = Poseidon2Transcript::new(b"neo.fold/tests/rv32m_sidecar_linkage");
    tr.append_message(b"num_steps", &(num_steps as u64).to_le_bytes());

    // The prover may either:
    // - reject because the witness no longer matches the commitment, or
    // - produce a proof that fails verification.
    let Ok((me_out, proof)) = pi_ccs_prove_simple(&mut tr, run.params(), &rv32m_ccs, &mcs_insts, &mcs_wits, run.committer())
    else {
        return;
    };

    let mut tr = Poseidon2Transcript::new(b"neo.fold/tests/rv32m_sidecar_linkage");
    tr.append_message(b"num_steps", &(num_steps as u64).to_le_bytes());
    let ok = pi_ccs_verify(&mut tr, run.params(), &rv32m_ccs, &mcs_insts, &[], &me_out, &proof)
        .expect("rv32m sidecar verify");
    assert!(
        !ok,
        "rv32m sidecar verification unexpectedly succeeded with a tampered witness"
    );
}
