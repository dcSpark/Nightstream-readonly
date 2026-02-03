use neo_ajtai::Commitment as Cmt;
use neo_fold::riscv_shard::{Rv32B1, Rv32B1Run};
use neo_fold::{pi_ccs_prove_simple, pi_ccs_verify};
use neo_memory::ajtai::encode_vector_balanced_to_mat;
use neo_memory::riscv::ccs::build_rv32_b1_decode_sidecar_ccs;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use neo_transcript::Poseidon2Transcript;
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

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

fn prove_run_addi_halt(imm: i32) -> Rv32B1Run {
    let program_bytes = addi_halt_program_bytes(imm);
    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_size(1)
        .max_steps(2)
        .ram_bytes(0x200)
        .prove()
        .expect("prove");
    run.verify().expect("baseline verify");
    run
}

fn collect_mcs(run: &Rv32B1Run) -> (Vec<neo_ccs::McsInstance<Cmt, F>>, Vec<neo_ccs::McsWitness<F>>) {
    let mut insts = Vec::with_capacity(run.steps_witness().len());
    let mut wits = Vec::with_capacity(run.steps_witness().len());
    for step in run.steps_witness() {
        let (inst, wit) = &step.mcs;
        insts.push(inst.clone());
        wits.push(wit.clone());
    }
    (insts, wits)
}

fn tamper_step0_witness(
    run: &Rv32B1Run,
    sidecar_ccs_m: usize,
    mcs_insts: &[neo_ccs::McsInstance<Cmt, F>],
    mcs_wits: &mut [neo_ccs::McsWitness<F>],
    idx_to_tamper: usize,
) {
    let m_in = mcs_insts[0].m_in;
    assert!(
        idx_to_tamper >= m_in,
        "expected idx_to_tamper to be in private witness region (idx={idx_to_tamper}, m_in={m_in})"
    );

    let mut z0 = Vec::with_capacity(m_in + mcs_wits[0].w.len());
    z0.extend_from_slice(&mcs_insts[0].x);
    z0.extend_from_slice(&mcs_wits[0].w);
    assert_eq!(z0.len(), sidecar_ccs_m, "unexpected witness width");

    z0[idx_to_tamper] += F::ONE;
    let z0_tampered = encode_vector_balanced_to_mat(run.params(), &z0);
    mcs_wits[0].w = z0[m_in..].to_vec();
    mcs_wits[0].Z = z0_tampered;
}

#[test]
fn rv32_b1_semantics_sidecar_tampered_pc_out_must_not_verify() {
    let run = prove_run_addi_halt(/*imm=*/ 1);
    // In the current RV32 B1 implementation, the “decode sidecar” CCS contains the full step semantics.
    let semantics_ccs = build_rv32_b1_decode_sidecar_ccs(run.layout(), run.mem_layouts()).expect("sidecar ccs");

    let (mcs_insts, mut mcs_wits) = collect_mcs(&run);
    let idx = run.layout().pc_out(0);
    tamper_step0_witness(&run, semantics_ccs.m, &mcs_insts, &mut mcs_wits, idx);

    let num_steps = mcs_insts.len();
    let mut tr = Poseidon2Transcript::new(b"neo.fold/rv32_b1/decode_sidecar_batch");
    tr.append_message(b"decode_sidecar/num_steps", &(num_steps as u64).to_le_bytes());

    // Prover may reject (commitment mismatch) or produce a proof that fails verification.
    let Ok((me_out, proof)) = pi_ccs_prove_simple(
        &mut tr,
        run.params(),
        &semantics_ccs,
        &mcs_insts,
        &mcs_wits,
        run.committer(),
    ) else {
        return;
    };

    let mut tr = Poseidon2Transcript::new(b"neo.fold/rv32_b1/decode_sidecar_batch");
    tr.append_message(b"decode_sidecar/num_steps", &(num_steps as u64).to_le_bytes());
    let Ok(ok) = pi_ccs_verify(&mut tr, run.params(), &semantics_ccs, &mcs_insts, &[], &me_out, &proof) else {
        return;
    };
    assert!(
        !ok,
        "semantics sidecar verification unexpectedly succeeded with a tampered witness"
    );
}

#[test]
fn rv32_b1_semantics_sidecar_splicing_across_runs_must_fail() {
    let run_a = prove_run_addi_halt(/*imm=*/ 1);
    let run_b = prove_run_addi_halt(/*imm=*/ 2);

    let semantics_ccs = build_rv32_b1_decode_sidecar_ccs(run_a.layout(), run_a.mem_layouts()).expect("sidecar ccs");

    let (mcs_insts_a, mcs_wits_a) = collect_mcs(&run_a);
    let num_steps = mcs_insts_a.len();
    let mut tr = Poseidon2Transcript::new(b"neo.fold/rv32_b1/decode_sidecar_batch");
    tr.append_message(b"decode_sidecar/num_steps", &(num_steps as u64).to_le_bytes());
    let (me_out_a, proof_a) = pi_ccs_prove_simple(
        &mut tr,
        run_a.params(),
        &semantics_ccs,
        &mcs_insts_a,
        &mcs_wits_a,
        run_a.committer(),
    )
    .expect("prove semantics sidecar");

    // Sanity: semantics sidecar should verify for the matching run.
    let mut tr = Poseidon2Transcript::new(b"neo.fold/rv32_b1/decode_sidecar_batch");
    tr.append_message(b"decode_sidecar/num_steps", &(num_steps as u64).to_le_bytes());
    let ok = pi_ccs_verify(
        &mut tr,
        run_a.params(),
        &semantics_ccs,
        &mcs_insts_a,
        &[],
        &me_out_a,
        &proof_a,
    )
    .expect("semantics sidecar verify (baseline)");
    assert!(ok, "baseline semantics sidecar proof should verify");

    let assert_verify_fails =
        |domain_sep: &'static [u8], num_steps_msg: u64, insts: &[neo_ccs::McsInstance<Cmt, F>], label: &str| {
            let mut tr = Poseidon2Transcript::new(domain_sep);
            tr.append_message(b"decode_sidecar/num_steps", &num_steps_msg.to_le_bytes());
            match pi_ccs_verify(&mut tr, run_a.params(), &semantics_ccs, insts, &[], &me_out_a, &proof_a) {
                Ok(true) => panic!("{label}: semantics sidecar verification unexpectedly succeeded"),
                Ok(false) | Err(_) => {}
            }
        };

    // Wrong transcript domain separator must fail (or error).
    assert_verify_fails(
        b"neo.fold/rv32_b1/decode_sidecar_batch/wrong_domain",
        num_steps as u64,
        &mcs_insts_a,
        "wrong transcript domain",
    );

    // Wrong num_steps binding must fail (or error).
    assert_verify_fails(
        b"neo.fold/rv32_b1/decode_sidecar_batch",
        num_steps.saturating_add(1) as u64,
        &mcs_insts_a,
        "wrong num_steps message",
    );

    // Swapping step order must fail (or error).
    assert!(num_steps >= 2, "expected at least 2 steps for swap test");
    let mut mcs_insts_swapped = mcs_insts_a.clone();
    mcs_insts_swapped.swap(0, 1);
    assert_verify_fails(
        b"neo.fold/rv32_b1/decode_sidecar_batch",
        num_steps as u64,
        &mcs_insts_swapped,
        "swapped step order",
    );

    // Attempt to verify run A's sidecar proof against run B's commitments must fail (or error).
    let (mcs_insts_b, _mcs_wits_b) = collect_mcs(&run_b);
    assert_eq!(mcs_insts_b.len(), num_steps, "expected same step count");
    assert_verify_fails(
        b"neo.fold/rv32_b1/decode_sidecar_batch",
        num_steps as u64,
        &mcs_insts_b,
        "spliced commitments",
    );
}
