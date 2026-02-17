use neo_ajtai::{s_lincomb, s_mul, Commitment as Cmt};
use neo_ccs::poly::SparsePoly;
use neo_ccs::relations::{CcsStructure, McsInstance, McsWitness, MeInstance};
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::pi_ccs_prove_simple;
use neo_fold::riscv_shard::{fold_shard_verify_rv32_b1_with_statement_mem_init, Rv32B1, Rv32B1ProofBundle, Rv32B1Run};
use neo_fold::shard::CommitMixers;
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F, K};
use neo_memory::output_check::ProgramIO;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    use neo_math::ring::cf_inv;

    debug_assert_eq!(mat.rows(), D);
    debug_assert_eq!(mat.cols(), D);

    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

fn default_mixers() -> CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt> {
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Cmt]) -> Cmt {
        if cs.len() == 1 {
            return cs[0].clone();
        }
        let rq_els: Vec<RqEl> = rhos.iter().map(rot_matrix_to_rq).collect();
        s_lincomb(&rq_els, cs).expect("s_lincomb should succeed")
    }

    fn combine_b_pows(cs: &[Cmt], b: u32) -> Cmt {
        let mut acc = cs[0].clone();
        let mut pow = F::from_u64(b as u64);
        for c in cs.iter().skip(1) {
            let rq_pow = RqEl::from_field_scalar(pow);
            let term = s_mul(&rq_pow, c);
            acc.add_inplace(&term);
            pow *= F::from_u64(b as u64);
        }
        acc
    }

    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

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

fn prove_basic_run() -> Rv32B1Run {
    let program_bytes = addi_halt_program_bytes(/*imm=*/ 7);
    Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(32)
        .ram_bytes(0x200)
        .chunk_size(1)
        .max_steps(2)
        .shout_ops([RiscvOpcode::Add])
        .prove()
        .expect("prove")
}

fn prove_output_run() -> Rv32B1Run {
    let program_bytes = addi_sw_halt_program_bytes(/*value=*/ 42, /*addr=*/ 0x100);
    Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(32)
        .ram_bytes(0x400)
        .chunk_size(1)
        .max_steps(3)
        .shout_ops([RiscvOpcode::Add])
        .output(/*output_addr=*/ 0x100, /*expected_output=*/ F::from_u64(42))
        .prove()
        .expect("prove")
}

fn collect_mcs(run: &Rv32B1Run) -> (Vec<McsInstance<Cmt, F>>, Vec<McsWitness<F>>) {
    let mut insts = Vec::with_capacity(run.steps_witness().len());
    let mut wits = Vec::with_capacity(run.steps_witness().len());
    for step in run.steps_witness() {
        let (inst, wit) = &step.mcs;
        insts.push(inst.clone());
        wits.push(wit.clone());
    }
    (insts, wits)
}

fn make_trivial_ccs(m: usize) -> CcsStructure<F> {
    let a = Mat::zero(1, m, F::ZERO);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![a], f).expect("build trivial CCS")
}

fn swap_decode_plumbing_for_trivial_ccs(run: &Rv32B1Run, bundle: &mut Rv32B1ProofBundle) {
    let (mcs_insts, mcs_wits) = collect_mcs(run);
    let num_steps = mcs_insts.len();
    let trivial_ccs = make_trivial_ccs(run.ccs().m);

    let mut tr = Poseidon2Transcript::new(b"neo.fold/rv32_b1/decode_plumbing_sidecar_batch");
    tr.append_message(b"decode_plumbing_sidecar/num_steps", &(num_steps as u64).to_le_bytes());

    let (me_out, proof) = pi_ccs_prove_simple(
        &mut tr,
        run.params(),
        &trivial_ccs,
        &mcs_insts,
        &mcs_wits,
        run.committer(),
    )
    .expect("prove trivial decode plumbing sidecar");

    bundle.decode_plumbing.num_steps = num_steps;
    bundle.decode_plumbing.me_out = me_out;
    bundle.decode_plumbing.proof = proof;
}

#[test]
fn redteam_output_claim_path_should_not_accept_without_sidecar_enforcement() {
    let run = prove_output_run();

    let mut bad_bundle = run.proof().clone();
    bad_bundle.semantics.me_out.clear();
    assert!(
        run.verify_proof_bundle(&bad_bundle).is_err(),
        "sanity: full bundle verification must fail for a corrupted semantics sidecar"
    );

    assert!(
        run.verify_output_claim_in_bundle(&bad_bundle, 0x100, F::from_u64(42))
            .is_err(),
        "output-claim verification accepted a bundle with corrupted sidecar proofs"
    );
}

#[test]
fn redteam_output_claim_variants_should_not_accept_without_sidecar_enforcement() {
    let run = prove_output_run();

    let mut bad_bundle = run.proof().clone();
    bad_bundle.semantics.me_out.clear();
    assert!(
        run.verify_proof_bundle(&bad_bundle).is_err(),
        "sanity: full bundle verification must fail for a corrupted semantics sidecar"
    );

    assert!(
        run.verify_default_output_claim_in_bundle(&bad_bundle)
            .is_err(),
        "default output-claim verification accepted a bundle with corrupted sidecar proofs"
    );

    let output_claims = ProgramIO::new().with_output(0x100, F::from_u64(42));
    assert!(
        run.verify_output_claims_in_bundle(&bad_bundle, output_claims)
            .is_err(),
        "multi-output-claim verification accepted a bundle with corrupted sidecar proofs"
    );
}

#[test]
fn redteam_verifier_should_reject_prover_selected_decode_ccs() {
    let mut run = prove_basic_run();
    run.verify().expect("baseline verify");

    let mut bad_bundle = run.proof().clone();
    swap_decode_plumbing_for_trivial_ccs(&run, &mut bad_bundle);

    assert!(
        run.verify_proof_bundle(&bad_bundle).is_err(),
        "verifier accepted a prover-supplied decode CCS shape"
    );
}

#[test]
fn redteam_legacy_main_only_verifier_should_not_accept_without_sidecars() {
    let mut run = prove_basic_run();
    run.verify().expect("baseline verify");

    let mut bad_bundle = run.proof().clone();
    bad_bundle.semantics.me_out.clear();
    assert!(
        run.verify_proof_bundle(&bad_bundle).is_err(),
        "sanity: full bundle verification must fail for a corrupted semantics sidecar"
    );

    let steps_public = run.steps_public();
    let mut tr = Poseidon2Transcript::new(b"neo.fold/session");
    let res = fold_shard_verify_rv32_b1_with_statement_mem_init(
        FoldingMode::Optimized,
        &mut tr,
        run.params(),
        run.ccs(),
        run.mem_layouts(),
        run.initial_mem(),
        &steps_public,
        &[] as &[MeInstance<Cmt, F, K>],
        &bad_bundle.main,
        default_mixers(),
        run.layout(),
    );

    assert!(
        res.is_err(),
        "legacy verifier accepted main proof without sidecar semantics checks"
    );
}
