#![allow(non_snake_case)]

use std::marker::PhantomData;

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{matrix::Mat, poly::SparsePoly, relations::CcsStructure, relations::McsInstance, relations::McsWitness};
use neo_ccs::traits::SModuleHomomorphism;
use neo_math::{D, K};
use neo_memory::encode::{encode_lut_for_shout, encode_mem_for_twist};
use neo_memory::plain::{PlainLutTrace, PlainMemLayout, PlainMemTrace};
use neo_memory::witness::StepWitnessBundle;
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::engines::utils;
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_fold::folding::CommitMixers;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

/// Dummy commit that produces zero commitments.
#[derive(Clone, Copy, Default)]
struct DummyCommit;

impl SModuleHomomorphism<F, Cmt> for DummyCommit {
    fn commit(&self, z: &Mat<F>) -> Cmt {
        Cmt::zeros(z.rows(), 1)
    }

    fn project_x(&self, z: &Mat<F>, m_in: usize) -> Mat<F> {
        let rows = z.rows();
        let mut out = Mat::zero(rows, m_in, F::ZERO);
        for r in 0..rows {
            for c in 0..m_in.min(z.cols()) {
                out[(r, c)] = z[(r, c)];
            }
        }
        out
    }
}

fn decompose_z_to_Z(params: &NeoParams, z: &[F]) -> Mat<F> {
    let d = D;
    let m = z.len();
    let digits = neo_ajtai::decomp_b(z, params.b, d, neo_ajtai::DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; d * m];
    for c in 0..m {
        for r in 0..d {
            row_major[r * m + c] = digits[c * d + r];
        }
    }
    Mat::from_row_major(d, m, row_major)
}

fn build_add_ccs_mcs(
    params: &NeoParams,
    l: &DummyCommit,
    const_one: F,
    lhs0: F,
    lhs1: F,
    out: F,
) -> (CcsStructure<F>, McsInstance<Cmt, F>, McsWitness<F>) {
    // CCS with n=1, m=4, constraint: const_one + x1 + x2 - x3 = 0.
    // Columns: [const_one (public input), lhs0, lhs1, out]
    let mut m0 = Mat::zero(1, 4, F::ZERO);
    m0[(0, 0)] = F::ONE; // picks const_one (public input)
    let mut m1 = Mat::zero(1, 4, F::ZERO);
    m1[(0, 1)] = F::ONE; // picks lhs0
    let mut m2 = Mat::zero(1, 4, F::ZERO);
    m2[(0, 2)] = F::ONE; // picks lhs1
    let mut m3 = Mat::zero(1, 4, F::ZERO);
    m3[(0, 3)] = F::ONE; // picks out

    let term_const = neo_ccs::poly::Term { coeff: F::ONE, exps: vec![1, 0, 0, 0] };
    let term_x1 = neo_ccs::poly::Term { coeff: F::ONE, exps: vec![0, 1, 0, 0] };
    let term_x2 = neo_ccs::poly::Term { coeff: F::ONE, exps: vec![0, 0, 1, 0] };
    let term_neg_out = neo_ccs::poly::Term { coeff: F::ZERO - F::ONE, exps: vec![0, 0, 0, 1] };
    let f = SparsePoly::new(4, vec![term_const, term_x1, term_x2, term_neg_out]);

    let s = CcsStructure::new(vec![m0, m1, m2, m3], f).expect("CCS");

    // Public input x = [const_one]; witness w = [lhs0, lhs1, out]
    let z = vec![const_one, lhs0, lhs1, out];
    let Z = decompose_z_to_Z(params, &z);
    let c = l.commit(&Z);
    let w = z.clone(); // all witness, m_in=0

    let inst = McsInstance { c, x: vec![], m_in: 0 };
    let wit = McsWitness { w, Z };
    (s, inst, wit)
}

fn default_mixers() -> CommitMixers<fn(&[Mat<F>], &[Cmt]) -> Cmt, fn(&[Cmt], u32) -> Cmt> {
    fn mix_rhos_commits(_rhos: &[Mat<F>], _cs: &[Cmt]) -> Cmt {
        Cmt::zeros(D, 1)
    }
    fn combine_b_pows(_cs: &[Cmt], _b: u32) -> Cmt {
        Cmt::zeros(D, 1)
    }
    CommitMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

#[test]
fn full_folding_integration_single_chunk() {
    let m = 4usize; // const_one (public), lhs0, lhs1, out
    let base_params = NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");
    let params = NeoParams::new(
        base_params.q,
        base_params.eta,
        base_params.d,
        base_params.kappa,
        base_params.m,
        base_params.b,
        16, // bump k_rho to satisfy Π_RLC norm bound comfortably for this test
        base_params.T,
        base_params.s,
        base_params.lambda,
    )
    .expect("params");
    let l = DummyCommit::default();
    let mixers = default_mixers();

    // Program values: lookup_val + write_val = out
    let const_one = F::ONE;
    let write_val = F::from_u64(1);
    let lookup_val = F::from_u64(1);
    let out_val = const_one + write_val + lookup_val;

    // Build CCS (single chunk) enforcing out = write_val + lookup_val
    let (ccs, mcs_inst, mcs_wit) = build_add_ccs_mcs(&params, &l, const_one, lookup_val, write_val, out_val);
    let dims = utils::build_dims_and_policy(&params, &ccs).expect("dims");
    let _ell_n = dims.ell_n;

    let acc_init: Vec<neo_ccs::relations::MeInstance<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    // Plain memory trace for one step
    let mem_layout = PlainMemLayout { k: 2, d: 1, n_side: 2 };
    let plain_mem = PlainMemTrace {
        init_vals: vec![F::ZERO; mem_layout.k],
        steps: 1,
        // One write to addr 0 with value 1 (no reads).
        has_read: vec![F::ZERO],
        has_write: vec![F::ONE],
        read_addr: vec![0],
        write_addr: vec![0],
        read_val: vec![F::ZERO],
        write_val: vec![F::from_u64(1)],
        // inc tracks per-address delta; only addr 0 changes by +1 in this step.
        inc: vec![vec![F::from_u64(1)], vec![F::ZERO]],
    };

    // Plain lookup trace for one step
    let plain_lut = PlainLutTrace {
        // One lookup: key 0 -> value 1 (must match table content).
        has_lookup: vec![F::ONE],
        addr: vec![0],
        val: vec![F::from_u64(1)],
    };
    let lut_table = neo_memory::plain::LutTable {
        table_id: 0,
        k: 2,
        d: 1,
        n_side: 2,
        content: vec![F::from_u64(1), F::from_u64(2)],
    };

    // Encode memory/lookup for this chunk
    let commit_fn = |mat: &Mat<F>| l.commit(mat);
    let (mem_inst, mem_wit) = encode_mem_for_twist(&params, &mem_layout, &plain_mem, &commit_fn);
    let (lut_inst, lut_wit) = encode_lut_for_shout(&params, &lut_table, &plain_lut, &commit_fn);

    let step_bundle = StepWitnessBundle {
        mcs: (mcs_inst.clone(), mcs_wit.clone()),
        lut_instances: vec![(lut_inst.clone(), lut_wit)],
        mem_instances: vec![(mem_inst.clone(), mem_wit)],
        _phantom: PhantomData::<K>,
    };

    let mut tr_prove = Poseidon2Transcript::new(b"full-fold");
    let proof = fold_shard_prove(
        FoldingMode::PaperExact,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    let mut tr_verify = Poseidon2Transcript::new(b"full-fold");
    fold_shard_verify(
        FoldingMode::PaperExact,
        &mut tr_verify,
        &params,
        &ccs,
        &[step_bundle],
        &acc_init,
        &proof,
        &l,
        mixers,
    )
    .expect("verify should succeed");

    // Print a short summary so it's clear what was enforced.
    let step0 = &proof.steps[0];
    let mem_me = step0.mem.me_claims.len();
    let ccs_me = step0.fold.ccs_out.len();
    let total_me = mem_me + ccs_me;
    let children = step0.fold.dec_children.len();
    println!("Full folding step:");
    println!("  CCS ME count: {}", ccs_me);
    println!("  Twist+Shout ME count: {}", mem_me);
    println!("  Total ME into RLC: {}", total_me);
    println!("  Children after DEC: {}", children);
    println!("  Lookup enforced: key 0 -> val 1 from table [1, 2]");
    println!("  Memory enforced: write addr 0 := 1 (inc +1)");

    // Program output comes from CCS: out = const_one + lookup + write = 3.
    println!("  Program output (CCS out) = {}", out_val.as_canonical_u64());

    // Show a small, deterministic slice of the folded output.
    let final_children = proof.compute_final_children(&acc_init);
    if let Some(first) = final_children.first() {
        let r_len = first.r.len();
        let y0_prefix: Vec<K> = first.y.get(0).map(|row| row.iter().take(2).cloned().collect()).unwrap_or_default();
        let y_scalars_prefix: Vec<K> = first.y_scalars.iter().take(2).cloned().collect();
        println!("  First child: r_len={}, y[0][..2]={:?}, y_scalars[..2]={:?}", r_len, y0_prefix, y_scalars_prefix);
    }
}
