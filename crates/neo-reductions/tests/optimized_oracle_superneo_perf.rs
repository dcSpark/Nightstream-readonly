use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use neo_ccs::{CcsStructure, CcsWitness, Mat, SparsePoly, Term};
use neo_math::{from_complex, D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::optimized_engine::oracle::{OptimizedOracle, SparseCache};
use neo_reductions::engines::optimized_engine::Challenges;
use neo_reductions::engines::paper_exact_engine::q_eval_at_ext_point_fe_paper_exact_with_inputs;
use neo_reductions::sumcheck::{poly_eval_k, run_sumcheck_prover};
use neo_reductions::superneo_eval::build_superneo_eval_cache;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn k(re: u64, im: u64) -> K {
    from_complex(F::from_u64(re), F::from_u64(im))
}

fn dense_mat<Ff: PrimeCharacteristicRing + Copy>(rows: usize, cols: usize, seed: u64) -> Mat<Ff> {
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let v = if ((r as u64 * 17) + (c as u64 * 13) + seed) % 37 == 0 {
                Ff::from_u64(((r + 3 * c + (seed as usize % 19)) % 29 + 1) as u64)
            } else {
                Ff::ZERO
            };
            data.push(v);
        }
    }
    Mat::from_row_major(rows, cols, data)
}

fn z_witness(seed: u64, m: usize) -> Mat<F> {
    let mut data = Vec::with_capacity(D * m);
    for rho in 0..D {
        for c in 0..m {
            data.push(F::from_u64(seed + (rho as u64) * 19 + (c as u64) * 29));
        }
    }
    Mat::from_row_major(D, m, data)
}

#[test]
#[ignore = "perf report only; run with --release -- --ignored --nocapture"]
fn report_optimized_oracle_superneo_canonical_perf() {
    let n = 64usize;
    let m = 4 * D;
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).unwrap();

    let matrices = vec![
        dense_mat::<F>(n, m, 111),
        dense_mat::<F>(n, m, 222),
        dense_mat::<F>(n, m, 333),
        dense_mat::<F>(n, m, 444),
    ];
    let f = SparsePoly::new(
        4,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![0, 1, 1, 0],
            },
            Term {
                coeff: F::ONE,
                exps: vec![1, 0, 0, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 0, 1],
            },
        ],
    );
    let s = CcsStructure::new(matrices, f).unwrap();
    let dims = neo_reductions::engines::utils::build_dims_and_policy(&params, &s).unwrap();

    let mcs_witnesses = vec![
        CcsWitness {
            w: vec![],
            Z: z_witness(10_000, m),
        },
        CcsWitness {
            w: vec![],
            Z: z_witness(20_000, m),
        },
        CcsWitness {
            w: vec![],
            Z: z_witness(30_000, m),
        },
        CcsWitness {
            w: vec![],
            Z: z_witness(40_000, m),
        },
    ];
    let me_witnesses = vec![z_witness(50_000, m), z_witness(60_000, m)];

    let ch = Challenges {
        alpha: (0..dims.ell_d)
            .map(|i| k(1_000 + i as u64, 2_000 + i as u64))
            .collect(),
        beta_a: (0..dims.ell_d)
            .map(|i| k(3_000 + i as u64, 4_000 + i as u64))
            .collect(),
        beta_r: (0..dims.ell_n)
            .map(|i| k(5_000 + i as u64, 6_000 + i as u64))
            .collect(),
        beta_m: Vec::new(),
        gamma: k(7_777, 8_888),
    };
    let r_inputs: Vec<K> = (0..dims.ell_n)
        .map(|i| k(9_000 + i as u64, 10_000 + i as u64))
        .collect();

    let d_sz = 1usize << dims.ell_d;
    let n_sz = 1usize << dims.ell_n;
    let mut initial_sum = K::ZERO;
    for xa in 0..d_sz {
        let alpha_bool: Vec<K> = (0..dims.ell_d)
            .map(|bit| if ((xa >> bit) & 1) == 1 { K::ONE } else { K::ZERO })
            .collect();
        for xr in 0..n_sz {
            let r_bool: Vec<K> = (0..dims.ell_n)
                .map(|bit| if ((xr >> bit) & 1) == 1 { K::ONE } else { K::ZERO })
                .collect();
            let (q_at_bool, _) = q_eval_at_ext_point_fe_paper_exact_with_inputs(
                &s,
                &params,
                &mcs_witnesses,
                &me_witnesses,
                &alpha_bool,
                &r_bool,
                &ch,
                Some(&r_inputs),
            );
            initial_sum += q_at_bool;
        }
    }

    let sparse = Arc::new(SparseCache::build(&s));
    let superneo_cache = build_superneo_eval_cache(&s)
        .map(Arc::new)
        .unwrap_or_else(|| panic!("expected D-compatible SuperNeo cache"));

    let iters = 40usize;

    let run_mode = || -> (f64, K) {
        let mut checksum = K::ZERO;
        let start = Instant::now();
        for iter in 0..iters {
            let mut oracle = OptimizedOracle::new_with_sparse_and_superneo_cache(
                &s,
                &params,
                &mcs_witnesses,
                &me_witnesses,
                ch.clone(),
                dims.ell_d,
                dims.ell_n,
                dims.d_sc,
                Some(&r_inputs),
                sparse.clone(),
                superneo_cache.clone(),
            );
            let mut tr = Poseidon2Transcript::new(b"optimized_oracle_superneo_perf".as_slice());
            let (rounds, challenges) = run_sumcheck_prover(&mut tr, &mut oracle, initial_sum).unwrap();

            let mut running = initial_sum;
            for (coeffs, &chal) in rounds.iter().zip(challenges.iter()) {
                running = poly_eval_k(coeffs, chal);
            }
            checksum += running;
            black_box(iter);
        }
        (start.elapsed().as_secs_f64(), checksum)
    };

    let (pass1_s, checksum_1) = run_mode();
    let (pass2_s, checksum_2) = run_mode();
    assert_eq!(checksum_1, checksum_2, "repeated canonical SuperNeo runs diverged");

    let speedup = pass1_s / pass2_s;
    eprintln!(
        "\\n[optimized-oracle-superneo-perf] workload: n={n}, m={m}, k_mcs={}, k_me={}, t={}, iters={iters}",
        mcs_witnesses.len(),
        me_witnesses.len(),
        s.t()
    );
    eprintln!("[optimized-oracle-superneo-perf] pass1 total: {:.6}s", pass1_s);
    eprintln!("[optimized-oracle-superneo-perf] pass2 total: {:.6}s", pass2_s);
    eprintln!("[optimized-oracle-superneo-perf] pass2/pass1 ratio: {:.3}x", speedup);
}
