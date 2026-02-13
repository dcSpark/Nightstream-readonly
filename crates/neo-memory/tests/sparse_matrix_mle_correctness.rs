use neo_math::K;
use neo_memory::mle::chi_at_index;
use neo_memory::sparse_matrix::{SparseMat, SparseMatEntry};
use p3_field::PrimeCharacteristicRing;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn sparse_matrix_mle_eval_by_folding_matches_dense() {
    let mut rng = ChaCha8Rng::seed_from_u64(12345);

    let ell_row = 3usize;
    let ell_col = 4usize;
    let row_len = 1usize << ell_row;
    let col_len = 1usize << ell_col;

    for _trial in 0..50usize {
        // Random sparse entries with possible duplicates.
        let nnz = rng.random_range(0..=40usize);
        let mut entries: Vec<SparseMatEntry<K>> = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            let row = rng.random_range(0..row_len) as u64;
            let col = rng.random_range(0..col_len) as u64;
            let value = K::from_u64(rng.random::<u64>());
            entries.push(SparseMatEntry { row, col, value });
        }

        // Dense materialization (with duplicate summation).
        let mut dense = vec![K::ZERO; row_len * col_len];
        for e in entries.iter() {
            dense[e.row as usize * col_len + e.col as usize] += e.value;
        }

        let sparse = SparseMat::from_entries(ell_row, ell_col, entries);

        // Random evaluation point.
        let r_row: Vec<K> = (0..ell_row)
            .map(|_| K::from_u64(rng.random::<u64>()))
            .collect();
        let r_col: Vec<K> = (0..ell_col)
            .map(|_| K::from_u64(rng.random::<u64>()))
            .collect();

        // Dense MLE eval: Σ_{i,j} M[i,j] χ_r_row(i) χ_r_col(j).
        let mut expected = K::ZERO;
        for row in 0..row_len {
            let chi_r = chi_at_index(&r_row, row);
            if chi_r == K::ZERO {
                continue;
            }
            for col in 0..col_len {
                let v = dense[row * col_len + col];
                if v == K::ZERO {
                    continue;
                }
                expected += v * chi_r * chi_at_index(&r_col, col);
            }
        }

        let got = sparse
            .mle_eval_by_folding(&r_row, &r_col)
            .expect("mle_eval_by_folding");
        let got_direct = sparse
            .mle_eval_direct(&r_row, &r_col)
            .expect("mle_eval_direct");
        assert_eq!(got, expected);
        assert_eq!(got_direct, expected);
    }
}
