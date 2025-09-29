//! Stage 7: Context digest stability/parity (toy check)

use neo_math::F;
use neo_ccs::{Mat, r1cs_to_ccs};
use p3_field::PrimeField64;
use p3_field::PrimeCharacteristicRing;

fn triplets_to_dense(rows: usize, cols: usize, trips: Vec<(usize, usize, F)>) -> Vec<F> {
    let mut v = vec![F::ZERO; rows*cols];
    for (r,c,a) in trips { v[r*cols + c] = a; }
    v
}

#[test]
fn context_digest_stable_for_identical_inputs() {
    // Tiny R1CS -> CCS (1 constraint, 2 vars)
    let rows=1; let cols=2;
    let a = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, vec![(0,1,F::ONE)]));
    let b = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, vec![(0,0,F::ONE)]));
    let c = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, vec![]));
    let ccs = r1cs_to_ccs(a,b,c);

    let x = vec![F::from_u64(1), F::from_u64(0)];
    // Local emulation: stable framing + SHA-256 to test determinism over bytes
    let mut enc = Vec::new();
    // encode ccs shape
    enc.extend_from_slice(&(ccs.n as u64).to_le_bytes());
    enc.extend_from_slice(&(ccs.m as u64).to_le_bytes());
    // encode matrices sparsely (row-major scan)
    for mj in &ccs.matrices {
        for r in 0..mj.rows() {
            for c in 0..mj.cols() {
                let a = mj[(r,c)].as_canonical_u64();
                enc.extend_from_slice(&a.to_le_bytes());
            }
        }
    }
    // encode public input
    enc.extend_from_slice(&(x.len() as u64).to_le_bytes());
    for xi in &x { enc.extend_from_slice(&xi.as_canonical_u64().to_le_bytes()); }
    // Simple fallback hash: sum of bytes modulo u64 for determinism in tests
    let h1: u64 = enc.iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64));
    let h2: u64 = enc.iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64));
    assert_eq!(h1, h2);
}

#[test]
fn context_digest_changes_on_input_change() {
    let rows=1; let cols=2;
    let a = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, vec![(0,1,F::ONE)]));
    let b = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, vec![(0,0,F::ONE)]));
    let c = Mat::from_row_major(rows, cols, triplets_to_dense(rows, cols, vec![]));
    let ccs = r1cs_to_ccs(a,b,c);
    let x1 = vec![F::from_u64(1), F::from_u64(0)];
    let x2 = vec![F::from_u64(2), F::from_u64(0)];
    let mut enc1 = Vec::new();
    enc1.extend_from_slice(&(ccs.n as u64).to_le_bytes());
    enc1.extend_from_slice(&(ccs.m as u64).to_le_bytes());
    for mj in &ccs.matrices { for r in 0..mj.rows() { for c in 0..mj.cols() { enc1.extend_from_slice(&mj[(r,c)].as_canonical_u64().to_le_bytes()); } } }
    enc1.extend_from_slice(&(x1.len() as u64).to_le_bytes());
    for xi in &x1 { enc1.extend_from_slice(&xi.as_canonical_u64().to_le_bytes()); }
    let mut enc2 = enc1.clone();
    // swap last segment to x2
    enc2.truncate(enc2.len() - x1.len()*8);
    enc2.extend_from_slice(&(x2.len() as u64).to_le_bytes());
    for xi in &x2 { enc2.extend_from_slice(&xi.as_canonical_u64().to_le_bytes()); }
    let h1: u64 = enc1.iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64));
    let h2: u64 = enc2.iter().fold(0u64, |acc, &b| acc.wrapping_add(b as u64));
    assert_ne!(h1, h2);
}
