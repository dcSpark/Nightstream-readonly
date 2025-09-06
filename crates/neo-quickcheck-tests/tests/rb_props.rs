#![cfg(feature = "quickcheck")]
//! QuickCheck: \sum_j w_j(r) == 1 for tensor weights over the Boolean hypercube.
//! Basic MLE sanity that underpins sum-check and evaluation identities.

use quickcheck_macros::quickcheck;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

fn tensor_rb(r: &[K]) -> Vec<K> {
    let ell = r.len();
    let n = 1usize << ell;
    let mut rb = vec![K::ONE; n];
    for j in 0..n {
        let mut w = K::ONE;
        for i in 0..ell {
            let bit = (j >> i) & 1;
            w *= if bit == 1 { r[i] } else { K::ONE - r[i] };
        }
        rb[j] = w;
    }
    rb
}

fn clamp(x: u8, lo: usize, hi: usize) -> usize {
    let span = hi - lo + 1;
    lo + (x as usize % span)
}

#[quickcheck]
fn rb_is_partition_of_unity(ell_raw: u8, rs: Vec<u64>) -> bool {
    let ell = clamp(ell_raw, 1, 5);       // small hypercubes
    let mut r = Vec::with_capacity(ell);
    for i in 0..ell {
        let u = *rs.get(i).unwrap_or(&1);
        r.push(K::from(F::from_u64(u)));
    }
    let rb = tensor_rb(&r);
    let sum = rb.into_iter().fold(K::ZERO, |acc, w| acc + w);
    sum == K::ONE
}
