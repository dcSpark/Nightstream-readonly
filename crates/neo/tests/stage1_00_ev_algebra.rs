//! Stage 1: Algebra-only unit tests (no Spartan), quick sanity for EV algebra.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn commit_evo_vector_algebra_holds() {
    // Small vectors to keep it fast and readable
    let n = 8usize;
    let rho = F::from_u64(7);
    let mut c_prev = Vec::with_capacity(n);
    let mut c_step = Vec::with_capacity(n);
    for i in 0..n as u64 {
        c_prev.push(F::from_u64(100 + i));
        c_step.push(F::from_u64(3 * i + 1));
    }
    let mut c_next = vec![F::ZERO; n];
    for i in 0..n { c_next[i] = c_prev[i] + rho * c_step[i]; }

    for i in 0..n {
        assert_eq!(c_next[i], c_prev[i] + rho * c_step[i], "commit-evo mismatch at {}", i);
    }
}

