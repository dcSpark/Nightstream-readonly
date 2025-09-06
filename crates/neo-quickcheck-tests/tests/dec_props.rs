#![cfg(feature = "quickcheck")]
//! Base‑b split/recombine properties over F and K + simple range polynomial check for b=2.

#![allow(deprecated)]

use proptest::prelude::*;
use neo_math::{F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn arb_f_vec(len: usize) -> impl Strategy<Value = Vec<F>> {
    prop::collection::vec(any::<u64>().prop_map(F::from_u64), len)
}

fn arb_k_vec(len: usize) -> impl Strategy<Value = Vec<K>> {
    prop::collection::vec(any::<u64>().prop_map(|u| K::from(F::from_u64(u))), len)
}

proptest! {
    // F: recomposition succeeds and a single flip breaks it
    #[test]
    fn recomposition_f_roundtrip_and_neg(
        parent in prop::collection::vec(arb_f_vec(1), 1..5).prop_flat_map(|mut vs| {
            if vs.is_empty() { vs.push(vec![F::ZERO]); }
            let m = vs[0].len();
            (Just(vs[0].clone()), 1..5usize, 2..7u64, Just(m))
        }).prop_map(|(parent, k, base, m)| (parent, k, base, m))
    ) {
        let (parent, k, base, m) = parent;
        
        // Create simple child limbs where first digit has the parent values
        let mut limbs = vec![vec![F::ZERO; m]; k];
        if k > 0 {
            limbs[0] = parent.clone();
        }
        
        prop_assert!(
            neo_fold::pi_dec::verify_recomposition_f(
                F::from_u64(base), 
                &parent, 
                &limbs
            )
        );

        // Negative: if m>0, flip one limb and expect failure  
        if m > 0 && k > 0 {
            let mut bad = limbs.clone();
            bad[0][0] += F::ONE;
            prop_assert!(
                !neo_fold::pi_dec::verify_recomposition_f(
                    F::from_u64(base), 
                    &parent, 
                    &bad
                )
            );
        }
    }

    // K: recomposition succeeds and a single flip breaks it (K elements embedded from F)
    #[test]
    fn recomposition_k_roundtrip_and_neg(
        parent in prop::collection::vec(arb_k_vec(1), 1..5).prop_flat_map(|mut vs| {
            if vs.is_empty() { vs.push(vec![K::ZERO]); }
            let m = vs[0].len();
            (Just(vs[0].clone()), 1..5usize, 2..7u64, Just(m))
        }).prop_map(|(parent, k, base, m)| (parent, k, base, m))
    ) {
        let (parent, k, base, m) = parent;
        
        // Create simple child limbs where first digit has the parent values
        let mut limbs = vec![vec![K::ZERO; m]; k];
        if k > 0 {
            limbs[0] = parent.clone();
        }

        prop_assert!(
            neo_fold::pi_dec::verify_recomposition_k(
                F::from_u64(base), 
                &parent, 
                &limbs
            )
        );

        if m > 0 && k > 0 {
            let mut bad = limbs.clone();
            bad[0][0] += K::from(F::ONE);
            prop_assert!(
                !neo_fold::pi_dec::verify_recomposition_k(
                    F::from_u64(base), 
                    &parent, 
                    &bad
                )
            );
        }
    }
}

// ---- Range polynomial for b=2: v(v-1)(v+1)=0 ⇔ v∈{-1,0,1} over F ----

use quickcheck_macros::quickcheck;

fn range_poly_b2(v: F) -> F {
    v * (v - F::ONE) * (v + F::ONE)
}

#[quickcheck]
fn b2_range_poly_vanishes_on_digits(z_input: i8) -> bool {
    // Use i8 to avoid overflow issues and limit the range
    let z = z_input as i64;
    
    // Map a small integer into F
    let v = if z >= 0 { 
        F::from_u64(z as u64) 
    } else { 
        // Safely handle negative numbers without overflow
        let abs_z = z.unsigned_abs();
        -F::from_u64(abs_z)
    };
    let zero = range_poly_b2(v).as_canonical_u64() == 0;
    // Check iff z ∈ {-1,0,1} (mod p this matches the intended subset for small |z|)
    zero == (z == -1 || z == 0 || z == 1)
}
