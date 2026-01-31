use neo_math::{F, K};
use neo_memory::identity::eval_identity_mle_le;
use neo_memory::mle::chi_at_index;
use p3_field::PrimeCharacteristicRing;

fn k(v: u64) -> K {
    K::from(F::from_u64(v))
}

#[test]
fn identity_mle_matches_bruteforce_small_ell() {
    for ell in [1usize, 2, 4, 8, 12] {
        let pow2 = 1usize << ell;

        for case in 0u64..4 {
            let r_addr: Vec<K> = (0..ell).map(|i| k(1 + case * 13 + i as u64 * 7)).collect();

            let fast = eval_identity_mle_le(&r_addr);

            let mut brute = K::ZERO;
            for a in 0..pow2 {
                brute += k(a as u64) * chi_at_index(&r_addr, a);
            }

            assert_eq!(fast, brute, "ell={ell}, case={case}");
        }
    }
}
