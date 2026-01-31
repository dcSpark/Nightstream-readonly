use neo_math::{KExtensions, F, K};
use neo_memory::identity::shout_oracle::IdentityAddressLookupOracleSparse;
use neo_memory::sparse_time::SparseIdxVec;
use neo_memory::twist_oracle::AddressLookupOracle;
use neo_reductions::sumcheck::{run_batched_sumcheck_prover, verify_batched_sumcheck_rounds, BatchedClaim};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn k(v: u64) -> K {
    K::from(F::from_u64(v))
}

fn build_sparse_lookup_lane(
    pow2_cycle: usize,
    ell_addr: usize,
    lookups: &[(usize, u64)],
) -> (Vec<SparseIdxVec<K>>, SparseIdxVec<K>) {
    assert!(pow2_cycle.is_power_of_two());

    let mut has_entries = Vec::with_capacity(lookups.len());
    for &(t, _addr) in lookups {
        has_entries.push((t, K::ONE));
    }
    let has_lookup = SparseIdxVec::from_entries(pow2_cycle, has_entries);

    let mut addr_bits = Vec::with_capacity(ell_addr);
    for bit in 0..ell_addr {
        let mut entries = Vec::new();
        for &(t, addr) in lookups {
            if ((addr >> bit) & 1) == 1 {
                entries.push((t, K::ONE));
            }
        }
        addr_bits.push(SparseIdxVec::from_entries(pow2_cycle, entries));
    }

    (addr_bits, has_lookup)
}

#[test]
fn identity_shout_addr_oracle_sparse_matches_dense_for_small_domain() {
    let ell_addr = 8usize;
    let table: Vec<K> = (0u64..(1u64 << ell_addr)).map(k).collect();

    let r_cycle: Vec<K> = vec![k(3), k(5), k(7)];
    let pow2_cycle = 1usize << r_cycle.len();

    // Two lookups in a single lane, including a repeated address (weights must accumulate).
    let lookups: Vec<(usize, u64)> = vec![(1, 0xA5), (4, 0xA5)];
    let (addr_bits, has_lookup) = build_sparse_lookup_lane(pow2_cycle, ell_addr, &lookups);

    let (mut dense, dense_sum) = AddressLookupOracle::new(&addr_bits, &has_lookup, &table, &r_cycle, ell_addr);
    let (mut sparse, sparse_sum) =
        IdentityAddressLookupOracleSparse::new_sparse_time(ell_addr, &addr_bits, &has_lookup, &r_cycle)
            .expect("construct sparse oracle");

    assert_eq!(dense_sum, sparse_sum, "claimed sums must match");
    assert_ne!(dense_sum, K::ZERO, "sanity: nonzero claim");

    let mut tr_p = Poseidon2Transcript::new(b"test/identity/shout/oracle_sparse_vs_dense");
    tr_p.append_message(b"ell_addr", &(ell_addr as u64).to_le_bytes());
    tr_p.append_fields(b"claimed_sum/dense", &dense_sum.as_coeffs());
    tr_p.append_fields(b"claimed_sum/sparse", &sparse_sum.as_coeffs());
    let mut tr_v = tr_p.clone();

    let mut claims = vec![
        BatchedClaim {
            oracle: &mut dense,
            claimed_sum: dense_sum,
            label: b"shout/addr_pre",
        },
        BatchedClaim {
            oracle: &mut sparse,
            claimed_sum: sparse_sum,
            label: b"shout/addr_pre",
        },
    ];

    let (r_addr_p, per_claim_results) =
        run_batched_sumcheck_prover(&mut tr_p, claims.as_mut_slice()).expect("prover should succeed");
    let per_claim_rounds: Vec<Vec<Vec<K>>> = per_claim_results
        .iter()
        .map(|r| r.round_polys.clone())
        .collect();

    let claimed_sums = vec![dense_sum, sparse_sum];
    let labels: Vec<&[u8]> = vec![b"shout/addr_pre".as_slice(); 2];
    let degree_bounds: Vec<usize> = vec![2; 2];

    let (r_addr_v, finals, ok) =
        verify_batched_sumcheck_rounds(&mut tr_v, &per_claim_rounds, &claimed_sums, &labels, &degree_bounds);
    assert!(ok, "verifier should accept");
    assert_eq!(r_addr_v, r_addr_p, "verifier must derive the same r_addr");
    assert_eq!(finals.len(), 2);
    assert_eq!(finals[0], finals[1], "final evaluations must match");
}
