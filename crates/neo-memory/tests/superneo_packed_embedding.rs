use neo_math::{D, F, K};
use neo_memory::ajtai::{
    commit_cols_for_ccs_m, decode_vector_for_ccs_m, decode_vector_superneo_packed_from_mat, encode_vector_for_ccs_m,
    encode_vector_superneo_packed_to_mat, validate_packed_vector_nc_range,
};
use neo_params::NeoParams;
use neo_reductions::common::decode_z_from_witness_mat;
use p3_field::PrimeCharacteristicRing;

#[test]
fn superneo_packed_encode_decode_roundtrip_and_reductions_decode() {
    let params = NeoParams::goldilocks_127();
    let z: Vec<F> = (0..(2 * D))
        .map(|i| F::from_u64(10_000u64 + i as u64))
        .collect();

    let packed = encode_vector_superneo_packed_to_mat(&z).expect("superneo packed encode");
    assert_eq!(packed.rows(), D);
    assert_eq!(packed.cols(), 2);

    let roundtrip = decode_vector_superneo_packed_from_mat(&packed).expect("superneo packed decode");
    assert_eq!(roundtrip, z, "packed encode/decode must be identity");

    let z_k = decode_z_from_witness_mat(&params, &packed, z.len()).expect("reductions decode");
    for (idx, v) in z.iter().enumerate() {
        assert_eq!(z_k[idx], K::from(*v), "reductions decode mismatch at idx={idx}");
    }
}

#[test]
fn superneo_packed_encode_rejects_non_multiple_of_d() {
    let z: Vec<F> = (0..(D + 1)).map(|i| F::from_u64(i as u64)).collect();
    let err = encode_vector_superneo_packed_to_mat(&z).expect_err("shape must be rejected");
    assert!(err.contains("not divisible by D"));
}

#[test]
fn superneo_packed_decode_rejects_wrong_row_count() {
    let bad = neo_ccs::Mat::zero(D - 1, 3, F::ZERO);
    let err = decode_vector_superneo_packed_from_mat(&bad).expect_err("wrong rows must fail");
    assert!(err.contains("expected D"));
}

#[test]
fn encode_vector_for_ccs_m_selects_layout_by_width() {
    assert_eq!(commit_cols_for_ccs_m(2 * D), 2, "packed width should compress by D");
    assert_eq!(
        commit_cols_for_ccs_m(D + 1),
        2,
        "non-divisible width should use packed ceil(m/D) columns"
    );

    let params = NeoParams::goldilocks_127();

    let z_packed: Vec<F> = (0..(2 * D))
        .map(|i| match i % 3 {
            0 => -F::ONE,
            1 => F::ZERO,
            _ => F::ONE,
        })
        .collect();
    let packed = encode_vector_for_ccs_m(&params, z_packed.len(), &z_packed).expect("packed encode");
    assert_eq!(packed.rows(), D);
    assert_eq!(packed.cols(), 2);
    let packed_roundtrip = decode_vector_superneo_packed_from_mat(&packed).expect("packed decode");
    assert_eq!(packed_roundtrip, z_packed);

    let z_nondiv: Vec<F> = (0..(D + 1)).map(|i| F::from_u64(i as u64 + 11)).collect();
    let nondiv_mat = encode_vector_for_ccs_m(&params, z_nondiv.len(), &z_nondiv).expect("non-divisible encode");
    assert_eq!(nondiv_mat.rows(), D);
    assert_eq!(nondiv_mat.cols(), 2);
    let nondiv_roundtrip = decode_vector_for_ccs_m(&params, z_nondiv.len(), &nondiv_mat).expect("non-divisible decode");
    assert_eq!(nondiv_roundtrip, z_nondiv);
}

#[test]
fn validate_packed_vector_nc_range_rejects_large_coeffs() {
    let mut params = NeoParams::goldilocks_127();
    params.b = 2;

    let z_ok: Vec<F> = (0..(2 * D))
        .map(|i| match i % 3 {
            0 => -F::ONE,
            1 => F::ZERO,
            _ => F::ONE,
        })
        .collect();
    validate_packed_vector_nc_range(&params, z_ok.len(), &z_ok, "test")
        .expect("small packed coefficients must pass NC range check");

    let mut z_bad = z_ok.clone();
    z_bad[7] = F::from_u64(1u64 << 60);
    let err = validate_packed_vector_nc_range(&params, z_bad.len(), &z_bad, "test")
        .expect_err("large packed coefficient must fail");
    assert!(err.contains("not representable"));
}

#[test]
fn encode_vector_for_ccs_m_rejects_out_of_range_packed_coeffs() {
    let mut params = NeoParams::goldilocks_127();
    params.b = 2;

    let mut z_bad: Vec<F> = vec![F::ZERO; 2 * D];
    z_bad[3] = F::from_u64(1u64 << 60);

    let err = encode_vector_for_ccs_m(&params, z_bad.len(), &z_bad)
        .expect_err("packed encode must reject out-of-range coefficients");
    assert!(err.contains("not representable"));
}
