use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_memory::ajtai::{decode_vector_for_ccs_m, encode_vector_for_ccs_m};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn decode_to_k_padded(params: &NeoParams, ccs_m: usize, mat: &Mat<F>, pow2_len: usize) -> Result<Vec<K>, String> {
    let mut out: Vec<K> = decode_vector_for_ccs_m(params, ccs_m, mat)?
        .into_iter()
        .map(K::from)
        .collect();
    out.resize(pow2_len, K::ZERO);
    Ok(out)
}

#[test]
fn decode_mat_to_k_padded_superneo_shape_roundtrips_with_padding() {
    let params = NeoParams::goldilocks_127();
    let m = 2 * D;
    let z: Vec<F> = (0..m)
        .map(|i| match i % 3 {
            0 => -F::ONE,
            1 => F::ZERO,
            _ => F::ONE,
        })
        .collect();

    let z_mat = encode_vector_for_ccs_m(&params, m, &z).expect("encode packed witness");
    let pow2_len = m.next_power_of_two();
    let got = decode_to_k_padded(&params, m, &z_mat, pow2_len).expect("decode packed witness");

    assert_eq!(got.len(), pow2_len);
    for (idx, &v) in z.iter().enumerate() {
        assert_eq!(got[idx], K::from(v), "decoded prefix mismatch at idx={idx}");
    }
    for (idx, v) in got.iter().enumerate().skip(m) {
        assert_eq!(*v, K::ZERO, "decoded padding must be zero at idx={idx}");
    }
}

#[test]
fn decode_mat_to_k_padded_nondiv_shape_roundtrips_with_padding() {
    let params = NeoParams::goldilocks_127();
    let m = D + 1;
    let z: Vec<F> = (0..m).map(|i| F::from_u64(100 + i as u64)).collect();

    let z_mat = encode_vector_for_ccs_m(&params, m, &z).expect("encode packed witness");
    let pow2_len = m.next_power_of_two();
    let got = decode_to_k_padded(&params, m, &z_mat, pow2_len).expect("decode packed witness");

    assert_eq!(got.len(), pow2_len);
    for (idx, &v) in z.iter().enumerate() {
        assert_eq!(got[idx], K::from(v), "decoded prefix mismatch at idx={idx}");
    }
    for (idx, v) in got.iter().enumerate().skip(m) {
        assert_eq!(*v, K::ZERO, "decoded padding must be zero at idx={idx}");
    }
}

#[test]
fn decode_mat_to_k_padded_rejects_wrong_shape_for_superneo_width() {
    let params = NeoParams::goldilocks_127();
    let m = D;

    // Wrong shape for SuperNeo-compatible width: should be D×(m/D)=D×1, not D×m.
    let bad = Mat::zero(D, m, F::ZERO);
    let err = decode_to_k_padded(&params, m, &bad, m.next_power_of_two())
        .expect_err("decode must reject wrong witness shape");
    assert!(err.contains("packed layout expects cols=1"), "unexpected error: {err}");
}
