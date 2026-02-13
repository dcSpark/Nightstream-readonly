use neo_ajtai::Commitment;
use neo_math::ring::D;
use serde_json::Value;

fn valid_commitment_json(kappa: usize) -> Value {
    let c = Commitment::zeros(D, kappa);
    serde_json::to_value(&c).expect("serialize valid commitment")
}

#[test]
fn commitment_deserialize_rejects_wrong_d() {
    let mut value = valid_commitment_json(2);
    value["d"] = serde_json::json!(D - 1);

    let err = serde_json::from_value::<Commitment>(value).expect_err("wrong d must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("invalid Commitment.d"), "unexpected error message: {msg}");
}

#[test]
fn commitment_deserialize_rejects_data_len_mismatch() {
    let mut value = valid_commitment_json(2);
    let data = value
        .get_mut("data")
        .and_then(Value::as_array_mut)
        .expect("commitment.data array");
    data.pop().expect("non-empty data");

    let err = serde_json::from_value::<Commitment>(value).expect_err("shape mismatch must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("data.len()"), "unexpected error message: {msg}");
}

#[test]
fn commitment_deserialize_accepts_valid_shape() {
    let value = valid_commitment_json(3);
    let c: Commitment = serde_json::from_value(value).expect("valid shape should deserialize");
    assert_eq!(c.d, D);
    assert_eq!(c.kappa, 3);
    assert_eq!(c.data.len(), D * 3);
}
