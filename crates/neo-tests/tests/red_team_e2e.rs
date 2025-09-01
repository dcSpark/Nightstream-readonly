// neo-tests/tests/red_team_e2e.rs
#![cfg_attr(debug_assertions, allow(unused))]
use neo_orchestrator::{prove_single, verify_single};
use neo_ccs::{Mat, r1cs::r1cs_to_ccs};
use p3_goldilocks::Goldilocks as F;
use p3_field::PrimeCharacteristicRing;

fn tiny_ccs() -> neo_ccs::CcsStructure<F> {
    // 1-row: (z0 - z1) * 1 = 0  → forces z0 = z1
    let a = Mat::from_row_major(1, 2, vec![F::ONE, -F::ONE]);
    let b = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let c = Mat::from_row_major(1, 2, vec![F::ZERO, F::ZERO]);
    r1cs_to_ccs(a, b, c)
}

#[ignore = "enable after real SNARK verifier is wired"]
#[test]
fn e2e_rejects_tampered_proof() {
    type Inst = neo_ccs::McsInstance<Vec<u8>, F>;
    type Wit  = neo_ccs::McsWitness<F>;

    let ccs = tiny_ccs();

    // witness: [1, x], choose x=5 → satisfies (z0 - z1) = 0 with z0=z1
    let w = vec![F::ONE, F::from_u64(5)];
    let d = neo_math::ring::D;
    let decomp = neo_ajtai::decomp_b(&w, 2, d, neo_ajtai::DecompStyle::Balanced);

    // minimal Ajtai commitment bytes
    let pp = neo_ajtai::setup(&mut rand::rngs::StdRng::from_seed([3u8;32]), d, 8, w.len());
    let c  = neo_ajtai::commit(&pp, &decomp);
    let mut c_bytes = Vec::new();
    c_bytes.extend_from_slice(&c.d.to_le_bytes());
    c_bytes.extend_from_slice(&c.kappa.to_le_bytes());
    for x in &c.data { c_bytes.extend_from_slice(&x.as_canonical_u64().to_le_bytes()); }

    let inst = Inst { c: c_bytes, x: vec![], m_in: 0 };
    let wit  = Wit  { w: w.clone(), Z: Mat::from_row_major(d, w.len(), decomp) };

    // produce proof
    let (proof, _metrics) = prove_single(&ccs, &inst, &wit).expect("prove");

    // tamper: flip one byte
    let mut forged = proof.clone();
    if !forged.is_empty() { forged[forged.len()/2] ^= 1; }

    assert!(!verify_single(&ccs, &inst, &forged), "tampered proof must be rejected");
}
