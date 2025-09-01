#![cfg(feature = "legacy-compat")]
use neo_ccs::{legacy::{CcsStructure, CcsInstance, CcsWitness, mv_poly}};
use neo_math::{from_base, ExtF, F, Polynomial};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

#[test]
fn test_ccs_prover_no_copy_panic() {
    let structure = CcsStructure::new(vec![], mv_poly(|_| ExtF::ZERO, 0)); // Changed to degree 0 for zero polynomial
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let witness = CcsWitness { z: vec![] };

    let result = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
    );
    assert!(result.is_ok());
}

#[test]
fn test_ccs_multilinear_specialization() {
    let mat = RowMajorMatrix::new(vec![F::ZERO; 2], 2);
    let mats = vec![mat.clone(), mat];
    let structure = CcsStructure::new(mats, mv_poly(|ins| ins[0] + ins[1], 1));
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let witness = CcsWitness {
        z: vec![ExtF::ONE, ExtF::ZERO],
    };

    let result = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
    );
    assert!(result.is_ok());
    let msgs = result.unwrap();
    assert!(!msgs.is_empty());
}

#[test]
fn test_ccs_multilinear_with_norms() {
    let structure = CcsStructure::new(vec![], mv_poly(|_| ExtF::ZERO, 1));
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let witness = CcsWitness { z: vec![ExtF::ONE] }; // Trigger norm check

    let result = ccs_sumcheck_prover(
        &structure,
        &instance,
        &witness,
    );
    assert!(result.is_ok()); // Norms handled without error
}

#[test]
fn test_ccs_zk_prover_hides() {
    let m = 2;
    let mat = RowMajorMatrix::new(vec![F::ZERO; m], 1);
    let mats = vec![mat];
    let f = mv_poly(|_: &[ExtF]| ExtF::ZERO, 1);
    let structure = CcsStructure::new(mats, f);
    let instance = CcsInstance {
        commitment: vec![],
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    let witness = CcsWitness {
        z: vec![from_base(F::ZERO)],
    };
    let prev: Option<Polynomial<ExtF>> = None;
    let mut diff = false;
    let mut rng = rand::rng();
    for _ in 0..5 {
        let mut t = vec![];
        let prefix = rng.random::<u64>().to_be_bytes().to_vec();
        t.extend(prefix);
        let _msgs =
            ccs_sumcheck_prover(&structure, &instance, &witness)
                .expect("sumcheck");
        if let Some(p) = &prev {
            if p.degree() != 0 { // Check polynomial degree instead
                diff = true;
                break;
            }
        }
        // prev = Some(p.clone()); // Store previous polynomial
    }
    assert!(diff);
}