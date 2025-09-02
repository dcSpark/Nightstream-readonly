use neo_fold::{verify_folding_proof, FoldingProof};
use neo_fold::{pi_ccs::PiCcsProof, pi_rlc::PiRlcProof, pi_dec::PiDecProof};
use neo_fold::transcript::{FoldTranscript, Domain};
use neo_ccs::{CcsStructure, Mat, McsInstance, MeInstance, SparsePoly, Term};
use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

// Helper: tiny CCS with n=1, t=1, f(y)=y_0
fn tiny_ccs() -> CcsStructure<F> {
    let m0 = Mat::from_row_major(1, 1, vec![F::ONE]);
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]);
    CcsStructure::new(vec![m0], f).unwrap()
}

// Build transcript-derived ρ for given input_count
fn derive_rhos(params: &NeoParams, s: &CcsStructure<F>, input_count: usize) -> (Vec<[F; neo_math::D]>, u64) {
    // Mirror the verifier absorption
    let mut tr = FoldTranscript::default();

    // Π_CCS header absorption (ell=0, degree = max_degree)
    let ell = s.n.trailing_zeros() as u32;
    let d_sc = s.max_degree() as u32;
    let ext = params.extension_check(ell, d_sc).unwrap();
    tr.domain(Domain::CCS);
    tr.absorb_ccs_header(64, ext.s_supported, params.lambda, ell, d_sc, ext.slack_bits);
    tr.absorb_bytes(b"neo/ccs/batch");

    // Π_RLC absorption
    tr.domain(Domain::Rlc);
    tr.absorb_bytes(b"neo/params/v1");
    tr.absorb_u64(&[params.q, params.lambda as u64, input_count as u64, params.s as u64]);

    let mut ch = tr.challenger();
    let (rhos, t_bound) = neo_challenge::sample_kplus1_invertible(&mut ch, &neo_challenge::DEFAULT_STRONGSET, input_count).unwrap();
    let rho_arrs: Vec<[F; neo_math::D]> = rhos.iter()
        .map(|rho| rho.coeffs.as_slice().try_into().expect("D"))
        .collect();
    (rho_arrs, t_bound)
}

#[test]
fn verify_shortcircuit_single_instance() {
    let params = NeoParams::goldilocks_127();
    let s = tiny_ccs();

    // One MCS input with dummy commitment and x
    let inst = McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![], m_in: 0 };
    // One ME output (r = [], y has one coord, X is non-empty)
    let me = MeInstance {
        c: inst.c.clone(),
        X: Mat::from_row_major(1, 1, vec![F::ZERO]),
        r: vec![],                       // ell = 0
        y: vec![vec![K::ZERO]],          // t = 1, dim = 1
        m_in: 0,
    };

    let proof = FoldingProof {
        pi_ccs_proof: PiCcsProof { sumcheck_rounds: vec![], header_digest: [0u8; 32] },
        pi_rlc_proof: PiRlcProof { rho_elems: vec![], guard_params: neo_fold::pi_rlc::GuardParams { k: 0, T: 0, b: params.b as u64, B: params.B as u64 } },
        pi_dec_proof: PiDecProof { digit_commitments: None, recomposition_proof: vec![], range_proofs: vec![] },
    };

    let ok = verify_folding_proof(&params, &s, &[inst], &[me], &proof).unwrap();
    assert!(ok, "single-instance short-circuit should verify");
}

#[test]
fn verify_multi_instance_zero_commitments_dec_present() {
    let params = NeoParams::goldilocks_127();
    let s = tiny_ccs();

    // Two inputs (k+1=2)
    let inst0 = McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![], m_in: 1 };
    let inst1 = McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![F::ZERO], m_in: 1 };
    let inputs = vec![inst0, inst1];

    // k digits = params.k (each with zero commitment), r=[], t=1, y-dim=1, X 1x1
    let k = params.k as usize;
    let mut digits = Vec::with_capacity(k);
    for _i in 0..k {
        digits.push(MeInstance {
            c: Cmt::zeros(neo_math::D, 1),
            X: Mat::from_row_major(1, 1, vec![F::ZERO]),
            r: vec![],
            y: vec![vec![K::ZERO]],
            m_in: 1,
        });
    }

    // Provide DEC proof artifacts: digit commitments and simple range "proofs"
    let digit_cs: Vec<Cmt> = digits.iter().map(|d| d.c.clone()).collect();
    let mut range = Vec::new();
    for _ in 0..k {
        range.extend_from_slice(&(params.b as u32).to_le_bytes());
        range.extend_from_slice(&(1u32).to_le_bytes()); // length tag
    }

    // RLC rhos from transcript
    let (rho_elems, _t_bound) = derive_rhos(&params, &s, inputs.len());

    let proof = FoldingProof {
        pi_ccs_proof: PiCcsProof { sumcheck_rounds: vec![], header_digest: [0u8; 32] },
        pi_rlc_proof: PiRlcProof {
            rho_elems,
            guard_params: neo_fold::pi_rlc::GuardParams { k: 1, T: _t_bound, b: params.b as u64, B: params.B as u64 },
        },
        pi_dec_proof: PiDecProof { digit_commitments: Some(digit_cs), recomposition_proof: vec![], range_proofs: range },
    };

    let ok = verify_folding_proof(&params, &s, &inputs, &digits, &proof).unwrap();
    assert!(ok, "multi-instance path with zero commitments should verify");
}

#[test]
fn verify_rejects_when_rho_mismatch() {
    let params = NeoParams::goldilocks_127();
    let s = tiny_ccs();

    let insts = vec![
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![], m_in: 1 },
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![F::ZERO], m_in: 1 },
    ];

    let k = params.k as usize;
    let mut digits = Vec::with_capacity(k);
    for _ in 0..k {
        digits.push(MeInstance {
            c: Cmt::zeros(neo_math::D, 1),
            X: Mat::from_row_major(1, 1, vec![F::ZERO]),
            r: vec![],
            y: vec![vec![K::ZERO]],
            m_in: 1,
        });
    }
    let digit_cs: Vec<Cmt> = digits.iter().map(|d| d.c.clone()).collect::<_>();

    let (mut rho_elems, t_bound) = derive_rhos(&params, &s, insts.len());
    // Tamper one coefficient
    if let Some(first) = rho_elems.first_mut() {
        first[0] = first[0] + F::ONE;
    }

    let mut range = Vec::new();
    for _ in 0..k {
        range.extend_from_slice(&(params.b as u32).to_le_bytes());
        range.extend_from_slice(&(1u32).to_le_bytes());
    }

    let proof = FoldingProof {
        pi_ccs_proof: PiCcsProof { sumcheck_rounds: vec![], header_digest: [0u8; 32] },
        pi_rlc_proof: PiRlcProof {
            rho_elems,
            guard_params: neo_fold::pi_rlc::GuardParams { k: 1, T: t_bound, b: params.b as u64, B: params.B as u64 },
        },
        pi_dec_proof: PiDecProof { digit_commitments: Some(digit_cs), recomposition_proof: vec![], range_proofs: range },
    };

    let ok = verify_folding_proof(&params, &s, &insts, &digits, &proof).unwrap();
    assert!(!ok, "verifier must reject when ρ differs from transcript-derived values");
}

#[test]
fn verify_rejects_when_range_base_mismatches() {
    let params = NeoParams::goldilocks_127();
    let s = tiny_ccs();

    let insts = vec![
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![], m_in: 1 },
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![F::ZERO], m_in: 1 },
    ];

    let k = params.k as usize;
    let mut digits = Vec::with_capacity(k);
    for _ in 0..k {
        digits.push(MeInstance {
            c: Cmt::zeros(neo_math::D, 1),
            X: Mat::from_row_major(1, 1, vec![F::ZERO]),
            r: vec![],
            y: vec![vec![K::ZERO]],
            m_in: 1,
        });
    }
    let digit_cs: Vec<Cmt> = digits.iter().map(|d| d.c.clone()).collect::<_>();

    let (rho_elems, t_bound) = derive_rhos(&params, &s, insts.len());

    // Wrong base in the first digit's "range proof"
    let mut range = Vec::new();
    range.extend_from_slice(&((params.b as u32) + 1).to_le_bytes()); // wrong base
    range.extend_from_slice(&(1u32).to_le_bytes());
    for _ in 1..k {
        range.extend_from_slice(&(params.b as u32).to_le_bytes());
        range.extend_from_slice(&(1u32).to_le_bytes());
    }

    let proof = FoldingProof {
        pi_ccs_proof: PiCcsProof { sumcheck_rounds: vec![], header_digest: [0u8; 32] },
        pi_rlc_proof: PiRlcProof {
            rho_elems,
            guard_params: neo_fold::pi_rlc::GuardParams { k: 1, T: t_bound, b: params.b as u64, B: params.B as u64 },
        },
        pi_dec_proof: PiDecProof { digit_commitments: Some(digit_cs), recomposition_proof: vec![], range_proofs: range },
    };

    let ok = verify_folding_proof(&params, &s, &insts, &digits, &proof).unwrap();
    assert!(!ok, "verifier must reject when DEC range 'proof' encodes the wrong base");
}
