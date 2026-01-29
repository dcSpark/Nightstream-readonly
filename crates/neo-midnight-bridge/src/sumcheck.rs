use crate::goldilocks::OuterScalar;
use crate::k_field::{assert_k_eq, k_add_mod_var, k_mul_mod_var, k_sum_mod_var, KVar};
use midnight_proofs::circuit::Layouter;
use midnight_proofs::plonk::Error;
use midnight_zk_stdlib::ZkStdLib;

pub fn sumcheck_round_check(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    coeffs: &[KVar],
    claimed_sum: &KVar,
) -> Result<(), Error> {
    if coeffs.is_empty() {
        return Err(Error::Synthesis("sumcheck_round_check: empty coeffs".into()));
    }

    // p(0) + p(1) == c0 + (sum_i c_i) == (sum_i c_i) + c0
    // We compute this as a single modular reduction by summing `c0` twice.
    let mut sum_terms = Vec::with_capacity(coeffs.len() + 1);
    sum_terms.extend_from_slice(coeffs);
    sum_terms.push(coeffs[0].clone());
    let sum = k_sum_mod_var(std, layouter, &sum_terms)?;
    assert_k_eq(std, layouter, &sum, claimed_sum)?;
    Ok(())
}

pub fn sumcheck_eval_horner(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    coeffs: &[KVar],
    challenge: &KVar,
    delta: u64,
) -> Result<KVar, Error> {
    if coeffs.is_empty() {
        return Err(Error::Synthesis("sumcheck_eval_horner: empty coeffs".into()));
    }

    let mut acc = coeffs[coeffs.len() - 1].clone();
    for c in coeffs[..coeffs.len() - 1].iter().rev() {
        let tmp = k_mul_mod_var(std, layouter, &acc, challenge, delta)?;
        acc = k_add_mod_var(std, layouter, &tmp, c)?;
    }
    Ok(acc)
}
