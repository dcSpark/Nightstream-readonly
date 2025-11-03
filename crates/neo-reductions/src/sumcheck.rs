//! Sumcheck protocol interface

use neo_math::{K, KExtensions};
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;

/// Trait for round oracles in the sumcheck protocol
pub trait RoundOracle {
    /// Evaluate the oracle at multiple points for the current round
    fn evals_at(&mut self, points: &[K]) -> Vec<K>;
    
    /// Get the number of rounds in the sumcheck protocol
    fn num_rounds(&self) -> usize;
    
    /// Get the degree bound for each round
    fn degree_bound(&self) -> usize;
    
    /// Fold the oracle with the given challenge
    fn fold(&mut self, r: K);
    
    /// Alias for fold - bind to a specific value and advance to the next round
    fn bind(&mut self, r: K) {
        self.fold(r);
    }
}

/// Evaluate a polynomial (given as coefficients) at a point
pub fn poly_eval_k(coeffs: &[K], x: K) -> K {
    if coeffs.is_empty() {
        return K::ZERO;
    }
    // Horner's method: p(x) = c_0 + x*(c_1 + x*(c_2 + ...))
    let mut result = coeffs[coeffs.len() - 1];
    for &c in coeffs.iter().rev().skip(1) {
        result = result * x + c;
    }
    result
}

/// Verify sumcheck rounds against a transcript
///
/// Returns (challenges, running_sum, is_valid)
pub fn verify_sumcheck_rounds<Tr: Transcript>(
    tr: &mut Tr,
    degree_bound: usize,
    initial_sum: K,
    rounds: &[Vec<K>],
) -> (Vec<K>, K, bool) {
    let mut challenges = Vec::with_capacity(rounds.len());
    let mut running_sum = initial_sum;
    
    for (_i, round_poly) in rounds.iter().enumerate() {
        // Check degree bound
        if round_poly.len() > degree_bound + 1 {
            return (challenges, running_sum, false);
        }
        
        // Verify that round_poly(0) + round_poly(1) = running_sum
        let eval_0 = poly_eval_k(round_poly, K::ZERO);
        let eval_1 = poly_eval_k(round_poly, K::ONE);
        if eval_0 + eval_1 != running_sum {
            return (challenges, running_sum, false);
        }
        
        // Append round polynomial to transcript
        for &coeff in round_poly.iter() {
            tr.append_fields(b"sumcheck/round/coeff", &coeff.as_coeffs());
        }
        
        // Sample challenge for this round: extension field element
        // Sample 2 base field elements and combine them
        let c = tr.challenge_field(b"sumcheck/challenge/0");
        let d = tr.challenge_field(b"sumcheck/challenge/1");
        let challenge = neo_math::from_complex(c, d);
        challenges.push(challenge);
        
        // Update running sum: running_sum := round_poly(challenge)
        running_sum = poly_eval_k(round_poly, challenge);
    }
    
    (challenges, running_sum, true)
}

