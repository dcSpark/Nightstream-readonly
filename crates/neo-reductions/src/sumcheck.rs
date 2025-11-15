//! Sumcheck protocol interface

use neo_math::{K, KExtensions};
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;

/// Format K value compactly for logging
#[cfg(feature = "debug-logs")]
fn format_k(k: &K) -> String {
    use p3_field::PrimeField64;
    let coeffs = k.as_coeffs();
    format!("K[{}, {}]", coeffs[0].as_canonical_u64(), coeffs[1].as_canonical_u64())
}

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
    
    #[cfg(feature = "debug-logs")]
    eprintln!("VERIFIER: Starting sumcheck with initial_sum={}", format_k(&initial_sum));
    
    for (i, round_poly) in rounds.iter().enumerate() {
        // Check degree bound
        if round_poly.len() > degree_bound + 1 {
            eprintln!("Round {} failed: degree check. len={}, degree_bound={}", i, round_poly.len(), degree_bound);
            return (challenges, running_sum, false);
        }
        
        // Verify that round_poly(0) + round_poly(1) = running_sum
        let eval_0 = poly_eval_k(round_poly, K::ZERO);
        let eval_1 = poly_eval_k(round_poly, K::ONE);
        
        #[cfg(feature = "debug-logs")]
        if i <= 1 {
            eprintln!("VERIFIER Round {}:", i);
            eprintln!("  Received {} coefficients", round_poly.len());
            if i == 0 {
                eprintln!("  coeffs=[{}]", round_poly.iter().map(format_k).collect::<Vec<_>>().join(", "));
            }
            eprintln!("  eval_0={}, eval_1={}, sum={}", format_k(&eval_0), format_k(&eval_1), format_k(&(eval_0 + eval_1)));
            eprintln!("  expected running_sum={}", format_k(&running_sum));
        }
        
        if eval_0 + eval_1 != running_sum {
            eprintln!("Round {} failed: invariant check. eval_0={:?}, eval_1={:?}, sum={:?}, running_sum={:?}", 
                      i, eval_0, eval_1, eval_0 + eval_1, running_sum);
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
        
        #[cfg(feature = "debug-logs")]
        if i <= 1 {
            eprintln!("  challenge={}", format_k(&challenge));
            eprintln!("  new_running_sum={}", format_k(&running_sum));
        }
    }
    
    (challenges, running_sum, true)
}

