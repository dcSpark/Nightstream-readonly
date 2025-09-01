use ff::PrimeField;

/// Map 32 bytes to a scalar using a uniform distribution.
/// 
/// # Security Warning
/// This function MUST produce a uniformly random field element to avoid bias
/// that could compromise cryptographic security. The current implementation
/// is NOT uniform and is only suitable for testing.
pub fn scalar_from_uniform<F: PrimeField>(bytes32: &[u8; 32]) -> F {
    #[cfg(feature = "uniform-map")]
    {
        // Use proper uniform map when available
        F::from_uniform_bytes(bytes32)
    }
    
    #[cfg(not(feature = "uniform-map"))]
    {
        // CRITICAL: Refuse to produce biased scalars in release builds
        #[cfg(not(debug_assertions))]
        panic!("SECURITY ERROR: Uniform scalar map not available. Enable 'uniform-map' feature or use debug build only.");
        
        // DEV-ONLY: Biased fallback for testing (NEVER use in production)
        #[cfg(debug_assertions)]
        {
            eprintln!("WARNING: Using biased scalar derivation - FOR TESTING ONLY");
            let mut acc = F::ZERO;
            let mut pow = F::ONE;
            for &b in bytes32.iter() {
                let limb = F::from(b as u64);
                acc += limb * pow;
                pow = pow * F::from(256u64);
            }
            acc
        }
    }
}

/// Canonical small lift from u64 (for Goldilocks elements encoded as < 2^64).
#[inline]
pub fn scalar_from_u64<F: PrimeField>(x: u64) -> F { F::from(x) }

/// Signed lift: negative maps to field negation of |x|.
#[inline]
pub fn scalar_from_i64<F: PrimeField>(x: i64) -> F {
    if x >= 0 { F::from(x as u64) } else { -F::from((-x) as u64) }
}
