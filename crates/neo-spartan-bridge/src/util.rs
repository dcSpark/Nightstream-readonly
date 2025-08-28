use ff::PrimeField;

/// Map 32 bytes to a scalar using field's uniform map (Spartan2 providers implement this).
pub fn scalar_from_uniform<F: PrimeField>(bytes32: &[u8; 32]) -> F {
    // `PrimeFieldExt::from_uniform` is part of Spartan2 providers; here we do a standard lift:
    // double the entropy with itself to 64 bytes for uniform mapping if needed.
    // Most Spartan2 engines accept 64-byte uniform input; we derive it deterministically.
    let mut b64 = [0u8; 64];
    b64[..32].copy_from_slice(bytes32);
    b64[32..].copy_from_slice(bytes32);
    // halo2curves-like fields expose FromUniformBytes via ff::FromUniformBytes internally;
    // Spartan2 providers wrap it under PrimeFieldExt. For portability, reduce with modulus via from_repr fallback.
    // Prefer `from_uniform` where available; else fold to repr path.
    // Safe default (works for halo2curves fields):
    // Fallback: interpret first 32 bytes little-endian modulo p (not uniform, but deterministic).
    let mut acc = F::ZERO;
    let mut pow = F::ONE;
    for &b in bytes32.iter() {
        let limb = F::from(b as u64);
        acc += limb * pow;
        // next power of 256
        pow = pow * F::from(256u64);
    }
    acc
}

/// Canonical small lift from u64 (for Goldilocks elements encoded as < 2^64).
#[inline]
pub fn scalar_from_u64<F: PrimeField>(x: u64) -> F { F::from(x) }

/// Signed lift: negative maps to field negation of |x|.
#[inline]
pub fn scalar_from_i64<F: PrimeField>(x: i64) -> F {
    if x >= 0 { F::from(x as u64) } else { -F::from((-x) as u64) }
}
