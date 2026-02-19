//! Goldilocks field arithmetic for Nightstream guests.
//!
//! Field: p = 2^64 - 2^32 + 1 = 0xFFFF_FFFF_0000_0001
//!
//! On RISC-V targets, `gl_mul`, `gl_add`, and `gl_sub` use ECALL precompiles
//! (3 ECALLs each: 1 compute + 2 reads for the 64-bit result). On other targets
//! the software implementations are used.

#![allow(dead_code)]

/// Goldilocks prime: p = 2^64 - 2^32 + 1.
pub const GL_P: u64 = 0xFFFF_FFFF_0000_0001;

/// A Goldilocks field element digest: 4 elements (32 bytes).
pub type GlDigest = [u64; 4];

// ---------------------------------------------------------------------------
// ECALL-based implementations (RISC-V only)
// ---------------------------------------------------------------------------

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const GL_MUL_ECALL_NUM: u32 = 0x474C4D;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const GL_ADD_ECALL_NUM: u32 = 0x474C41;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const GL_SUB_ECALL_NUM: u32 = 0x474C53;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const GL_READ_ECALL_NUM: u32 = 0x80474C52;

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline]
fn gl_ecall_compute(ecall_id: u32, a: u64, b: u64) {
    let a_lo = a as u32;
    let a_hi = (a >> 32) as u32;
    let b_lo = b as u32;
    let b_hi = (b >> 32) as u32;
    unsafe {
        core::arch::asm!(
            "ecall",
            in("a0") ecall_id,
            in("a1") a_lo,
            in("a2") a_hi,
            in("a3") b_lo,
            in("a4") b_hi,
            options(nostack),
        );
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline]
fn gl_ecall_read() -> u32 {
    let result: u32;
    unsafe {
        core::arch::asm!(
            "ecall",
            inout("a0") GL_READ_ECALL_NUM => result,
            options(nostack),
        );
    }
    result
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline]
fn gl_ecall_op(ecall_id: u32, a: u64, b: u64) -> u64 {
    gl_ecall_compute(ecall_id, a, b);
    let lo = gl_ecall_read() as u64;
    let hi = gl_ecall_read() as u64;
    lo | (hi << 32)
}

// ---------------------------------------------------------------------------
// Software fallbacks (non-RISC-V)
// ---------------------------------------------------------------------------

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
#[inline]
fn reduce128(x: u128) -> u64 {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let hi_times_correction = (hi as u128) * 0xFFFF_FFFFu128;
    let sum = lo as u128 + hi_times_correction;
    let lo2 = sum as u64;
    let hi2 = (sum >> 64) as u64;
    let (mut result, overflow) = lo2.overflowing_add(hi2.wrapping_mul(0xFFFF_FFFF));
    if overflow || result >= GL_P {
        result = result.wrapping_sub(GL_P);
    }
    result
}

/// Additive identity.
pub const GL_ZERO: u64 = 0;

/// Multiplicative identity.
pub const GL_ONE: u64 = 1;

/// Field addition: (a + b) mod p.
#[inline]
pub fn gl_add(a: u64, b: u64) -> u64 {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    { gl_ecall_op(GL_ADD_ECALL_NUM, a, b) }

    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    {
        let sum = a as u128 + b as u128;
        let s = sum as u64;
        let carry = (sum >> 64) as u64;
        let (mut r, overflow) = s.overflowing_add(carry.wrapping_mul(0xFFFF_FFFF));
        if overflow || r >= GL_P {
            r = r.wrapping_sub(GL_P);
        }
        r
    }
}

/// Field subtraction: (a - b) mod p.
#[inline]
pub fn gl_sub(a: u64, b: u64) -> u64 {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    { gl_ecall_op(GL_SUB_ECALL_NUM, a, b) }

    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    {
        if a >= b {
            let diff = a - b;
            if diff >= GL_P { diff - GL_P } else { diff }
        } else {
            GL_P - (b - a)
        }
    }
}

/// Field negation: (-a) mod p.
#[inline]
pub fn gl_neg(a: u64) -> u64 {
    if a == 0 { 0 } else { GL_P - a }
}

/// Field multiplication: (a * b) mod p.
#[inline]
pub fn gl_mul(a: u64, b: u64) -> u64 {
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    { gl_ecall_op(GL_MUL_ECALL_NUM, a, b) }

    #[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
    {
        let prod = (a as u128) * (b as u128);
        reduce128(prod)
    }
}

/// Field squaring: (a * a) mod p.
#[inline]
pub fn gl_sqr(a: u64) -> u64 {
    gl_mul(a, a)
}

/// Field exponentiation: a^exp mod p via square-and-multiply.
pub fn gl_pow(mut base: u64, mut exp: u64) -> u64 {
    let mut result = GL_ONE;
    while exp > 0 {
        if exp & 1 == 1 {
            result = gl_mul(result, base);
        }
        base = gl_sqr(base);
        exp >>= 1;
    }
    result
}

/// Field inverse: a^(p-2) mod p (Fermat's little theorem).
///
/// Panics (halts) if a == 0.
pub fn gl_inv(a: u64) -> u64 {
    debug_assert!(a != 0, "inverse of zero");
    gl_pow(a, GL_P - 2)
}

/// Check equality of two digests.
pub fn digest_eq(a: &GlDigest, b: &GlDigest) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
}

/// Zero digest.
pub const ZERO_DIGEST: GlDigest = [0; 4];
