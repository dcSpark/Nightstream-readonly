//! Poseidon2-Goldilocks hash via ECALL precompile.
//!
//! Provides the guest-side API for computing Poseidon2 hashes over Goldilocks
//! field elements. On RISC-V targets, this issues ECALLs to the host:
//! 1. A "compute" ECALL that reads inputs (via untraced loads) and computes the
//!    Poseidon2 hash, storing the digest in host-side CPU state.
//! 2. Eight "read" ECALLs that each return one u32 word of the digest in
//!    register a0.
//!
//! On non-RISC-V targets, a stub panics.

use crate::goldilocks::GlDigest;

/// Poseidon2 compute ECALL number (must match neo-memory's POSEIDON2_ECALL_NUM).
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const POSEIDON2_ECALL_NUM: u32 = 0x504F53;

/// Poseidon2 read ECALL number (must match neo-memory's POSEIDON2_READ_ECALL_NUM).
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const POSEIDON2_READ_ECALL_NUM: u32 = 0x80504F53;

/// Digest length in Goldilocks elements (matches neo-params DIGEST_LEN).
const DIGEST_LEN: usize = 4;

/// Scratch buffer size: supports hashing up to 64 Goldilocks elements per call.
/// Each element occupies 2 u32 words (8 bytes).
const MAX_INPUT_ELEMENTS: usize = 64;

/// Hash an arbitrary-length slice of Goldilocks field elements.
///
/// Packs elements to a stack-allocated scratch buffer, issues the Poseidon2
/// compute ECALL, then retrieves the 4-element digest via 8 read ECALLs
/// (each returning one u32 word in register a0).
///
/// # Panics
///
/// Panics if `input.len() > MAX_INPUT_ELEMENTS` (64).
pub fn poseidon2_hash(input: &[u64]) -> GlDigest {
    assert!(
        input.len() <= MAX_INPUT_ELEMENTS,
        "poseidon2_hash: too many input elements"
    );

    // Scratch buffer for input (as u32 words).
    let mut input_buf: [u32; MAX_INPUT_ELEMENTS * 2] = [0; MAX_INPUT_ELEMENTS * 2];

    // Pack Goldilocks elements as pairs of u32 words (little-endian).
    for (i, &elem) in input.iter().enumerate() {
        input_buf[i * 2] = elem as u32;
        input_buf[i * 2 + 1] = (elem >> 32) as u32;
    }

    // Compute ECALL: host reads inputs via untraced loads and stores digest in CPU state.
    poseidon2_ecall_compute(input.len() as u32, input_buf.as_ptr() as u32);

    // Read 8 u32 words of the digest via register a0.
    let d: [u32; 8] = core::array::from_fn(|_| poseidon2_ecall_read());

    // Unpack into 4 Goldilocks elements.
    let mut digest = [0u64; DIGEST_LEN];
    for i in 0..DIGEST_LEN {
        digest[i] = (d[i * 2] as u64) | ((d[i * 2 + 1] as u64) << 32);
    }
    digest
}

/// Issue the Poseidon2 compute ECALL.
///
/// Registers: a0 = ECALL ID, a1 = element count, a2 = input addr.
/// No output via registers; digest is stored in host CPU state.
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
fn poseidon2_ecall_compute(n_elements: u32, input_addr: u32) {
    unsafe {
        core::arch::asm!(
            "ecall",
            in("a0") POSEIDON2_ECALL_NUM,
            in("a1") n_elements,
            in("a2") input_addr,
            options(nostack),
        );
    }
}

/// Issue the Poseidon2 read ECALL and return the next digest word.
///
/// The ECALL number is passed in a0, and the host returns the next u32 word
/// of the pending digest in a0.
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
fn poseidon2_ecall_read() -> u32 {
    let result: u32;
    unsafe {
        core::arch::asm!(
            "ecall",
            inout("a0") POSEIDON2_READ_ECALL_NUM => result,
            options(nostack),
        );
    }
    result
}

/// Stub for non-RISC-V targets (used in native tests via the reference impl).
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
fn poseidon2_ecall_compute(_n_elements: u32, _input_addr: u32) {
    unimplemented!("poseidon2_ecall_compute is only available on RISC-V targets")
}

/// Stub for non-RISC-V targets.
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
fn poseidon2_ecall_read() -> u32 {
    unimplemented!("poseidon2_ecall_read is only available on RISC-V targets")
}
