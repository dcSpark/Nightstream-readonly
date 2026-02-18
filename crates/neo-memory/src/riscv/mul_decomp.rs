//! RV32M multiply decomposition into byte-level subtable lookups.
//!
//! Decomposes a 32×32→64-bit unsigned multiplication into:
//! - 16 `Mul8` lookups: 8-bit × 8-bit → (lo8, hi8)
//! - Up to 32 `Add8Acc` lookups: byte add with carry accumulation
//!
//! This avoids the need for a 2^64-row MUL lookup table by using two
//! small, fully-enumerable subtables (64K and 512K rows respectively).

use neo_vm_trace::ShoutId;

/// Shout table ID for Mul8 (8×8 → 16-bit product).
/// Chosen to avoid collision with RiscvOpcode IDs (0–30).
pub const MUL8_TABLE_ID: u32 = 100;

/// Shout table ID for Add8Acc (byte add with carry accumulation).
pub const ADD8ACC_TABLE_ID: u32 = 101;

pub const MUL8_SHOUT_ID: ShoutId = ShoutId(MUL8_TABLE_ID);
pub const ADD8ACC_SHOUT_ID: ShoutId = ShoutId(ADD8ACC_TABLE_ID);

/// Address width for Mul8: 8 bits a + 8 bits b = 16.
pub const MUL8_ADDR_BITS: usize = 16;

/// Address width for Add8Acc: 8 bits sum_in + 8 bits add + 3 bits carry_in = 19.
pub const ADD8ACC_ADDR_BITS: usize = 19;

/// Number of Mul8 lookups per 32×32 multiply (4×4 byte pairs).
pub const MUL8_LANES: usize = 16;

/// Number of Add8Acc lookups per 32×32 multiply (16 low + 16 high accumulations).
pub const ADD8ACC_LANES: usize = 32;

/// A single lookup row emitted during multiplication decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulLookupRow {
    Mul8 {
        a: u8,
        b: u8,
        lo: u8,
        hi: u8,
    },
    Add8Acc {
        sum_in: u8,
        add: u8,
        carry_in: u8,
        sum_out: u8,
        carry_out: u8,
    },
}

/// Result of decomposing a 32×32 unsigned multiplication.
#[derive(Debug, Clone)]
pub struct Mul32Decomp {
    /// Full 64-bit product as 8 little-endian bytes.
    pub p_bytes: [u8; 8],
    /// Lower 32 bits of product (MUL result).
    pub lo: u32,
    /// Upper 32 bits of unsigned product (MULHU result).
    pub hi_u: u32,
    /// MULH result (signed × signed high word).
    pub hi_mulh: u32,
    /// MULHSU result (signed × unsigned high word).
    pub hi_mulhsu: u32,
    /// All lookup rows in emission order: 16 Mul8 then up to 32 Add8Acc.
    pub lookups: Vec<MulLookupRow>,
}

/// Diagonal index sets for schoolbook byte multiplication.
/// `DIAGS[k]` lists the (i, j) pairs where i + j == k.
const DIAGS: [&[(usize, usize)]; 8] = [
    &[(0, 0)],
    &[(0, 1), (1, 0)],
    &[(0, 2), (1, 1), (2, 0)],
    &[(0, 3), (1, 2), (2, 1), (3, 0)],
    &[(1, 3), (2, 2), (3, 1)],
    &[(2, 3), (3, 2)],
    &[(3, 3)],
    &[],
];

#[inline]
fn mul8(a: u8, b: u8) -> (u8, u8) {
    let prod = (a as u16) * (b as u16);
    ((prod & 0xFF) as u8, (prod >> 8) as u8)
}

#[inline]
fn add8_acc(sum_in: u8, add: u8, carry_in: u8) -> (u8, u8) {
    debug_assert!(carry_in <= 7);
    let t = (sum_in as u16) + (add as u16);
    let sum_out = (t & 0xFF) as u8;
    let carry_out = carry_in + ((t >> 8) as u8);
    debug_assert!(carry_out <= 7);
    (sum_out, carry_out)
}

/// Decompose `rs1 * rs2` into byte-level lookup rows.
///
/// Returns the full 64-bit product decomposition and all Mul8/Add8Acc
/// lookup rows needed to verify it via the shared CPU bus.
pub fn decompose_mul32(rs1: u32, rs2: u32) -> Mul32Decomp {
    let a = rs1.to_le_bytes();
    let b = rs2.to_le_bytes();

    let mut plo = [[0u8; 4]; 4];
    let mut phi = [[0u8; 4]; 4];
    let mut lookups = Vec::with_capacity(MUL8_LANES + ADD8ACC_LANES);

    for i in 0..4 {
        for j in 0..4 {
            let (lo, hi) = mul8(a[i], b[j]);
            plo[i][j] = lo;
            phi[i][j] = hi;
            lookups.push(MulLookupRow::Mul8 { a: a[i], b: b[j], lo, hi });
        }
    }

    let mut p = [0u8; 8];
    let mut carry_lo: u8 = 0;
    let mut carry_hi: u8 = 0;

    for k in 0..8 {
        // Low-byte accumulation: sum partial product low bytes, track overflow.
        let mut sum: u8 = carry_lo;
        let mut c1: u8 = 0;

        for &(i, j) in DIAGS[k] {
            let (sum_out, carry_out) = add8_acc(sum, plo[i][j], c1);
            lookups.push(MulLookupRow::Add8Acc {
                sum_in: sum,
                add: plo[i][j],
                carry_in: c1,
                sum_out,
                carry_out,
            });
            sum = sum_out;
            c1 = carry_out;
        }

        p[k] = sum;

        // High-byte accumulation: base = c1 + carry_hi, then add partial product high bytes.
        let base = c1 + carry_hi;
        debug_assert!(base <= 7);

        let mut sum2: u8 = base;
        let mut c2: u8 = 0;

        for &(i, j) in DIAGS[k] {
            let (sum_out, carry_out) = add8_acc(sum2, phi[i][j], c2);
            lookups.push(MulLookupRow::Add8Acc {
                sum_in: sum2,
                add: phi[i][j],
                carry_in: c2,
                sum_out,
                carry_out,
            });
            sum2 = sum_out;
            c2 = carry_out;
        }

        carry_lo = sum2;
        carry_hi = c2;
    }

    debug_assert!(carry_lo == 0 && carry_hi == 0, "final carry must be zero");

    let lo = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
    let hi_u = u32::from_le_bytes([p[4], p[5], p[6], p[7]]);

    let a_neg = ((rs1 >> 31) & 1) != 0;
    let b_neg = ((rs2 >> 31) & 1) != 0;

    let hi_mulhsu = hi_u.wrapping_sub(if a_neg { rs2 } else { 0 });
    let hi_mulh = hi_mulhsu.wrapping_sub(if b_neg { rs1 } else { 0 });

    Mul32Decomp {
        p_bytes: p,
        lo,
        hi_u,
        hi_mulh,
        hi_mulhsu,
        lookups,
    }
}

/// Encode a Mul8 lookup address as a flat 16-bit key: `a | (b << 8)`.
#[inline]
pub fn mul8_key(a: u8, b: u8) -> u64 {
    (a as u64) | ((b as u64) << 8)
}

/// Encode a Mul8 lookup value: full 16-bit product `lo | (hi << 8)`.
#[inline]
pub fn mul8_value(lo: u8, hi: u8) -> u64 {
    (lo as u64) | ((hi as u64) << 8)
}

/// Encode an Add8Acc lookup address as a flat 19-bit key:
/// `sum_in | (add << 8) | (carry_in << 16)`.
#[inline]
pub fn add8acc_key(sum_in: u8, add: u8, carry_in: u8) -> u64 {
    (sum_in as u64) | ((add as u64) << 8) | ((carry_in as u64) << 16)
}

/// Encode an Add8Acc lookup value: `sum_out | (carry_out << 8)`.
#[inline]
pub fn add8acc_value(sum_out: u8, carry_out: u8) -> u64 {
    (sum_out as u64) | ((carry_out as u64) << 8)
}

/// Build the full Mul8 truth table as a dense vector of 2^16 field-encoded values.
pub fn build_mul8_table() -> Vec<u64> {
    let size = 1 << MUL8_ADDR_BITS;
    let mut table = vec![0u64; size];
    for a in 0u16..=255 {
        for b in 0u16..=255 {
            let addr = (a | (b << 8)) as usize;
            let prod = a * b;
            table[addr] = prod as u64;
        }
    }
    table
}

/// Build the full Add8Acc truth table as a dense vector of 2^19 field-encoded values.
pub fn build_add8acc_table() -> Vec<u64> {
    let size = 1 << ADD8ACC_ADDR_BITS;
    let mut table = vec![0u64; size];
    for sum_in in 0u32..=255 {
        for add in 0u32..=255 {
            for carry_in in 0u32..=7 {
                let addr = (sum_in | (add << 8) | (carry_in << 16)) as usize;
                let t = sum_in + add;
                let sum_out = t & 0xFF;
                let carry_out = carry_in + (t >> 8);
                table[addr] = (sum_out as u64) | ((carry_out as u64) << 8);
            }
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul8() {
        assert_eq!(mul8(0, 0), (0, 0));
        assert_eq!(mul8(1, 1), (1, 0));
        assert_eq!(mul8(255, 255), (1, 254)); // 65025 = 0xFE01
        assert_eq!(mul8(16, 16), (0, 1)); // 256
    }

    #[test]
    fn test_add8_acc() {
        assert_eq!(add8_acc(0, 0, 0), (0, 0));
        assert_eq!(add8_acc(255, 1, 0), (0, 1));
        assert_eq!(add8_acc(255, 255, 0), (254, 1));
        assert_eq!(add8_acc(200, 100, 3), (44, 4)); // 200+100=300, sum_out=44, carry_out=3+1=4
    }

    #[test]
    fn test_decompose_simple() {
        let d = decompose_mul32(3, 7);
        assert_eq!(d.lo, 21);
        assert_eq!(d.hi_u, 0);
    }

    #[test]
    fn test_decompose_overflow() {
        let d = decompose_mul32(0xFFFF_FFFF, 0xFFFF_FFFF);
        let expected = 0xFFFF_FFFFu64 * 0xFFFF_FFFFu64;
        assert_eq!(d.lo, expected as u32);
        assert_eq!(d.hi_u, (expected >> 32) as u32);
    }

    #[test]
    fn test_decompose_mul_matches_wrapping() {
        let pairs: &[(u32, u32)] = &[
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (123, 456),
            (0xDEAD, 0xBEEF),
            (0xFFFF_FFFF, 2),
            (0x8000_0000, 0x8000_0000),
            (0x1234_5678, 0x9ABC_DEF0),
            (0xFFFF_FFFF, 0xFFFF_FFFF),
        ];
        for &(a, b) in pairs {
            let d = decompose_mul32(a, b);
            assert_eq!(d.lo, a.wrapping_mul(b), "MUL mismatch for {a:#x} * {b:#x}");

            let full = (a as u64) * (b as u64);
            assert_eq!(d.hi_u, (full >> 32) as u32, "MULHU mismatch for {a:#x} * {b:#x}");
        }
    }

    #[test]
    fn test_decompose_mulh_signed() {
        let pairs: &[(u32, u32)] = &[
            (0xFFFF_FFFF, 0xFFFF_FFFF), // -1 * -1 = 1, hi = 0
            (0x8000_0000, 2),            // -2^31 * 2, hi = -1
            (0x7FFF_FFFF, 0x7FFF_FFFF), // max_pos * max_pos
            (100, 200),
        ];
        for &(a, b) in pairs {
            let d = decompose_mul32(a, b);
            let expected = ((a as i32 as i64) * (b as i32 as i64)) >> 32;
            assert_eq!(
                d.hi_mulh, expected as u32,
                "MULH mismatch for {a:#x} * {b:#x}"
            );
        }
    }

    #[test]
    fn test_decompose_mulhsu() {
        let pairs: &[(u32, u32)] = &[
            (0xFFFF_FFFF, 1), // -1 * 1u = -1, hi = -1
            (0x8000_0000, 2), // -2^31 * 2u
            (100, 200),
        ];
        for &(a, b) in pairs {
            let d = decompose_mul32(a, b);
            let expected = ((a as i32 as i64) * (b as u32 as i64)) >> 32;
            assert_eq!(
                d.hi_mulhsu, expected as u32,
                "MULHSU mismatch for {a:#x} * {b:#x}"
            );
        }
    }

    #[test]
    fn test_lookup_counts() {
        let d = decompose_mul32(12345, 67890);
        let mul8_count = d.lookups.iter().filter(|r| matches!(r, MulLookupRow::Mul8 { .. })).count();
        let acc_count = d.lookups.iter().filter(|r| matches!(r, MulLookupRow::Add8Acc { .. })).count();
        assert_eq!(mul8_count, 16);
        assert_eq!(acc_count, d.lookups.len() - 16);
        assert!(acc_count <= ADD8ACC_LANES);
    }

    #[test]
    fn test_build_mul8_table() {
        let table = build_mul8_table();
        assert_eq!(table.len(), 1 << 16);
        assert_eq!(table[mul8_key(0, 0) as usize], 0);
        assert_eq!(table[mul8_key(3, 7) as usize], 21);
        assert_eq!(table[mul8_key(255, 255) as usize], 65025);
    }

    #[test]
    fn test_build_add8acc_table() {
        let table = build_add8acc_table();
        assert_eq!(table.len(), 1 << 19);
        assert_eq!(table[add8acc_key(0, 0, 0) as usize], add8acc_value(0, 0));
        assert_eq!(table[add8acc_key(255, 1, 0) as usize], add8acc_value(0, 1));
        assert_eq!(table[add8acc_key(200, 100, 3) as usize], add8acc_value(44, 4));
    }
}
