use midnight_circuits::{
    instructions::{
        ArithInstructions, AssertionInstructions, AssignmentInstructions, ConversionInstructions,
        DecompositionInstructions, PublicInputInstructions, RangeCheckInstructions,
    },
    types::{AssignedBit, AssignedByte, AssignedNative},
};
use midnight_proofs::circuit::{Layouter, Value};
use midnight_proofs::plonk::Error;
use midnight_zk_stdlib::ZkStdLib;
use num_bigint::BigUint;

/// Goldilocks prime: `2^64 - 2^32 + 1`.
pub const GOLDILOCKS_P_U64: u64 = 0xFFFF_FFFF_0000_0001;

pub type OuterScalar = midnight_curves::Fq;

#[derive(Clone, Debug)]
pub struct GlVar {
    pub assigned: AssignedNative<OuterScalar>,
    pub value: Value<u64>,
}

/// Constrain `x` to fit in 64 bits.
pub fn assert_u64(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    x: &AssignedNative<OuterScalar>,
) -> Result<(), Error> {
    // Decompose into 8×8-bit chunks (little endian) to range-check to 64 bits.
    //
    // `MidnightCircuit` may auto-select a small pow2range table (often `max_bit_len=8`),
    // in which case byte-based decomposition stays on the fast lookup path.
    let bytes = std.assigned_to_le_bytes(layouter, x, Some(8))?;
    if bytes.len() != 8 {
        return Err(Error::Synthesis(format!(
            "assigned_to_le_bytes(8) returned {} bytes",
            bytes.len()
        )));
    }
    Ok(())
}

/// Constrain `x` to be a canonical Goldilocks element: `0 <= x < p`.
pub fn assert_canonical_goldilocks(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    x: &AssignedNative<OuterScalar>,
) -> Result<(), Error> {
    // Canonicality check: `0 <= x < p`.
    //
    // Use ZkStdLib's range-check instruction so the underlying decomposition
    // chip can pick an optimal limb schedule for the circuit's pow2range table.
    std.assert_lower_than_fixed(layouter, x, &BigUint::from(GOLDILOCKS_P_U64))?;
    Ok(())
}

/// Constrain `z == x * y (mod p)` by providing a private quotient witness `k` such that
/// `x*y = z + k*p`, with `k` range-checked to 64 bits.
pub fn gl_mul_mod_check_with_quotient(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    x: &AssignedNative<OuterScalar>,
    y: &AssignedNative<OuterScalar>,
    z: &AssignedNative<OuterScalar>,
    k_u64: Value<u64>,
) -> Result<(), Error> {
    let k_val = k_u64.map(OuterScalar::from);
    let k = std.assign(layouter, k_val)?;
    assert_u64(std, layouter, &k)?;

    let lhs = std.mul(layouter, x, y, None)?;
    let rhs = std.linear_combination(
        layouter,
        &[
            (OuterScalar::from(1u64), z.clone()),
            (OuterScalar::from(GOLDILOCKS_P_U64), k),
        ],
        OuterScalar::from(0u64),
    )?;
    std.assert_equal(layouter, &lhs, &rhs)?;
    Ok(())
}

/// Reduce an outer-field value `t` modulo the Goldilocks prime `p`, producing a canonical
/// `r < p` and proving `t = r + p*q`.
///
/// This uses a 72-bit quotient encoding `q = q0 + 2^24*q1 + 2^48*q2` (3×24-bit limbs),
/// which is enough for common "small degree" batched reductions (e.g., sums of a few
/// 64×64-bit products).
pub fn gl_reduce_mod_p_quotient_72(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    t: &AssignedNative<OuterScalar>,
    t_val: Value<BigUint>,
) -> Result<GlVar, Error> {
    let p_big = BigUint::from(GOLDILOCKS_P_U64);
    let limb_mask = BigUint::from((1u64 << 24) - 1);

    // Compute (q0, q1, q2, r) from the host value (when known).
    let qr = t_val.map(|tv| {
        let q = &tv / &p_big;
        let r = &tv % &p_big;

        let q0 = (&q & &limb_mask)
            .to_u64_digits()
            .first()
            .copied()
            .unwrap_or(0);
        let q1 = ((&q >> 24usize) & &limb_mask)
            .to_u64_digits()
            .first()
            .copied()
            .unwrap_or(0);
        let q2 = ((&q >> 48usize) & &limb_mask)
            .to_u64_digits()
            .first()
            .copied()
            .unwrap_or(0);

        let r_u64 = r.to_u64_digits().first().copied().unwrap_or(0);
        (q0, q1, q2, r_u64)
    });

    let q0_u64 = qr.clone().map(|(q0, _, _, _)| q0);
    let q1_u64 = qr.clone().map(|(_, q1, _, _)| q1);
    let q2_u64 = qr.clone().map(|(_, _, q2, _)| q2);
    let r_u64 = qr.map(|(_, _, _, r)| r);

    let q0 = std.assign(layouter, q0_u64.map(OuterScalar::from))?;
    let q1 = std.assign(layouter, q1_u64.map(OuterScalar::from))?;
    let q2 = std.assign(layouter, q2_u64.map(OuterScalar::from))?;

    // Range-check quotient limbs to 24 bits each.
    let _ = std.assigned_to_le_bytes(layouter, &q0, Some(3))?;
    let _ = std.assigned_to_le_bytes(layouter, &q1, Some(3))?;
    let _ = std.assigned_to_le_bytes(layouter, &q2, Some(3))?;

    // Allocate canonical remainder.
    let r_assigned = std.assign(layouter, r_u64.map(OuterScalar::from))?;
    assert_u64(std, layouter, &r_assigned)?;
    assert_canonical_goldilocks(std, layouter, &r_assigned)?;

    let p = OuterScalar::from(GOLDILOCKS_P_U64);
    let p_2_24 = p * OuterScalar::from(1u64 << 24);
    let p_2_48 = p * OuterScalar::from(1u64 << 48);

    // Enforce: t == r + p*q0 + p*2^24*q1 + p*2^48*q2.
    let rhs = std.linear_combination(
        layouter,
        &[
            (OuterScalar::from(1u64), r_assigned.clone()),
            (p, q0),
            (p_2_24, q1),
            (p_2_48, q2),
        ],
        OuterScalar::from(0u64),
    )?;
    std.assert_equal(layouter, t, &rhs)?;

    Ok(GlVar {
        assigned: r_assigned,
        value: r_u64,
    })
}

/// Compute `(k, r)` such that `x*y = r + k*p` over unsigned integers, where `x,y < p`.
pub fn host_mul_quotient_and_remainder(x: u64, y: u64) -> (u64, u64) {
    let p = GOLDILOCKS_P_U64 as u128;
    let prod = (x as u128) * (y as u128);
    let k = (prod / p) as u64;
    let r = (prod % p) as u64;
    (k, r)
}

pub fn host_add_mod(x: u64, y: u64) -> u64 {
    let p = GOLDILOCKS_P_U64 as u128;
    let sum = (x as u128) + (y as u128);
    (sum % p) as u64
}

pub fn host_sub_mod(x: u64, y: u64) -> u64 {
    if x >= y {
        x - y
    } else {
        // x + p - y fits in u128.
        let p = GOLDILOCKS_P_U64 as u128;
        ((x as u128) + p - (y as u128)) as u64
    }
}

/// Allocate a Goldilocks element as a private witness.
pub fn alloc_gl_private(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    x_u64: Value<u64>,
) -> Result<GlVar, Error> {
    let assigned = std.assign(layouter, x_u64.map(OuterScalar::from))?;
    assert_u64(std, layouter, &assigned)?;
    assert_canonical_goldilocks(std, layouter, &assigned)?;
    Ok(GlVar { assigned, value: x_u64 })
}

/// Allocate a Goldilocks element as a private witness, only constraining it to fit in 64 bits.
///
/// This is useful for high-volume witness data (e.g., Ajtai digit rows) where the circuit only
/// needs a well-defined 64-bit integer representative, not a canonical `x < p` check at the
/// allocation site.
pub fn alloc_gl_private_u64(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    x_u64: Value<u64>,
) -> Result<GlVar, Error> {
    let assigned = std.assign(layouter, x_u64.map(OuterScalar::from))?;
    assert_u64(std, layouter, &assigned)?;
    Ok(GlVar { assigned, value: x_u64 })
}

/// Allocate a Goldilocks element as a public input.
pub fn alloc_gl_public(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    x_u64: Value<u64>,
) -> Result<GlVar, Error> {
    let var = alloc_gl_private(std, layouter, x_u64)?;
    std.constrain_as_public_input(layouter, &var.assigned)?;
    Ok(var)
}

pub fn gl_add_mod_var(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    x: &GlVar,
    y: &GlVar,
) -> Result<GlVar, Error> {
    let carry = x.value.zip(y.value).map(|(xu, yu)| {
        let sum = (xu as u128) + (yu as u128);
        sum >= (GOLDILOCKS_P_U64 as u128)
    });
    let carry_bit: AssignedBit<OuterScalar> = std.assign(layouter, carry)?;
    let carry_field: AssignedNative<OuterScalar> = std.convert(layouter, &carry_bit)?;

    let assigned = std.linear_combination(
        layouter,
        &[
            (OuterScalar::from(1u64), x.assigned.clone()),
            (OuterScalar::from(1u64), y.assigned.clone()),
            (-OuterScalar::from(GOLDILOCKS_P_U64), carry_field),
        ],
        OuterScalar::from(0u64),
    )?;
    assert_u64(std, layouter, &assigned)?;
    assert_canonical_goldilocks(std, layouter, &assigned)?;

    let value = x.value.zip(y.value).map(|(xu, yu)| host_add_mod(xu, yu));
    Ok(GlVar { assigned, value })
}

pub fn gl_sub_mod_var(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    x: &GlVar,
    y: &GlVar,
) -> Result<GlVar, Error> {
    let borrow = x.value.zip(y.value).map(|(xu, yu)| xu < yu);
    let borrow_bit: AssignedBit<OuterScalar> = std.assign(layouter, borrow)?;
    let borrow_field: AssignedNative<OuterScalar> = std.convert(layouter, &borrow_bit)?;

    let assigned = std.linear_combination(
        layouter,
        &[
            (OuterScalar::from(1u64), x.assigned.clone()),
            (-OuterScalar::from(1u64), y.assigned.clone()),
            (OuterScalar::from(GOLDILOCKS_P_U64), borrow_field),
        ],
        OuterScalar::from(0u64),
    )?;
    assert_u64(std, layouter, &assigned)?;
    assert_canonical_goldilocks(std, layouter, &assigned)?;

    let value = x.value.zip(y.value).map(|(xu, yu)| host_sub_mod(xu, yu));
    Ok(GlVar { assigned, value })
}

/// Sum a list of canonical Goldilocks elements modulo `p`.
///
/// This is much cheaper than chaining `gl_add_mod_var` because we do the accumulation
/// in the outer field (no modular reductions), and reduce once with a small quotient.
pub fn gl_sum_mod_var(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    terms: &[GlVar],
) -> Result<GlVar, Error> {
    if terms.is_empty() {
        return Err(Error::Synthesis("gl_sum_mod_var: empty terms".into()));
    }
    if terms.len() == 1 {
        return Ok(terms[0].clone());
    }

    // Sum in the outer field (exact, since values are < 64 bits and the outer modulus is huge).
    // Use a small fan-in to reduce the number of `add` rows.
    let mut partials: Vec<AssignedNative<OuterScalar>> = Vec::new();
    for chunk in terms.chunks(4) {
        if chunk.len() == 1 {
            partials.push(chunk[0].assigned.clone());
        } else {
            let lc_terms = chunk
                .iter()
                .map(|t| (OuterScalar::from(1u64), t.assigned.clone()))
                .collect::<Vec<_>>();
            partials.push(std.linear_combination(layouter, &lc_terms, OuterScalar::from(0u64))?);
        }
    }
    let mut sum_assigned = partials[0].clone();
    for p in partials.iter().skip(1) {
        sum_assigned = std.add(layouter, &sum_assigned, p)?;
    }

    // Host-side quotient and remainder: sum = r + q*p, with q < terms.len().
    let p_u128 = GOLDILOCKS_P_U64 as u128;
    let sum_u128 = terms.iter().fold(Value::known(0u128), |acc, t| {
        acc.zip(t.value).map(|(a, b)| a + (b as u128))
    });
    let q_u64 = sum_u128.map(|s| (s / p_u128) as u64);
    let r_u64 = sum_u128.map(|s| (s % p_u128) as u64);

    // q is always < terms.len(), so for our typical small fan-in we can assign it as a byte
    // (which includes the range check) and avoid an extra constraint-heavy decomposition.
    let max_q = (terms.len() as u64).saturating_sub(1);
    let q: AssignedNative<OuterScalar> = if max_q <= (u8::MAX as u64) {
        let q_byte: AssignedByte<OuterScalar> = std.assign(layouter, q_u64.map(|q| q as u8))?;
        std.convert(layouter, &q_byte)?
    } else {
        let q = std.assign(layouter, q_u64.map(OuterScalar::from))?;
        let q_bits: usize = 64 - max_q.leading_zeros() as usize;
        let _ = std.assigned_to_le_chunks(layouter, &q, q_bits, Some(1))?;
        q
    };

    // Allocate canonical remainder.
    let assigned_r = std.assign(layouter, r_u64.map(OuterScalar::from))?;
    assert_u64(std, layouter, &assigned_r)?;
    assert_canonical_goldilocks(std, layouter, &assigned_r)?;

    // Enforce: sum_assigned == assigned_r + q*p.
    let rhs = std.linear_combination(
        layouter,
        &[
            (OuterScalar::from(1u64), assigned_r.clone()),
            (OuterScalar::from(GOLDILOCKS_P_U64), q),
        ],
        OuterScalar::from(0u64),
    )?;
    std.assert_equal(layouter, &sum_assigned, &rhs)?;

    Ok(GlVar {
        assigned: assigned_r,
        value: r_u64,
    })
}

pub fn gl_mul_mod_var(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    x: &GlVar,
    y: &GlVar,
) -> Result<GlVar, Error> {
    let r_u64 = x
        .value
        .zip(y.value)
        .map(|(xu, yu)| host_mul_quotient_and_remainder(xu, yu).1);
    let k_u64 = x
        .value
        .zip(y.value)
        .map(|(xu, yu)| host_mul_quotient_and_remainder(xu, yu).0);

    let assigned = std.assign(layouter, r_u64.map(OuterScalar::from))?;
    assert_u64(std, layouter, &assigned)?;

    gl_mul_mod_check_with_quotient(std, layouter, &x.assigned, &y.assigned, &assigned, k_u64)?;
    Ok(GlVar { assigned, value: r_u64 })
}

pub fn gl_mul_mod_var_by_constant(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    x: &GlVar,
    c: u64,
) -> Result<GlVar, Error> {
    match c {
        0 => Ok(GlVar {
            assigned: std.assign_fixed(layouter, OuterScalar::from(0u64))?,
            value: Value::known(0),
        }),
        1 => Ok(x.clone()),
        _ => {
            let r_u64 = x.value.map(|xu| host_mul_quotient_and_remainder(xu, c).1);
            let k_u64 = x.value.map(|xu| host_mul_quotient_and_remainder(xu, c).0);

            let assigned_r = std.assign(layouter, r_u64.map(OuterScalar::from))?;
            assert_u64(std, layouter, &assigned_r)?;

            // Since `x < p`, the quotient satisfies `k = floor(c*x / p) < c`.
            // For small constants (like δ=7), assign it as a byte to get the range-check "for free".
            let k: AssignedNative<OuterScalar> = if c <= (u8::MAX as u64) {
                let k_byte: AssignedByte<OuterScalar> = std.assign(layouter, k_u64.map(|k| k as u8))?;
                std.convert(layouter, &k_byte)?
            } else {
                let k = std.assign(layouter, k_u64.map(OuterScalar::from))?;
                let k_bits: usize = 64 - (c - 1).leading_zeros() as usize;
                let _ = std.assigned_to_le_chunks(layouter, &k, k_bits, Some(1))?;
                k
            };

            // Enforce: c*x = r + k*p.
            let cx = std.mul_by_constant(layouter, &x.assigned, OuterScalar::from(c))?;
            let rhs = std.linear_combination(
                layouter,
                &[
                    (OuterScalar::from(1u64), assigned_r.clone()),
                    (OuterScalar::from(GOLDILOCKS_P_U64), k),
                ],
                OuterScalar::from(0u64),
            )?;
            std.assert_equal(layouter, &cx, &rhs)?;

            Ok(GlVar {
                assigned: assigned_r,
                value: r_u64,
            })
        }
    }
}
