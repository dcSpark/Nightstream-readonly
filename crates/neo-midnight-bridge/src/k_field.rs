use crate::goldilocks::{
    alloc_gl_private, alloc_gl_private_u64, alloc_gl_public, gl_add_mod_var, gl_reduce_mod_p_quotient_72,
    gl_sub_mod_var, gl_sum_mod_var, host_add_mod, host_mul_quotient_and_remainder, GlVar, OuterScalar,
    GOLDILOCKS_P_U64,
};
use midnight_circuits::instructions::{ArithInstructions, AssertionInstructions, AssignmentInstructions};
use midnight_proofs::circuit::{Layouter, Value};
use midnight_proofs::plonk::Error;
use midnight_zk_stdlib::ZkStdLib;
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};

/// Quadratic extension field `K = F_p[u]/(u^2 - δ)` with `δ = 7` used by Neo.
pub const K_DELTA_U64: u64 = 7;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KRepr {
    pub c0: u64,
    pub c1: u64,
}

impl KRepr {
    pub const ZERO: Self = Self { c0: 0, c1: 0 };
}

#[derive(Clone, Debug)]
pub struct KVar {
    pub c0: GlVar,
    pub c1: GlVar,
}

pub fn alloc_k_private(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    k: Value<KRepr>,
) -> Result<KVar, Error> {
    let c0 = alloc_gl_private(std, layouter, k.as_ref().map(|v| v.c0))?;
    let c1 = alloc_gl_private(std, layouter, k.as_ref().map(|v| v.c1))?;
    Ok(KVar { c0, c1 })
}

/// Allocate a `K` element as a private witness, constraining coordinates only to 64 bits.
pub fn alloc_k_private_u64(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    k: Value<KRepr>,
) -> Result<KVar, Error> {
    let c0 = alloc_gl_private_u64(std, layouter, k.as_ref().map(|v| v.c0))?;
    let c1 = alloc_gl_private_u64(std, layouter, k.as_ref().map(|v| v.c1))?;
    Ok(KVar { c0, c1 })
}

pub fn alloc_k_public(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    k: Value<KRepr>,
) -> Result<KVar, Error> {
    let c0 = alloc_gl_public(std, layouter, k.as_ref().map(|v| v.c0))?;
    let c1 = alloc_gl_public(std, layouter, k.as_ref().map(|v| v.c1))?;
    Ok(KVar { c0, c1 })
}

pub fn assert_k_eq(std: &ZkStdLib, layouter: &mut impl Layouter<OuterScalar>, a: &KVar, b: &KVar) -> Result<(), Error> {
    std.assert_equal(layouter, &a.c0.assigned, &b.c0.assigned)?;
    std.assert_equal(layouter, &a.c1.assigned, &b.c1.assigned)?;
    Ok(())
}

pub fn k_add_mod_var(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    a: &KVar,
    b: &KVar,
) -> Result<KVar, Error> {
    let c0 = gl_add_mod_var(std, layouter, &a.c0, &b.c0)?;
    let c1 = gl_add_mod_var(std, layouter, &a.c1, &b.c1)?;
    Ok(KVar { c0, c1 })
}

pub fn k_sub_mod_var(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    a: &KVar,
    b: &KVar,
) -> Result<KVar, Error> {
    let c0 = gl_sub_mod_var(std, layouter, &a.c0, &b.c0)?;
    let c1 = gl_sub_mod_var(std, layouter, &a.c1, &b.c1)?;
    Ok(KVar { c0, c1 })
}

pub fn gl_const(std: &ZkStdLib, layouter: &mut impl Layouter<OuterScalar>, x: u64) -> Result<GlVar, Error> {
    let assigned = std.assign_fixed(layouter, OuterScalar::from(x))?;
    Ok(GlVar {
        assigned,
        value: Value::known(x),
    })
}

pub fn k_const(std: &ZkStdLib, layouter: &mut impl Layouter<OuterScalar>, c0: u64, c1: u64) -> Result<KVar, Error> {
    let c0 = gl_const(std, layouter, c0)?;
    let c1 = gl_const(std, layouter, c1)?;
    Ok(KVar { c0, c1 })
}

pub fn k_zero(std: &ZkStdLib, layouter: &mut impl Layouter<OuterScalar>) -> Result<KVar, Error> {
    k_const(std, layouter, 0, 0)
}

pub fn k_one(std: &ZkStdLib, layouter: &mut impl Layouter<OuterScalar>) -> Result<KVar, Error> {
    k_const(std, layouter, 1, 0)
}

pub fn k_sum_mod_var(std: &ZkStdLib, layouter: &mut impl Layouter<OuterScalar>, terms: &[KVar]) -> Result<KVar, Error> {
    if terms.is_empty() {
        return Err(Error::Synthesis("k_sum_mod_var: empty terms".into()));
    }

    let c0_terms: Vec<GlVar> = terms.iter().map(|t| t.c0.clone()).collect();
    let c1_terms: Vec<GlVar> = terms.iter().map(|t| t.c1.clone()).collect();
    let c0 = gl_sum_mod_var(std, layouter, &c0_terms)?;
    let c1 = gl_sum_mod_var(std, layouter, &c1_terms)?;
    Ok(KVar { c0, c1 })
}

pub fn k_mul_mod_var(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    a: &KVar,
    b: &KVar,
    delta: u64,
) -> Result<KVar, Error> {
    // Faster K multiplication: compute the coordinate expressions in the outer field
    // (integer arithmetic), then reduce each coordinate mod p once.
    //
    // (a0 + a1*u) * (b0 + b1*u) = (a0*b0 + δ*a1*b1) + (a0*b1 + a1*b0)*u.

    // Outer-field products (no mod-p reduction here).
    let a0b0 = std.mul(layouter, &a.c0.assigned, &b.c0.assigned, None)?;
    let a1b1 = std.mul(layouter, &a.c1.assigned, &b.c1.assigned, None)?;
    let a0b1 = std.mul(layouter, &a.c0.assigned, &b.c1.assigned, None)?;
    let a1b0 = std.mul(layouter, &a.c1.assigned, &b.c0.assigned, None)?;

    let delta_a1b1 = std.mul_by_constant(layouter, &a1b1, OuterScalar::from(delta))?;
    let t0 = std.add(layouter, &a0b0, &delta_a1b1)?;
    let t1 = std.add(layouter, &a0b1, &a1b0)?;

    let t0_val: Value<BigUint> =
        a.c0.value
            .zip(b.c0.value)
            .zip(a.c1.value)
            .zip(b.c1.value)
            .map(|(((a0, b0), a1), b1)| {
                let prod00 = BigUint::from((a0 as u128) * (b0 as u128));
                let prod11 = BigUint::from((a1 as u128) * (b1 as u128));
                prod00 + (BigUint::from(delta) * prod11)
            });

    let t1_val: Value<BigUint> =
        a.c0.value
            .zip(b.c1.value)
            .zip(a.c1.value)
            .zip(b.c0.value)
            .map(|(((a0, b1), a1), b0)| {
                let prod01 = BigUint::from((a0 as u128) * (b1 as u128));
                let prod10 = BigUint::from((a1 as u128) * (b0 as u128));
                prod01 + prod10
            });

    let c0 = gl_reduce_mod_p_quotient_72(std, layouter, &t0, t0_val)?;
    let c1 = gl_reduce_mod_p_quotient_72(std, layouter, &t1, t1_val)?;
    Ok(KVar { c0, c1 })
}

/// Compute `v0 + a*(v1 - v0)` in K while paying only one mod-`p` reduction per coordinate.
///
/// This is the core hot-path for multilinear folding: it replaces `(sub, mul, add)` sequences
/// with a single batched reduction of a small-degree outer-field expression.
pub fn k_mle_fold_step(
    std: &ZkStdLib,
    layouter: &mut impl Layouter<OuterScalar>,
    v0: &KVar,
    v1: &KVar,
    a: &KVar,
    delta: u64,
) -> Result<KVar, Error> {
    // Add a large multiple of p^2 to keep the integer expressions non-negative (BigUint-friendly).
    // This does not change the result modulo p.
    let p_big = BigUint::from(GOLDILOCKS_P_U64);
    let offset_big = BigUint::from(8u64) * &p_big * &p_big; // 8*p^2
    let p_fe = OuterScalar::from(GOLDILOCKS_P_U64);
    let offset_fe = OuterScalar::from(8u64) * p_fe * p_fe;

    // next.c0 = v0.c0 + a0*(v1.c0 - v0.c0) + δ*a1*(v1.c1 - v0.c1)
    //         = v0.c0 + a0*v1.c0 - a0*v0.c0 + δ*a1*v1.c1 - δ*a1*v0.c1
    let a0_v1c0 = std.mul(layouter, &a.c0.assigned, &v1.c0.assigned, None)?;
    let a0_v0c0 = std.mul(layouter, &a.c0.assigned, &v0.c0.assigned, None)?;
    let a1_v1c1 = std.mul(layouter, &a.c1.assigned, &v1.c1.assigned, None)?;
    let a1_v0c1 = std.mul(layouter, &a.c1.assigned, &v0.c1.assigned, None)?;
    let delta_a1_v1c1 = std.mul_by_constant(layouter, &a1_v1c1, OuterScalar::from(delta))?;
    let delta_a1_v0c1 = std.mul_by_constant(layouter, &a1_v0c1, OuterScalar::from(delta))?;

    // Use a single (possibly multi-row) linear combination instead of (lc + lc + add).
    // This saves rows in the hot-path of multilinear folding.
    let t0 = std.linear_combination(
        layouter,
        &[
            (OuterScalar::from(1u64), v0.c0.assigned.clone()),
            (OuterScalar::from(1u64), a0_v1c0),
            (-OuterScalar::from(1u64), a0_v0c0),
            (OuterScalar::from(1u64), delta_a1_v1c1),
            (-OuterScalar::from(1u64), delta_a1_v0c1),
        ],
        offset_fe,
    )?;

    let t0_val: Value<BigUint> = v0
        .c0
        .value
        .zip(v0.c1.value)
        .zip(v1.c0.value)
        .zip(v1.c1.value)
        .zip(a.c0.value)
        .zip(a.c1.value)
        .map(|(((((v0c0, v0c1), v1c0), v1c1), a0), a1)| {
            let mut acc = offset_big.clone();
            acc += BigUint::from(v0c0);
            acc += BigUint::from((a0 as u128) * (v1c0 as u128));
            acc += BigUint::from(delta) * BigUint::from((a1 as u128) * (v1c1 as u128));
            acc -= BigUint::from((a0 as u128) * (v0c0 as u128));
            acc -= BigUint::from(delta) * BigUint::from((a1 as u128) * (v0c1 as u128));
            // v0c1 is only used to subtract the δ*a1*v0.c1 term; keep v0c1 in scope.
            let _ = v0c1;
            acc
        });

    // next.c1 = v0.c1 + a0*(v1.c1 - v0.c1) + a1*(v1.c0 - v0.c0)
    //         = v0.c1 + a0*v1.c1 - a0*v0.c1 + a1*v1.c0 - a1*v0.c0
    let a0_v1c1 = std.mul(layouter, &a.c0.assigned, &v1.c1.assigned, None)?;
    let a0_v0c1 = std.mul(layouter, &a.c0.assigned, &v0.c1.assigned, None)?;
    let a1_v1c0 = std.mul(layouter, &a.c1.assigned, &v1.c0.assigned, None)?;
    let a1_v0c0 = std.mul(layouter, &a.c1.assigned, &v0.c0.assigned, None)?;

    let t1 = std.linear_combination(
        layouter,
        &[
            (OuterScalar::from(1u64), v0.c1.assigned.clone()),
            (OuterScalar::from(1u64), a0_v1c1),
            (-OuterScalar::from(1u64), a0_v0c1),
            (OuterScalar::from(1u64), a1_v1c0),
            (-OuterScalar::from(1u64), a1_v0c0),
        ],
        offset_fe,
    )?;

    let t1_val: Value<BigUint> = v0
        .c0
        .value
        .zip(v0.c1.value)
        .zip(v1.c0.value)
        .zip(v1.c1.value)
        .zip(a.c0.value)
        .zip(a.c1.value)
        .map(|(((((v0c0, v0c1), v1c0), v1c1), a0), a1)| {
            let mut acc = offset_big.clone();
            acc += BigUint::from(v0c1);
            acc += BigUint::from((a0 as u128) * (v1c1 as u128));
            acc += BigUint::from((a1 as u128) * (v1c0 as u128));
            acc -= BigUint::from((a0 as u128) * (v0c1 as u128));
            acc -= BigUint::from((a1 as u128) * (v0c0 as u128));
            acc
        });

    let c0 = gl_reduce_mod_p_quotient_72(std, layouter, &t0, t0_val)?;
    let c1 = gl_reduce_mod_p_quotient_72(std, layouter, &t1, t1_val)?;
    Ok(KVar { c0, c1 })
}

fn host_gl_mul_mod(x: u64, y: u64) -> u64 {
    host_mul_quotient_and_remainder(x, y).1
}

pub fn host_k_add(a: KRepr, b: KRepr) -> KRepr {
    KRepr {
        c0: host_add_mod(a.c0, b.c0),
        c1: host_add_mod(a.c1, b.c1),
    }
}

pub fn host_k_mul(a: KRepr, b: KRepr, delta: u64) -> KRepr {
    debug_assert!(a.c0 < GOLDILOCKS_P_U64);
    debug_assert!(a.c1 < GOLDILOCKS_P_U64);
    debug_assert!(b.c0 < GOLDILOCKS_P_U64);
    debug_assert!(b.c1 < GOLDILOCKS_P_U64);
    debug_assert!(delta < GOLDILOCKS_P_U64);

    let a0b0 = host_gl_mul_mod(a.c0, b.c0);
    let a1b1 = host_gl_mul_mod(a.c1, b.c1);
    let delta_a1b1 = host_gl_mul_mod(a1b1, delta);
    let c0 = host_add_mod(a0b0, delta_a1b1);

    let a0b1 = host_gl_mul_mod(a.c0, b.c1);
    let a1b0 = host_gl_mul_mod(a.c1, b.c0);
    let c1 = host_add_mod(a0b1, a1b0);

    KRepr { c0, c1 }
}

pub fn host_k_eval_horner(coeffs: &[KRepr], x: KRepr, delta: u64) -> KRepr {
    assert!(!coeffs.is_empty(), "coeffs must not be empty");
    let mut acc = *coeffs.last().expect("non-empty");
    for c in coeffs[..coeffs.len() - 1].iter().rev() {
        acc = host_k_add(host_k_mul(acc, x, delta), *c);
    }
    acc
}

pub fn host_sumcheck_round_claim(coeffs: &[KRepr]) -> KRepr {
    assert!(!coeffs.is_empty(), "coeffs must not be empty");
    let p_at_0 = coeffs[0];
    let p_at_1 = coeffs.iter().copied().fold(KRepr::ZERO, host_k_add);
    host_k_add(p_at_0, p_at_1)
}
