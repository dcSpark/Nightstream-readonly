use neo_reductions::error::PiCcsError;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::riscv::lookups::{compute_op, RiscvOpcode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedBitnessRole {
    HasLookup,
    Val,
    PackedCol(usize),
}

#[inline]
pub fn rv32_packed_supported_opcode(op: RiscvOpcode) -> bool {
    matches!(
        op,
        RiscvOpcode::And
            | RiscvOpcode::Andn
            | RiscvOpcode::Or
            | RiscvOpcode::Xor
            | RiscvOpcode::Add
            | RiscvOpcode::Sub
            | RiscvOpcode::Eq
            | RiscvOpcode::Neq
            | RiscvOpcode::Slt
            | RiscvOpcode::Sll
            | RiscvOpcode::Srl
            | RiscvOpcode::Sra
            | RiscvOpcode::Sltu
            | RiscvOpcode::Mul
            | RiscvOpcode::Mulh
            | RiscvOpcode::Mulhu
            | RiscvOpcode::Mulhsu
            | RiscvOpcode::Div
            | RiscvOpcode::Divu
            | RiscvOpcode::Rem
            | RiscvOpcode::Remu
    )
}

#[inline]
pub fn rv32_packed_rollout_opcode(op: RiscvOpcode) -> bool {
    matches!(
        op,
        RiscvOpcode::Mul
            | RiscvOpcode::Mulh
            | RiscvOpcode::Mulhu
            | RiscvOpcode::Mulhsu
            | RiscvOpcode::Div
            | RiscvOpcode::Divu
            | RiscvOpcode::Rem
            | RiscvOpcode::Remu
    )
}

pub fn rv32_packed_d(op: RiscvOpcode) -> Result<usize, PiCcsError> {
    Ok(match op {
        RiscvOpcode::And | RiscvOpcode::Andn | RiscvOpcode::Or | RiscvOpcode::Xor => 34usize,
        RiscvOpcode::Add | RiscvOpcode::Sub => 3usize,
        RiscvOpcode::Eq | RiscvOpcode::Neq => 35usize,
        RiscvOpcode::Slt => 37usize,
        RiscvOpcode::Sll | RiscvOpcode::Srl | RiscvOpcode::Sra => 38usize,
        RiscvOpcode::Sltu => 35usize,
        RiscvOpcode::Mul => 34usize,
        RiscvOpcode::Mulh => 38usize,
        RiscvOpcode::Mulhu => 34usize,
        RiscvOpcode::Mulhsu => 37usize,
        RiscvOpcode::Div | RiscvOpcode::Rem => 43usize,
        RiscvOpcode::Divu | RiscvOpcode::Remu => 38usize,
        _ => {
            return Err(PiCcsError::InvalidInput(format!(
                "packed RV32 opcode is unsupported: opcode={op:?}"
            )));
        }
    })
}

fn push_col_range(out: &mut Vec<PackedBitnessRole>, start: usize, len: usize) {
    for idx in start..start + len {
        out.push(PackedBitnessRole::PackedCol(idx));
    }
}

pub fn rv32_packed_bitness_roles(op: RiscvOpcode) -> Result<Vec<PackedBitnessRole>, PiCcsError> {
    let mut out = Vec::new();
    match op {
        RiscvOpcode::And | RiscvOpcode::Andn | RiscvOpcode::Or | RiscvOpcode::Xor => {
            out.push(PackedBitnessRole::HasLookup);
        }
        RiscvOpcode::Add | RiscvOpcode::Sub => {
            out.push(PackedBitnessRole::PackedCol(2));
            out.push(PackedBitnessRole::HasLookup);
        }
        RiscvOpcode::Eq | RiscvOpcode::Neq => {
            out.push(PackedBitnessRole::HasLookup);
            out.push(PackedBitnessRole::Val);
            out.push(PackedBitnessRole::PackedCol(2));
            push_col_range(&mut out, 3, 32);
        }
        RiscvOpcode::Mul | RiscvOpcode::Mulhu => {
            out.push(PackedBitnessRole::HasLookup);
            push_col_range(&mut out, 2, 32);
        }
        RiscvOpcode::Mulh => {
            out.push(PackedBitnessRole::HasLookup);
            out.push(PackedBitnessRole::PackedCol(3));
            out.push(PackedBitnessRole::PackedCol(4));
            push_col_range(&mut out, 6, 32);
        }
        RiscvOpcode::Mulhsu => {
            out.push(PackedBitnessRole::HasLookup);
            out.push(PackedBitnessRole::PackedCol(3));
            out.push(PackedBitnessRole::PackedCol(4));
            push_col_range(&mut out, 5, 32);
        }
        RiscvOpcode::Slt => {
            out.push(PackedBitnessRole::Val);
            out.push(PackedBitnessRole::HasLookup);
            out.push(PackedBitnessRole::PackedCol(3));
            out.push(PackedBitnessRole::PackedCol(4));
            push_col_range(&mut out, 5, 32);
        }
        RiscvOpcode::Sll | RiscvOpcode::Srl => {
            out.push(PackedBitnessRole::HasLookup);
            push_col_range(&mut out, 1, 5);
            push_col_range(&mut out, 6, 32);
        }
        RiscvOpcode::Sra => {
            out.push(PackedBitnessRole::HasLookup);
            push_col_range(&mut out, 1, 5);
            out.push(PackedBitnessRole::PackedCol(6));
            push_col_range(&mut out, 7, 31);
        }
        RiscvOpcode::Sltu => {
            out.push(PackedBitnessRole::Val);
            out.push(PackedBitnessRole::HasLookup);
            push_col_range(&mut out, 3, 32);
        }
        RiscvOpcode::Divu | RiscvOpcode::Remu => {
            out.push(PackedBitnessRole::HasLookup);
            out.push(PackedBitnessRole::PackedCol(4));
            push_col_range(&mut out, 6, 32);
        }
        RiscvOpcode::Div | RiscvOpcode::Rem => {
            out.push(PackedBitnessRole::HasLookup);
            out.push(PackedBitnessRole::PackedCol(5));
            out.push(PackedBitnessRole::PackedCol(6));
            out.push(PackedBitnessRole::PackedCol(7));
            out.push(PackedBitnessRole::PackedCol(9));
            push_col_range(&mut out, 11, 32);
        }
        _ => {
            return Err(PiCcsError::InvalidInput(format!(
                "packed RV32 bitness roles are unsupported: opcode={op:?}"
            )));
        }
    }
    Ok(out)
}

pub fn rv32_collect_packed_bitness_terms<T: Clone>(
    op: RiscvOpcode,
    packed_cols: &[T],
    has_lookup: T,
    val: T,
) -> Result<Vec<T>, PiCcsError> {
    let roles = rv32_packed_bitness_roles(op)?;
    let mut out = Vec::with_capacity(roles.len());
    for role in roles {
        match role {
            PackedBitnessRole::HasLookup => out.push(has_lookup.clone()),
            PackedBitnessRole::Val => out.push(val.clone()),
            PackedBitnessRole::PackedCol(idx) => {
                let col = packed_cols.get(idx).ok_or_else(|| {
                    PiCcsError::InvalidInput(format!(
                        "packed RV32 bitness role index out of bounds for opcode={op:?}: idx={idx}, packed_cols={}",
                        packed_cols.len()
                    ))
                })?;
                out.push(col.clone());
            }
        }
    }
    Ok(out)
}

#[inline]
fn f_bool<F: PrimeCharacteristicRing>(bit: bool) -> F {
    if bit {
        F::ONE
    } else {
        F::ZERO
    }
}

pub fn build_rv32_packed_cols<F: Field + PrimeCharacteristicRing>(
    op: RiscvOpcode,
    lhs: u32,
    rhs: u32,
    val: u32,
) -> Result<Vec<F>, PiCcsError> {
    if !rv32_packed_rollout_opcode(op) {
        return Err(PiCcsError::InvalidInput(format!(
            "packed RV32 col synthesis is unsupported for opcode={op:?}"
        )));
    }
    let expected = compute_op(op, lhs as u64, rhs as u64, 32) as u32;
    if val != expected {
        return Err(PiCcsError::InvalidInput(format!(
            "packed RV32 col synthesis value mismatch for opcode={op:?}: got={val:#x}, expected={expected:#x}"
        )));
    }

    match op {
        RiscvOpcode::Mul => {
            let wide = (lhs as u64) * (rhs as u64);
            let hi = (wide >> 32) as u32;
            let mut packed = Vec::with_capacity(34);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((hi >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::Mulhu => {
            let wide = (lhs as u64) * (rhs as u64);
            let lo = (wide & 0xffff_ffff) as u32;
            let mut packed = Vec::with_capacity(34);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((lo >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::Mulh => {
            let uprod = (lhs as u64) * (rhs as u64);
            let lo = (uprod & 0xffff_ffff) as u32;
            let hi = (uprod >> 32) as u32;
            let lhs_sign = (lhs >> 31) & 1;
            let rhs_sign = (rhs >> 31) & 1;

            let diff =
                (val as i128) - (hi as i128) + (lhs_sign as i128) * (rhs as i128) + (rhs_sign as i128) * (lhs as i128);
            let two32 = 1_i128 << 32;
            if diff < 0 || diff % two32 != 0 {
                return Err(PiCcsError::InvalidInput(format!(
                    "packed MULH helper: invalid k decomposition (diff={diff})"
                )));
            }
            let k = (diff / two32) as u32;
            if k > 2 {
                return Err(PiCcsError::InvalidInput(format!(
                    "packed MULH helper: k out of range (k={k})"
                )));
            }

            let mut packed = Vec::with_capacity(38);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(hi as u64));
            packed.push(f_bool::<F>(lhs_sign == 1));
            packed.push(f_bool::<F>(rhs_sign == 1));
            packed.push(F::from_u64(k as u64));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((lo >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::Mulhsu => {
            let uprod = (lhs as u64) * (rhs as u64);
            let lo = (uprod & 0xffff_ffff) as u32;
            let hi = (uprod >> 32) as u32;
            let lhs_sign = (lhs >> 31) & 1;

            let diff = (val as i128) - (hi as i128) + (lhs_sign as i128) * (rhs as i128);
            let two32 = 1_i128 << 32;
            if diff < 0 || diff % two32 != 0 {
                return Err(PiCcsError::InvalidInput(format!(
                    "packed MULHSU helper: invalid borrow decomposition (diff={diff})"
                )));
            }
            let borrow = (diff / two32) as u32;
            if borrow > 1 {
                return Err(PiCcsError::InvalidInput(format!(
                    "packed MULHSU helper: borrow out of range (borrow={borrow})"
                )));
            }

            let mut packed = Vec::with_capacity(37);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(hi as u64));
            packed.push(f_bool::<F>(lhs_sign == 1));
            packed.push(f_bool::<F>(borrow == 1));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((lo >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::Div => {
            let lhs_sign = (lhs >> 31) & 1;
            let rhs_sign = (rhs >> 31) & 1;
            let lhs_abs = if lhs_sign == 0 { lhs } else { lhs.wrapping_neg() };
            let rhs_abs = if rhs == 0 {
                0u32
            } else if rhs_sign == 0 {
                rhs
            } else {
                rhs.wrapping_neg()
            };

            let rhs_f = F::from_u64(rhs as u64);
            let rhs_inv = if rhs_f == F::ZERO { F::ZERO } else { rhs_f.inverse() };
            let rhs_is_zero = rhs == 0;

            let (q_abs, r_abs) = if rhs == 0 {
                (0u32, 0u32)
            } else {
                (lhs_abs / rhs_abs, lhs_abs % rhs_abs)
            };
            let q_is_zero = q_abs == 0;
            let q_f = F::from_u64(q_abs as u64);
            let q_inv = if q_f == F::ZERO { F::ZERO } else { q_f.inverse() };
            let diff = if rhs == 0 { 0u32 } else { r_abs.wrapping_sub(rhs_abs) };

            let mut packed = Vec::with_capacity(43);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(q_abs as u64));
            packed.push(F::from_u64(r_abs as u64));
            packed.push(rhs_inv);
            packed.push(f_bool::<F>(rhs_is_zero));
            packed.push(f_bool::<F>(lhs_sign == 1));
            packed.push(f_bool::<F>(rhs_sign == 1));
            packed.push(q_inv);
            packed.push(f_bool::<F>(q_is_zero));
            packed.push(F::from_u64(diff as u64));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((diff >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::Divu => {
            let rhs_f = F::from_u64(rhs as u64);
            let rhs_inv = if rhs_f == F::ZERO { F::ZERO } else { rhs_f.inverse() };
            let rhs_is_zero = rhs == 0;

            let rem = if rhs == 0 {
                0u32
            } else {
                ((lhs as u64) % (rhs as u64)) as u32
            };
            if rhs != 0 {
                let rem_check = (lhs as u64).wrapping_sub((rhs as u64).wrapping_mul(val as u64)) as u32;
                if rem_check != rem {
                    return Err(PiCcsError::InvalidInput(format!(
                        "packed DIVU helper: invalid quotient/remainder relation (rem_check={rem_check:#x}, rem={rem:#x})"
                    )));
                }
            }
            let diff = rem.wrapping_sub(rhs);

            let mut packed = Vec::with_capacity(38);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(rem as u64));
            packed.push(rhs_inv);
            packed.push(f_bool::<F>(rhs_is_zero));
            packed.push(F::from_u64(diff as u64));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((diff >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::Rem => {
            let lhs_sign = (lhs >> 31) & 1;
            let rhs_sign = (rhs >> 31) & 1;
            let lhs_abs = if lhs_sign == 0 { lhs } else { lhs.wrapping_neg() };
            let rhs_abs = if rhs == 0 {
                0u32
            } else if rhs_sign == 0 {
                rhs
            } else {
                rhs.wrapping_neg()
            };

            let rhs_f = F::from_u64(rhs as u64);
            let rhs_inv = if rhs_f == F::ZERO { F::ZERO } else { rhs_f.inverse() };
            let rhs_is_zero = rhs == 0;

            let (q_abs, r_abs) = if rhs == 0 {
                (0u32, 0u32)
            } else {
                (lhs_abs / rhs_abs, lhs_abs % rhs_abs)
            };
            let r_is_zero = r_abs == 0;
            let r_f = F::from_u64(r_abs as u64);
            let r_inv = if r_f == F::ZERO { F::ZERO } else { r_f.inverse() };
            let diff = if rhs == 0 { 0u32 } else { r_abs.wrapping_sub(rhs_abs) };

            let mut packed = Vec::with_capacity(43);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(q_abs as u64));
            packed.push(F::from_u64(r_abs as u64));
            packed.push(rhs_inv);
            packed.push(f_bool::<F>(rhs_is_zero));
            packed.push(f_bool::<F>(lhs_sign == 1));
            packed.push(f_bool::<F>(rhs_sign == 1));
            packed.push(r_inv);
            packed.push(f_bool::<F>(r_is_zero));
            packed.push(F::from_u64(diff as u64));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((diff >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        RiscvOpcode::Remu => {
            let rhs_f = F::from_u64(rhs as u64);
            let rhs_inv = if rhs_f == F::ZERO { F::ZERO } else { rhs_f.inverse() };
            let rhs_is_zero = rhs == 0;

            let quot = if rhs == 0 {
                0u32
            } else {
                (lhs as u64 / rhs as u64) as u32
            };
            if rhs != 0 {
                let rem_check = ((lhs as u64) % (rhs as u64)) as u32;
                if rem_check != val {
                    return Err(PiCcsError::InvalidInput(format!(
                        "packed REMU helper: invalid remainder relation (rem_check={rem_check:#x}, val={val:#x})"
                    )));
                }
            }
            let diff = val.wrapping_sub(rhs);

            let mut packed = Vec::with_capacity(38);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(quot as u64));
            packed.push(rhs_inv);
            packed.push(f_bool::<F>(rhs_is_zero));
            packed.push(F::from_u64(diff as u64));
            for bit in 0..32usize {
                packed.push(f_bool::<F>(((diff >> bit) & 1) == 1));
            }
            Ok(packed)
        }
        _ => Err(PiCcsError::InvalidInput(format!(
            "packed RV32 col synthesis is unsupported for opcode={op:?}"
        ))),
    }
}
