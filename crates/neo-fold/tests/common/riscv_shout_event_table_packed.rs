use neo_math::F;
use neo_memory::cpu::build_bus_layout_for_instances_with_shout_and_twist_lanes;
use neo_memory::riscv::exec_table::Rv32ShoutEventRow;
use neo_memory::riscv::lookups::RiscvOpcode;
use p3_field::{Field, PrimeCharacteristicRing};

pub fn ell_n_from_ccs_n(n: usize) -> usize {
    let n_pad = n.next_power_of_two().max(2);
    n_pad.trailing_zeros() as usize
}

pub fn rv32_packed_base_d(op: RiscvOpcode) -> Result<usize, String> {
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
            return Err(format!(
                "event-table packed: unsupported opcode={op:?} (no packed layout)"
            ));
        }
    })
}

fn mulh_hi_signed(lhs: u32, rhs: u32) -> u32 {
    let a = lhs as i32 as i64;
    let b = rhs as i32 as i64;
    let p = a * b;
    (p >> 32) as i32 as u32
}

fn mulhsu_hi_signed(lhs: u32, rhs: u32) -> u32 {
    let a = lhs as i32 as i64;
    let b = rhs as i64;
    let p = a * b;
    (p >> 32) as i32 as u32
}

fn div_signed(lhs: u32, rhs: u32) -> u32 {
    let lhs_i = lhs as i32;
    let rhs_i = rhs as i32;
    if rhs_i == 0 {
        return u32::MAX;
    }
    if lhs_i == i32::MIN && rhs_i == -1 {
        return lhs; // overflow case: quotient = MIN_INT
    }
    (lhs_i / rhs_i) as u32
}

fn rem_signed(lhs: u32, rhs: u32) -> u32 {
    let lhs_i = lhs as i32;
    let rhs_i = rhs as i32;
    if rhs_i == 0 {
        return lhs;
    }
    if lhs_i == i32::MIN && rhs_i == -1 {
        return 0; // overflow case: remainder = 0
    }
    (lhs_i % rhs_i) as u32
}

fn divu(lhs: u32, rhs: u32) -> u32 {
    if rhs == 0 {
        return u32::MAX;
    }
    (lhs as u64 / rhs as u64) as u32
}

fn remu(lhs: u32, rhs: u32) -> u32 {
    if rhs == 0 {
        return lhs;
    }
    ((lhs as u64) % (rhs as u64)) as u32
}

pub fn rv32_expected_val(op: RiscvOpcode, lhs: u32, rhs: u32) -> Result<u32, String> {
    Ok(match op {
        RiscvOpcode::And => lhs & rhs,
        RiscvOpcode::Andn => lhs & !rhs,
        RiscvOpcode::Or => lhs | rhs,
        RiscvOpcode::Xor => lhs ^ rhs,
        RiscvOpcode::Add => lhs.wrapping_add(rhs),
        RiscvOpcode::Sub => lhs.wrapping_sub(rhs),
        RiscvOpcode::Eq => (lhs == rhs) as u32,
        RiscvOpcode::Neq => (lhs != rhs) as u32,
        RiscvOpcode::Sltu => (lhs < rhs) as u32,
        RiscvOpcode::Slt => ((lhs as i32) < (rhs as i32)) as u32,
        RiscvOpcode::Sll => lhs.wrapping_shl(rhs & 0x1F),
        RiscvOpcode::Srl => lhs.wrapping_shr(rhs & 0x1F),
        RiscvOpcode::Sra => ((lhs as i32) >> (rhs & 0x1F)) as u32,
        RiscvOpcode::Mul => lhs.wrapping_mul(rhs),
        RiscvOpcode::Mulhu => (((lhs as u64) * (rhs as u64)) >> 32) as u32,
        RiscvOpcode::Mulh => mulh_hi_signed(lhs, rhs),
        RiscvOpcode::Mulhsu => mulhsu_hi_signed(lhs, rhs),
        RiscvOpcode::Div => div_signed(lhs, rhs),
        RiscvOpcode::Rem => rem_signed(lhs, rhs),
        RiscvOpcode::Divu => divu(lhs, rhs),
        RiscvOpcode::Remu => remu(lhs, rhs),
        _ => {
            return Err(format!(
                "event-table packed: expected value unsupported for opcode={op:?}"
            ));
        }
    })
}

pub fn build_rv32_event_table_packed_cols(op: RiscvOpcode, lhs: u32, rhs: u32, val: u32) -> Result<Vec<F>, String> {
    match op {
        RiscvOpcode::And | RiscvOpcode::Andn | RiscvOpcode::Or | RiscvOpcode::Xor => {
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!(
                    "packed {op:?}: val mismatch (got {val:#x}, expected {expected:#x})"
                ));
            }

            let mut lhs_digits = Vec::with_capacity(16);
            let mut rhs_digits = Vec::with_capacity(16);
            for i in 0..16usize {
                lhs_digits.push(F::from_u64(((lhs >> (2 * i)) & 3) as u64));
                rhs_digits.push(F::from_u64(((rhs >> (2 * i)) & 3) as u64));
            }
            let mut packed = Vec::with_capacity(34);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.extend_from_slice(&lhs_digits);
            packed.extend_from_slice(&rhs_digits);
            if packed.len() != 34 {
                return Err("packed bitwise: digit packing length mismatch".into());
            }
            Ok(packed)
        }
        RiscvOpcode::Add => {
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!(
                    "packed ADD: val mismatch (got {val:#x}, expected {expected:#x})"
                ));
            }
            let carry = ((lhs as u64 + rhs as u64) >> 32) & 1;
            Ok(vec![
                F::from_u64(lhs as u64),
                F::from_u64(rhs as u64),
                F::from_u64(carry),
            ])
        }
        RiscvOpcode::Sub => {
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!(
                    "packed SUB: val mismatch (got {val:#x}, expected {expected:#x})"
                ));
            }
            let borrow = if lhs < rhs { 1u64 } else { 0u64 };
            Ok(vec![
                F::from_u64(lhs as u64),
                F::from_u64(rhs as u64),
                F::from_u64(borrow),
            ])
        }
        RiscvOpcode::Eq | RiscvOpcode::Neq => {
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!("packed {op:?}: val mismatch (got {val}, expected {expected})"));
            }
            let borrow = if lhs < rhs { 1u64 } else { 0u64 };
            let diff = lhs.wrapping_sub(rhs);

            let mut packed = Vec::with_capacity(35);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(if borrow == 1 { F::ONE } else { F::ZERO });
            for bit in 0..32usize {
                packed.push(if ((diff >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            if packed.len() != 35 {
                return Err("packed EQ/NEQ: length mismatch".into());
            }
            Ok(packed)
        }
        RiscvOpcode::Sltu => {
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!("packed SLTU: val mismatch (got {val}, expected {expected})"));
            }
            let diff = lhs.wrapping_sub(rhs);
            let mut packed = Vec::with_capacity(35);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(diff as u64));
            for bit in 0..32usize {
                packed.push(if ((diff >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            if packed.len() != 35 {
                return Err("packed SLTU: length mismatch".into());
            }
            Ok(packed)
        }
        RiscvOpcode::Slt => {
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!("packed SLT: val mismatch (got {val}, expected {expected})"));
            }

            let lhs_sign = (lhs >> 31) & 1;
            let rhs_sign = (rhs >> 31) & 1;
            let lhs_b = lhs ^ 0x8000_0000;
            let rhs_b = rhs ^ 0x8000_0000;
            let diff = lhs_b.wrapping_sub(rhs_b);

            let mut packed = Vec::with_capacity(37);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(diff as u64));
            packed.push(if lhs_sign == 1 { F::ONE } else { F::ZERO });
            packed.push(if rhs_sign == 1 { F::ONE } else { F::ZERO });
            for bit in 0..32usize {
                packed.push(if ((diff >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            if packed.len() != 37 {
                return Err("packed SLT: length mismatch".into());
            }
            Ok(packed)
        }
        RiscvOpcode::Sll => {
            let shamt = rhs & 0x1F;
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!(
                    "packed SLL: val mismatch (got {val:#x}, expected {expected:#x})"
                ));
            }
            let wide = (lhs as u64) << shamt;
            let carry = (wide >> 32) as u32;

            let mut packed = Vec::with_capacity(38);
            packed.push(F::from_u64(lhs as u64));
            for bit in 0..5usize {
                packed.push(if ((shamt >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            for bit in 0..32usize {
                packed.push(if ((carry >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            if packed.len() != 38 {
                return Err("packed SLL: length mismatch".into());
            }
            Ok(packed)
        }
        RiscvOpcode::Srl => {
            let shamt = rhs & 0x1F;
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!(
                    "packed SRL: val mismatch (got {val:#x}, expected {expected:#x})"
                ));
            }
            let rem: u32 = if shamt == 0 {
                0
            } else {
                let mask = (1u64 << shamt) - 1;
                ((lhs as u64) & mask) as u32
            };

            let mut packed = Vec::with_capacity(38);
            packed.push(F::from_u64(lhs as u64));
            for bit in 0..5usize {
                packed.push(if ((shamt >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            for bit in 0..32usize {
                packed.push(if ((rem >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            if packed.len() != 38 {
                return Err("packed SRL: length mismatch".into());
            }
            Ok(packed)
        }
        RiscvOpcode::Sra => {
            let shamt = rhs & 0x1F;
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!(
                    "packed SRA: val mismatch (got {val:#x}, expected {expected:#x})"
                ));
            }
            let sign = (lhs >> 31) & 1;

            let lhs_signed: i64 = if sign == 1 {
                (lhs as i64) - (1i64 << 32)
            } else {
                lhs as i64
            };
            let val_signed: i64 = (val as i64) - (sign as i64) * (1i64 << 32);
            let pow2: i64 = 1i64 << shamt;
            let rem_i64 = lhs_signed - val_signed * pow2;
            if rem_i64 < 0 {
                return Err("packed SRA: negative remainder".into());
            }
            let rem = rem_i64 as u64;

            let mut packed = Vec::with_capacity(38);
            packed.push(F::from_u64(lhs as u64));
            for bit in 0..5usize {
                packed.push(if ((shamt >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            packed.push(if sign == 1 { F::ONE } else { F::ZERO });
            for bit in 0..31usize {
                packed.push(if ((rem >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            if packed.len() != 38 {
                return Err("packed SRA: length mismatch".into());
            }
            Ok(packed)
        }
        RiscvOpcode::Mul => {
            let wide = (lhs as u64) * (rhs as u64);
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!(
                    "packed MUL: val mismatch (got {val:#x}, expected {expected:#x})"
                ));
            }
            let hi = (wide >> 32) as u32;

            let mut packed = Vec::with_capacity(34);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            for bit in 0..32usize {
                packed.push(if ((hi >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            if packed.len() != 34 {
                return Err("packed MUL: length mismatch".into());
            }
            Ok(packed)
        }
        RiscvOpcode::Mulhu => {
            let wide = (lhs as u64) * (rhs as u64);
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!(
                    "packed MULHU: val mismatch (got {val:#x}, expected {expected:#x})"
                ));
            }
            let lo = (wide & 0xffff_ffff) as u32;

            let mut packed = Vec::with_capacity(34);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            for bit in 0..32usize {
                packed.push(if ((lo >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            if packed.len() != 34 {
                return Err("packed MULHU: length mismatch".into());
            }
            Ok(packed)
        }
        RiscvOpcode::Mulh => {
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!(
                    "packed MULH: val mismatch (got {val:#x}, expected {expected:#x})"
                ));
            }

            let uprod = (lhs as u64) * (rhs as u64);
            let lo = (uprod & 0xffff_ffff) as u32;
            let hi = (uprod >> 32) as u32;
            let lhs_sign = (lhs >> 31) & 1;
            let rhs_sign = (rhs >> 31) & 1;

            let diff =
                (val as i128) - (hi as i128) + (lhs_sign as i128) * (rhs as i128) + (rhs_sign as i128) * (lhs as i128);
            let two32 = 1_i128 << 32;
            if diff < 0 || diff % two32 != 0 {
                return Err(format!("packed MULH: invalid k (diff={diff})"));
            }
            let k = (diff / two32) as u32;
            if k > 2 {
                return Err(format!("packed MULH: k out of range (k={k})"));
            }

            let mut packed = Vec::with_capacity(38);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(hi as u64));
            packed.push(if lhs_sign == 1 { F::ONE } else { F::ZERO });
            packed.push(if rhs_sign == 1 { F::ONE } else { F::ZERO });
            packed.push(F::from_u64(k as u64));
            for bit in 0..32usize {
                packed.push(if ((lo >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            if packed.len() != 38 {
                return Err("packed MULH: length mismatch".into());
            }
            Ok(packed)
        }
        RiscvOpcode::Mulhsu => {
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!(
                    "packed MULHSU: val mismatch (got {val:#x}, expected {expected:#x})"
                ));
            }

            let uprod = (lhs as u64) * (rhs as u64);
            let lo = (uprod & 0xffff_ffff) as u32;
            let hi = (uprod >> 32) as u32;
            let lhs_sign = (lhs >> 31) & 1;

            let diff = (val as i128) - (hi as i128) + (lhs_sign as i128) * (rhs as i128);
            let two32 = 1_i128 << 32;
            if diff < 0 || diff % two32 != 0 {
                return Err(format!("packed MULHSU: invalid borrow (diff={diff})"));
            }
            let borrow = (diff / two32) as u32;
            if borrow > 1 {
                return Err(format!("packed MULHSU: borrow out of range (borrow={borrow})"));
            }

            let mut packed = Vec::with_capacity(37);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(hi as u64));
            packed.push(if lhs_sign == 1 { F::ONE } else { F::ZERO });
            packed.push(if borrow == 1 { F::ONE } else { F::ZERO });
            for bit in 0..32usize {
                packed.push(if ((lo >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            if packed.len() != 37 {
                return Err("packed MULHSU: length mismatch".into());
            }
            Ok(packed)
        }
        RiscvOpcode::Div => {
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!(
                    "packed DIV: val mismatch (got {val:#x}, expected {expected:#x})"
                ));
            }

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
            packed.push(if rhs_is_zero { F::ONE } else { F::ZERO });
            packed.push(if lhs_sign == 1 { F::ONE } else { F::ZERO });
            packed.push(if rhs_sign == 1 { F::ONE } else { F::ZERO });
            packed.push(q_inv);
            packed.push(if q_is_zero { F::ONE } else { F::ZERO });
            packed.push(F::from_u64(diff as u64));
            for bit in 0..32usize {
                packed.push(if ((diff >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            if packed.len() != 43 {
                return Err("packed DIV: length mismatch".into());
            }
            Ok(packed)
        }
        RiscvOpcode::Rem => {
            let expected = rv32_expected_val(op, lhs, rhs)?;
            if val != expected {
                return Err(format!(
                    "packed REM: val mismatch (got {val:#x}, expected {expected:#x})"
                ));
            }

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
            packed.push(if rhs_is_zero { F::ONE } else { F::ZERO });
            packed.push(if lhs_sign == 1 { F::ONE } else { F::ZERO });
            packed.push(if rhs_sign == 1 { F::ONE } else { F::ZERO });
            packed.push(r_inv);
            packed.push(if r_is_zero { F::ONE } else { F::ZERO });
            packed.push(F::from_u64(diff as u64));
            for bit in 0..32usize {
                packed.push(if ((diff >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            if packed.len() != 43 {
                return Err("packed REM: length mismatch".into());
            }
            Ok(packed)
        }
        RiscvOpcode::Divu => {
            let expected_quot = rv32_expected_val(op, lhs, rhs)?;
            if val != expected_quot {
                return Err(format!(
                    "packed DIVU: val mismatch (got {val:#x}, expected {expected_quot:#x})"
                ));
            }

            let rhs_f = F::from_u64(rhs as u64);
            let rhs_inv = if rhs_f == F::ZERO { F::ZERO } else { rhs_f.inverse() };
            let rhs_is_zero = rhs == 0;

            let rem = if rhs == 0 {
                0u32
            } else {
                let r = ((lhs as u64) % (rhs as u64)) as u32;
                // Cross-check with quotient:
                let r2 = (lhs as u64).wrapping_sub((rhs as u64).wrapping_mul(val as u64)) as u32;
                if r2 != r {
                    return Err(format!(
                        "packed DIVU: remainder mismatch (lhs={lhs:#x}, rhs={rhs:#x}, quot={val:#x}, r2={r2:#x}, r={r:#x})"
                    ));
                }
                r
            };
            let diff = rem.wrapping_sub(rhs);

            let mut packed = Vec::with_capacity(38);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(rem as u64));
            packed.push(rhs_inv);
            packed.push(if rhs_is_zero { F::ONE } else { F::ZERO });
            packed.push(F::from_u64(diff as u64));
            for bit in 0..32usize {
                packed.push(if ((diff >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            if packed.len() != 38 {
                return Err("packed DIVU: length mismatch".into());
            }
            Ok(packed)
        }
        RiscvOpcode::Remu => {
            let expected_rem = rv32_expected_val(op, lhs, rhs)?;
            if val != expected_rem {
                return Err(format!(
                    "packed REMU: val mismatch (got {val:#x}, expected {expected_rem:#x})"
                ));
            }

            let rhs_f = F::from_u64(rhs as u64);
            let rhs_inv = if rhs_f == F::ZERO { F::ZERO } else { rhs_f.inverse() };
            let rhs_is_zero = rhs == 0;

            let quot = if rhs == 0 {
                0u32
            } else {
                (lhs as u64 / rhs as u64) as u32
            };
            if rhs != 0 {
                let rem2 = ((lhs as u64) % (rhs as u64)) as u32;
                if rem2 != val {
                    return Err(format!(
                        "packed REMU: remainder mismatch (lhs={lhs:#x}, rhs={rhs:#x}, quot={quot:#x}, rem={val:#x}, rem2={rem2:#x})"
                    ));
                }
            }
            let diff = val.wrapping_sub(rhs);

            let mut packed = Vec::with_capacity(38);
            packed.push(F::from_u64(lhs as u64));
            packed.push(F::from_u64(rhs as u64));
            packed.push(F::from_u64(quot as u64));
            packed.push(rhs_inv);
            packed.push(if rhs_is_zero { F::ONE } else { F::ZERO });
            packed.push(F::from_u64(diff as u64));
            for bit in 0..32usize {
                packed.push(if ((diff >> bit) & 1) == 1 { F::ONE } else { F::ZERO });
            }
            if packed.len() != 38 {
                return Err("packed REMU: length mismatch".into());
            }
            Ok(packed)
        }
        _ => Err(format!("event-table packed cols: unsupported opcode={op:?}")),
    }
}

pub fn build_shout_event_table_bus_z(
    m: usize,
    m_in: usize,
    steps: usize,
    ell_n: usize,
    op: RiscvOpcode,
    rows: &[Rv32ShoutEventRow],
    x_prefix: &[F],
) -> Result<Vec<F>, String> {
    if steps == 0 {
        return Err("build_shout_event_table_bus_z: steps=0".into());
    }
    if rows.len() != steps {
        return Err("build_shout_event_table_bus_z: rows/steps mismatch".into());
    }
    if x_prefix.len() != m_in {
        return Err(format!(
            "build_shout_event_table_bus_z: x_prefix.len()={} != m_in={}",
            x_prefix.len(),
            m_in
        ));
    }

    let base_d = rv32_packed_base_d(op)?;
    let ell_addr = ell_n
        .checked_add(base_d)
        .ok_or_else(|| "build_shout_event_table_bus_z: ell_addr overflow".to_string())?;

    let bus = build_bus_layout_for_instances_with_shout_and_twist_lanes(
        m,
        m_in,
        steps,
        core::iter::once((ell_addr, /*lanes=*/ 1usize)),
        core::iter::empty::<(usize, usize)>(),
    )?;
    if bus.shout_cols.len() != 1 || !bus.twist_cols.is_empty() {
        return Err("build_shout_event_table_bus_z: expected 1 shout instance and 0 twist instances".into());
    }

    let mut z = vec![F::ZERO; m];
    z[..m_in].copy_from_slice(x_prefix);

    let cols = &bus.shout_cols[0].lanes[0];
    for (j, row) in rows.iter().enumerate() {
        z[bus.bus_cell(cols.has_lookup, j)] = F::ONE;
        z[bus.bus_cell(cols.primary_val(), j)] = F::from_u64(row.value as u64);

        let t_idx = m_in
            .checked_add(row.row_idx)
            .ok_or_else(|| "build_shout_event_table_bus_z: time index overflow".to_string())?;
        let mut time_bits = vec![F::ZERO; ell_n];
        for b in 0..ell_n {
            let bit = ((t_idx as u64) >> b) & 1;
            time_bits[b] = if bit == 1 { F::ONE } else { F::ZERO };
        }

        let lhs = row.lhs as u32;
        let rhs = row.rhs as u32;
        let val = row.value as u32;
        let packed_cols = build_rv32_event_table_packed_cols(op, lhs, rhs, val)?;
        if packed_cols.len() != base_d {
            return Err("build_shout_event_table_bus_z: packed cols length mismatch".into());
        }

        for (idx, col_id) in cols.addr_bits.clone().enumerate() {
            let v = if idx < ell_n {
                time_bits[idx]
            } else {
                packed_cols[idx - ell_n]
            };
            z[bus.bus_cell(col_id, j)] = v;
        }
    }

    Ok(z)
}
