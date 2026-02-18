use core::ops::Range;
use neo_reductions::error::PiCcsError;
use p3_field::PrimeCharacteristicRing;

use crate::cpu::BusLayout;
use crate::riscv::shout_oracle::RiscvAddressLookupOracleSparse;
use crate::witness::{LutInstance, LutTableSpec, MemInstance};

pub fn for_each_addr_bit_dim_major_le(addr: u64, d: usize, n_side: usize, ell: usize, mut f: impl FnMut(usize, bool)) {
    assert!(n_side > 0, "n_side must be > 0");
    let mut tmp = addr;
    for dim in 0..d {
        let comp = (tmp % (n_side as u64)) as u64;
        tmp /= n_side as u64;
        for bit in 0..ell {
            let idx = dim * ell + bit;
            let is_one = ((comp >> bit) & 1) == 1;
            f(idx, is_one);
        }
    }
}

pub fn write_addr_bits_dim_major_le<F: PrimeCharacteristicRing>(
    out_bits: &mut [F],
    addr: u64,
    d: usize,
    n_side: usize,
    ell: usize,
) {
    assert_eq!(out_bits.len(), d * ell, "addr_bits output length mismatch");
    for_each_addr_bit_dim_major_le(addr, d, n_side, ell, |idx, is_one| {
        out_bits[idx] = if is_one { F::ONE } else { F::ZERO };
    });
}

pub fn write_addr_bits_dim_major_le_into_bus<F: PrimeCharacteristicRing>(
    z: &mut [F],
    bus: &BusLayout,
    bit_cols: Range<usize>,
    j: usize,
    addr: u64,
    d: usize,
    n_side: usize,
    ell: usize,
) {
    assert_eq!(bit_cols.end - bit_cols.start, d * ell, "addr_bits bus range mismatch");
    for_each_addr_bit_dim_major_le(addr, d, n_side, ell, |idx, is_one| {
        let col_id = bit_cols.start + idx;
        z[bus.bus_cell(col_id, j)] = if is_one { F::ONE } else { F::ZERO };
    });
}

pub fn validate_pow2_bit_addressing(
    proto: &'static str,
    n_side: usize,
    d: usize,
    ell: usize,
    k: usize,
) -> Result<(), PiCcsError> {
    if n_side == 0 {
        return Err(PiCcsError::InvalidInput(format!("{proto}: n_side must be > 0")));
    }
    if !n_side.is_power_of_two() {
        return Err(PiCcsError::InvalidInput(format!(
            "{proto}: n_side={n_side} must be a power of two under bit addressing (otherwise a range proof is required)"
        )));
    }

    let expected_ell = n_side.trailing_zeros() as usize;
    if ell != expected_ell {
        return Err(PiCcsError::InvalidInput(format!(
            "{proto}: ell={ell} must equal log2(n_side)={expected_ell} for power-of-two n_side"
        )));
    }

    let expected_k = n_side
        .checked_pow(d as u32)
        .ok_or_else(|| PiCcsError::InvalidInput(format!("{proto}: n_side^d overflow")))?;
    if k != expected_k {
        return Err(PiCcsError::InvalidInput(format!(
            "{proto}: k={k} must equal n_side^d = {n_side}^{d} = {expected_k} for bit addressing"
        )));
    }

    Ok(())
}

pub fn validate_pow2_bit_addressing_shape(proto: &'static str, n_side: usize, ell: usize) -> Result<(), PiCcsError> {
    if n_side == 0 {
        return Err(PiCcsError::InvalidInput(format!("{proto}: n_side must be > 0")));
    }
    if !n_side.is_power_of_two() {
        return Err(PiCcsError::InvalidInput(format!(
            "{proto}: n_side={n_side} must be a power of two under bit addressing (otherwise a range proof is required)"
        )));
    }

    let expected_ell = n_side.trailing_zeros() as usize;
    if ell != expected_ell {
        return Err(PiCcsError::InvalidInput(format!(
            "{proto}: ell={ell} must equal log2(n_side)={expected_ell} for power-of-two n_side"
        )));
    }

    Ok(())
}

pub fn validate_shout_bit_addressing<Cmt, F>(inst: &LutInstance<Cmt, F>) -> Result<(), PiCcsError> {
    // Virtual/implicit tables may not have a materialized `k = n_side^d` table.
    if let Some(spec) = &inst.table_spec {
        let rv32_packed_expected_d = |opcode: crate::riscv::lookups::RiscvOpcode| -> Result<usize, PiCcsError> {
            Ok(match opcode {
                crate::riscv::lookups::RiscvOpcode::And
                | crate::riscv::lookups::RiscvOpcode::Andn
                | crate::riscv::lookups::RiscvOpcode::Xor
                | crate::riscv::lookups::RiscvOpcode::Or => 34usize,
                crate::riscv::lookups::RiscvOpcode::Add | crate::riscv::lookups::RiscvOpcode::Sub => 3usize,
                crate::riscv::lookups::RiscvOpcode::Eq | crate::riscv::lookups::RiscvOpcode::Neq => 35usize,
                crate::riscv::lookups::RiscvOpcode::Slt => 37usize,
                crate::riscv::lookups::RiscvOpcode::Sll => 38usize,
                crate::riscv::lookups::RiscvOpcode::Srl => 38usize,
                crate::riscv::lookups::RiscvOpcode::Sra => 38usize,
                crate::riscv::lookups::RiscvOpcode::Sltu => 35usize,
                crate::riscv::lookups::RiscvOpcode::Mul => 34usize,
                crate::riscv::lookups::RiscvOpcode::Mulh => 38usize,
                crate::riscv::lookups::RiscvOpcode::Mulhu => 34usize,
                crate::riscv::lookups::RiscvOpcode::Mulhsu => 37usize,
                crate::riscv::lookups::RiscvOpcode::Div => 43usize,
                crate::riscv::lookups::RiscvOpcode::Divu => 38usize,
                crate::riscv::lookups::RiscvOpcode::Rem => 43usize,
                crate::riscv::lookups::RiscvOpcode::Remu => 38usize,
                _ => {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V packed): unsupported opcode={opcode:?}"
                    )));
                }
            })
        };

        validate_pow2_bit_addressing_shape("Shout", inst.n_side, inst.ell)?;
        if inst.k != 0 {
            return Err(PiCcsError::InvalidInput(
                "Shout: k must be 0 when table_spec is set (implicit table)".into(),
            ));
        }
        if !inst.table.is_empty() {
            return Err(PiCcsError::InvalidInput(
                "Shout: table must be empty when table_spec is set".into(),
            ));
        }

        match spec {
            LutTableSpec::RiscvOpcode { opcode, xlen } => {
                RiscvAddressLookupOracleSparse::validate_spec(*opcode, *xlen)?;
                if inst.n_side != 2 || inst.ell != 1 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V): expected n_side=2, ell=1, got n_side={}, ell={}",
                        inst.n_side, inst.ell
                    )));
                }
                let expected_d = xlen
                    .checked_mul(2)
                    .ok_or_else(|| PiCcsError::InvalidInput("Shout(RISC-V): 2*xlen overflow".into()))?;
                if inst.d != expected_d {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V): expected d=2*xlen={}, got d={}",
                        expected_d, inst.d
                    )));
                }
            }
            LutTableSpec::RiscvOpcodePacked { opcode, xlen } => {
                if *xlen != 32 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V packed): expected xlen=32, got xlen={xlen}"
                    )));
                }
                let expected_d = rv32_packed_expected_d(*opcode)?;
                // Packed-key Shout lanes are not bit-addressed: we repurpose the addr-bit slice as
                // `[lhs_u32, rhs_u32, aux...]` and keep `[has_lookup, val_u32]`.
                if inst.n_side != 2 || inst.ell != 1 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V packed): expected n_side=2, ell=1, got n_side={}, ell={}",
                        inst.n_side, inst.ell
                    )));
                }
                if inst.d != expected_d {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V packed): expected d={expected_d}, got d={}",
                        inst.d,
                    )));
                }
            }
            LutTableSpec::RiscvOpcodeEventTablePacked {
                opcode,
                xlen,
                time_bits,
            } => {
                if *xlen != 32 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V event-table packed): expected xlen=32, got xlen={xlen}"
                    )));
                }
                if *time_bits == 0 {
                    return Err(PiCcsError::InvalidInput(
                        "Shout(RISC-V event-table packed): time_bits must be >= 1".into(),
                    ));
                }
                if *time_bits > 64 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V event-table packed): time_bits={time_bits} too large (max 64)"
                    )));
                }
                let base_d = rv32_packed_expected_d(*opcode)?;
                let expected_d = time_bits
                    .checked_add(base_d)
                    .ok_or_else(|| PiCcsError::InvalidInput("Shout(RISC-V event-table packed): d overflow".into()))?;

                // Event-table packed Shout lanes are not bit-addressed: addr_bits is repurposed as
                // `[time_bits_le, lhs_u32, rhs_u32, aux...]` and we keep `[has_lookup, val_u32]`.
                if inst.n_side != 2 || inst.ell != 1 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V event-table packed): expected n_side=2, ell=1, got n_side={}, ell={}",
                        inst.n_side, inst.ell
                    )));
                }
                if inst.d != expected_d {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(RISC-V event-table packed): expected d={expected_d} (= time_bits({time_bits}) + base_d({base_d})), got d={}",
                        inst.d,
                    )));
                }
            }
            LutTableSpec::IdentityU32 => {
                if inst.n_side != 2 || inst.ell != 1 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(IdentityU32): expected n_side=2, ell=1, got n_side={}, ell={}",
                        inst.n_side, inst.ell
                    )));
                }
                if inst.d != 32 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(IdentityU32): expected d=32, got d={}",
                        inst.d
                    )));
                }
            }
            LutTableSpec::Mul8 => {
                if inst.n_side != 2 || inst.ell != 1 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(Mul8): expected n_side=2, ell=1, got n_side={}, ell={}",
                        inst.n_side, inst.ell
                    )));
                }
                if inst.d != crate::riscv::mul_decomp::MUL8_ADDR_BITS {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(Mul8): expected d={}, got d={}",
                        crate::riscv::mul_decomp::MUL8_ADDR_BITS,
                        inst.d
                    )));
                }
            }
            LutTableSpec::Add8Acc => {
                if inst.n_side != 2 || inst.ell != 1 {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(Add8Acc): expected n_side=2, ell=1, got n_side={}, ell={}",
                        inst.n_side, inst.ell
                    )));
                }
                if inst.d != crate::riscv::mul_decomp::ADD8ACC_ADDR_BITS {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(Add8Acc): expected d={}, got d={}",
                        crate::riscv::mul_decomp::ADD8ACC_ADDR_BITS,
                        inst.d
                    )));
                }
            }
        }

        return Ok(());
    }

    // Explicit table mode (legacy).
    validate_pow2_bit_addressing("Shout", inst.n_side, inst.d, inst.ell, inst.k)?;
    if inst.table.len() != inst.k {
        return Err(PiCcsError::InvalidInput(format!(
            "Shout: table.len()={} must equal k={} for bit addressing",
            inst.table.len(),
            inst.k
        )));
    }
    Ok(())
}

pub fn validate_twist_bit_addressing<Cmt, F: PrimeCharacteristicRing>(
    inst: &MemInstance<Cmt, F>,
) -> Result<(), PiCcsError> {
    validate_pow2_bit_addressing("Twist", inst.n_side, inst.d, inst.ell, inst.k)?;
    inst.init.validate(inst.k)?;
    Ok(())
}
