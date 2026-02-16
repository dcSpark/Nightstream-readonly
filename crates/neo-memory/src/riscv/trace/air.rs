use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use super::{layout::Rv32TraceLayout, witness::Rv32TraceWitness};

#[derive(Clone, Debug)]
pub struct Rv32TraceAir {
    pub layout: Rv32TraceLayout,
}

impl Rv32TraceAir {
    pub fn new() -> Self {
        Self {
            layout: Rv32TraceLayout::new(),
        }
    }

    #[inline]
    fn is_zero(x: F) -> bool {
        x == F::ZERO
    }

    #[inline]
    fn bool_check(x: F) -> F {
        x * (x - F::ONE)
    }

    #[inline]
    fn gated_zero(gate: F, x: F) -> F {
        gate * x
    }

    #[inline]
    fn gated_eq(gate: F, a: F, b: F) -> F {
        gate * (a - b)
    }

    pub fn assert_satisfied(&self, wit: &Rv32TraceWitness) -> Result<(), String> {
        let l = &self.layout;
        if wit.cols.len() != l.cols {
            return Err(format!(
                "trace witness width mismatch: got {} cols, expected {}",
                wit.cols.len(),
                l.cols
            ));
        }
        for (c, col) in wit.cols.iter().enumerate() {
            if col.len() != wit.t {
                return Err(format!(
                    "trace witness column length mismatch at col {c}: got {}, expected {}",
                    col.len(),
                    wit.t
                ));
            }
        }

        let col = |c: usize, i: usize| -> F { wit.cols[c][i] };

        // Row-wise constraints.
        for i in 0..wit.t {
            let one = col(l.one, i);
            if one != F::ONE {
                return Err(format!("row {i}: one != 1"));
            }

            let active = col(l.active, i);
            let halted = col(l.halted, i);
            let rd_has_write = col(l.rd_has_write, i);
            let ram_has_read = col(l.ram_has_read, i);
            let ram_has_write = col(l.ram_has_write, i);
            let shout_has_lookup = col(l.shout_has_lookup, i);

            // Booleans.
            for (name, v) in [
                ("active", active),
                ("halted", halted),
                ("rd_has_write", rd_has_write),
                ("ram_has_read", ram_has_read),
                ("ram_has_write", ram_has_write),
                ("shout_has_lookup", shout_has_lookup),
            ] {
                let e = Self::bool_check(v);
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: {name} not boolean"));
                }
            }
            for (bit, c) in l.rd_bit.iter().copied().enumerate() {
                let e = Self::bool_check(col(c, i));
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: rd_bit[{bit}] not boolean"));
                }
            }
            for (bit, c) in l.funct3_bit.iter().copied().enumerate() {
                let e = Self::bool_check(col(c, i));
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: funct3_bit[{bit}] not boolean"));
                }
            }
            for (bit, c) in l.rs1_bit.iter().copied().enumerate() {
                let e = Self::bool_check(col(c, i));
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: rs1_bit[{bit}] not boolean"));
                }
            }
            for (bit, c) in l.rs2_bit.iter().copied().enumerate() {
                let e = Self::bool_check(col(c, i));
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: rs2_bit[{bit}] not boolean"));
                }
            }
            for (bit, c) in l.funct7_bit.iter().copied().enumerate() {
                let e = Self::bool_check(col(c, i));
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: funct7_bit[{bit}] not boolean"));
                }
            }
            // Padding invariants: inactive rows must not carry "hidden" values.
            let inv_active = F::ONE - active;
            for (name, c) in [
                ("instr_word", l.instr_word),
                ("opcode", l.opcode),
                ("funct3", l.funct3),
                ("prog_addr", l.prog_addr),
                ("prog_value", l.prog_value),
                ("rs1_addr", l.rs1_addr),
                ("rs1_val", l.rs1_val),
                ("rs2_addr", l.rs2_addr),
                ("rs2_val", l.rs2_val),
                ("rd_has_write", l.rd_has_write),
                ("rd_addr", l.rd_addr),
                ("rd_val", l.rd_val),
                ("ram_has_read", l.ram_has_read),
                ("ram_has_write", l.ram_has_write),
                ("ram_addr", l.ram_addr),
                ("ram_rv", l.ram_rv),
                ("ram_wv", l.ram_wv),
                ("shout_has_lookup", l.shout_has_lookup),
                ("shout_val", l.shout_val),
                ("shout_lhs", l.shout_lhs),
                ("shout_rhs", l.shout_rhs),
            ] {
                let e = Self::gated_zero(inv_active, col(c, i));
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: inactive padding violated ({name} != 0)"));
                }
            }

            // rd_is_zero prefix products.
            {
                let b0 = col(l.rd_bit[0], i);
                let b1 = col(l.rd_bit[1], i);
                let b2 = col(l.rd_bit[2], i);
                let b3 = col(l.rd_bit[3], i);
                let b4 = col(l.rd_bit[4], i);

                let z01 = col(l.rd_is_zero_01, i);
                let z012 = col(l.rd_is_zero_012, i);
                let z0123 = col(l.rd_is_zero_0123, i);
                let z = col(l.rd_is_zero, i);

                let e = z01 - (F::ONE - b0) * (F::ONE - b1);
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: rd_is_zero_01 mismatch"));
                }
                let e = z012 - z01 * (F::ONE - b2);
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: rd_is_zero_012 mismatch"));
                }
                let e = z0123 - z012 * (F::ONE - b3);
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: rd_is_zero_0123 mismatch"));
                }
                let e = z - z0123 * (F::ONE - b4);
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: rd_is_zero mismatch"));
                }
            }

            // Sound x0 invariant: if rd_has_write==1 then rd != 0.
            {
                let e = rd_has_write * col(l.rd_is_zero, i);
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: rd_has_write implies rd != 0 violated"));
                }
            }

            // If rd_has_write==0, write fields must be 0.
            {
                let inv = F::ONE - rd_has_write;
                if !Self::is_zero(Self::gated_zero(inv, col(l.rd_addr, i))) {
                    return Err(format!("row {i}: rd_addr must be 0 when rd_has_write=0"));
                }
                if !Self::is_zero(Self::gated_zero(inv, col(l.rd_val, i))) {
                    return Err(format!("row {i}: rd_val must be 0 when rd_has_write=0"));
                }
            }

            // RAM bus padding: inactive values must be 0 when their flags are 0.
            {
                if !Self::is_zero(Self::gated_zero(F::ONE - ram_has_read, col(l.ram_rv, i))) {
                    return Err(format!("row {i}: ram_rv must be 0 when ram_has_read=0"));
                }
                if !Self::is_zero(Self::gated_zero(F::ONE - ram_has_write, col(l.ram_wv, i))) {
                    return Err(format!("row {i}: ram_wv must be 0 when ram_has_write=0"));
                }
            }

            // Shout padding: if no lookup, the lookup output must be 0.
            {
                if !Self::is_zero(Self::gated_zero(F::ONE - shout_has_lookup, col(l.shout_val, i))) {
                    return Err(format!("row {i}: shout_val must be 0 when shout_has_lookup=0"));
                }
                if !Self::is_zero(Self::gated_zero(F::ONE - shout_has_lookup, col(l.shout_lhs, i))) {
                    return Err(format!("row {i}: shout_lhs must be 0 when shout_has_lookup=0"));
                }
                if !Self::is_zero(Self::gated_zero(F::ONE - shout_has_lookup, col(l.shout_rhs, i))) {
                    return Err(format!("row {i}: shout_rhs must be 0 when shout_has_lookup=0"));
                }
            }

            // Active → PROG fetch binds (pc_before, instr_word).
            {
                if !Self::is_zero(Self::gated_eq(active, col(l.prog_addr, i), col(l.pc_before, i))) {
                    return Err(format!("row {i}: PROG addr mismatch"));
                }
                if !Self::is_zero(Self::gated_eq(active, col(l.prog_value, i), col(l.instr_word, i))) {
                    return Err(format!("row {i}: PROG value mismatch"));
                }
            }

            // Active → REG addr bindings; rd_has_write → rd_addr binding.
            {
                let rs1_bits = col(l.rs1_bit[0], i)
                    + F::from_u64(2) * col(l.rs1_bit[1], i)
                    + F::from_u64(4) * col(l.rs1_bit[2], i)
                    + F::from_u64(8) * col(l.rs1_bit[3], i)
                    + F::from_u64(16) * col(l.rs1_bit[4], i);
                if !Self::is_zero(Self::gated_eq(active, col(l.rs1_addr, i), rs1_bits)) {
                    return Err(format!("row {i}: rs1_addr != packed rs1 bits"));
                }
                let rs2_bits = col(l.rs2_bit[0], i)
                    + F::from_u64(2) * col(l.rs2_bit[1], i)
                    + F::from_u64(4) * col(l.rs2_bit[2], i)
                    + F::from_u64(8) * col(l.rs2_bit[3], i)
                    + F::from_u64(16) * col(l.rs2_bit[4], i);
                if !Self::is_zero(Self::gated_eq(active, col(l.rs2_addr, i), rs2_bits)) {
                    return Err(format!("row {i}: rs2_addr != packed rs2 bits"));
                }
                let rd_bits = col(l.rd_bit[0], i)
                    + F::from_u64(2) * col(l.rd_bit[1], i)
                    + F::from_u64(4) * col(l.rd_bit[2], i)
                    + F::from_u64(8) * col(l.rd_bit[3], i)
                    + F::from_u64(16) * col(l.rd_bit[4], i);
                if !Self::is_zero(Self::gated_eq(rd_has_write, col(l.rd_addr, i), rd_bits)) {
                    return Err(format!("row {i}: rd_addr != packed rd bits when rd_has_write=1"));
                }
            }
        }

        // Transition constraints.
        for i in 0..wit.t.saturating_sub(1) {
            let e = col(l.pc_after, i) - col(l.pc_before, i + 1);
            if !Self::is_zero(e) {
                return Err(format!("pc chain mismatch at row {i}"));
            }

            let e = col(l.cycle, i + 1) - (col(l.cycle, i) + F::ONE);
            if !Self::is_zero(e) {
                return Err(format!("cycle chain mismatch at row {i}"));
            }

            // Once inactive, remain inactive.
            let a0 = col(l.active, i);
            let a1 = col(l.active, i + 1);
            if !Self::is_zero(a1 * (F::ONE - a0)) {
                return Err(format!("active monotonicity violated at row {i}"));
            }

            // Once halted, remain halted.
            let h0 = col(l.halted, i);
            let h1 = col(l.halted, i + 1);
            if !Self::is_zero(h0 * (F::ONE - h1)) {
                return Err(format!("halted monotonicity violated at row {i}"));
            }

            // HALT terminates execution: halted[i] => active[i+1] == 0.
            if !Self::is_zero(h0 * a1) {
                return Err(format!("halted tail quiescence violated at row {i}"));
            }
        }

        Ok(())
    }
}
