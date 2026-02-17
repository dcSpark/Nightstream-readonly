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
            let shout_has_lookup = col(l.shout_has_lookup, i);

            // Booleans.
            for (name, v) in [
                ("active", active),
                ("halted", halted),
                ("shout_has_lookup", shout_has_lookup),
            ] {
                let e = Self::bool_check(v);
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: {name} not boolean"));
                }
            }
            // Padding invariants: inactive rows must not carry "hidden" values.
            let inv_active = F::ONE - active;
            for (name, c) in [
                ("instr_word", l.instr_word),
                ("rs1_addr", l.rs1_addr),
                ("rs1_val", l.rs1_val),
                ("rs2_addr", l.rs2_addr),
                ("rs2_val", l.rs2_val),
                ("rd_addr", l.rd_addr),
                ("rd_val", l.rd_val),
                ("ram_addr", l.ram_addr),
                ("ram_rv", l.ram_rv),
                ("ram_wv", l.ram_wv),
                ("shout_has_lookup", l.shout_has_lookup),
                ("shout_val", l.shout_val),
                ("shout_lhs", l.shout_lhs),
                ("shout_rhs", l.shout_rhs),
                ("jalr_drop_bit", l.jalr_drop_bit),
            ] {
                let e = Self::gated_zero(inv_active, col(c, i));
                if !Self::is_zero(e) {
                    return Err(format!("row {i}: inactive padding violated ({name} != 0)"));
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
