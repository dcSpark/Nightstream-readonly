use neo_ccs::relations::CcsStructure;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use crate::riscv::exec_table::Rv32ExecTable;
use crate::riscv::trace::{Rv32TraceLayout, Rv32TraceWitness};

use super::constraint_builder::{build_r1cs_ccs, Constraint, UniformConstraintKey, UniformConstraintRow};

/// Fixed-width, time-in-rows trace CCS layout.
///
/// This is a Tier 2.1 trace CCS with fixed columns over time (`t` rows),
/// AIR-like wiring invariants, and a compact subset of ISA semantics guards.
#[derive(Clone, Debug)]
pub struct Rv32TraceCcsLayout {
    pub t: usize,
    pub m_in: usize,
    pub m: usize,

    // Public scalars.
    pub const_one: usize,
    pub pc0: usize,
    pub pc_final: usize,
    pub halted_in: usize,
    pub halted_out: usize,

    pub trace_base: usize,
    pub trace: Rv32TraceLayout,
}

impl Rv32TraceCcsLayout {
    pub fn new(t: usize) -> Result<Self, String> {
        Self::new_internal(t)
    }

    /// Uniform Route-A kernel layout:
    /// - physical witness width is column-based (`m_in + trace.cols`),
    /// - time is handled by the batched time-domain sumcheck path.
    pub fn new_uniform(t: usize) -> Result<Self, String> {
        Self::new_internal(t)
    }

    fn new_internal(t: usize) -> Result<Self, String> {
        if t == 0 {
            return Err("Rv32TraceCcsLayout: t must be >= 1".into());
        }

        let const_one: usize = 0;
        let pc0: usize = 1;
        let pc_final: usize = 2;
        let halted_in: usize = 3;
        let halted_out: usize = 4;
        let m_in: usize = 5;

        let trace = Rv32TraceLayout::new();
        let trace_base = m_in;
        let trace_len = trace.cols;
        let m = trace_base
            .checked_add(trace_len)
            .ok_or_else(|| "Rv32TraceCcsLayout: m overflow".to_string())?;

        Ok(Self {
            t,
            m_in,
            m,
            const_one,
            pc0,
            pc_final,
            halted_in,
            halted_out,
            trace_base,
            trace,
        })
    }

    /// Physical witness index for a trace column in uniform mode.
    ///
    /// `row` is kept in the API for call-site compatibility and must satisfy `row < t`,
    /// but does not affect the physical index.
    #[inline]
    pub fn cell(&self, trace_col: usize, row: usize) -> usize {
        debug_assert!(trace_col < self.trace.cols);
        debug_assert!(row < self.t);
        assert!(
            row == 0,
            "uniform kernel physical cell indexing must not use per-row offsets"
        );
        self.trace_base + trace_col
    }

    #[inline]
    pub fn is_uniform_kernel(&self) -> bool {
        true
    }
}

/// Build the public inputs `x` and witness `w` for the trace CCS from an exec table.
pub fn rv32_trace_ccs_witness_from_exec_table(
    layout: &Rv32TraceCcsLayout,
    exec: &Rv32ExecTable,
) -> Result<(Vec<F>, Vec<F>), String> {
    let wit = Rv32TraceWitness::from_exec_table(&layout.trace, exec)?;
    rv32_trace_ccs_witness_from_trace_witness(layout, &wit)
}

/// Build the public inputs `x` and witness `w` for the trace CCS from a trace witness.
pub fn rv32_trace_ccs_witness_from_trace_witness(
    layout: &Rv32TraceCcsLayout,
    wit: &Rv32TraceWitness,
) -> Result<(Vec<F>, Vec<F>), String> {
    if wit.t != layout.t {
        return Err(format!(
            "trace CCS witness: t mismatch (wit.t={} layout.t={})",
            wit.t, layout.t
        ));
    }
    if wit.cols.len() != layout.trace.cols {
        return Err(format!(
            "trace CCS witness: width mismatch (wit.cols={} trace.cols={})",
            wit.cols.len(),
            layout.trace.cols
        ));
    }

    let mut x = vec![F::ZERO; layout.m_in];
    x[layout.const_one] = F::ONE;
    x[layout.pc0] = wit.cols[layout.trace.pc_before][0];
    x[layout.pc_final] = wit.cols[layout.trace.pc_after][layout.t - 1];
    x[layout.halted_in] = wit.cols[layout.trace.halted][0];
    x[layout.halted_out] = wit.cols[layout.trace.halted][layout.t - 1];

    // Uniform kernel keeps one physical slot per trace column and delegates time checks
    // to Route-A time-domain sumchecks.
    let mut w = vec![F::ZERO; layout.m - layout.m_in];
    for trace_col in 0..layout.trace.cols {
        w[trace_col] = wit.cols[trace_col][0];
    }

    Ok((x, w))
}

/// Build the base trace CCS (wiring invariants + partial ISA semantics guards).
pub fn build_rv32_trace_wiring_ccs(layout: &Rv32TraceCcsLayout) -> Result<CcsStructure<F>, String> {
    build_rv32_trace_wiring_ccs_with_reserved_rows(layout, 0)
}

pub fn build_rv32_trace_wiring_ccs_with_reserved_rows(
    layout: &Rv32TraceCcsLayout,
    reserved_rows: usize,
) -> Result<CcsStructure<F>, String> {
    build_rv32_trace_uniform_kernel_ccs_with_reserved_rows(layout, reserved_rows)
}

fn legacy_trace_core_rows(t: usize) -> Result<usize, String> {
    t.checked_mul(10)
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| "RV32 trace CCS: core row count overflow".to_string())
}

fn build_rv32_trace_uniform_kernel_ccs_with_reserved_rows(
    layout: &Rv32TraceCcsLayout,
    reserved_rows: usize,
) -> Result<CcsStructure<F>, String> {
    let one = layout.const_one;
    let tr = |c: usize| -> usize { layout.cell(c, 0) };
    let l = &layout.trace;

    // Keep a minimal anchor relation in the physical CCS and push full time semantics
    // into the Route-A time-domain sumchecks.
    let cons: Vec<Constraint<F>> = vec![
        Constraint::terms(one, false, vec![(tr(l.one), F::ONE), (one, -F::ONE)]),
        Constraint::terms(one, false, vec![(layout.pc0, F::ONE), (tr(l.pc_before), -F::ONE)]),
        Constraint::terms(one, false, vec![(layout.halted_in, F::ONE), (tr(l.halted), -F::ONE)]),
        Constraint::terms(one, false, vec![(tr(l.active), F::ONE), (one, -F::ONE)]),
        Constraint::terms(tr(l.shout_has_lookup), true, vec![(tr(l.shout_val), F::ONE)]),
        Constraint::terms(tr(l.shout_has_lookup), true, vec![(tr(l.shout_lhs), F::ONE)]),
        Constraint::terms(tr(l.shout_has_lookup), true, vec![(tr(l.shout_rhs), F::ONE)]),
    ];

    let n = legacy_trace_core_rows(layout.t)?
        .checked_add(reserved_rows)
        .ok_or_else(|| "RV32 trace CCS: n overflow".to_string())?;
    build_r1cs_ccs(&cons, n, layout.m, layout.const_one)
}

/// Build a time-independent per-step key for the RV32 trace constraints.
///
/// Column ids are per-step ids:
/// - `0..m_in` are public columns,
/// - `m_in..(m_in + trace.cols)` are trace columns.
///
/// Shift constraints reference next-step values using virtual ids:
/// `next(col) = m_cols + col`.
pub fn build_rv32_uniform_constraint_key() -> UniformConstraintKey<F> {
    build_rv32_uniform_constraint_key_with_m_in(5)
}

/// Build a time-independent per-step key for the RV32 trace constraints with a caller-provided
/// public-input prefix width.
pub fn build_rv32_uniform_constraint_key_with_m_in(m_in: usize) -> UniformConstraintKey<F> {
    assert!(
        m_in >= 5,
        "build_rv32_uniform_constraint_key_with_m_in requires m_in >= 5"
    );
    let trace = Rv32TraceLayout::new();
    let m_cols = m_in + trace.cols;
    let one = 0usize;
    let pc0 = 1usize;
    let pc_final = 2usize;
    let halted_in = 3usize;
    let halted_out = 4usize;
    let tr = |c: usize| -> usize { m_in + c };
    let next = |c: usize| -> usize { m_cols + c };

    let mut key = UniformConstraintKey::new(m_cols);

    // Boundary/public rows (evaluated outside the per-step local sum).
    key.boundary_rows.push(UniformConstraintRow::from_terms(
        [(one, F::ONE)],
        [(pc0, F::ONE), (tr(trace.pc_before), -F::ONE)],
        [],
    ));
    key.boundary_rows.push(UniformConstraintRow::from_terms(
        [(one, F::ONE)],
        [(pc_final, F::ONE), (tr(trace.pc_after), -F::ONE)],
        [],
    ));
    key.boundary_rows.push(UniformConstraintRow::from_terms(
        [(one, F::ONE)],
        [(halted_in, F::ONE), (tr(trace.halted), -F::ONE)],
        [],
    ));
    key.boundary_rows.push(UniformConstraintRow::from_terms(
        [(one, F::ONE)],
        [(halted_out, F::ONE), (tr(trace.halted), -F::ONE)],
        [],
    ));

    // Local rows.
    key.local_rows.push(UniformConstraintRow::from_terms(
        [(one, F::ONE)],
        [(tr(trace.one), F::ONE), (one, -F::ONE)],
        [],
    ));

    // Shout padding: (1 - has_lookup) * val == 0
    key.local_rows.push(UniformConstraintRow::from_terms(
        [(one, F::ONE), (tr(trace.shout_has_lookup), -F::ONE)],
        [(tr(trace.shout_val), F::ONE)],
        [],
    ));
    key.local_rows.push(UniformConstraintRow::from_terms(
        [(one, F::ONE), (tr(trace.shout_has_lookup), -F::ONE)],
        [(tr(trace.shout_lhs), F::ONE)],
        [],
    ));
    key.local_rows.push(UniformConstraintRow::from_terms(
        [(one, F::ONE), (tr(trace.shout_has_lookup), -F::ONE)],
        [(tr(trace.shout_rhs), F::ONE)],
        [],
    ));

    // Shift rows over j -> j+1.
    key.shift_rows.push(UniformConstraintRow::from_terms(
        [(one, F::ONE)],
        [(tr(trace.pc_after), F::ONE), (next(tr(trace.pc_before)), -F::ONE)],
        [],
    ));
    key.shift_rows.push(UniformConstraintRow::from_terms(
        [(one, F::ONE)],
        [
            (next(tr(trace.cycle)), F::ONE),
            (tr(trace.cycle), -F::ONE),
            (one, -F::ONE),
        ],
        [],
    ));
    key.shift_rows.push(UniformConstraintRow::from_terms(
        [(next(tr(trace.active)), F::ONE)],
        [(one, F::ONE), (tr(trace.active), -F::ONE)],
        [],
    ));
    key.shift_rows.push(UniformConstraintRow::from_terms(
        [(tr(trace.halted), F::ONE)],
        [(one, F::ONE), (next(tr(trace.halted)), -F::ONE)],
        [],
    ));
    key.shift_rows.push(UniformConstraintRow::from_terms(
        [(tr(trace.halted), F::ONE)],
        [(next(tr(trace.active)), F::ONE)],
        [],
    ));
    key.shift_rows.push(UniformConstraintRow::from_terms(
        [(tr(trace.halted), F::ONE)],
        [(tr(trace.pc_after), F::ONE), (next(tr(trace.pc_after)), -F::ONE)],
        [],
    ));

    key
}
