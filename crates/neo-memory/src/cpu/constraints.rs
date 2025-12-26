//! CPU constraint builder for Neo.
//!
//! This module provides R1CS/CCS constraints that bind CPU semantics to the shared
//! memory/lookup bus. These constraints are **critical for security** - without them,
//! a malicious prover can create divergent CPU and memory states.
//!
//! # Credits
//!
//! The constraint logic in this module is ported from the Jolt zkVM project:
//! - Repository: https://github.com/a16z/jolt
//! - Original file: `jolt-core/src/zkvm/r1cs/constraints.rs`
//! - License: Apache-2.0 / MIT
//!
//! Jolt's R1CS constraint system defines the binding between CPU instruction semantics
//! and memory/lookup values. We adapt this for Neo's CCS-based proving system and
//! shared CPU bus architecture.
//!
//! # Architecture
//!
//! Neo's shared CPU bus places memory/lookup columns in the tail of the CPU witness:
//!
//! ```text
//! z = [x | w_cpu | ... | bus_tail ]
//!      └─┬─┘     └──────┬────────┘
//!     public       private witness
//!
//! bus_tail layout (per step):
//!   - Shout instances: [addr_bits..., has_lookup, val]
//!   - Twist instances: [ra_bits..., wa_bits..., has_read, has_write, wv, rv, inc]
//! ```
//!
//! The constraints in this module enforce that CPU semantic columns (e.g., the value
//! a load instruction writes to a register) are equal to the corresponding bus columns
//! (e.g., `rv` from Twist).

use neo_ccs::matrix::Mat;
use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::relations::CcsStructure;
use p3_field::Field;
use std::ops::Range;

use crate::witness::{LutInstance, MemInstance};

/// Configuration for a single Twist (memory) instance in the bus.
#[derive(Clone, Debug)]
pub struct TwistBusConfig {
    /// Number of address bits (d * ell).
    pub ell_addr: usize,
    /// Range of ra_bits columns (relative to this Twist's start in bus).
    pub ra_bits: Range<usize>,
    /// Range of wa_bits columns (relative to this Twist's start in bus).
    pub wa_bits: Range<usize>,
    /// Index of has_read column (relative to this Twist's start).
    pub has_read: usize,
    /// Index of has_write column (relative to this Twist's start).
    pub has_write: usize,
    /// Index of wv (write value) column.
    pub wv: usize,
    /// Index of rv (read value) column.
    pub rv: usize,
    /// Index of inc_at_write_addr column.
    pub inc_at_write_addr: usize,
}

impl TwistBusConfig {
    /// Create a new TwistBusConfig from address bit count.
    pub fn new(ell_addr: usize) -> Self {
        Self {
            ell_addr,
            ra_bits: 0..ell_addr,
            wa_bits: ell_addr..(2 * ell_addr),
            has_read: 2 * ell_addr,
            has_write: 2 * ell_addr + 1,
            wv: 2 * ell_addr + 2,
            rv: 2 * ell_addr + 3,
            inc_at_write_addr: 2 * ell_addr + 4,
        }
    }

    /// Total number of columns for this Twist instance.
    pub fn total_cols(&self) -> usize {
        2 * self.ell_addr + 5
    }
}

/// Configuration for a single Shout (lookup) instance in the bus.
#[derive(Clone, Debug)]
pub struct ShoutBusConfig {
    /// Number of address bits (d * ell).
    pub ell_addr: usize,
    /// Range of addr_bits columns.
    pub addr_bits: Range<usize>,
    /// Index of has_lookup column.
    pub has_lookup: usize,
    /// Index of val column.
    pub val: usize,
}

impl ShoutBusConfig {
    /// Create a new ShoutBusConfig from address bit count.
    pub fn new(ell_addr: usize) -> Self {
        Self {
            ell_addr,
            addr_bits: 0..ell_addr,
            has_lookup: ell_addr,
            val: ell_addr + 1,
        }
    }

    /// Total number of columns for this Shout instance.
    pub fn total_cols(&self) -> usize {
        self.ell_addr + 2
    }
}

/// CPU column layout for binding to the bus.
///
/// This mirrors Jolt's `JoltR1CSInputs` but is specific to Neo's witness structure.
///
/// # Credits
/// Adapted from Jolt's `JoltR1CSInputs` enum in `jolt-core/src/zkvm/r1cs/inputs.rs`.
#[derive(Clone, Debug)]
pub struct CpuColumnLayout {
    /// Column index of the CPU's "is_load" selector (1 when executing a load instruction).
    pub is_load: usize,
    /// Column index of the CPU's "is_store" selector (1 when executing a store instruction).
    pub is_store: usize,
    /// Column index of the CPU's effective address (rs1 + imm for RISC-V).
    pub effective_addr: usize,
    /// Column index of the CPU's destination register write value (for loads).
    pub rd_write_value: usize,
    /// Column index of the CPU's source register value (for stores, typically rs2).
    pub rs2_value: usize,
    /// Column index for the CPU's lookup selector (1 when executing a lookup).
    pub is_lookup: usize,
    /// Column index for the CPU's lookup key (the table index / operand).
    pub lookup_key: usize,
    /// Column index for the CPU's lookup output value.
    pub lookup_output: usize,
}

/// Constraint label for debugging and logging.
///
/// # Credits
/// Adapted from Jolt's `R1CSConstraintLabel` in `jolt-core/src/zkvm/r1cs/constraints.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CpuConstraintLabel {
    /// Load: RamReadValue == RdWriteValue (when Load flag is set)
    LoadValueBinding,
    /// Store: Rs2Value == RamWriteValue (when Store flag is set)
    StoreValueBinding,
    /// Load: Packed bus `ra_bits` matches CPU effective address.
    LoadAddressBinding,
    /// Store: Packed bus `wa_bits` matches CPU effective address.
    StoreAddressBinding,
    /// Padding: rv == 0 (when NOT has_read)
    ReadValueZeroPadding,
    /// Padding: wv == 0 (when NOT has_write)
    WriteValueZeroPadding,
    /// Padding: ra_bits == 0 (when NOT has_read)
    ReadAddressBitsZeroPadding,
    /// Padding: wa_bits == 0 (when NOT has_write)
    WriteAddressBitsZeroPadding,
    /// Padding: inc_at_write_addr == 0 (when NOT has_write)
    IncrementZeroPadding,
    /// Lookup: val == lookup_output (when has_lookup)
    LookupValueBinding,
    /// Lookup: packed bus `addr_bits` matches CPU lookup key.
    LookupKeyBinding,
    /// Padding: lookup val == 0 (when NOT has_lookup)
    LookupValueZeroPadding,
    /// Padding: lookup addr_bits == 0 (when NOT has_lookup)
    LookupAddressBitsZeroPadding,
    /// Selector binding: is_load == has_read
    LoadSelectorBinding,
    /// Selector binding: is_store == has_write
    StoreSelectorBinding,
    /// Selector binding: is_lookup == has_lookup
    LookupSelectorBinding,
}

/// A single constraint in the CPU binding system.
///
/// Represents: `condition * (left - right) = 0`
///
/// # Credits
/// Adapted from Jolt's `R1CSConstraint` and `r1cs_eq_conditional!` macro.
#[derive(Clone, Debug)]
pub struct CpuConstraint<F> {
    /// Human-readable label for debugging.
    pub label: CpuConstraintLabel,
    /// Condition column index (selector).
    pub condition_col: usize,
    /// Whether to negate the condition (1 - condition).
    pub negate_condition: bool,
    /// Optional: additional columns to add to the condition (for OR-style conditions).
    pub additional_condition_cols: Vec<usize>,
    /// Linear expression for the B side: Σ coeffᵢ · z[colᵢ].
    ///
    /// The constraint enforced is:
    ///   A(z) * B(z) = 0
    /// where A(z) is the (possibly negated / OR'ed) condition expression.
    pub b_terms: Vec<(usize, F)>,
}

impl<F: Field> CpuConstraint<F> {
    /// Create a constraint: `condition * (left - right) = 0`.
    pub fn new_eq(
        label: CpuConstraintLabel,
        condition_col: usize,
        left_col: usize,
        right_col: usize,
    ) -> Self {
        Self {
            label,
            condition_col,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(left_col, F::ONE), (right_col, -F::ONE)],
        }
    }

    /// Create a constraint: `(1 - condition) * (left - right) = 0`.
    pub fn new_eq_negated(
        label: CpuConstraintLabel,
        condition_col: usize,
        left_col: usize,
        right_col: usize,
    ) -> Self {
        Self {
            label,
            condition_col,
            negate_condition: true,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(left_col, F::ONE), (right_col, -F::ONE)],
        }
    }

    /// Create a constraint with OR condition: `(cond1 + cond2 + ...) * (left - right) = 0`.
    pub fn new_with_or_eq(
        label: CpuConstraintLabel,
        condition_cols: &[usize],
        left_col: usize,
        right_col: usize,
    ) -> Self {
        assert!(!condition_cols.is_empty(), "need at least one condition");
        Self {
            label,
            condition_col: condition_cols[0],
            negate_condition: false,
            additional_condition_cols: condition_cols[1..].to_vec(),
            b_terms: vec![(left_col, F::ONE), (right_col, -F::ONE)],
        }
    }

    /// Create a constraint: `condition * (col) = 0`.
    pub fn new_zero(label: CpuConstraintLabel, condition_col: usize, col: usize) -> Self {
        Self {
            label,
            condition_col,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(col, F::ONE)],
        }
    }

    /// Create a constraint: `(1 - condition) * (col) = 0`.
    pub fn new_zero_negated(label: CpuConstraintLabel, condition_col: usize, col: usize) -> Self {
        Self {
            label,
            condition_col,
            negate_condition: true,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(col, F::ONE)],
        }
    }

    /// Create a constraint with an arbitrary B-side linear expression:
    /// `condition * (Σ coeffᵢ · z[colᵢ]) = 0`.
    pub fn new_terms(
        label: CpuConstraintLabel,
        condition_col: usize,
        negate_condition: bool,
        b_terms: Vec<(usize, F)>,
    ) -> Self {
        Self {
            label,
            condition_col,
            negate_condition,
            additional_condition_cols: Vec::new(),
            b_terms,
        }
    }
}

/// Builder for CPU-to-bus binding constraints.
///
/// Generates CCS constraints that enforce the binding between CPU semantics
/// and the shared memory/lookup bus.
///
/// # Credits
/// The constraint logic is adapted from Jolt's `R1CS_CONSTRAINTS` static array
/// in `jolt-core/src/zkvm/r1cs/constraints.rs`.
///
/// # Example
/// ```ignore
/// let mut builder = CpuConstraintBuilder::new(n, m, bus_base, const_one_col);
/// builder.add_twist_instance(twist_cfg, cpu_layout);
/// builder.add_shout_instance(shout_cfg, cpu_layout);
/// let ccs = builder.build()?;
/// ```
#[derive(Clone, Debug)]
pub struct CpuConstraintBuilder<F: Field> {
    /// Number of constraint rows.
    pub n: usize,
    /// Number of witness columns.
    pub m: usize,
    /// Base index where bus columns start in the witness.
    pub bus_base: usize,
    /// Column index that must be fixed to 1 (typically the R1CS constant-one column).
    pub const_one_col: usize,
    /// Accumulated constraints.
    constraints: Vec<CpuConstraint<F>>,
    /// Current offset within the bus region.
    bus_offset: usize,
}

impl<F: Field> CpuConstraintBuilder<F> {
    /// Create a new constraint builder.
    ///
    /// # Arguments
    /// * `n` - Number of constraint rows
    /// * `m` - Number of witness columns (including bus tail)
    /// * `bus_base` - Starting index of bus columns in witness
    /// * `const_one_col` - Column index that is fixed to 1 (public or otherwise constrained)
    pub fn new(n: usize, m: usize, bus_base: usize, const_one_col: usize) -> Self {
        Self {
            n,
            m,
            bus_base,
            const_one_col,
            constraints: Vec::new(),
            bus_offset: 0,
        }
    }

    /// Get the current bus column offset.
    pub fn current_bus_offset(&self) -> usize {
        self.bus_offset
    }

    /// Add constraints for a Twist (memory) instance.
    ///
    /// # Constraints Added (from Jolt)
    ///
    /// **Value Binding:**
    /// - `is_load * (bus_rv - rd_write_value) = 0` (Load: memory value → register)
    /// - `is_store * (bus_wv - rs2_value) = 0` (Store: register value → memory)
    ///
    /// **Selector Binding:**
    /// - `is_load - has_read = 0` (CPU load flag must match bus has_read)
    /// - `is_store - has_write = 0` (CPU store flag must match bus has_write)
    ///
    /// **Padding Constraints:**
    /// - `(1 - has_read) * rv = 0`
    /// - `(1 - has_write) * wv = 0`
    /// - `(1 - has_read) * ra_bits[i] = 0` for all i
    /// - `(1 - has_write) * wa_bits[i] = 0` for all i
    /// - `(1 - has_write) * inc_at_write_addr = 0`
    ///
    /// # Credits
    /// Constraints adapted from:
    /// - `RamReadEqRdWriteIfLoad` in Jolt
    /// - `Rs2EqRamWriteIfStore` in Jolt
    /// - `RamAddrEqZeroIfNotLoadStore` in Jolt
    pub fn add_twist_instance(&mut self, twist_cfg: &TwistBusConfig, cpu_layout: &CpuColumnLayout) {
        let base = self.bus_base + self.bus_offset;

        // Bus column indices (absolute in witness)
        let bus_has_read = base + twist_cfg.has_read;
        let bus_has_write = base + twist_cfg.has_write;
        let bus_rv = base + twist_cfg.rv;
        let bus_wv = base + twist_cfg.wv;
        let bus_inc = base + twist_cfg.inc_at_write_addr;

        // Value binding constraints (from Jolt's RamReadEqRdWriteIfLoad, Rs2EqRamWriteIfStore)
        // is_load * (rd_write_value - bus_rv) = 0
        self.constraints.push(CpuConstraint::new_eq(
            CpuConstraintLabel::LoadValueBinding,
            cpu_layout.is_load,
            cpu_layout.rd_write_value,
            bus_rv,
        ));

        // is_store * (rs2_value - bus_wv) = 0
        self.constraints.push(CpuConstraint::new_eq(
            CpuConstraintLabel::StoreValueBinding,
            cpu_layout.is_store,
            cpu_layout.rs2_value,
            bus_wv,
        ));

        // Selector binding: is_load == has_read, is_store == has_write
        self.add_equality_constraint(
            CpuConstraintLabel::LoadSelectorBinding,
            cpu_layout.is_load,
            bus_has_read,
        );
        self.add_equality_constraint(
            CpuConstraintLabel::StoreSelectorBinding,
            cpu_layout.is_store,
            bus_has_write,
        );

        // Address binding (bit-pack):
        // - is_load  * (effective_addr - pack(ra_bits)) = 0
        // - is_store * (effective_addr - pack(wa_bits)) = 0
        self.constraints.push(CpuConstraint::new_terms(
            CpuConstraintLabel::LoadAddressBinding,
            cpu_layout.is_load,
            false,
            pack_addr_bits::<F>(cpu_layout.effective_addr, twist_cfg.ra_bits.clone(), base),
        ));
        self.constraints.push(CpuConstraint::new_terms(
            CpuConstraintLabel::StoreAddressBinding,
            cpu_layout.is_store,
            false,
            pack_addr_bits::<F>(cpu_layout.effective_addr, twist_cfg.wa_bits.clone(), base),
        ));

        // Padding: (1 - has_read) * rv = 0
        self.constraints.push(CpuConstraint::new_zero_negated(
            CpuConstraintLabel::ReadValueZeroPadding,
            bus_has_read,
            bus_rv,
        ));

        // Padding: (1 - has_write) * wv = 0
        self.constraints.push(CpuConstraint::new_zero_negated(
            CpuConstraintLabel::WriteValueZeroPadding,
            bus_has_write,
            bus_wv,
        ));

        // Padding: (1 - has_write) * inc_at_write_addr = 0
        self.constraints.push(CpuConstraint::new_zero_negated(
            CpuConstraintLabel::IncrementZeroPadding,
            bus_has_write,
            bus_inc,
        ));

        // Padding: (1 - has_read) * ra_bits[i] = 0 for all i
        for i in twist_cfg.ra_bits.clone() {
            self.constraints.push(CpuConstraint::new_zero_negated(
                CpuConstraintLabel::ReadAddressBitsZeroPadding,
                bus_has_read,
                base + i,
            ));
        }

        // Padding: (1 - has_write) * wa_bits[i] = 0 for all i
        for i in twist_cfg.wa_bits.clone() {
            self.constraints.push(CpuConstraint::new_zero_negated(
                CpuConstraintLabel::WriteAddressBitsZeroPadding,
                bus_has_write,
                base + i,
            ));
        }

        // Advance bus offset
        self.bus_offset += twist_cfg.total_cols();
    }

    /// Add constraints for a Shout (lookup) instance.
    ///
    /// # Constraints Added
    ///
    /// **Value Binding:**
    /// - `is_lookup * (lookup_output - bus_val) = 0`
    ///
    /// **Padding Constraints:**
    /// - `(1 - has_lookup) * val = 0`
    /// - `(1 - has_lookup) * addr_bits[i] = 0` for all i
    ///
    /// # Credits
    /// Constraints adapted from Jolt's `RdWriteEqLookupIfWriteLookupToRd`.
    pub fn add_shout_instance(&mut self, shout_cfg: &ShoutBusConfig, cpu_layout: &CpuColumnLayout) {
        let base = self.bus_base + self.bus_offset;

        // Bus column indices
        let bus_has_lookup = base + shout_cfg.has_lookup;
        let bus_val = base + shout_cfg.val;

        // Value binding: is_lookup * (lookup_output - bus_val) = 0
        self.constraints.push(CpuConstraint::new_eq(
            CpuConstraintLabel::LookupValueBinding,
            cpu_layout.is_lookup,
            cpu_layout.lookup_output,
            bus_val,
        ));

        // Selector binding: is_lookup == has_lookup
        self.add_equality_constraint(
            CpuConstraintLabel::LookupSelectorBinding,
            cpu_layout.is_lookup,
            bus_has_lookup,
        );

        // Key binding (bit-pack): is_lookup * (lookup_key - pack(addr_bits)) = 0
        self.constraints.push(CpuConstraint::new_terms(
            CpuConstraintLabel::LookupKeyBinding,
            cpu_layout.is_lookup,
            false,
            pack_addr_bits::<F>(cpu_layout.lookup_key, shout_cfg.addr_bits.clone(), base),
        ));

        // Padding: (1 - has_lookup) * val = 0
        self.constraints.push(CpuConstraint::new_zero_negated(
            CpuConstraintLabel::LookupValueZeroPadding,
            bus_has_lookup,
            bus_val,
        ));

        // Padding: (1 - has_lookup) * addr_bits[i] = 0 for all i
        for i in shout_cfg.addr_bits.clone() {
            self.constraints.push(CpuConstraint::new_zero_negated(
                CpuConstraintLabel::LookupAddressBitsZeroPadding,
                bus_has_lookup,
                base + i,
            ));
        }

        // Advance bus offset
        self.bus_offset += shout_cfg.total_cols();
    }

    /// Add an unconditional equality constraint: `left == right` (always).
    ///
    /// This is used for selector binding (is_load == has_read, etc.).
    /// Internally represented as: `1 * (left - right) = 0`
    pub fn add_equality_constraint(
        &mut self,
        label: CpuConstraintLabel,
        left_col: usize,
        right_col: usize,
    ) {
        self.constraints.push(CpuConstraint::new_eq(
            label,
            self.const_one_col,
            left_col,
            right_col,
        ));
    }

    /// Get the accumulated constraints.
    pub fn constraints(&self) -> &[CpuConstraint<F>] {
        &self.constraints
    }

    /// Build the CCS structure from accumulated constraints.
    ///
    /// Each constraint `A(z) * B(z) = 0` becomes an R1CS constraint:
    /// - A matrix: the condition linear form
    /// - B matrix: the linear expression to gate
    /// - C matrix: all zeros
    ///
    /// The resulting CCS uses f(x1, x2, x3) = x1 * x2 - x3 (standard R1CS embedding).
    pub fn build(&self) -> Result<CcsStructure<F>, String> {
        if self.constraints.is_empty() {
            return Err("no constraints added".to_string());
        }

        let n = self.n;
        let m = self.m;
        let num_constraints = self.constraints.len();

        // We need n >= num_constraints for square CCS
        if num_constraints > n {
            return Err(format!(
                "too many constraints ({}) for CCS with n={}",
                num_constraints, n
            ));
        }

        // Build A, B, C matrices for R1CS
        let mut a_data = vec![F::ZERO; n * m];
        let mut b_data = vec![F::ZERO; n * m];
        let c_data = vec![F::ZERO; n * m]; // C is all zeros for our constraints

        for (row, constraint) in self.constraints.iter().enumerate() {
            // A matrix: condition column(s)
            if constraint.negate_condition {
                // (1 - condition)
                a_data[row * m + self.const_one_col] = F::ONE;
                a_data[row * m + constraint.condition_col] = -F::ONE;
                for &col in &constraint.additional_condition_cols {
                    a_data[row * m + col] = -F::ONE;
                }
            } else {
                // Just condition
                a_data[row * m + constraint.condition_col] = F::ONE;
                for &col in &constraint.additional_condition_cols {
                    a_data[row * m + col] = F::ONE;
                }
            }

            // B matrix: Σ coeffᵢ · z[colᵢ]
            for &(col, coeff) in &constraint.b_terms {
                b_data[row * m + col] += coeff;
            }
        }

        // Create matrices
        let a = Mat::from_row_major(n, m, a_data);
        let b = Mat::from_row_major(n, m, b_data);
        let c = Mat::from_row_major(n, m, c_data);

        // Convert to CCS: f(x1, x2, x3) = x1 * x2 - x3
        // For identity-first CCS (square), we add I_n as M_0
        if n == m {
            let i_n = Mat::identity(n);
            let f = SparsePoly::new(
                4,
                vec![
                    Term {
                        coeff: F::ONE,
                        exps: vec![0, 1, 1, 0], // x1 * x2
                    },
                    Term {
                        coeff: -F::ONE,
                        exps: vec![0, 0, 0, 1], // -x3
                    },
                ],
            );
            CcsStructure::new(vec![i_n, a, b, c], f)
                .map_err(|e| format!("failed to create CCS: {:?}", e))
        } else {
            let f = SparsePoly::new(
                3,
                vec![
                    Term {
                        coeff: F::ONE,
                        exps: vec![1, 1, 0], // x1 * x2
                    },
                    Term {
                        coeff: -F::ONE,
                        exps: vec![0, 0, 1], // -x3
                    },
                ],
            );
            CcsStructure::new(vec![a, b, c], f)
                .map_err(|e| format!("failed to create CCS: {:?}", e))
        }
    }

    /// Extend an existing CCS with bus binding constraints.
    ///
    /// This is the recommended way to add bus constraints to an existing CPU CCS.
    /// The function merges the constraint matrices while preserving the original
    /// CPU constraints.
    ///
    /// # Arguments
    /// * `base_ccs` - The original CPU CCS
    /// * `constraint_rows` - Which rows in the CCS to place the new constraints
    ///
    /// # Returns
    /// Extended CCS with bus binding constraints added.
    pub fn extend_ccs(
        &self,
        base_ccs: &CcsStructure<F>,
        constraint_start_row: usize,
    ) -> Result<CcsStructure<F>, String> {
        if self.constraints.is_empty() {
            return Ok(base_ccs.clone());
        }

        let num_constraints = self.constraints.len();
        if constraint_start_row + num_constraints > base_ccs.n {
            return Err(format!(
                "constraint rows {} + {} exceed CCS n={}",
                constraint_start_row, num_constraints, base_ccs.n
            ));
        }

        // Ensure base CCS has at least 3 matrices (A, B, C from R1CS)
        if base_ccs.matrices.len() < 3 {
            return Err("base CCS must have at least 3 matrices (R1CS structure)".to_string());
        }

        let _m = base_ccs.m;
        let _n = base_ccs.n;

        // Clone base matrices
        let mut matrices: Vec<Mat<F>> = base_ccs.matrices.clone();

        // Determine which matrix indices are A, B, C
        // For identity-first CCS: M_0 = I_n, M_1 = A, M_2 = B, M_3 = C
        // For non-identity-first: M_0 = A, M_1 = B, M_2 = C
        let (a_idx, b_idx, _c_idx) =
            if base_ccs.matrices.len() >= 4 && base_ccs.matrices[0].is_identity() {
                (1, 2, 3)
            } else {
                (0, 1, 2)
            };

        // Add constraints to A and B matrices
        for (i, constraint) in self.constraints.iter().enumerate() {
            let row = constraint_start_row + i;

            // A matrix: condition
            if constraint.negate_condition {
                let current = matrices[a_idx][(row, self.const_one_col)];
                matrices[a_idx].set(row, self.const_one_col, current + F::ONE);
                let current = matrices[a_idx][(row, constraint.condition_col)];
                matrices[a_idx].set(row, constraint.condition_col, current - F::ONE);
                for &col in &constraint.additional_condition_cols {
                    let current = matrices[a_idx][(row, col)];
                    matrices[a_idx].set(row, col, current - F::ONE);
                }
            } else {
                let current = matrices[a_idx][(row, constraint.condition_col)];
                matrices[a_idx].set(row, constraint.condition_col, current + F::ONE);
                for &col in &constraint.additional_condition_cols {
                    let current = matrices[a_idx][(row, col)];
                    matrices[a_idx].set(row, col, current + F::ONE);
                }
            }

            // B matrix: Σ coeffᵢ · z[colᵢ]
            for &(col, coeff) in &constraint.b_terms {
                let current = matrices[b_idx][(row, col)];
                matrices[b_idx].set(row, col, current + coeff);
            }
        }

        CcsStructure::new(matrices, base_ccs.f.clone())
            .map_err(|e| format!("failed to extend CCS: {:?}", e))
    }
}

/// Extend an existing **CPU CCS** with the canonical shared-bus binding + padding constraints.
///
/// This is the "no footguns" helper: it computes the shared-bus tail layout from the provided
/// instance metadata, builds the Jolt-derived constraints via `CpuConstraintBuilder`, and injects
/// them into the **last** rows of the base CCS (which must be reserved / unused).
///
/// ## Requirements
/// - The base CCS must be R1CS-embedded (A/B/C matrices, optionally identity-first).
/// - The base CCS must reserve enough trailing **all-zero** rows in A/B/C to fit the injected constraints.
/// - Only `chunk_size == 1` is supported (i.e., all instances must have `steps == 1`).
///
/// ## Bus Ordering
/// This function assumes the caller passes instances in canonical bus order:
/// 1) all Shout instances (table_id-sorted order upstream),
/// 2) all Twist instances (mem_id-sorted order upstream).
pub fn extend_ccs_with_shared_cpu_bus_constraints<F: Field + Copy, Cmt>(
    base_ccs: &CcsStructure<F>,
    const_one_col: usize,
    cpu_layout: &CpuColumnLayout,
    lut_insts: &[LutInstance<Cmt, F>],
    mem_insts: &[MemInstance<Cmt, F>],
) -> Result<CcsStructure<F>, String> {
    // This helper is for the current shared-bus layout in `R1csCpu`, which is chunk_size==1.
    for (i, inst) in lut_insts.iter().enumerate() {
        if inst.steps != 1 {
            return Err(format!(
                "extend_ccs_with_shared_cpu_bus_constraints only supports chunk_size=1 (got lut_insts[{i}].steps={})",
                inst.steps
            ));
        }
    }
    for (i, inst) in mem_insts.iter().enumerate() {
        if inst.steps != 1 {
            return Err(format!(
                "extend_ccs_with_shared_cpu_bus_constraints only supports chunk_size=1 (got mem_insts[{i}].steps={})",
                inst.steps
            ));
        }
    }

    // Compute the total reserved bus tail length (in columns, since chunk_size==1).
    let mut bus_cols_total = 0usize;
    let mut shout_cfgs = Vec::<ShoutBusConfig>::with_capacity(lut_insts.len());
    let mut twist_cfgs = Vec::<TwistBusConfig>::with_capacity(mem_insts.len());

    for inst in lut_insts {
        let ell_addr = inst.d * inst.ell;
        let cfg = ShoutBusConfig::new(ell_addr);
        bus_cols_total = bus_cols_total
            .checked_add(cfg.total_cols())
            .ok_or_else(|| "bus_cols_total overflow (shout)".to_string())?;
        shout_cfgs.push(cfg);
    }
    for inst in mem_insts {
        let ell_addr = inst.d * inst.ell;
        let cfg = TwistBusConfig::new(ell_addr);
        bus_cols_total = bus_cols_total
            .checked_add(cfg.total_cols())
            .ok_or_else(|| "bus_cols_total overflow (twist)".to_string())?;
        twist_cfgs.push(cfg);
    }

    if bus_cols_total == 0 {
        return Ok(base_ccs.clone());
    }
    if bus_cols_total > base_ccs.m {
        return Err(format!(
            "shared-bus tail does not fit in CCS width: bus_cols_total({bus_cols_total}) > ccs.m({})",
            base_ccs.m
        ));
    }
    let bus_base = base_ccs.m - bus_cols_total;

    let mut builder = CpuConstraintBuilder::<F>::new(base_ccs.n, base_ccs.m, bus_base, const_one_col);
    for cfg in &shout_cfgs {
        builder.add_shout_instance(cfg, cpu_layout);
    }
    for cfg in &twist_cfgs {
        builder.add_twist_instance(cfg, cpu_layout);
    }

    if builder.current_bus_offset() != bus_cols_total {
        return Err(format!(
            "internal error: bus offset mismatch (builder wrote {}, expected {bus_cols_total})",
            builder.current_bus_offset()
        ));
    }

    let needed = builder.constraints().len();
    if needed == 0 {
        return Ok(base_ccs.clone());
    }
    if needed > base_ccs.n {
        return Err(format!(
            "not enough CCS rows to inject shared-bus constraints: need {needed}, have n={}",
            base_ccs.n
        ));
    }
    let start = base_ccs.n - needed;

    // Determine which matrix indices are A/B/C.
    let (a_idx, b_idx, c_idx) = if base_ccs.matrices.len() >= 4 && base_ccs.matrices[0].is_identity() {
        (1usize, 2usize, 3usize)
    } else if base_ccs.matrices.len() >= 3 {
        (0usize, 1usize, 2usize)
    } else {
        return Err("base CCS must have at least 3 matrices (R1CS A/B/C)".to_string());
    };

    // Refuse to overwrite any existing constraints: require trailing rows are empty in A/B/C.
    for row in start..base_ccs.n {
        for col in 0..base_ccs.m {
            if base_ccs.matrices[a_idx][(row, col)] != F::ZERO
                || base_ccs.matrices[b_idx][(row, col)] != F::ZERO
                || base_ccs.matrices[c_idx][(row, col)] != F::ZERO
            {
                return Err(format!(
                    "cannot inject shared-bus constraints: base CCS row {row} is not empty in A/B/C.\n\
                     Fix: reserve the LAST {needed} rows of the base CPU R1CS as padding rows (all-zero in A/B/C)."
                ));
            }
        }
    }

    builder.extend_ccs(base_ccs, start)
}

/// Create padding constraints for all bus columns.
///
/// This is a convenience function that adds all necessary padding constraints
/// to ensure inactive bus fields are zero.
///
/// # Credits
/// Implements the "padding row constraints" recommended by the Jolt team:
/// - `(1 - has_read) * rv = 0`
/// - `(1 - has_write) * wv = 0`
/// - etc.
pub fn create_twist_padding_constraints<F: Field>(
    bus_base: usize,
    twist_cfg: &TwistBusConfig,
) -> Vec<CpuConstraint<F>> {
    let base = bus_base;
    let bus_has_read = base + twist_cfg.has_read;
    let bus_has_write = base + twist_cfg.has_write;
    let bus_rv = base + twist_cfg.rv;
    let bus_wv = base + twist_cfg.wv;
    let bus_inc = base + twist_cfg.inc_at_write_addr;

    let mut constraints = Vec::new();

    // (1 - has_read) * rv = 0
    constraints.push(CpuConstraint::new_zero_negated(
        CpuConstraintLabel::ReadValueZeroPadding,
        bus_has_read,
        bus_rv,
    ));

    // (1 - has_write) * wv = 0
    constraints.push(CpuConstraint::new_zero_negated(
        CpuConstraintLabel::WriteValueZeroPadding,
        bus_has_write,
        bus_wv,
    ));

    // (1 - has_write) * inc_at_write_addr = 0
    constraints.push(CpuConstraint::new_zero_negated(
        CpuConstraintLabel::IncrementZeroPadding,
        bus_has_write,
        bus_inc,
    ));

    // (1 - has_read) * ra_bits[i] = 0 for all i
    for i in twist_cfg.ra_bits.clone() {
        constraints.push(CpuConstraint::new_zero_negated(
            CpuConstraintLabel::ReadAddressBitsZeroPadding,
            bus_has_read,
            base + i,
        ));
    }

    // (1 - has_write) * wa_bits[i] = 0 for all i
    for i in twist_cfg.wa_bits.clone() {
        constraints.push(CpuConstraint::new_zero_negated(
            CpuConstraintLabel::WriteAddressBitsZeroPadding,
            bus_has_write,
            base + i,
        ));
    }

    constraints
}

/// Create padding constraints for Shout (lookup) bus columns.
pub fn create_shout_padding_constraints<F: Field>(
    bus_base: usize,
    shout_cfg: &ShoutBusConfig,
) -> Vec<CpuConstraint<F>> {
    let base = bus_base;
    let bus_has_lookup = base + shout_cfg.has_lookup;
    let bus_val = base + shout_cfg.val;

    let mut constraints = Vec::new();

    // (1 - has_lookup) * val = 0
    constraints.push(CpuConstraint::new_zero_negated(
        CpuConstraintLabel::LookupValueZeroPadding,
        bus_has_lookup,
        bus_val,
    ));

    // (1 - has_lookup) * addr_bits[i] = 0 for all i
    for i in shout_cfg.addr_bits.clone() {
        constraints.push(CpuConstraint::new_zero_negated(
            CpuConstraintLabel::LookupAddressBitsZeroPadding,
            bus_has_lookup,
            base + i,
        ));
    }

    constraints
}

fn pack_addr_bits<F: Field>(addr_col: usize, bit_range: Range<usize>, base: usize) -> Vec<(usize, F)> {
    let len = bit_range.end.saturating_sub(bit_range.start);
    let mut terms = Vec::with_capacity(1 + len);
    terms.push((addr_col, F::ONE));
    let mut weight = F::ONE;
    for bit_idx in bit_range {
        terms.push((base + bit_idx, -weight));
        weight = weight + weight; // *= 2
    }
    terms
}
