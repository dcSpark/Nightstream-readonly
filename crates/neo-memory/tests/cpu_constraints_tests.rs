//! Tests for the CPU constraint builder.
//!
//! # Credits
//!
//! The constraint logic tested here is ported from the Jolt zkVM project:
//! - Repository: https://github.com/a16z/jolt
//! - Original file: `jolt-core/src/zkvm/r1cs/constraints.rs`
//! - License: Apache-2.0 / MIT

use neo_memory::cpu::build_bus_layout_for_instances;
use neo_memory::cpu::constraints::{
    create_twist_padding_constraints, CpuColumnLayout, CpuConstraint, CpuConstraintBuilder, CpuConstraintLabel,
};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;

type F = Goldilocks;

#[test]
fn test_twist_bus_config() {
    let bus = build_bus_layout_for_instances(64, 0, 1, [], [4]).expect("layout");
    assert_eq!(bus.twist_cols.len(), 1);
    let cfg = &bus.twist_cols[0].lanes[0];
    assert_eq!(cfg.ra_bits, 0..4);
    assert_eq!(cfg.wa_bits, 4..8);
    assert_eq!(cfg.has_read, 8);
    assert_eq!(cfg.has_write, 9);
    assert_eq!(cfg.wv, 10);
    assert_eq!(cfg.rv, 11);
    assert_eq!(cfg.inc, 12);
    assert_eq!(bus.bus_cols, 13);
}

#[test]
fn test_shout_bus_config() {
    let bus = build_bus_layout_for_instances(64, 0, 1, [4], []).expect("layout");
    assert_eq!(bus.shout_cols.len(), 1);
    let cfg = &bus.shout_cols[0].lanes[0];
    assert_eq!(cfg.addr_bits, 0..4);
    assert_eq!(cfg.has_lookup, 4);
    assert_eq!(cfg.primary_val(), 5);
    assert_eq!(bus.bus_cols, 6);
}

#[test]
fn test_constraint_builder_basic() {
    // Create a minimal CPU layout
    let cpu_layout = CpuColumnLayout {
        is_load: 1,
        is_store: 2,
        effective_addr: 3,
        rd_write_value: 4,
        rs2_value: 5,
        is_lookup: 6,
        lookup_key: 7,
        lookup_output: 8,
    };

    let n = 32;
    let m = 64;
    let bus = build_bus_layout_for_instances(m, 0, 1, [], [4]).expect("layout");

    let mut builder = CpuConstraintBuilder::<F>::new(n, m, 0);
    builder.add_twist_instance(&bus, &bus.twist_cols[0].lanes[0], &cpu_layout);

    // Should have added value binding + padding constraints
    let constraints = builder.constraints();
    assert!(!constraints.is_empty());

    // Check that we have load value binding
    let has_load_binding = constraints
        .iter()
        .any(|c| matches!(c.label, CpuConstraintLabel::LoadValueBinding));
    assert!(has_load_binding, "should have load value binding constraint");

    // Check that we have store value binding
    let has_store_binding = constraints
        .iter()
        .any(|c| matches!(c.label, CpuConstraintLabel::StoreValueBinding));
    assert!(has_store_binding, "should have store value binding constraint");

    // Check that we have padding constraints
    let has_rv_padding = constraints
        .iter()
        .any(|c| matches!(c.label, CpuConstraintLabel::ReadValueZeroPadding));
    assert!(has_rv_padding, "should have rv padding constraint");
}

#[test]
fn test_build_ccs() {
    let cpu_layout = CpuColumnLayout {
        is_load: 1,
        is_store: 2,
        effective_addr: 3,
        rd_write_value: 4,
        rs2_value: 5,
        is_lookup: 6,
        lookup_key: 7,
        lookup_output: 8,
    };

    let n = 64;
    let m = 64; // Square for identity-first
    let bus = build_bus_layout_for_instances(m, 0, 1, [], [4]).expect("layout");

    let mut builder = CpuConstraintBuilder::<F>::new(n, m, 0);
    builder.add_twist_instance(&bus, &bus.twist_cols[0].lanes[0], &cpu_layout);

    let ccs = builder.build().expect("should build CCS");

    // Verify CCS structure
    assert_eq!(ccs.n, n);
    assert_eq!(ccs.m, m);
    assert!(ccs.matrices.len() >= 3, "should have at least A, B, C matrices");
}

#[test]
fn test_padding_constraints_generation() {
    let bus = build_bus_layout_for_instances(64, 0, 1, [], [4]).expect("layout");
    let constraints: Vec<CpuConstraint<F>> = create_twist_padding_constraints(&bus, &bus.twist_cols[0].lanes[0]);

    // Should have:
    // - 1 for rv padding
    // - 1 for wv padding
    // - 1 for inc padding
    // - 4 for ra_bits padding
    // - 4 for wa_bits padding
    assert_eq!(constraints.len(), 1 + 1 + 1 + 4 + 4);

    // All should be negated (1 - has_xxx)
    for c in &constraints {
        assert!(c.negate_condition, "padding constraints should be negated");
    }
}

fn eval_constraint(const_one_col: usize, c: &CpuConstraint<F>, z: &[F]) -> F {
    let mut a = z[c.condition_col];
    for &col in &c.additional_condition_cols {
        a += z[col];
    }
    if c.negate_condition {
        a = z[const_one_col] - a;
    }

    let mut b = F::ZERO;
    for &(col, coeff) in &c.b_terms {
        b += z[col] * coeff;
    }
    a * b
}

#[test]
fn test_twist_write_addr_domain_split_enforces_virtual_flag_vs_wa_bit5() {
    let n = 16usize;
    let m = 96usize;
    let const_one_col = 0usize;
    let flag_col = 20usize;

    // One Twist lane with ell_addr=6 => write address bit[5] exists.
    let bus = build_bus_layout_for_instances(m, 0, 1, [], [6]).expect("layout");
    let lane = &bus.twist_cols[0].lanes[0];

    let mut builder = CpuConstraintBuilder::<F>::new(n, m, const_one_col);
    builder.add_twist_write_addr_domain_split(&bus, lane, flag_col, /*split_bit_idx=*/ 5);
    assert_eq!(
        builder.constraints().len(),
        1,
        "chunk_size=1 should produce one domain-split constraint"
    );

    let mut z = vec![F::ZERO; m];
    z[const_one_col] = F::ONE;
    z[bus.bus_cell(lane.has_write, 0)] = F::ONE;
    z[flag_col] = F::ONE;

    // Valid virtual write: wa_bit[5] = 1 => addr >= 32.
    let wa_bit5_col = lane.wa_bits.start + 5;
    z[bus.bus_cell(wa_bit5_col, 0)] = F::ONE;
    for c in builder.constraints() {
        assert_eq!(eval_constraint(const_one_col, c, &z), F::ZERO);
        assert_eq!(c.label, CpuConstraintLabel::TwistWriteAddressDomainSplit);
    }

    // Tamper: keep has_write=1 and is_virtual=1, but force wa_bit[5]=0.
    // This must violate the domain split.
    z[bus.bus_cell(wa_bit5_col, 0)] = F::ZERO;
    let violated = builder
        .constraints()
        .iter()
        .any(|c| eval_constraint(const_one_col, c, &z) != F::ZERO);
    assert!(violated, "expected domain-split violation when flag != wa_bit[5]");

    // Gate check: when has_write=0, mismatch must be ignored.
    z[bus.bus_cell(lane.has_write, 0)] = F::ZERO;
    for c in builder.constraints() {
        assert_eq!(
            eval_constraint(const_one_col, c, &z),
            F::ZERO,
            "constraint should be gated off when has_write=0"
        );
    }
}

#[test]
fn test_twist_write_addr_domain_split_fallback_without_bit5() {
    let n = 16usize;
    let m = 96usize;
    let const_one_col = 0usize;
    let flag_col = 24usize;

    // One Twist lane with ell_addr=5 => bit[5] is absent.
    let bus = build_bus_layout_for_instances(m, 0, 1, [], [5]).expect("layout");
    let lane = &bus.twist_cols[0].lanes[0];
    let wa_len = lane.wa_bits.end - lane.wa_bits.start;
    assert_eq!(wa_len, 5, "expected wa_bits length without bit[5]");

    let mut builder = CpuConstraintBuilder::<F>::new(n, m, const_one_col);
    builder.add_twist_write_addr_domain_split(&bus, lane, flag_col, /*split_bit_idx=*/ 5);

    let mut z = vec![F::ZERO; m];
    z[const_one_col] = F::ONE;
    z[bus.bus_cell(lane.has_write, 0)] = F::ONE;

    // Fallback path enforces has_write * flag = 0 when bit[5] is unavailable.
    z[flag_col] = F::ZERO;
    for c in builder.constraints() {
        assert_eq!(eval_constraint(const_one_col, c, &z), F::ZERO);
    }

    z[flag_col] = F::ONE;
    let violated = builder
        .constraints()
        .iter()
        .any(|c| eval_constraint(const_one_col, c, &z) != F::ZERO);
    assert!(
        violated,
        "expected fallback domain-split violation when has_write=1 and flag=1"
    );
}

#[test]
fn test_twist_read_addr_domain_split_nonvirtual_enforces_bit5_zero() {
    let n = 16usize;
    let m = 96usize;
    let const_one_col = 0usize;
    let is_virtual_col = 28usize;

    // One Twist lane with ell_addr=6 => read address bit[5] exists.
    let bus = build_bus_layout_for_instances(m, 0, 1, [], [6]).expect("layout");
    let lane = &bus.twist_cols[0].lanes[0];

    let mut builder = CpuConstraintBuilder::<F>::new(n, m, const_one_col);
    builder.add_twist_read_addr_domain_split_nonvirtual(&bus, lane, is_virtual_col, /*split_bit_idx=*/ 5);
    assert_eq!(builder.constraints().len(), 1);

    let mut z = vec![F::ZERO; m];
    z[const_one_col] = F::ONE;
    let ra_bit5_col = lane.ra_bits.start + 5;

    // Non-virtual read: bit[5] must be 0.
    z[is_virtual_col] = F::ZERO;
    z[bus.bus_cell(ra_bit5_col, 0)] = F::ZERO;
    for c in builder.constraints() {
        assert_eq!(eval_constraint(const_one_col, c, &z), F::ZERO);
        assert_eq!(c.label, CpuConstraintLabel::TwistReadAddressDomainSplit);
    }

    // Tamper: non-virtual row with bit[5]=1 must violate.
    z[bus.bus_cell(ra_bit5_col, 0)] = F::ONE;
    let violated = builder
        .constraints()
        .iter()
        .any(|c| eval_constraint(const_one_col, c, &z) != F::ZERO);
    assert!(
        violated,
        "expected non-virtual read-address domain-split violation when bit[5]=1"
    );

    // Virtual row: unconstrained by this split.
    z[is_virtual_col] = F::ONE;
    for c in builder.constraints() {
        assert_eq!(
            eval_constraint(const_one_col, c, &z),
            F::ZERO,
            "virtual rows should not be constrained by non-virtual read split"
        );
    }
}

#[test]
fn test_twist_write_mirror_group_constraints() {
    // Two Twist instances with ell_addr=4 each.
    let n = 32;
    let m = 64;
    let const_one_col = 0;
    let bus = build_bus_layout_for_instances(m, 0, 1, [], [4, 4]).expect("layout");
    assert_eq!(bus.twist_cols.len(), 2);

    let mut builder = CpuConstraintBuilder::<F>::new(n, m, const_one_col);
    let twists = vec![bus.twist_cols[0].lanes[0].clone(), bus.twist_cols[1].lanes[0].clone()];
    builder.add_twist_write_mirror_group(&bus, &twists);

    // Per mirrored pair (chunk_size=1):
    // - has_write equality (1)
    // - wv equality (1)
    // - inc equality (1)
    // - wa_bits equality (ell_addr=4)
    assert_eq!(builder.constraints().len(), 1 + 1 + 1 + 4);

    // Build a witness that satisfies the mirror constraints.
    let mut z = vec![F::ZERO; m];
    z[const_one_col] = F::ONE;

    let t0 = &bus.twist_cols[0].lanes[0];
    let t1 = &bus.twist_cols[1].lanes[0];

    // Write stream: active write to some address with some value/inc.
    z[bus.bus_cell(t0.has_write, 0)] = F::ONE;
    z[bus.bus_cell(t1.has_write, 0)] = F::ONE;

    z[bus.bus_cell(t0.wv, 0)] = F::from_u64(7);
    z[bus.bus_cell(t1.wv, 0)] = F::from_u64(7);

    z[bus.bus_cell(t0.inc, 0)] = F::from_u64(3);
    z[bus.bus_cell(t1.inc, 0)] = F::from_u64(3);

    // wa_bits: 0b1010 (little-endian bit order) for both.
    for (i, (c0, c1)) in t0.wa_bits.clone().zip(t1.wa_bits.clone()).enumerate() {
        let bit = if i % 2 == 0 { F::ZERO } else { F::ONE };
        z[bus.bus_cell(c0, 0)] = bit;
        z[bus.bus_cell(c1, 0)] = bit;
    }

    for c in builder.constraints() {
        assert_eq!(
            eval_constraint(const_one_col, c, &z),
            F::ZERO,
            "constraint {:?} should hold",
            c.label
        );
    }

    // Break a mirrored column and ensure a violation is detected.
    z[bus.bus_cell(t1.wv, 0)] = F::from_u64(8);
    let violated = builder
        .constraints()
        .iter()
        .any(|c| eval_constraint(const_one_col, c, &z) != F::ZERO);
    assert!(violated, "expected at least one constraint violation");
}
