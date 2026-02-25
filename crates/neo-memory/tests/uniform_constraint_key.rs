use neo_memory::riscv::ccs::build_rv32_uniform_constraint_key;
use neo_memory::riscv::trace::Rv32TraceLayout;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[test]
fn uniform_key_has_fixed_step_width() {
    let key = build_rv32_uniform_constraint_key();
    // m_in=5 plus the canonical RV32 trace width.
    assert_eq!(key.m_cols, 5 + Rv32TraceLayout::new().cols);
    assert!(!key.local_rows.is_empty());
    assert!(!key.shift_rows.is_empty());
    assert!(!key.boundary_rows.is_empty());
}

#[test]
fn uniform_key_eval_helpers_are_sparse_and_total() {
    let key = build_rv32_uniform_constraint_key();
    let zero_a = key.eval_local_a(0, 10_000);
    let zero_b = key.eval_local_b(0, 10_000);
    let zero_c = key.eval_local_c(0, 10_000);
    assert_eq!(zero_a, F::ZERO);
    assert_eq!(zero_b, F::ZERO);
    assert_eq!(zero_c, F::ZERO);
}
