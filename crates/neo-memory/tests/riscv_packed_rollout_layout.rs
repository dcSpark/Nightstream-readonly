use neo_memory::riscv::lookups::{compute_op, RiscvOpcode};
use neo_memory::riscv::packed::{build_rv32_packed_cols, rv32_packed_d};
use p3_goldilocks::Goldilocks;

#[test]
fn packed_rollout_cols_len_matches_declared_d() {
    let cases: &[(RiscvOpcode, u32, u32)] = &[
        (RiscvOpcode::Mul, 13, 5),
        (RiscvOpcode::Mulh, 0xFFFF_FFFF, 0xFFFF_FFFD),
        (RiscvOpcode::Mulhu, 0xFFFF_FFFF, 0xFFFF_FFFD),
        (RiscvOpcode::Mulhsu, 0xFFFF_FFFF, 5),
        (RiscvOpcode::Div, 0x8000_0000, 0xFFFF_FFFF),
        (RiscvOpcode::Div, 13, 0),
        (RiscvOpcode::Divu, 13, 0),
        (RiscvOpcode::Divu, 13, 5),
        (RiscvOpcode::Rem, 0x8000_0000, 0xFFFF_FFFF),
        (RiscvOpcode::Rem, 13, 0),
        (RiscvOpcode::Remu, 13, 0),
        (RiscvOpcode::Remu, 13, 5),
    ];

    for &(op, lhs, rhs) in cases {
        let val = compute_op(op, lhs as u64, rhs as u64, 32) as u32;
        let cols = build_rv32_packed_cols::<Goldilocks>(op, lhs, rhs, val)
            .unwrap_or_else(|e| panic!("packed col synthesis failed for {op:?}: {e}"));
        let d = rv32_packed_d(op).unwrap_or_else(|e| panic!("packed d failed for {op:?}: {e}"));
        assert_eq!(cols.len(), d, "packed columns length mismatch for opcode={op:?}");
    }
}

#[test]
fn packed_rollout_rejects_incorrect_value() {
    let err = build_rv32_packed_cols::<Goldilocks>(RiscvOpcode::Mul, 13, 5, 64)
        .expect_err("packed col synthesis must reject mismatched val");
    let msg = err.to_string();
    assert!(msg.contains("value mismatch"), "unexpected mismatch error: {msg}");
}
