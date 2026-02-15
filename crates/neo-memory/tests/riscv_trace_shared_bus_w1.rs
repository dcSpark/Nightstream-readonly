use std::collections::HashMap;

use neo_memory::riscv::ccs::{
    rv32_trace_shared_bus_requirements, rv32_trace_shared_cpu_bus_config, Rv32TraceCcsLayout, RV32_B1_SHOUT_PROFILE_FULL20,
};
use p3_goldilocks::Goldilocks as F;

#[test]
fn rv32_trace_shared_bus_config_uses_padding_only_shout_bindings_for_all_tables() {
    let layout = Rv32TraceCcsLayout::new(/*t=*/ 4).expect("trace CCS layout");
    let cfg = rv32_trace_shared_cpu_bus_config(
        &layout,
        RV32_B1_SHOUT_PROFILE_FULL20,
        HashMap::new(),
        HashMap::<(u32, u64), F>::new(),
    )
    .expect("trace shared bus config");

    for &table_id in RV32_B1_SHOUT_PROFILE_FULL20 {
        let lanes = cfg
            .shout_cpu
            .get(&table_id)
            .expect("missing shout_cpu entry for table");
        assert!(
            lanes.is_empty(),
            "trace shared bus must use padding-only shout bindings (table_id={table_id})"
        );
    }
}

#[test]
fn rv32_trace_shared_bus_requirements_accept_rv32m_table_ids() {
    let layout = Rv32TraceCcsLayout::new(/*t=*/ 4).expect("trace CCS layout");
    let (bus_region_len, reserved_rows) =
        rv32_trace_shared_bus_requirements(&layout, RV32_B1_SHOUT_PROFILE_FULL20, &HashMap::new())
            .expect("trace shared bus requirements");
    assert!(bus_region_len > 0, "expected non-zero bus region for full table profile");
    assert!(reserved_rows > 0, "expected injected bus constraints for shout padding rows");
}

#[test]
fn rv32_trace_shared_bus_requirements_reject_unknown_table_id() {
    let layout = Rv32TraceCcsLayout::new(/*t=*/ 4).expect("trace CCS layout");
    let err = rv32_trace_shared_bus_requirements(&layout, &[999u32], &HashMap::new())
        .expect_err("unknown table id must be rejected");
    assert!(
        err.contains("unsupported shout table_id=999"),
        "unexpected error: {err}"
    );
}
