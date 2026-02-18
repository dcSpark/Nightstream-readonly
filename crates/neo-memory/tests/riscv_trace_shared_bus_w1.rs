use std::collections::HashMap;

use neo_memory::cpu::CPU_BUS_COL_DISABLED;
use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{
    rv32_trace_shared_bus_requirements_with_specs, rv32_trace_shared_cpu_bus_config_with_specs, Rv32TraceCcsLayout,
    TraceShoutBusSpec,
};
use neo_memory::riscv::lookups::{RiscvOpcode, RiscvShoutTables, PROG_ID, RAM_ID, REG_ID};
use neo_memory::riscv::trace::{
    rv32_decode_lookup_backed_cols, rv32_decode_lookup_table_id_for_col, rv32_trace_lookup_addr_group_for_table_id,
    rv32_trace_lookup_selector_group_for_table_id, rv32_width_lookup_backed_cols, rv32_width_lookup_table_id_for_col,
    Rv32DecodeSidecarLayout, Rv32WidthSidecarLayout,
};
use p3_goldilocks::Goldilocks as F;

fn sample_mem_layouts() -> HashMap<u32, PlainMemLayout> {
    HashMap::from([
        (
            PROG_ID.0,
            PlainMemLayout {
                k: 16,
                d: 4,
                n_side: 2,
                lanes: 1,
            },
        ),
        (
            REG_ID.0,
            PlainMemLayout {
                k: 32,
                d: 5,
                n_side: 2,
                lanes: 2,
            },
        ),
        (
            RAM_ID.0,
            PlainMemLayout {
                k: 16,
                d: 4,
                n_side: 2,
                lanes: 1,
            },
        ),
    ])
}

fn decode_selector_specs(prog_d: usize) -> Vec<TraceShoutBusSpec> {
    let decode = Rv32DecodeSidecarLayout::new();
    [decode.rd_has_write, decode.ram_has_read, decode.ram_has_write]
        .into_iter()
        .map(|col| TraceShoutBusSpec {
            table_id: rv32_decode_lookup_table_id_for_col(col),
            ell_addr: prog_d,
            n_vals: 1usize,
        })
        .collect()
}

fn width_selector_specs(cycle_d: usize) -> Vec<TraceShoutBusSpec> {
    let width = Rv32WidthSidecarLayout::new();
    [width.ram_rv_q16, width.rs2_q16]
        .into_iter()
        .map(|col| TraceShoutBusSpec {
            table_id: rv32_width_lookup_table_id_for_col(col),
            ell_addr: cycle_d,
            n_vals: 1usize,
        })
        .collect()
}

fn full_table_ids() -> Vec<u32> {
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    [
        RiscvOpcode::And,
        RiscvOpcode::Xor,
        RiscvOpcode::Or,
        RiscvOpcode::Add,
        RiscvOpcode::Sub,
        RiscvOpcode::Slt,
        RiscvOpcode::Sltu,
        RiscvOpcode::Sll,
        RiscvOpcode::Srl,
        RiscvOpcode::Sra,
        RiscvOpcode::Eq,
        RiscvOpcode::Neq,
        RiscvOpcode::Mul,
        RiscvOpcode::Mulh,
        RiscvOpcode::Mulhu,
        RiscvOpcode::Mulhsu,
        RiscvOpcode::Div,
        RiscvOpcode::Divu,
        RiscvOpcode::Rem,
        RiscvOpcode::Remu,
    ]
    .into_iter()
    .map(|op| shout.opcode_to_id(op).0)
    .collect()
}

#[test]
fn rv32_trace_shared_bus_config_uses_padding_only_shout_bindings_for_all_tables() {
    let mut layout = Rv32TraceCcsLayout::new(/*t=*/ 4).expect("trace CCS layout");
    let mem_layouts = sample_mem_layouts();
    let decode_specs = decode_selector_specs(mem_layouts[&PROG_ID.0].d);
    let table_ids = full_table_ids();
    let (bus_region_len, _) =
        rv32_trace_shared_bus_requirements_with_specs(&layout, &table_ids, &decode_specs, &mem_layouts)
            .expect("trace shared bus requirements");
    layout.m += bus_region_len;
    let cfg = rv32_trace_shared_cpu_bus_config_with_specs(
        &layout,
        &table_ids,
        &decode_specs,
        mem_layouts,
        HashMap::<(u32, u64), F>::new(),
    )
    .expect("trace shared bus config");

    for &table_id in &table_ids {
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
    let mem_layouts = sample_mem_layouts();
    let decode_specs = decode_selector_specs(mem_layouts[&PROG_ID.0].d);
    let table_ids = full_table_ids();
    let (bus_region_len, reserved_rows) =
        rv32_trace_shared_bus_requirements_with_specs(&layout, &table_ids, &decode_specs, &mem_layouts)
            .expect("trace shared bus requirements");
    assert!(
        bus_region_len > 0,
        "expected non-zero bus region for full table profile"
    );
    assert!(
        reserved_rows > 0,
        "expected injected bus constraints for shout padding rows"
    );
}

#[test]
fn rv32_trace_shared_bus_requirements_reject_unknown_table_id() {
    let layout = Rv32TraceCcsLayout::new(/*t=*/ 4).expect("trace CCS layout");
    let mem_layouts = sample_mem_layouts();
    let decode_specs = decode_selector_specs(mem_layouts[&PROG_ID.0].d);
    let err = rv32_trace_shared_bus_requirements_with_specs(&layout, &[999u32], &decode_specs, &mem_layouts)
        .expect_err("unknown table id must be rejected");
    assert!(
        err.contains("unsupported shout table_id=999"),
        "unexpected error: {err}"
    );
}

#[test]
fn rv32_trace_shared_bus_with_specs_adds_custom_shout_width() {
    let layout = Rv32TraceCcsLayout::new(/*t=*/ 4).expect("trace CCS layout");
    let mem_layouts = sample_mem_layouts();
    let mut specs = decode_selector_specs(mem_layouts[&PROG_ID.0].d);
    let (bus_region_base, _) = rv32_trace_shared_bus_requirements_with_specs(&layout, &[3u32], &specs, &mem_layouts)
        .expect("trace shared bus baseline requirements");
    specs.push(TraceShoutBusSpec {
        table_id: 1000,
        ell_addr: 13,
        n_vals: 1usize,
    });
    let (bus_region_len, reserved_rows) =
        rv32_trace_shared_bus_requirements_with_specs(&layout, &[3u32], &specs, &mem_layouts)
            .expect("trace shared bus requirements with extra spec");

    let expected_extra_cols = 13 + 2;
    assert_eq!(
        bus_region_len - bus_region_base,
        expected_extra_cols * layout.t,
        "bus width delta must include custom extra shout ell_addr"
    );
    assert!(reserved_rows > 0, "expected injected padding constraints");
}

#[test]
fn rv32_trace_shared_cpu_bus_config_with_specs_keeps_padding_only_bindings() {
    let mut layout = Rv32TraceCcsLayout::new(/*t=*/ 4).expect("trace CCS layout");
    let mem_layouts = sample_mem_layouts();
    let mut specs = decode_selector_specs(mem_layouts[&PROG_ID.0].d);
    specs.push(TraceShoutBusSpec {
        table_id: 1001,
        ell_addr: 17,
        n_vals: 1usize,
    });
    let (bus_region_len, _) = rv32_trace_shared_bus_requirements_with_specs(&layout, &[3u32], &specs, &mem_layouts)
        .expect("trace shared bus requirements with extra spec");
    layout.m += bus_region_len;
    let cfg = rv32_trace_shared_cpu_bus_config_with_specs(
        &layout,
        &[3u32],
        &specs,
        mem_layouts,
        HashMap::<(u32, u64), F>::new(),
    )
    .expect("trace shared bus config with extra spec");

    let base = cfg.shout_cpu.get(&3u32).expect("missing base shout table");
    assert!(base.is_empty(), "base table must stay padding-only");
    let custom = cfg
        .shout_cpu
        .get(&1001u32)
        .expect("missing custom shout table");
    assert!(custom.is_empty(), "custom table must use padding-only bindings");
}

#[test]
fn rv32_trace_shared_bus_with_specs_rejects_conflicting_ell_addr() {
    let layout = Rv32TraceCcsLayout::new(/*t=*/ 4).expect("trace CCS layout");
    let mem_layouts = sample_mem_layouts();
    let mut extra = decode_selector_specs(mem_layouts[&PROG_ID.0].d);
    extra.push(TraceShoutBusSpec {
        table_id: 3,
        ell_addr: 63,
        n_vals: 1usize,
    });
    let err = rv32_trace_shared_bus_requirements_with_specs(&layout, &[3u32], &extra, &mem_layouts)
        .expect_err("conflicting table width must fail");
    assert!(err.contains("conflicting ell_addr"), "unexpected error: {err}");
}

#[test]
fn rv32_trace_shared_cpu_bus_config_with_specs_binds_decode_lookup_key_to_pc_before() {
    let mut layout = Rv32TraceCcsLayout::new(/*t=*/ 4).expect("trace CCS layout");
    let mem_layouts = sample_mem_layouts();
    let decode_specs = decode_selector_specs(mem_layouts[&PROG_ID.0].d);
    let table_ids = full_table_ids();
    let (bus_region_len, _) =
        rv32_trace_shared_bus_requirements_with_specs(&layout, &table_ids, &decode_specs, &mem_layouts)
            .expect("trace shared bus requirements");
    layout.m += bus_region_len;
    let cfg = rv32_trace_shared_cpu_bus_config_with_specs(
        &layout,
        &table_ids,
        &decode_specs,
        mem_layouts,
        HashMap::<(u32, u64), F>::new(),
    )
    .expect("trace shared bus config");

    let decode = Rv32DecodeSidecarLayout::new();
    let table_id = rv32_decode_lookup_table_id_for_col(decode.rd_has_write);
    let lanes = cfg
        .shout_cpu
        .get(&table_id)
        .expect("missing decode shout_cpu binding");
    assert_eq!(lanes.len(), 1, "decode lookup should bind one shout lane");
    let lane = &lanes[0];
    assert_eq!(
        lane.has_lookup, CPU_BUS_COL_DISABLED,
        "decode lookup should use key-only linkage (selector disabled)"
    );
    assert_eq!(
        lane.val, CPU_BUS_COL_DISABLED,
        "decode lookup should use key-only linkage (value disabled)"
    );
    assert_eq!(
        lane.addr,
        Some(layout.cell(layout.trace.pc_before, 0)),
        "decode lookup key must bind to committed pc_before"
    );
}

#[test]
fn rv32_trace_shared_cpu_bus_config_with_specs_binds_width_lookup_key_to_cycle() {
    let mut layout = Rv32TraceCcsLayout::new(/*t=*/ 4).expect("trace CCS layout");
    let mem_layouts = sample_mem_layouts();
    let mut specs = decode_selector_specs(mem_layouts[&PROG_ID.0].d);
    specs.extend(width_selector_specs(/*cycle_d=*/ 8));
    let table_ids = full_table_ids();
    let (bus_region_len, _) = rv32_trace_shared_bus_requirements_with_specs(&layout, &table_ids, &specs, &mem_layouts)
        .expect("trace shared bus requirements");
    layout.m += bus_region_len;
    let cfg = rv32_trace_shared_cpu_bus_config_with_specs(
        &layout,
        &table_ids,
        &specs,
        mem_layouts,
        HashMap::<(u32, u64), F>::new(),
    )
    .expect("trace shared bus config");

    let width = Rv32WidthSidecarLayout::new();
    let table_id = rv32_width_lookup_table_id_for_col(width.ram_rv_q16);
    let lanes = cfg
        .shout_cpu
        .get(&table_id)
        .expect("missing width shout_cpu binding");
    assert_eq!(lanes.len(), 1, "width lookup should bind one shout lane");
    let lane = &lanes[0];
    assert_eq!(
        lane.has_lookup, CPU_BUS_COL_DISABLED,
        "width lookup should use key-only linkage (selector disabled)"
    );
    assert_eq!(
        lane.val, CPU_BUS_COL_DISABLED,
        "width lookup should use key-only linkage (value disabled)"
    );
    assert_eq!(
        lane.addr,
        Some(layout.cell(layout.trace.cycle, 0)),
        "width lookup key must bind to committed cycle"
    );
}

#[test]
fn rv32_trace_lookup_addr_group_coalesces_all_decode_lookup_backed_tables() {
    let decode = Rv32DecodeSidecarLayout::new();
    let cols = rv32_decode_lookup_backed_cols(&decode);
    assert!(!cols.is_empty(), "decode lookup-backed set must be non-empty");

    let mut groups = std::collections::BTreeSet::new();
    for col in cols {
        let table_id = rv32_decode_lookup_table_id_for_col(col);
        let group = rv32_trace_lookup_addr_group_for_table_id(table_id);
        assert!(group.is_some(), "decode table_id={table_id} must have an addr group");
        groups.insert(group);
    }
    assert_eq!(
        groups.len(),
        1,
        "all decode lookup-backed tables should share one address group"
    );
}

#[test]
fn rv32_trace_lookup_addr_group_coalesces_all_width_lookup_tables() {
    let width = Rv32WidthSidecarLayout::new();
    let cols = rv32_width_lookup_backed_cols(&width);
    assert!(!cols.is_empty(), "width lookup-backed set must be non-empty");

    let mut groups = std::collections::BTreeSet::new();
    for col in cols {
        let table_id = rv32_width_lookup_table_id_for_col(col);
        let group = rv32_trace_lookup_addr_group_for_table_id(table_id);
        assert!(group.is_some(), "width table_id={table_id} must have an addr group");
        groups.insert(group);
    }
    assert_eq!(
        groups.len(),
        1,
        "all width lookup-backed tables should share one address group"
    );
}

#[test]
fn rv32_trace_lookup_selector_group_coalesces_all_decode_lookup_backed_tables() {
    let decode = Rv32DecodeSidecarLayout::new();
    let cols = rv32_decode_lookup_backed_cols(&decode);
    assert!(!cols.is_empty(), "decode lookup-backed set must be non-empty");

    let mut groups = std::collections::BTreeSet::new();
    for col in cols {
        let table_id = rv32_decode_lookup_table_id_for_col(col);
        let group = rv32_trace_lookup_selector_group_for_table_id(table_id);
        assert!(group.is_some(), "decode table_id={table_id} must have a selector group");
        groups.insert(group);
    }
    assert_eq!(
        groups.len(),
        1,
        "all decode lookup-backed tables should share one selector group"
    );
}

#[test]
fn rv32_trace_lookup_selector_group_coalesces_all_width_lookup_tables() {
    let width = Rv32WidthSidecarLayout::new();
    let cols = rv32_width_lookup_backed_cols(&width);
    assert!(!cols.is_empty(), "width lookup-backed set must be non-empty");

    let mut groups = std::collections::BTreeSet::new();
    for col in cols {
        let table_id = rv32_width_lookup_table_id_for_col(col);
        let group = rv32_trace_lookup_selector_group_for_table_id(table_id);
        assert!(group.is_some(), "width table_id={table_id} must have a selector group");
        groups.insert(group);
    }
    assert_eq!(
        groups.len(),
        1,
        "all width lookup-backed tables should share one selector group"
    );
}
