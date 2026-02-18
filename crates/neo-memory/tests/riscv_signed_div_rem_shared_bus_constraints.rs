use std::collections::HashMap;

use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{
    rv32_trace_shared_bus_requirements_with_specs, rv32_trace_shared_cpu_bus_config_with_specs, Rv32TraceCcsLayout,
    TraceShoutBusSpec,
};
use neo_memory::riscv::lookups::{RiscvOpcode, RiscvShoutTables, PROG_ID, RAM_ID, REG_ID};
use neo_memory::riscv::trace::{rv32_decode_lookup_table_id_for_col, Rv32DecodeSidecarLayout};
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
            lanes: 1,
        })
        .collect()
}

fn div_rem_table_ids() -> Vec<u32> {
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    [RiscvOpcode::Div, RiscvOpcode::Divu, RiscvOpcode::Rem, RiscvOpcode::Remu]
        .into_iter()
        .map(|op| shout.opcode_to_id(op).0)
        .collect()
}

#[test]
fn rv32_trace_shared_bus_requirements_accept_div_and_rem_tables() {
    let layout = Rv32TraceCcsLayout::new(/*t=*/ 4).expect("trace CCS layout");
    let mem_layouts = sample_mem_layouts();
    let decode_specs = decode_selector_specs(mem_layouts[&PROG_ID.0].d);
    let table_ids = div_rem_table_ids();

    let (bus_region_len, reserved_rows) =
        rv32_trace_shared_bus_requirements_with_specs(&layout, &table_ids, &decode_specs, &mem_layouts)
            .expect("trace shared bus requirements for DIV/REM tables");

    assert!(bus_region_len > 0, "expected non-zero bus region for DIV/REM tables");
    assert!(
        reserved_rows > 0,
        "expected injected constraints when shared-bus rows are reserved"
    );
}

#[test]
fn rv32_trace_shared_bus_config_keeps_div_and_rem_tables_padding_only() {
    let mut layout = Rv32TraceCcsLayout::new(/*t=*/ 4).expect("trace CCS layout");
    let mem_layouts = sample_mem_layouts();
    let decode_specs = decode_selector_specs(mem_layouts[&PROG_ID.0].d);
    let table_ids = div_rem_table_ids();

    let (bus_region_len, _) =
        rv32_trace_shared_bus_requirements_with_specs(&layout, &table_ids, &decode_specs, &mem_layouts)
            .expect("trace shared bus requirements for DIV/REM tables");
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
            .expect("missing shout_cpu entry for DIV/REM table");
        assert!(
            lanes.is_empty(),
            "trace mode must keep DIV/REM tables as padding-only shout bindings (table_id={table_id})"
        );
    }
}
