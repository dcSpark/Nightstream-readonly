use std::collections::HashMap;

use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{
    build_rv32_trace_wiring_ccs, build_rv32_trace_wiring_ccs_with_reserved_rows,
    rv32_trace_shared_bus_requirements_with_specs, Rv32TraceCcsLayout, TraceShoutBusSpec,
};
use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID,
    RAM_ID, REG_ID,
};
use neo_memory::riscv::trace::{rv32_decode_lookup_table_id_for_col, Rv32DecodeSidecarLayout};
use neo_vm_trace::trace_program;

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

fn trace_addi_halt_exec_table(min_len: usize) -> Rv32ExecTable {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes).expect("decode_program");

    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);

    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, min_len).expect("from_trace_padded_pow2");
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");
    exec
}

#[test]
fn rv32_trace_ccs_counts_follow_layout_shape() {
    let exec = trace_addi_halt_exec_table(/*min_len=*/ 4);
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    assert_eq!(
        layout.m,
        layout.m_in + layout.trace.cols * layout.t,
        "layout width must equal public + flattened trace region"
    );
    assert_eq!(ccs.m, layout.m, "CCS witness width must match layout width");
    assert!(ccs.n > layout.t, "CCS should include transition + wiring constraints");
}

#[test]
fn rv32_trace_reserved_rows_increase_constraint_count_exactly() {
    let exec = trace_addi_halt_exec_table(/*min_len=*/ 4);
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let ccs_base = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS base");

    let mem_layouts = sample_mem_layouts();
    let decode_specs = decode_selector_specs(mem_layouts[&PROG_ID.0].d);
    let add_table_id = RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Add).0;
    let (_bus_region_len, reserved_rows) =
        rv32_trace_shared_bus_requirements_with_specs(&layout, &[add_table_id], &decode_specs, &mem_layouts)
            .expect("trace shared bus requirements");

    let ccs_reserved =
        build_rv32_trace_wiring_ccs_with_reserved_rows(&layout, reserved_rows).expect("trace CCS reserved");

    assert!(reserved_rows > 0, "expected reserved rows from shared bus requirements");
    assert_eq!(
        ccs_reserved.n,
        ccs_base.n + reserved_rows,
        "reserved rows must add directly to CCS row count"
    );
}
