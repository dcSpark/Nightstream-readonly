use std::collections::HashMap;

use neo_ccs::relations::check_ccs_rowwise_zero;
use neo_memory::cpu::CPU_BUS_COL_DISABLED;
use neo_memory::plain::PlainMemLayout;
use neo_memory::riscv::ccs::{
    build_rv32_trace_wiring_ccs, rv32_trace_ccs_witness_from_exec_table,
    rv32_trace_shared_bus_requirements_with_specs, rv32_trace_shared_cpu_bus_config_with_specs, Rv32TraceCcsLayout,
    TraceShoutBusSpec,
};
use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemOp, RiscvMemory, RiscvOpcode, RiscvShoutTables,
    PROG_ID, RAM_ID, REG_ID,
};
use neo_memory::riscv::trace::{rv32_decode_lookup_table_id_for_col, Rv32DecodeSidecarLayout};
use neo_vm_trace::trace_program;
use p3_field::PrimeCharacteristicRing;
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

fn full_rv32i_table_ids() -> Vec<u32> {
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
    ]
    .into_iter()
    .map(|op| shout.opcode_to_id(op).0)
    .collect()
}

fn exec_table_for(program: Vec<RiscvInstruction>, min_len: usize, max_steps: usize) -> Rv32ExecTable {
    let program_bytes = encode_program(&program);
    let decoded_program = decode_program(&program_bytes).expect("decode_program");

    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, max_steps).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, min_len).expect("from_trace_padded_pow2");
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty().expect("inactive rows");
    exec
}

#[test]
fn rv32_trace_ccs_happy_path_addi_halt() {
    let exec = exec_table_for(
        vec![
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 1,
            },
            RiscvInstruction::Halt,
        ],
        /*min_len=*/ 4,
        /*max_steps=*/ 16,
    );

    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    check_ccs_rowwise_zero(&ccs, &x, &w).expect("trace CCS satisfied");
}

#[test]
fn rv32_trace_ccs_happy_path_addi_sw_lw_halt() {
    let exec = exec_table_for(
        vec![
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 7,
            },
            RiscvInstruction::Store {
                op: RiscvMemOp::Sw,
                rs1: 0,
                rs2: 1,
                imm: 0,
            },
            RiscvInstruction::Load {
                op: RiscvMemOp::Lw,
                rd: 2,
                rs1: 0,
                imm: 0,
            },
            RiscvInstruction::Halt,
        ],
        /*min_len=*/ 4,
        /*max_steps=*/ 32,
    );

    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    check_ccs_rowwise_zero(&ccs, &x, &w).expect("trace CCS satisfied");
}

#[test]
fn rv32_trace_ccs_rejects_tampered_pc_transition() {
    let exec = exec_table_for(
        vec![
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 1,
            },
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 2,
                rs1: 1,
                imm: 1,
            },
            RiscvInstruction::Halt,
        ],
        /*min_len=*/ 4,
        /*max_steps=*/ 16,
    );

    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    let idx = layout.cell(layout.trace.pc_before, 1) - layout.m_in;
    w[idx] = w[idx] + F::ONE;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered pc_before on row 1 must violate trace transition wiring"
    );
}

#[test]
fn rv32_trace_ccs_rejects_first_row_inactive() {
    let exec = exec_table_for(
        vec![
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm: 1,
            },
            RiscvInstruction::Halt,
        ],
        /*min_len=*/ 4,
        /*max_steps=*/ 16,
    );

    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    let idx = layout.cell(layout.trace.active, 0) - layout.m_in;
    w[idx] = F::ZERO;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "trace execution anchor requires active[0] == 1"
    );
}

#[test]
fn rv32_trace_shared_bus_config_uses_padding_only_shout_bindings() {
    let mut layout = Rv32TraceCcsLayout::new(/*t=*/ 4).expect("trace CCS layout");
    let mem_layouts = sample_mem_layouts();
    let decode_specs = decode_selector_specs(mem_layouts[&PROG_ID.0].d);
    let table_ids = full_rv32i_table_ids();

    let (bus_region_len, reserved_rows) =
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

    assert!(reserved_rows > 0, "expected reserved shared-bus constraints");
    for &table_id in &table_ids {
        let lanes = cfg
            .shout_cpu
            .get(&table_id)
            .expect("missing shout_cpu entry for table");
        assert!(
            lanes.is_empty(),
            "trace shared bus uses padding-only shout bindings (table_id={table_id})"
        );
    }
}

#[test]
fn rv32_trace_shared_bus_decode_lookup_binds_to_pc_before() {
    let mut layout = Rv32TraceCcsLayout::new(/*t=*/ 4).expect("trace CCS layout");
    let mem_layouts = sample_mem_layouts();
    let decode_specs = decode_selector_specs(mem_layouts[&PROG_ID.0].d);
    let table_ids = full_rv32i_table_ids();

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
        .expect("missing decode shout_cpu entry");

    assert_eq!(lanes.len(), 1, "decode lookup should bind exactly one lane");
    let lane = &lanes[0];
    assert_eq!(lane.has_lookup, CPU_BUS_COL_DISABLED);
    assert_eq!(lane.val, CPU_BUS_COL_DISABLED);
    assert_eq!(lane.addr, Some(layout.cell(layout.trace.pc_before, 0)));
}
