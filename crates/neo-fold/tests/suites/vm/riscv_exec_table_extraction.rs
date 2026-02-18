use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::exec_table::{
    Rv32MEventTable, Rv32RamEventKind, Rv32RamEventTable, Rv32RegEventKind, Rv32RegEventTable,
};
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode};
use std::collections::HashMap;

#[test]
fn exec_table_extracts_from_trace_run_and_pads() {
    // Program exercises:
    // - REG reads (rs1/rs2) on every step
    // - ALU op in the middle of the trace
    // - RAM store/load
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 3,
        }, // x1 = 3
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 4,
        }, // x2 = 4
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Add,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = 7
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 3,
            imm: 0,
        }, // mem[0] = x3
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 4,
            rs1: 0,
            imm: 0,
        }, // x4 = mem[0]
        RiscvInstruction::Halt,
    ];

    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(32)
        .min_trace_len(8)
        .chunk_rows(4)
        .max_steps(program.len())
        .shout_auto_minimal()
        .prove()
        .expect("prove");
    run.verify().expect("verify");

    let exec = run.exec_table();
    assert_eq!(exec.rows.len(), 8);
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows are empty");
    exec.validate_halted_tail().expect("halted tail");

    let active = exec.rows.iter().filter(|r| r.active).count();
    assert_eq!(active, program.len());

    // Padded rows must be inactive and have no fetched/proven events.
    for r in exec.rows.iter().skip(active) {
        assert!(!r.active);
        assert!(r.prog_read.is_none());
        assert!(r.reg_read_lane0.is_none());
        assert!(r.reg_read_lane1.is_none());
        assert!(r.reg_write_lane0.is_none());
        assert!(r.ram_events.is_empty());
        assert!(r.shout_events.is_empty());
    }

    // Validate regfile/RAM semantics against the statement initial memory.
    let init_regs: HashMap<u64, u64> = HashMap::new();
    let init_ram: HashMap<u64, u64> = HashMap::new();
    exec.validate_regfile_semantics(&init_regs)
        .expect("regfile semantics");
    exec.validate_ram_semantics(&init_ram)
        .expect("ram semantics");

    // Extract reg/RAM event tables (sparse-over-time representation).
    let reg_table = Rv32RegEventTable::from_exec_table(&exec, &init_regs).expect("reg event table");
    assert_eq!(reg_table.rows.len(), 16); // 2 reads per row + 4 writes
    assert_eq!(
        reg_table
            .rows
            .iter()
            .filter(|r| r.kind == Rv32RegEventKind::ReadLane0)
            .count(),
        active
    );
    assert_eq!(
        reg_table
            .rows
            .iter()
            .filter(|r| r.kind == Rv32RegEventKind::ReadLane1)
            .count(),
        active
    );
    assert_eq!(
        reg_table
            .rows
            .iter()
            .filter(|r| r.kind == Rv32RegEventKind::WriteLane0)
            .count(),
        4
    );

    let ram_table = Rv32RamEventTable::from_exec_table(&exec, &init_ram).expect("ram event table");
    assert_eq!(ram_table.rows.len(), 2);
    assert!(ram_table
        .rows
        .iter()
        .any(|r| { r.kind == Rv32RamEventKind::Write && r.addr == 0 && r.prev_val == 0 && r.next_val == 7 }));
    assert!(ram_table
        .rows
        .iter()
        .any(|r| { r.kind == Rv32RamEventKind::Read && r.addr == 0 && r.prev_val == 7 && r.next_val == 7 }));

    // No RV32M ops in this program.
    let m = Rv32MEventTable::from_exec_table(&exec).expect("rv32m event table");
    assert_eq!(m.rows.len(), 0);
}
