use neo_fold::riscv_shard::Rv32B1;
use neo_memory::riscv::exec_table::{
    Rv32MEventTable, Rv32RamEventKind, Rv32RamEventTable, Rv32RegEventKind, Rv32RegEventTable,
};
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode};
use p3_field::PrimeField64;
use std::collections::HashMap;

#[test]
fn exec_table_extracts_from_chunked_run_and_pads() {
    // Program exercises:
    // - REG reads (rs1/rs2) on every step
    // - one RV32M op (MUL) for event-table extraction
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
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        }, // x3 = 12
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
    let mut run = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
        .xlen(32)
        .ram_bytes(0x40)
        .chunk_size(4)
        .max_steps(program.len())
        .shout_auto_minimal()
        .prove()
        .expect("prove");
    run.verify().expect("verify");

    // Sanity: per-chunk RV32M count should match expected (only the MUL chunk).
    let steps = run.steps_public();
    assert_eq!(steps.len(), 2);
    let counts: Vec<u64> = steps
        .iter()
        .map(|s| s.mcs_inst.x[run.layout().rv32m_count].as_canonical_u64())
        .collect();
    assert_eq!(counts, vec![1, 0]);

    // Build a padded-to-pow2 exec table from the replayed trace.
    let exec = run
        .exec_table_padded_pow2(/*min_len=*/ 8)
        .expect("exec table");
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
    let mut init_regs: HashMap<u64, u64> = HashMap::new();
    let mut init_ram: HashMap<u64, u64> = HashMap::new();
    for (&(mem_id, addr), value) in run.initial_mem() {
        let v = value.as_canonical_u64();
        if mem_id == neo_memory::riscv::lookups::REG_ID.0 {
            init_regs.insert(addr, v);
        } else if mem_id == neo_memory::riscv::lookups::RAM_ID.0 {
            init_ram.insert(addr, v);
        }
    }
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
        .any(|r| { r.kind == Rv32RamEventKind::Write && r.addr == 0 && r.prev_val == 0 && r.next_val == 12 }));
    assert!(ram_table
        .rows
        .iter()
        .any(|r| { r.kind == Rv32RamEventKind::Read && r.addr == 0 && r.prev_val == 12 && r.next_val == 12 }));

    // Extract RV32M events from the exec table (time-in-rows view).
    let m = Rv32MEventTable::from_exec_table(&exec).expect("rv32m event table");
    assert_eq!(m.rows.len(), 1);
    let row = &m.rows[0];
    assert_eq!(row.opcode, RiscvOpcode::Mul);
    assert_eq!(row.rs1_val, 3);
    assert_eq!(row.rs2_val, 4);
    assert_eq!(row.expected_rd_val, 12);

    // The trace should have written rd (x3), and it must match the expected result.
    let Some(wrote) = row.rd_write_val else {
        panic!("expected an rd write event for MUL");
    };
    assert_eq!(wrote, 12);
}
