use neo_memory::riscv::exec_table::{Rv32ExecTable, Rv32ShoutEventTable};
use neo_memory::riscv::lookups::{
    decode_program, encode_program, interleave_bits, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode,
    RiscvShoutTables, PROG_ID,
};
use neo_vm_trace::trace_program;

#[test]
fn rv32_exec_table_matches_rv32_b1_lane_conventions_addi_halt() {
    // Program: ADDI x1, x0, 1; HALT
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

    // Initialize only PROG; RAM starts empty/zeroed and REG starts with all-zero regs.
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");
    assert_eq!(trace.steps.len(), 2, "expected ADDI + HALT trace");

    let table = Rv32ExecTable::from_trace(&trace).expect("Rv32ExecTable::from_trace");
    table.validate_pc_chain().expect("pc chain");
    assert_eq!(table.rows.len(), 2);

    // Step 0: ADDI x1,x0,1
    {
        let row0 = &table.rows[0];
        assert!(row0.active);
        assert_eq!(row0.pc_before, 0);
        assert_eq!(row0.pc_after, 4);
        assert_eq!(row0.fields.opcode, 0x13);
        assert_eq!(row0.fields.rs1, 0);
        assert_eq!(row0.fields.rd, 1);

        // PROG fetch matches the instruction word for this row.
        let prog_read = row0.prog_read.as_ref().expect("expected PROG read");
        assert_eq!(prog_read.addr, row0.pc_before);
        assert_eq!(prog_read.value, row0.instr_word as u64);

        // REG lane policy: lane0 reads rs1_field, lane1 reads rs2_field.
        let rs1 = row0.reg_read_lane0.as_ref().expect("expected rs1 read");
        let rs2 = row0.reg_read_lane1.as_ref().expect("expected rs2 read");
        assert_eq!(rs1.addr, 0);
        assert_eq!(rs1.value, 0);
        assert_eq!(rs2.addr, row0.fields.rs2 as u64);

        // Writeback: rd_field=1 should be written with value 1.
        let w = row0.reg_write_lane0.as_ref().expect("expected rd write");
        assert_eq!(w.addr, 1);
        assert_eq!(w.value, 1);

        // ADDI uses one ADD shout lookup: key = interleave(rs1_val, imm_u32), value = rd.
        let add_id = RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Add);
        assert_eq!(row0.shout_events.len(), 1);
        let ev = &row0.shout_events[0];
        assert_eq!(ev.shout_id, add_id);

        let imm_u32 = 1u64;
        let expected_key = interleave_bits(rs1.value, imm_u32) as u64;
        assert_eq!(ev.key, expected_key);
        assert_eq!(ev.value, 1);
    }

    // Step 1: HALT (ECALL). Lane1 still reads rs2_field (which is 0 for ECALL).
    {
        let row1 = &table.rows[1];
        assert!(row1.active);
        assert_eq!(row1.pc_before, 4);
        assert_eq!(row1.fields.opcode, 0x73);
        assert!(row1.halted);

        let prog_read = row1.prog_read.as_ref().expect("expected PROG read");
        assert_eq!(prog_read.addr, row1.pc_before);
        assert_eq!(prog_read.value, row1.instr_word as u64);

        let rs1 = row1.reg_read_lane0.as_ref().expect("expected rs1 read");
        let rs2 = row1.reg_read_lane1.as_ref().expect("expected rs2 read");
        assert_eq!(rs1.addr, row1.fields.rs1 as u64);
        assert_eq!(rs2.addr, row1.fields.rs2 as u64);
        assert!(row1.reg_write_lane0.is_none());
        assert!(row1.shout_events.is_empty());
    }
}

#[test]
fn rv32_exec_table_padding_builds_inactive_rows() {
    // Program: ADDI x1, x0, 1; HALT
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
    assert_eq!(trace.steps.len(), 2, "expected ADDI + HALT trace");

    let table = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    assert_eq!(table.rows.len(), 4);
    table.validate_pc_chain().expect("pc chain");

    // First two rows are active; tail rows are inactive padding with no side effects.
    assert!(table.rows[0].active);
    assert!(table.rows[1].active);
    assert!(!table.rows[2].active);
    assert!(!table.rows[3].active);

    let halted_pc = table.rows[1].pc_after;
    for r in table.rows.iter().skip(2) {
        assert_eq!(r.pc_before, halted_pc);
        assert_eq!(r.pc_after, halted_pc);
        assert!(r.halted, "padded rows should stay halted");
        assert!(r.prog_read.is_none());
        assert!(r.reg_read_lane0.is_none());
        assert!(r.reg_read_lane1.is_none());
        assert!(r.reg_write_lane0.is_none());
        assert!(r.ram_events.is_empty());
        assert!(r.shout_events.is_empty());
    }

    let cols = table.to_columns();
    assert_eq!(cols.len(), 4);
    assert_eq!(cols.active, vec![true, true, false, false]);
    assert_eq!(cols.pc_before[2], halted_pc);
    assert_eq!(cols.pc_after[3], halted_pc);
    assert_eq!(cols.prog_value[2], 0);
    assert!(!cols.rd_has_write[3]);
}

#[test]
fn rv32_shout_event_table_includes_rv32m_rows() {
    // Target production behavior: RV32M Shout-backed ops should appear in the
    // trace-derived event table used by event-table packed proving paths.
    //
    // This test is expected to fail until RV32M event-table coverage is fully wired.
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 9,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
    let events = Rv32ShoutEventTable::from_exec_table(&exec).expect("Rv32ShoutEventTable::from_exec_table");

    assert!(
        events
            .rows
            .iter()
            .any(|row| row.opcode == Some(RiscvOpcode::Mul)),
        "expected RV32M (MUL) rows in trace shout event table"
    );
}

#[test]
fn rv32_exec_table_rejects_jalr_non_strict_target_tamper() {
    // Program:
    //   ADDI x1, x0, 8
    //   JALR x2, x1, 0
    //   HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 8,
        },
        RiscvInstruction::Jalr { rd: 2, rs1: 1, imm: 0 },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");
    let mut table = Rv32ExecTable::from_trace(&trace).expect("Rv32ExecTable::from_trace");

    // Tamper the JALR row target so it no longer equals rs1+imm under strict policy.
    let jalr_row = table
        .rows
        .iter_mut()
        .find(|r| matches!(r.decoded, Some(RiscvInstruction::Jalr { .. })))
        .expect("expected one JALR row");
    jalr_row.pc_after = jalr_row.pc_after.wrapping_add(4);

    assert!(
        table.validate_jalr_strict_alignment_policy().is_err(),
        "tampered JALR target should fail strict alignment policy validation"
    );
}
