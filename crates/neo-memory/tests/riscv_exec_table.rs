use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    encode_program, interleave_bits, decode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode,
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
        assert_eq!(row0.pc_before, 0);
        assert_eq!(row0.pc_after, 4);
        assert_eq!(row0.fields.opcode, 0x13);
        assert_eq!(row0.fields.rs1, 0);
        assert_eq!(row0.fields.rd, 1);

        // PROG fetch matches the instruction word for this row.
        assert_eq!(row0.prog_read.addr, row0.pc_before);
        assert_eq!(row0.prog_read.value, row0.instr_word as u64);

        // REG lane policy: lane0 reads rs1_field, lane1 reads rs2_field.
        assert_eq!(row0.reg_read_lane0.addr, 0);
        assert_eq!(row0.reg_read_lane0.value, 0);
        assert_eq!(row0.reg_read_lane1.addr, row0.fields.rs2 as u64);

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
        let expected_key = interleave_bits(row0.reg_read_lane0.value, imm_u32) as u64;
        assert_eq!(ev.key, expected_key);
        assert_eq!(ev.value, 1);
    }

    // Step 1: HALT (ECALL). Lane1 still reads rs2_field (which is 0 for ECALL).
    {
        let row1 = &table.rows[1];
        assert_eq!(row1.pc_before, 4);
        assert_eq!(row1.fields.opcode, 0x73);
        assert!(row1.halted);

        assert_eq!(row1.prog_read.addr, row1.pc_before);
        assert_eq!(row1.prog_read.value, row1.instr_word as u64);

        assert_eq!(row1.reg_read_lane0.addr, row1.fields.rs1 as u64);
        assert_eq!(row1.reg_read_lane1.addr, row1.fields.rs2 as u64);
        assert!(row1.reg_write_lane0.is_none());
        assert!(row1.shout_events.is_empty());
    }
}
