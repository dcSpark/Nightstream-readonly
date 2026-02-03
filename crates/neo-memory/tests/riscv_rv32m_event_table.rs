use neo_memory::riscv::exec_table::{Rv32ExecTable, Rv32MEventTable};
use neo_memory::riscv::lookups::{
    decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID,
};
use neo_vm_trace::trace_program;

#[test]
fn rv32m_event_table_extracts_and_matches_cpu_semantics() {
    // Program:
    //   ADDI x1,x0,3
    //   ADDI x2,x0,5
    //   MUL  x3,x1,x2        -> 15
    //   DIVU x4,x2,x1        -> 1
    //   REMU x5,x2,x1        -> 2
    //   HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 3,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mul,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 4,
            rs1: 2,
            rs2: 1,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Remu,
            rd: 5,
            rs1: 2,
            rs2: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);

    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");
    assert!(trace.did_halt(), "expected program to halt");

    let exec = Rv32ExecTable::from_trace(&trace).expect("Rv32ExecTable::from_trace");
    let events = Rv32MEventTable::from_exec_table(&exec).expect("Rv32MEventTable::from_exec_table");
    assert_eq!(events.rows.len(), 3, "expected MUL/DIVU/REMU events");

    // MUL x3,x1,x2: 3*5 = 15
    {
        let e = &events.rows[0];
        assert_eq!(e.opcode, RiscvOpcode::Mul);
        assert_eq!(e.rs1, 1);
        assert_eq!(e.rs2, 2);
        assert_eq!(e.rd, 3);
        assert_eq!(e.rs1_val, 3);
        assert_eq!(e.rs2_val, 5);
        assert_eq!(e.expected_rd_val, 15);
        assert_eq!(e.rd_write_val, Some(15));
    }

    // DIVU x4,x2,x1: 5/3 = 1
    {
        let e = &events.rows[1];
        assert_eq!(e.opcode, RiscvOpcode::Divu);
        assert_eq!(e.rs1_val, 5);
        assert_eq!(e.rs2_val, 3);
        assert_eq!(e.expected_rd_val, 1);
        assert_eq!(e.rd_write_val, Some(1));
    }

    // REMU x5,x2,x1: 5%3 = 2
    {
        let e = &events.rows[2];
        assert_eq!(e.opcode, RiscvOpcode::Remu);
        assert_eq!(e.rs1_val, 5);
        assert_eq!(e.rs2_val, 3);
        assert_eq!(e.expected_rd_val, 2);
        assert_eq!(e.rd_write_val, Some(2));
    }
}
