use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID,
};
use neo_memory::riscv::trace::{Rv32TraceAir, Rv32TraceWitness};
use neo_vm_trace::trace_program;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[test]
fn rv32_trace_air_satisfies_addi_halt() {
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

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");

    let air = Rv32TraceAir::new();
    let wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");
    air.assert_satisfied(&wit).expect("trace AIR satisfied");
}

#[test]
fn rv32_trace_air_rejects_halted_tail_reactivation() {
    // Program with at least one active transition after row 0.
    let program = vec![
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
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);

    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");

    let air = Rv32TraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");

    // Force halted=1 from row 0 onward, while keeping row 1 active=1 in the witness.
    // This should violate the halted tail quiescence transition check at row 0.
    for row in 0..wit.t {
        wit.cols[air.layout.halted][row] = F::ONE;
    }

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate halted tail quiescence");
    assert!(
        err.contains("halted tail quiescence violated"),
        "unexpected error: {err}"
    );
}

#[test]
fn rv32_trace_air_rejects_non_boolean_funct3_bit() {
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
    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");

    let air = Rv32TraceAir::new();
    let mut wit = Rv32TraceWitness::from_exec_table(&air.layout, &exec).expect("trace witness");
    wit.cols[air.layout.funct3_bit[0]][0] = F::from_u64(2);

    let err = air
        .assert_satisfied(&wit)
        .expect_err("mutated witness should violate bit booleanity");
    assert!(err.contains("funct3_bit[0] not boolean"), "unexpected error: {err}");
}
