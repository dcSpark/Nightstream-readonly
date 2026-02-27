use neo_ccs::relations::check_ccs_rowwise_zero;
use neo_memory::riscv::ccs::{build_rv32_trace_wiring_ccs, rv32_trace_ccs_witness_from_exec_table, Rv32TraceCcsLayout};
use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    decode_program, encode_program, RiscvCpu, RiscvInstruction, RiscvMemory, RiscvOpcode, RiscvShoutTables, PROG_ID,
    REG_ID,
};
use neo_vm_trace::trace_program;
use neo_vm_trace::Twist as _;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[test]
fn rv32_trace_wiring_ccs_rejects_virtual_row_marked_halted() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 5,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(REG_ID, 1, 0x8000_0000);
    twist.store(REG_ID, 2, 0xFFFF_FFFF);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    let virtual_row = (0..layout.t)
        .find(|&i| w[layout.cell(layout.trace.is_virtual, i) - layout.m_in] == F::ONE)
        .expect("expected at least one virtual row");
    let halted_idx = layout.cell(layout.trace.halted, virtual_row) - layout.m_in;
    w[halted_idx] = F::ONE;

    let err = check_ccs_rowwise_zero(&ccs, &x, &w).expect_err("mutated witness should violate virtual halted ban");
    assert!(
        err.to_string().contains("row"),
        "unexpected error (should include row context): {err:?}"
    );
}

#[test]
fn rv32_trace_wiring_ccs_accepts_virtual_commit_with_rd_x0() {
    let program = vec![
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 0,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    cpu.set_runtime_decomposition_enabled(true);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(REG_ID, 1, 0x8000_0000);
    twist.store(REG_ID, 2, 0xFFFF_FFFF);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 64).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    check_ccs_rowwise_zero(&ccs, &x, &w).expect("rd=x0 decomposed commit must satisfy CCS");
}
