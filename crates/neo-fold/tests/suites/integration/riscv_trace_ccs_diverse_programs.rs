//! CCS-level unit tests for the trace wiring CCS across diverse instruction types.
//!
//! These catch constraint design bugs by checking A*z . B*z = C*z on the raw CCS
//! without folding/proving, for every major instruction category.

use neo_ccs::relations::check_ccs_rowwise_zero;
use neo_memory::riscv::ccs::{build_rv32_trace_wiring_ccs, rv32_trace_ccs_witness_from_exec_table, Rv32TraceCcsLayout};
use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    decode_program, encode_program, BranchCondition, RiscvCpu, RiscvInstruction, RiscvMemOp, RiscvMemory, RiscvOpcode,
    RiscvShoutTables, PROG_ID,
};
use neo_vm_trace::trace_program;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

fn trace_and_check_ccs(program: Vec<RiscvInstruction>, label: &str) {
    let program_bytes = encode_program(&program);
    let decoded = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, decoded);
    let twist = RiscvMemory::with_program_in_twist(32, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(32);
    let trace = trace_program(cpu, twist, shout, 64).expect("trace_program");
    assert!(trace.did_halt(), "{label}: expected Halt");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, 4).expect("from_trace_padded_pow2");
    exec.validate_cycle_chain().unwrap_or_else(|e| panic!("{label}: {e}"));
    exec.validate_pc_chain().unwrap_or_else(|e| panic!("{label}: {e}"));
    exec.validate_halted_tail().unwrap_or_else(|e| panic!("{label}: {e}"));
    exec.validate_inactive_rows_are_empty().unwrap_or_else(|e| panic!("{label}: {e}"));

    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");
    check_ccs_rowwise_zero(&ccs, &x, &w).unwrap_or_else(|e| panic!("{label}: CCS not satisfied: {e:?}"));
}

fn trace_and_get_witness(program: Vec<RiscvInstruction>) -> (Rv32TraceCcsLayout, Vec<F>, Vec<F>) {
    let program_bytes = encode_program(&program);
    let decoded = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(32);
    cpu.load_program(0, decoded);
    let twist = RiscvMemory::with_program_in_twist(32, PROG_ID, 0, &program_bytes);
    let shout = RiscvShoutTables::new(32);
    let trace = trace_program(cpu, twist, shout, 64).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, 4).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("layout");
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("witness");
    (layout, x, w)
}

// ── Happy-path CCS satisfaction for diverse instruction types ──

#[test]
fn trace_ccs_happy_rv32i_alu_ops() {
    trace_and_check_ccs(
        vec![
            RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 5 },
            RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 3 },
            RiscvInstruction::RAlu { op: RiscvOpcode::Add, rd: 3, rs1: 1, rs2: 2 },
            RiscvInstruction::RAlu { op: RiscvOpcode::Sub, rd: 4, rs1: 1, rs2: 2 },
            RiscvInstruction::RAlu { op: RiscvOpcode::And, rd: 5, rs1: 1, rs2: 2 },
            RiscvInstruction::RAlu { op: RiscvOpcode::Or, rd: 6, rs1: 1, rs2: 2 },
            RiscvInstruction::RAlu { op: RiscvOpcode::Xor, rd: 7, rs1: 1, rs2: 2 },
            RiscvInstruction::RAlu { op: RiscvOpcode::Slt, rd: 8, rs1: 2, rs2: 1 },
            RiscvInstruction::RAlu { op: RiscvOpcode::Sltu, rd: 9, rs1: 2, rs2: 1 },
            RiscvInstruction::Halt,
        ],
        "rv32i_alu_ops",
    );
}

#[test]
fn trace_ccs_happy_shifts() {
    trace_and_check_ccs(
        vec![
            RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 0x0F },
            RiscvInstruction::IAlu { op: RiscvOpcode::Sll, rd: 2, rs1: 1, imm: 4 },
            RiscvInstruction::IAlu { op: RiscvOpcode::Srl, rd: 3, rs1: 2, imm: 2 },
            RiscvInstruction::IAlu { op: RiscvOpcode::Sra, rd: 4, rs1: 2, imm: 2 },
            RiscvInstruction::Halt,
        ],
        "shifts",
    );
}

#[test]
fn trace_ccs_happy_load_store_word() {
    trace_and_check_ccs(
        vec![
            RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 42 },
            RiscvInstruction::Store { op: RiscvMemOp::Sw, rs1: 0, rs2: 1, imm: 0x100 },
            RiscvInstruction::Load { op: RiscvMemOp::Lw, rd: 2, rs1: 0, imm: 0x100 },
            RiscvInstruction::Halt,
        ],
        "load_store_word",
    );
}

#[test]
fn trace_ccs_happy_byte_half_loads() {
    trace_and_check_ccs(
        vec![
            RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 0x1FF },
            RiscvInstruction::Store { op: RiscvMemOp::Sw, rs1: 0, rs2: 1, imm: 0x100 },
            RiscvInstruction::Load { op: RiscvMemOp::Lb, rd: 2, rs1: 0, imm: 0x100 },
            RiscvInstruction::Load { op: RiscvMemOp::Lbu, rd: 3, rs1: 0, imm: 0x100 },
            RiscvInstruction::Load { op: RiscvMemOp::Lh, rd: 4, rs1: 0, imm: 0x100 },
            RiscvInstruction::Load { op: RiscvMemOp::Lhu, rd: 5, rs1: 0, imm: 0x100 },
            RiscvInstruction::Halt,
        ],
        "byte_half_loads",
    );
}

#[test]
fn trace_ccs_happy_sub_word_stores() {
    trace_and_check_ccs(
        vec![
            RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 0xAB },
            RiscvInstruction::Store { op: RiscvMemOp::Sb, rs1: 0, rs2: 1, imm: 0x100 },
            RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 0x1234 },
            RiscvInstruction::Store { op: RiscvMemOp::Sh, rs1: 0, rs2: 2, imm: 0x104 },
            RiscvInstruction::Halt,
        ],
        "sub_word_stores",
    );
}

#[test]
fn trace_ccs_happy_branches_beq_bne() {
    trace_and_check_ccs(
        vec![
            RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 5 },
            RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 5 },
            RiscvInstruction::Branch { cond: BranchCondition::Eq, rs1: 1, rs2: 2, imm: 8 },
            RiscvInstruction::Halt,
            RiscvInstruction::Branch { cond: BranchCondition::Ne, rs1: 1, rs2: 0, imm: 8 },
            RiscvInstruction::Halt,
            RiscvInstruction::Halt,
        ],
        "branches_beq_bne",
    );
}

#[test]
fn trace_ccs_happy_branches_blt_bge() {
    trace_and_check_ccs(
        vec![
            RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 3 },
            RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 2, rs1: 0, imm: 5 },
            RiscvInstruction::Branch { cond: BranchCondition::Lt, rs1: 1, rs2: 2, imm: 8 },
            RiscvInstruction::Halt,
            RiscvInstruction::Branch { cond: BranchCondition::Geu, rs1: 2, rs2: 1, imm: 8 },
            RiscvInstruction::Halt,
            RiscvInstruction::Halt,
        ],
        "branches_blt_bge",
    );
}

#[test]
fn trace_ccs_happy_jal_jalr() {
    trace_and_check_ccs(
        vec![
            RiscvInstruction::Jal { rd: 1, imm: 8 },
            RiscvInstruction::Halt,
            RiscvInstruction::Jalr { rd: 2, rs1: 1, imm: 0 },
            RiscvInstruction::Halt,
        ],
        "jal_jalr",
    );
}

#[test]
fn trace_ccs_happy_lui_auipc() {
    trace_and_check_ccs(
        vec![
            RiscvInstruction::Lui { rd: 1, imm: 0x12 },
            RiscvInstruction::Auipc { rd: 2, imm: 0 },
            RiscvInstruction::Halt,
        ],
        "lui_auipc",
    );
}

#[test]
fn trace_ccs_happy_fence() {
    trace_and_check_ccs(
        vec![
            RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 1 },
            RiscvInstruction::Fence { pred: 0xF, succ: 0xF },
            RiscvInstruction::Halt,
        ],
        "fence",
    );
}

// ── CCS-level tamper tests (detect constraint violations without folding) ──

#[test]
fn trace_ccs_rejects_tampered_pc_after() {
    let program = vec![
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 1 },
        RiscvInstruction::Halt,
    ];
    let (layout, x, mut w) = trace_and_get_witness(program);

    let idx = layout.trace.pc_after * layout.t + 0;
    w[idx] += F::ONE;

    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");
    let res = check_ccs_rowwise_zero(&ccs, &x, &w);
    assert!(res.is_err(), "tampered pc_after should violate CCS");
}

// NOTE: shout_val is NOT constrained by the trace wiring CCS alone.
// It is bound through the Shout bus (Route-A claim). The full-pipeline
// tamper test lives in trace_bus_binding_redteam::trace_cpu_vs_bus_shout_val_mismatch_must_fail.

#[test]
fn trace_ccs_rejects_tampered_halted_flag() {
    let program = vec![
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 1 },
        RiscvInstruction::Halt,
    ];
    let (layout, x, mut w) = trace_and_get_witness(program);

    let idx = layout.trace.halted * layout.t + 0;
    w[idx] += F::ONE;

    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");
    let res = check_ccs_rowwise_zero(&ccs, &x, &w);
    assert!(res.is_err(), "tampered halted flag should violate CCS");
}

// NOTE: rd_val is NOT constrained by the trace wiring CCS alone.
// It is bound through the Twist bus (memory sidecar). The full-pipeline
// tamper test for memory values lives in trace_bus_binding_redteam.

#[test]
fn trace_ccs_rejects_tampered_cycle() {
    let program = vec![
        RiscvInstruction::IAlu { op: RiscvOpcode::Add, rd: 1, rs1: 0, imm: 1 },
        RiscvInstruction::Halt,
    ];
    let (layout, x, mut w) = trace_and_get_witness(program);

    let idx = layout.trace.cycle * layout.t + 0;
    w[idx] += F::ONE;

    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");
    let res = check_ccs_rowwise_zero(&ccs, &x, &w);
    assert!(res.is_err(), "tampered cycle counter should violate CCS");
}
