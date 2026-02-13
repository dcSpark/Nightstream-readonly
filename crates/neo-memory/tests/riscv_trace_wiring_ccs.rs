use neo_ccs::relations::check_ccs_rowwise_zero;
use neo_memory::riscv::ccs::{build_rv32_trace_wiring_ccs, rv32_trace_ccs_witness_from_exec_table, Rv32TraceCcsLayout};
use neo_memory::riscv::exec_table::Rv32ExecTable;
use neo_memory::riscv::lookups::{
    decode_program, encode_program, BranchCondition, RiscvCpu, RiscvInstruction, RiscvMemOp, RiscvMemory, RiscvOpcode,
    RiscvShoutTables, PROG_ID, RAM_ID,
};
use neo_vm_trace::trace_program;
use neo_vm_trace::Twist as _;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

#[test]
fn rv32_trace_wiring_ccs_satisfies_addi_halt() {
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
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");

    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    check_ccs_rowwise_zero(&ccs, &x, &w).expect("trace CCS satisfied");
}

#[test]
fn rv32_trace_wiring_ccs_satisfies_addi_sw_lw_halt() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
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
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 32).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    exec.validate_cycle_chain().expect("cycle chain");
    exec.validate_pc_chain().expect("pc chain");
    exec.validate_halted_tail().expect("halted tail");
    exec.validate_inactive_rows_are_empty()
        .expect("inactive rows");

    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    check_ccs_rowwise_zero(&ccs, &x, &w).expect("trace CCS satisfied");
}

#[test]
fn rv32_trace_wiring_ccs_satisfies_lui_x0_halt() {
    // Program: LUI x0, 1; HALT
    // Architecturally this must be satisfiable (x0 discards the writeback).
    let program = vec![RiscvInstruction::Lui { rd: 0, imm: 1 }, RiscvInstruction::Halt];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_ok(),
        "LUI x0 must be satisfiable in trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_all_inactive_padding_witness() {
    // Program: ADDI x1, x0, 1; HALT
    //
    // Red-team target: with no explicit execution anchor, an all-inactive witness can
    // satisfy most gated constraints while only honoring public bindings + chains.
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
    let t = exec.rows.len();
    let layout = Rv32TraceCcsLayout::new(t).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    let mut set = |col: usize, row: usize, value: F| {
        let idx = layout.cell(col, row);
        w[idx - layout.m_in] = value;
    };

    // Start from an all-zero trace region.
    for col in 0..layout.trace.cols {
        for row in 0..t {
            set(col, row, F::ZERO);
        }
    }

    // Public bindings that must continue to hold.
    set(layout.trace.pc_before, 0, x[layout.pc0]);
    set(layout.trace.pc_after, t - 1, x[layout.pc_final]);
    set(layout.trace.halted, 0, x[layout.halted_in]);
    set(layout.trace.halted, t - 1, x[layout.halted_out]);

    // Keep cycle and pc chains valid.
    for row in 0..t {
        set(layout.trace.cycle, row, F::from_u64(row as u64));
        if row > 0 {
            set(layout.trace.pc_before, row, F::ZERO);
        }
        if row < (t - 1) {
            set(layout.trace.pc_after, row, F::ZERO);
        }
    }

    // Force all rows inactive.
    for row in 0..t {
        set(layout.trace.active, row, F::ZERO);
        // rd helper chain is ungated and must stay algebraically consistent with rd_bit[*] = 0.
        set(layout.trace.rd_is_zero_01, row, F::ONE);
        set(layout.trace.rd_is_zero_012, row, F::ONE);
        set(layout.trace.rd_is_zero_0123, row, F::ONE);
        set(layout.trace.rd_is_zero, row, F::ONE);
    }

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "all-inactive witness should be rejected by an execution anchor"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_trace_one_column_tamper() {
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Tamper trace-local `one` column; production CCS should reject this.
    let one_idx = layout.cell(layout.trace.one, 0);
    w[one_idx - layout.m_in] = F::ZERO;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered trace.one should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_jalr_misaligned_pc_after() {
    // Program:
    //   ADDI x1, x0, 8
    //   JALR x2, x1, 0
    //   BEQ x0, x0, 4
    //   HALT
    //
    // Red-team target: force row1.pc_after to an odd value while keeping
    // pc_after + drop0 + 2*drop1 == rs1 + imm_i and global chaining intact.
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 8,
        },
        RiscvInstruction::Jalr { rd: 2, rs1: 1, imm: 0 },
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 0,
            rs2: 0,
            imm: 4,
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

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (mut x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    let t = exec.rows.len();

    // Shift JALR target chain by -1 from row1 forward (including padded tail),
    // while preserving local control-flow equations and pc chaining.
    for row in 1..t {
        let idx = layout.cell(layout.trace.pc_after, row);
        let new_pc_after = w[idx - layout.m_in] - F::ONE;
        w[idx - layout.m_in] = new_pc_after;
    }
    for row in 2..t {
        let pc_before_idx = layout.cell(layout.trace.pc_before, row);
        let new_pc_before = w[pc_before_idx - layout.m_in] - F::ONE;
        w[pc_before_idx - layout.m_in] = new_pc_before;
        if exec.rows[row].active {
            let prog_addr_idx = layout.cell(layout.trace.prog_addr, row);
            let new_prog_addr = w[prog_addr_idx - layout.m_in] - F::ONE;
            w[prog_addr_idx - layout.m_in] = new_prog_addr;
        }
    }

    // Keep JALR equation satisfied on row1 with an odd pc_after.
    let jalr_b0_idx = layout.cell(layout.trace.jalr_drop_bit[0], 1);
    let jalr_b1_idx = layout.cell(layout.trace.jalr_drop_bit[1], 1);
    w[jalr_b0_idx - layout.m_in] = F::ONE;
    w[jalr_b1_idx - layout.m_in] = F::ZERO;

    // Keep public pc_final consistent with the shifted tail.
    x[layout.pc_final] -= F::ONE;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "misaligned JALR pc_after should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_spurious_ram_addr_on_non_memory_row() {
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Row 0 is ALU (non-memory): keep RAM flags at 0 but inject a spurious address.
    let row0_ram_addr = layout.cell(layout.trace.ram_addr, 0);
    w[row0_ram_addr - layout.m_in] = F::from_u64(1234);

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "non-memory row with spurious ram_addr should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_prog_value_tamper() {
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Flip PROG value for the first row (active row), which should violate
    // active -> (prog_value == instr_word).
    let prog_value_idx = layout.cell(layout.trace.prog_value, 0);
    w[prog_value_idx - layout.m_in] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered witness should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_halted_tail_pc_drift() {
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Red-team: keep continuity constraints satisfied but drift the halted tail PC.
    // row2.pc_after += 1 and row3.pc_before += 1 preserves
    // `pc_after[2] == pc_before[3]` while violating halted-tail quiescence.
    let row2_pc_after_idx = layout.cell(layout.trace.pc_after, 2);
    let row3_pc_before_idx = layout.cell(layout.trace.pc_before, 3);
    w[row2_pc_after_idx - layout.m_in] += F::ONE;
    w[row3_pc_before_idx - layout.m_in] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "halted-tail PC drift should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_halt_flag_mismatch_on_active_row() {
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Row 1 is HALT/system; forge halted=0.
    let row1_halted_idx = layout.cell(layout.trace.halted, 1);
    w[row1_halted_idx - layout.m_in] = F::ZERO;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "active HALT row with halted=0 must fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_opcode_decode_tamper() {
    // Program: ADDI x1, x0, 1; HALT
    //
    // Target production behavior: opcode/decoded fields are semantically bound to instr_word.
    // This test is expected to fail until trace semantics are enforced (not just wiring).
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Tamper opcode on an active row while leaving instr_word unchanged.
    let opcode_idx = layout.cell(layout.trace.opcode, 0);
    w[opcode_idx - layout.m_in] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered opcode decode should not satisfy production-grade trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_lui_writeback_tamper() {
    // Program: LUI x1, 1; HALT
    //
    // Target production behavior: rd_val/writeback must satisfy ISA semantics.
    // This test is expected to fail until trace semantics are enforced.
    let program = vec![RiscvInstruction::Lui { rd: 1, imm: 1 }, RiscvInstruction::Halt];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Tamper rd writeback value on the LUI row.
    let rd_val_idx = layout.cell(layout.trace.rd_val, 0);
    w[rd_val_idx - layout.m_in] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered LUI writeback should not satisfy production-grade trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_auipc_writeback_tamper() {
    // Program: AUIPC x1, 1; HALT
    let program = vec![RiscvInstruction::Auipc { rd: 1, imm: 1 }, RiscvInstruction::Halt];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let twist = RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Tamper rd writeback value on the AUIPC row.
    let rd_val_idx = layout.cell(layout.trace.rd_val, 0);
    w[rd_val_idx - layout.m_in] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered AUIPC writeback should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_jal_link_writeback_tamper() {
    // Program: JAL x1, 8; ADDI x2, x0, 1; HALT
    // Jump skips over ADDI; JAL link value should be pc_before + 4.
    let program = vec![
        RiscvInstruction::Jal { rd: 1, imm: 8 },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Tamper rd writeback value on the JAL row.
    let rd_val_idx = layout.cell(layout.trace.rd_val, 0);
    w[rd_val_idx - layout.m_in] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered JAL link writeback should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_jalr_link_writeback_tamper() {
    // Program:
    //   ADDI x1, x0, 8
    //   JALR x2, x1, 0
    //   HALT
    // JALR link value should be pc_before + 4 on row 1.
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

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Tamper rd writeback value on the JALR row (row 1).
    let rd_val_idx = layout.cell(layout.trace.rd_val, 1);
    w[rd_val_idx - layout.m_in] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered JALR link writeback should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_non_branch_pc_update_tamper() {
    // Program: ADDI x1, x0, 1; ADDI x2, x1, 2; HALT
    //
    // Target production behavior: non-branch rows must apply the correct PC update rule.
    // This test is expected to fail until trace semantics are enforced.
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
            imm: 2,
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Keep wiring equalities intact but drift a straight-line PC transition:
    // row0.pc_after := row0.pc_after + 4
    // row1.pc_before := row1.pc_before + 4
    // row1.prog_addr := row1.prog_addr + 4 (to preserve active->prog_addr==pc_before)
    let row0_pc_after_idx = layout.cell(layout.trace.pc_after, 0);
    let row1_pc_before_idx = layout.cell(layout.trace.pc_before, 1);
    let row1_prog_addr_idx = layout.cell(layout.trace.prog_addr, 1);
    let delta = F::from_u64(4);
    w[row0_pc_after_idx - layout.m_in] += delta;
    w[row1_pc_before_idx - layout.m_in] += delta;
    w[row1_prog_addr_idx - layout.m_in] += delta;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered non-branch PC update should not satisfy production-grade trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_missing_writeback_on_addi() {
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Row 0 is ADDI with rd=1. Forge "no writeback" while keeping existing padding constraints.
    let row0_rd_has_write = layout.cell(layout.trace.rd_has_write, 0);
    let row0_rd_addr = layout.cell(layout.trace.rd_addr, 0);
    let row0_rd_val = layout.cell(layout.trace.rd_val, 0);
    w[row0_rd_has_write - layout.m_in] = F::ZERO;
    w[row0_rd_addr - layout.m_in] = F::ZERO;
    w[row0_rd_val - layout.m_in] = F::ZERO;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "ADDI row without required writeback must fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_load_without_ram_read() {
    // Program: LW x1, 0(x0); HALT
    let program = vec![
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 1,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(RAM_ID, /*addr=*/ 0, /*value=*/ 7);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Tamper load row to look like a non-memory row: clear the RAM read flag and value.
    let row0_ram_has_read = layout.cell(layout.trace.ram_has_read, 0);
    let row0_ram_rv = layout.cell(layout.trace.ram_rv, 0);
    w[row0_ram_has_read - layout.m_in] = F::ZERO;
    w[row0_ram_rv - layout.m_in] = F::ZERO;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "load row without RAM read must fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_store_without_ram_write() {
    // Program: ADDI x1, x0, 9; SW x1, 0(x0); HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 9,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 0,
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Row 1 is SW. Clear write flag and write value.
    let row1_ram_has_write = layout.cell(layout.trace.ram_has_write, 1);
    let row1_ram_wv = layout.cell(layout.trace.ram_wv, 1);
    w[row1_ram_has_write - layout.m_in] = F::ZERO;
    w[row1_ram_wv - layout.m_in] = F::ZERO;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "store row without RAM write must fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_store_with_spurious_rd_writeback() {
    // Program: ADDI x1, x0, 5; SW x1, 4(x0); HALT
    //
    // S-type encodes imm[4:0] in the "rd" field position. With imm=4 this field is non-zero,
    // so a forged rd writeback can satisfy rd_addr==rd unless we enforce store no-writeback policy.
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 4,
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Row 1 is SW. Forge a writeback event that is self-consistent with rd packing.
    let row1_rd_has_write = layout.cell(layout.trace.rd_has_write, 1);
    let row1_rd_addr = layout.cell(layout.trace.rd_addr, 1);
    w[row1_rd_has_write - layout.m_in] = F::ONE;
    w[row1_rd_addr - layout.m_in] = F::from_u64(4);

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "store row with forged rd writeback must fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_load_pc_update_tamper() {
    // Program: LW x1, 0(x0); LW x2, 0(x0); HALT
    let program = vec![
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 1,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 2,
            rs1: 0,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(RAM_ID, /*addr=*/ 0, /*value=*/ 13);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Preserve wiring equalities while drifting the first straight-line transition:
    // row0.pc_after += 4, row1.pc_before += 4, row1.prog_addr += 4.
    let row0_pc_after = layout.cell(layout.trace.pc_after, 0);
    let row1_pc_before = layout.cell(layout.trace.pc_before, 1);
    let row1_prog_addr = layout.cell(layout.trace.prog_addr, 1);
    let delta = F::from_u64(4);
    w[row0_pc_after - layout.m_in] += delta;
    w[row1_pc_before - layout.m_in] += delta;
    w[row1_prog_addr - layout.m_in] += delta;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered load PC update should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_jal_pc_target_tamper() {
    // Program:
    //   JAL x1, 8
    //   ADDI x2, x0, 1   (skipped)
    //   BEQ x0, x0, 4
    //   HALT
    // Row 0 JAL target is pc_before + 8.
    let program = vec![
        RiscvInstruction::Jal { rd: 1, imm: 8 },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 0,
            rs2: 0,
            imm: 4,
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

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Keep continuity and active->prog_addr intact while forging the JAL target.
    // row0.pc_after += 4, row1.pc_before += 4, row1.prog_addr += 4.
    // Row1 is a BRANCH control row, so existing non-control PC constraints do not catch this.
    let row0_pc_after = layout.cell(layout.trace.pc_after, 0);
    let row1_pc_before = layout.cell(layout.trace.pc_before, 1);
    let row1_prog_addr = layout.cell(layout.trace.prog_addr, 1);
    let delta = F::from_u64(4);
    w[row0_pc_after - layout.m_in] += delta;
    w[row1_pc_before - layout.m_in] += delta;
    w[row1_prog_addr - layout.m_in] += delta;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered JAL target should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_jalr_pc_target_tamper() {
    // Program:
    //   ADDI x1, x0, 8
    //   JALR x2, x1, 0
    //   BEQ x0, x0, 4
    //   HALT
    // Row 1 JALR target is (rs1 + imm_i) masked to 4-byte alignment.
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 8,
        },
        RiscvInstruction::Jalr { rd: 2, rs1: 1, imm: 0 },
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 0,
            rs2: 0,
            imm: 4,
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

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Keep continuity and active->prog_addr intact while forging the JALR target.
    // row1.pc_after += 4, row2.pc_before += 4, row2.prog_addr += 4.
    // Row2 is a BRANCH control row, so existing non-control PC constraints do not catch this.
    let row1_pc_after = layout.cell(layout.trace.pc_after, 1);
    let row2_pc_before = layout.cell(layout.trace.pc_before, 2);
    let row2_prog_addr = layout.cell(layout.trace.prog_addr, 2);
    let delta = F::from_u64(4);
    w[row1_pc_after - layout.m_in] += delta;
    w[row2_pc_before - layout.m_in] += delta;
    w[row2_prog_addr - layout.m_in] += delta;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered JALR target should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_branch_target_tamper() {
    // Program:
    //   BEQ x0, x0, 8
    //   ADDI x1, x0, 1   (skipped)
    //   BEQ x0, x0, 4
    //   HALT
    // Row 0 BEQ is always taken, so pc_after must be pc_before + 8.
    let program = vec![
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 0,
            rs2: 0,
            imm: 8,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 0,
            rs2: 0,
            imm: 4,
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

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Keep continuity and active->prog_addr intact while forging the BRANCH target.
    // row0.pc_after += 4, row1.pc_before += 4, row1.prog_addr += 4.
    // Row1 is another BRANCH control row, so existing non-control PC constraints do not catch this.
    let row0_pc_after = layout.cell(layout.trace.pc_after, 0);
    let row1_pc_before = layout.cell(layout.trace.pc_before, 1);
    let row1_prog_addr = layout.cell(layout.trace.prog_addr, 1);
    let delta = F::from_u64(4);
    w[row0_pc_after - layout.m_in] += delta;
    w[row1_pc_before - layout.m_in] += delta;
    w[row1_prog_addr - layout.m_in] += delta;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered branch target should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_load_ram_addr_tamper() {
    // Program: LW x1, 4(x0); HALT
    let program = vec![
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 1,
            rs1: 0,
            imm: 4,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(RAM_ID, /*addr=*/ 4, /*value=*/ 0x1234);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Row 0 is LW. Forge RAM addr while preserving current wiring and class policy constraints.
    let row0_ram_addr = layout.cell(layout.trace.ram_addr, 0);
    w[row0_ram_addr - layout.m_in] += F::from_u64(4);

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered load ram_addr should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_store_ram_addr_tamper() {
    // Program: ADDI x1, x0, 7; SW x1, 4(x0); HALT
    let program = vec![
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
            imm: 4,
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Row 1 is SW. Forge RAM addr while preserving current wiring and class policy constraints.
    let row1_ram_addr = layout.cell(layout.trace.ram_addr, 1);
    w[row1_ram_addr - layout.m_in] += F::from_u64(4);

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered store ram_addr should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_branch_condition_shout_tamper() {
    // Program: BEQ x0, x0, 8; ADDI x1, x0, 1; HALT
    // BEQ compares equal, so shout_val should drive taken=1.
    let program = vec![
        RiscvInstruction::Branch {
            cond: BranchCondition::Eq,
            rs1: 0,
            rs2: 0,
            imm: 8,
        },
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    // Keep PC target intact but forge branch compare output on the branch row.
    let row0_shout_val = layout.cell(layout.trace.shout_val, 0);
    w[row0_shout_val - layout.m_in] = F::ZERO;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered branch compare output should fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_alu_value_binding_tamper() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 7,
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    let rd_val_idx = layout.cell(layout.trace.rd_val, 0);
    w[rd_val_idx - layout.m_in] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered ALU rd value must fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_branch_table_id_tamper() {
    let program = vec![
        RiscvInstruction::Branch {
            cond: BranchCondition::Ltu,
            rs1: 0,
            rs2: 0,
            imm: 4,
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
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    let table_id_idx = layout.cell(layout.trace.shout_table_id, 0);
    w[table_id_idx - layout.m_in] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered branch shout table id must fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_load_writeback_tamper_all_widths() {
    let cases = [
        (RiscvMemOp::Lb, 0x0000_00FFu64, "LB"),
        (RiscvMemOp::Lbu, 0x0000_00FFu64, "LBU"),
        (RiscvMemOp::Lh, 0x0000_8001u64, "LH"),
        (RiscvMemOp::Lhu, 0x0000_8001u64, "LHU"),
        (RiscvMemOp::Lw, 0x1234_5678u64, "LW"),
    ];

    for (op, ram_value, name) in cases {
        let program = vec![
            RiscvInstruction::Load {
                op,
                rd: 1,
                rs1: 0,
                imm: 0,
            },
            RiscvInstruction::Halt,
        ];
        let program_bytes = encode_program(&program);

        let decoded_program = decode_program(&program_bytes).expect("decode_program");
        let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
        cpu.load_program(/*base=*/ 0, decoded_program);
        let mut twist =
            RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
        twist.store(RAM_ID, /*addr=*/ 0, /*value=*/ ram_value);
        let shout = RiscvShoutTables::new(/*xlen=*/ 32);
        let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

        let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
        let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
        let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
        let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

        let rd_val_idx = layout.cell(layout.trace.rd_val, 0);
        w[rd_val_idx - layout.m_in] += F::ONE;

        assert!(
            check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
            "tampered {name} writeback must fail trace CCS"
        );
    }
}

#[test]
fn rv32_trace_wiring_ccs_rejects_sw_store_value_tamper() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 9,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(RAM_ID, /*addr=*/ 0, /*value=*/ 0xAABB_CCDD);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    let row_store = 1usize;
    let ram_wv_idx = layout.cell(layout.trace.ram_wv, row_store);
    w[ram_wv_idx - layout.m_in] += F::ONE;

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "tampered SW store value must fail trace CCS"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_sb_sh_store_merge_tamper() {
    let cases = [(RiscvMemOp::Sb, 0x12i32, "SB"), (RiscvMemOp::Sh, 0x123i32, "SH")];

    for (op, imm, name) in cases {
        let program = vec![
            RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 1,
                rs1: 0,
                imm,
            },
            RiscvInstruction::Store {
                op,
                rs1: 0,
                rs2: 1,
                imm: 0,
            },
            RiscvInstruction::Halt,
        ];
        let program_bytes = encode_program(&program);

        let decoded_program = decode_program(&program_bytes).expect("decode_program");
        let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
        cpu.load_program(/*base=*/ 0, decoded_program);
        let mut twist =
            RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
        twist.store(RAM_ID, /*addr=*/ 0, /*value=*/ 0xA1B2_C3D4);
        let shout = RiscvShoutTables::new(/*xlen=*/ 32);
        let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

        let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 4).expect("from_trace_padded_pow2");
        let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
        let (x, mut w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
        let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

        let row_store = 1usize;
        let ram_wv_idx = layout.cell(layout.trace.ram_wv, row_store);
        w[ram_wv_idx - layout.m_in] += F::ONE;

        assert!(
            check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
            "tampered {name} merge store value must fail trace CCS"
        );
    }
}

#[test]
fn rv32_trace_wiring_ccs_rejects_rv32m_in_trace_scope() {
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
            imm: 4,
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
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "RV32M must be rejected in Tier 2.1 trace scope"
    );
}

#[test]
fn rv32_trace_wiring_ccs_rejects_amo_in_trace_scope() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoaddW,
            rd: 2,
            rs1: 0,
            rs2: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let decoded_program = decode_program(&program_bytes).expect("decode_program");
    let mut cpu = RiscvCpu::new(/*xlen=*/ 32);
    cpu.load_program(/*base=*/ 0, decoded_program);
    let mut twist =
        RiscvMemory::with_program_in_twist(/*xlen=*/ 32, PROG_ID, /*base_addr=*/ 0, &program_bytes);
    twist.store(RAM_ID, /*addr=*/ 0, /*value=*/ 0x44);
    let shout = RiscvShoutTables::new(/*xlen=*/ 32);
    let trace = trace_program(cpu, twist, shout, /*max_steps=*/ 16).expect("trace_program");

    let exec = Rv32ExecTable::from_trace_padded_pow2(&trace, /*min_len=*/ 8).expect("from_trace_padded_pow2");
    let layout = Rv32TraceCcsLayout::new(exec.rows.len()).expect("trace CCS layout");
    let (x, w) = rv32_trace_ccs_witness_from_exec_table(&layout, &exec).expect("trace CCS witness");
    let ccs = build_rv32_trace_wiring_ccs(&layout).expect("trace CCS");

    assert!(
        check_ccs_rowwise_zero(&ccs, &x, &w).is_err(),
        "AMO must be rejected in Tier 2.1 trace scope"
    );
}
