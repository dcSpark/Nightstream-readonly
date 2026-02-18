use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode};

fn run_trace(program: &[RiscvInstruction]) -> neo_fold::riscv_trace_shard::Rv32TraceWiringRun {
    let program_bytes = encode_program(program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(program.len())
        .min_trace_len(program.len())
        .max_steps(program.len())
        .prove()
        .expect("prove");
    run.verify().expect("verify");
    run
}

#[test]
fn trace_program_without_ram_ops_has_no_ram_events() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let run = run_trace(&program);
    let ram_rows = run
        .exec_table()
        .rows
        .iter()
        .filter(|row| !row.ram_events.is_empty())
        .count();
    assert_eq!(ram_rows, 0, "expected no RAM events in non-memory program");
}

#[test]
fn trace_rows_are_sparse_over_time_for_store_load() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 12,
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
    let run = run_trace(&program);

    let ram_rows: Vec<usize> = run
        .exec_table()
        .rows
        .iter()
        .enumerate()
        .filter_map(|(idx, row)| (!row.ram_events.is_empty()).then_some(idx))
        .collect();
    assert_eq!(ram_rows, vec![1, 2], "expected RAM rows only on SW/LW steps");
}

#[test]
fn trace_rows_select_only_expected_opcodes() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 3,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Or,
            rd: 2,
            rs1: 1,
            imm: 1,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Sub,
            rd: 3,
            rs1: 2,
            rs2: 1,
        },
        RiscvInstruction::Halt,
    ];
    let run = run_trace(&program);

    let op_rows: Vec<usize> = run
        .exec_table()
        .rows
        .iter()
        .enumerate()
        .filter_map(|(idx, row)| {
            matches!(
                row.decoded,
                Some(RiscvInstruction::IAlu {
                    op: RiscvOpcode::Or,
                    ..
                }) | Some(RiscvInstruction::RAlu {
                    op: RiscvOpcode::Sub,
                    ..
                })
            )
            .then_some(idx)
        })
        .collect();
    assert_eq!(op_rows, vec![1, 2], "expected OR/SUB rows at indices 1 and 2");
}
