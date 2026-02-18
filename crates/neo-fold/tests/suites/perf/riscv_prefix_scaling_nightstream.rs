use std::time::{Duration, Instant};

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::lookups::{encode_program, BranchCondition, RiscvInstruction, RiscvOpcode};

struct ScaleRow {
    n_instr: usize,

    ns_step_rows_raw: usize,
    ns_step_rows_p2: usize,
    ns_cols_raw: usize,
    ns_cols_p2: usize,
    ns_fold_chunks: usize,
    ns_rows_total_padded: usize,
    ns_prove_time: Duration,
    ns_verify_time: Duration,
    ns_total_time: Duration,
}

#[test]
#[ignore = "perf-style test: run with `cargo test -p neo-fold --test riscv_prefix_scaling_nightstream --release -- --ignored --nocapture`"]
fn nightstream_prefix_lengths_1_to_10_and_256_halt_terminated() {
    // Fixed instruction sequence; we benchmark prefixes of length 1..10, plus 256.
    //
    // We always append a HALT so each program terminates. We then prove the whole trace as a
    // single chunk by setting chunk_size = trace_len (no folding per instruction).
    let base_sequence: Vec<RiscvInstruction> = instruction_sequence();
    assert_eq!(base_sequence.len(), 10);

    let mut rows: Vec<ScaleRow> = Vec::with_capacity(11);
    let mut ns: Vec<usize> = (1..=10).collect();
    ns.push(256);

    for n in ns {
        let mut program: Vec<RiscvInstruction> = (0..n)
            .map(|i| base_sequence[i % base_sequence.len()].clone())
            .collect();
        program.push(RiscvInstruction::Halt);
        let program_bytes = encode_program(&program);

        let trace_len = n + 1;
        let ns_total_start = Instant::now();

        let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
            .min_trace_len(trace_len)
            .chunk_rows(trace_len)
            .max_steps(trace_len)
            .prove()
            .expect("Nightstream prove");

        let ns_step_rows_raw = run.ccs_num_constraints();
        let ns_cols_raw = run.ccs_num_variables();
        let ns_step_rows_p2 = ns_step_rows_raw.next_power_of_two();
        let ns_cols_p2 = ns_cols_raw.next_power_of_two();
        let ns_fold_chunks = run.fold_count();
        let ns_rows_total_padded = ns_step_rows_p2.saturating_mul(ns_fold_chunks);

        run.verify().expect("Nightstream verify");
        let ns_prove_time = run.prove_duration();
        let ns_verify_time = run.verify_duration().unwrap_or(Duration::ZERO);
        let ns_total_time = ns_total_start.elapsed();

        rows.push(ScaleRow {
            n_instr: n,

            ns_step_rows_raw,
            ns_step_rows_p2,
            ns_cols_raw,
            ns_cols_p2,
            ns_fold_chunks,
            ns_rows_total_padded,
            ns_prove_time,
            ns_verify_time,
            ns_total_time,
        });
    }

    println!();
    println!("{:=<110}", "");
    println!("NIGHTSTREAM — SCALING (prefix n=1..10, 256; prove as single chunk)");
    println!("{:=<110}", "");
    println!("Note: times include per-run setup; compare trends more than absolute intercept on tiny traces.");
    println!();

    println!("{:-<110}", "");
    println!(
        "{:>4}  {:>14} {:>10} {:>10} {:>10}  {:>9} {:>9} {:>9}  {:>9}",
        "n", "NS rows/chunk", "NS rowsTot", "NS cols", "NS cols(p2)", "chunks", "prove", "verify", "total",
    );
    println!("{:-<110}", "");
    for r in &rows {
        let ns_rows_step = format!("{}/{}", r.ns_step_rows_raw, r.ns_step_rows_p2);
        println!(
            "{:>4}  {:>14} {:>10} {:>10} {:>10}  {:>9} {:>9} {:>9}  {:>9}",
            r.n_instr,
            ns_rows_step,
            r.ns_rows_total_padded,
            r.ns_cols_raw,
            r.ns_cols_p2,
            r.ns_fold_chunks,
            fmt_duration(r.ns_prove_time),
            fmt_duration(r.ns_verify_time),
            fmt_duration(r.ns_total_time),
        );
    }
    println!("{:-<110}", "");
    println!();
}

fn instruction_sequence() -> Vec<RiscvInstruction> {
    vec![
        // ADDI x1,x0,1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        // ANDI x2,x0,1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::And,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        // ORI x3,x0,1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Or,
            rd: 3,
            rs1: 0,
            imm: 1,
        },
        // XORI x4,x0,1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Xor,
            rd: 4,
            rs1: 0,
            imm: 1,
        },
        // SLTI x6,x0,1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Slt,
            rd: 6,
            rs1: 0,
            imm: 1,
        },
        // SLTIU x7,x0,1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sltu,
            rd: 7,
            rs1: 0,
            imm: 1,
        },
        // SLLI x8,x0,1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sll,
            rd: 8,
            rs1: 0,
            imm: 1,
        },
        // SRLI x9,x0,1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Srl,
            rd: 9,
            rs1: 0,
            imm: 1,
        },
        // SRAI x10,x0,1
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sra,
            rd: 10,
            rs1: 0,
            imm: 1,
        },
        // BNE x0,x0,+8 (not taken)
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 0,
            rs2: 0,
            imm: 8,
        },
    ]
}

fn fmt_duration(d: Duration) -> String {
    if d.as_secs_f64() < 1.0 {
        format!("{:.3}ms", d.as_secs_f64() * 1000.0)
    } else {
        format!("{:.3}s", d.as_secs_f64())
    }
}
