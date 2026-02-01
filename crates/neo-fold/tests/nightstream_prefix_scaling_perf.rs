#![allow(non_snake_case)]

use std::time::{Duration, Instant};

use neo_fold::riscv_shard::Rv32B1;
use neo_memory::riscv::lookups::{encode_program, BranchCondition, RiscvInstruction, RiscvOpcode};

struct ScaleRow {
    n_instr: usize,

    ns_step_rows_raw: usize,
    ns_step_rows_p2: usize,
    ns_cols_p2: usize,
    ns_fold_chunks: usize,
    ns_rows_total_padded: usize,
    ns_prove_time: Duration,
    ns_verify_time: Duration,
    ns_total_time: Duration,
}

#[test]
#[ignore = "perf test; run with `cargo test -p neo-fold --test nightstream_prefix_scaling_perf --release -- --ignored --nocapture`"]
fn nightstream_prefix_lengths_1_to_10_and_256() {
    // Fixed instruction sequence; we benchmark prefixes of length 1..10 and 256.
    //
    // Nightstream: execute `n` RV32 instructions and prove them as a single chunk (chunk_size=n),
    //              so this is one proof per prefix length (no folding per instruction).
    let base_sequence = instruction_sequence();
    assert_eq!(base_sequence.len(), 10);

    let mut rows: Vec<ScaleRow> = Vec::with_capacity(11);
    let mut ns: Vec<usize> = (1..=10).collect();
    ns.push(256);

    for n in ns {
        let ns_program: Vec<RiscvInstruction> = (0..n)
            .map(|i| base_sequence[i % base_sequence.len()].clone())
            .collect();
        let ns_program_bytes = encode_program(&ns_program);

        let ns_total_start = Instant::now();
        let mut ns_run = Rv32B1::from_rom(/*program_base=*/ 0, &ns_program_bytes)
            // IMPORTANT: avoid "fold per instruction".
            // Use a single chunk that covers the entire prefix so this is one proof for `n` instructions.
            .chunk_size(n)
            .ram_bytes(4)
            .max_steps(n)
            .prove()
            .expect("Nightstream prove");

        let ns_step_rows_raw = ns_run.ccs_num_constraints();
        let ns_cols_raw = ns_run.ccs_num_variables();
        let ns_step_rows_p2 = ns_step_rows_raw.next_power_of_two();
        let ns_cols_p2 = ns_cols_raw.next_power_of_two();
        let ns_fold_chunks = ns_run.fold_count();
        let ns_rows_total_padded = ns_step_rows_p2.saturating_mul(ns_fold_chunks);

        ns_run.verify().expect("Nightstream verify");
        let ns_prove_time = ns_run.prove_duration();
        let ns_verify_time = ns_run.verify_duration().expect("Nightstream verify duration");
        let ns_total_time = ns_total_start.elapsed();

        rows.push(ScaleRow {
            n_instr: n,

            ns_step_rows_raw,
            ns_step_rows_p2,
            ns_cols_p2,
            ns_fold_chunks,
            ns_rows_total_padded,
            ns_prove_time,
            ns_verify_time,
            ns_total_time,
        });
    }

    println!();
    println!("{:=<105}", "");
    println!("NIGHTSTREAM — PREFIX SCALING (n=1..10, 256)");
    println!("{:=<105}", "");
    println!("Note: times include per-run setup; compare trends (slope) more than absolute intercept on tiny traces.");
    println!("Note: rowsTotal = next_pow2(ccs.n) * fold_chunks, cols(p2) = next_pow2(ccs.m).");
    println!();

    println!("{:-<105}", "");
    println!(
        "{:>4}  {:>13} {:>10} {:>10} {:>8}  {:>9} {:>9} {:>9}  {:>9}",
        "n", "rows/chunk", "rowsTot", "cols(p2)", "chunks", "prove", "verify", "total", "prove/n",
    );
    println!("{:-<105}", "");
    for r in &rows {
        let ns_rows_step = format!("{}/{}", r.ns_step_rows_raw, r.ns_step_rows_p2);
        println!(
            "{:>4}  {:>13} {:>10} {:>10} {:>8}  {:>9} {:>9} {:>9}  {:>9}",
            r.n_instr,
            ns_rows_step,
            r.ns_rows_total_padded,
            r.ns_cols_p2,
            r.ns_fold_chunks,
            fmt_duration(r.ns_prove_time),
            fmt_duration(r.ns_verify_time),
            fmt_duration(r.ns_total_time),
            fmt_duration(div_duration(r.ns_prove_time, r.n_instr)),
        );
    }
    println!("{:-<105}", "");
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

fn div_duration(d: Duration, denom: usize) -> Duration {
    if denom == 0 {
        return Duration::from_secs(0);
    }
    Duration::from_secs_f64(d.as_secs_f64() / denom as f64)
}

