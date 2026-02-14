use std::time::{Duration, Instant};

use neo_fold::riscv_shard::Rv32B1;
use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_memory::riscv::lookups::{encode_program, BranchCondition, RiscvInstruction, RiscvOpcode};

#[test]
#[ignore = "perf-style test: run with `cargo test -p neo-fold --release --test perf -- --ignored --nocapture compare_single_addi_metrics_nightstream_only`"]
fn compare_single_addi_metrics_nightstream_only() {
    let instruction_label = "ADDI x1,x0,1";

    let ns_program = vec![RiscvInstruction::IAlu {
        op: RiscvOpcode::Add,
        rd: 1,
        rs1: 0,
        imm: 1,
    }];
    let ns_program_bytes = encode_program(&ns_program);
    let ns_chunk_size = 1usize;
    let ns_max_steps = 1usize;
    let ns_ram_bytes = 4usize;

    let ns_total_start = Instant::now();
    let mut ns_run = Rv32B1::from_rom(/*program_base=*/ 0, &ns_program_bytes)
        .chunk_size(ns_chunk_size)
        .ram_bytes(ns_ram_bytes)
        .max_steps(ns_max_steps)
        .prove()
        .expect("Nightstream prove");

    let ns_constraints = ns_run.ccs_num_constraints();
    let ns_witness_cols = ns_run.ccs_num_variables();
    let ns_constraints_padded_pow2 = ns_constraints.next_power_of_two();
    let ns_witness_cols_padded_pow2 = ns_witness_cols.next_power_of_two();
    let ns_fold_count = ns_run.fold_count();
    let ns_trace_len = ns_run.riscv_trace_len().expect("Nightstream trace length");
    let ns_shout_lookups = ns_run
        .shout_lookup_count()
        .expect("Nightstream shout lookup count");
    let ns_step0 = ns_run
        .steps_public()
        .first()
        .cloned()
        .expect("Nightstream collected steps");
    let ns_m_in = ns_step0.mcs_inst.m_in;
    let ns_witness_private = ns_witness_cols.saturating_sub(ns_m_in);
    let ns_lut_instances = ns_step0.lut_insts.len();
    let ns_mem_instances = ns_step0.mem_insts.len();

    ns_run.verify().expect("Nightstream verify");
    let ns_prove_time = ns_run.prove_duration();
    let ns_verify_time = ns_run
        .verify_duration()
        .expect("Nightstream verify duration");
    let ns_total_duration = ns_total_start.elapsed();

    println!();
    println!("Instruction under test: {instruction_label}");
    println!();
    println!("**Nightstream (Neo RV32 B1)**");
    println!(
        "- CCS: n={} constraints (padded_pow2_n={}), m={} cols (padded_pow2_m={}) (m_in={} public, w={} private)",
        ns_constraints,
        ns_constraints_padded_pow2,
        ns_witness_cols,
        ns_witness_cols_padded_pow2,
        ns_m_in,
        ns_witness_private
    );
    println!(
        "- Trace: executed_steps={} (max_steps={}), fold_chunks={} (chunk_size={})",
        ns_trace_len, ns_max_steps, ns_fold_count, ns_chunk_size
    );
    println!(
        "- Sidecars: lut_instances={} mem_instances={} shout_lookups_used={}",
        ns_lut_instances, ns_mem_instances, ns_shout_lookups
    );
    println!(
        "- Time: prove={} verify={} total_end_to_end={}",
        fmt_duration(ns_prove_time),
        fmt_duration(ns_verify_time),
        fmt_duration(ns_total_duration)
    );
    println!();

    println!("{:-<80}", "");
    println!("{:<40} {:>18}", "Metric", "Nightstream");
    println!("{:<40} {:>18}", "", "(RV32 B1)");
    println!("{:-<80}", "");
    println!("{:<40} {:>18}", "Rows per step (raw)", ns_constraints);
    println!(
        "{:<40} {:>18}",
        "Rows per step (padded pow2)", ns_constraints_padded_pow2
    );
    println!(
        "{:<40} {:>18}",
        "Total rows in proof (padded)",
        ns_constraints_padded_pow2.saturating_mul(ns_fold_count)
    );
    println!(
        "{:<40} {:>18}",
        "Total rows (estimate, unpadded)",
        ns_constraints.saturating_mul(ns_trace_len)
    );
    println!("{:<40} {:>18}", "Cols / vars (raw)", ns_witness_cols);
    println!(
        "{:<40} {:>18}",
        "Cols / vars (padded pow2)", ns_witness_cols_padded_pow2
    );
    println!("{:<40} {:>18}", "Public inputs (m_in)", ns_m_in);
    println!(
        "{:<40} {:>18}",
        "Trace len (unpadded)",
        format!("{} steps", ns_trace_len)
    );
    println!("{:<40} {:>18}", "Lookup tables", format!("{} Shout", ns_lut_instances));
    println!("{:<40} {:>18}", "Lookups used", ns_shout_lookups);
    println!("{:<40} {:>18}", "Prove time", fmt_duration(ns_prove_time));
    println!("{:<40} {:>18}", "Verify time", fmt_duration(ns_verify_time));
    println!("{:-<80}", "");
}

fn fmt_duration(d: Duration) -> String {
    if d.as_secs_f64() < 1.0 {
        format!("{:.3}ms", d.as_secs_f64() * 1000.0)
    } else {
        format!("{:.3}s", d.as_secs_f64())
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) => v.parse::<usize>().unwrap_or(default),
        Err(_) => default,
    }
}

fn mixed_instruction_sequence() -> Vec<RiscvInstruction> {
    vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::And,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Or,
            rd: 3,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Xor,
            rd: 4,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Slt,
            rd: 6,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sltu,
            rd: 7,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sll,
            rd: 8,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Srl,
            rd: 9,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sra,
            rd: 10,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 0,
            rs2: 0,
            imm: 8,
        },
    ]
}

#[test]
#[ignore = "perf-style test: NS_DEBUG_N=256 cargo test -p neo-fold --release --test perf -- --ignored --nocapture debug_trace_single_n_addi_only"]
fn debug_trace_single_n_addi_only() {
    let n = env_usize("NS_DEBUG_N", 256);
    let chunk_rows = env_usize("NS_TRACE_CHUNK_ROWS", n + 1);
    assert!(n > 0);
    assert!(chunk_rows > 0);

    let mut program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 1,
            imm: 1,
        };
        n
    ];
    program.push(RiscvInstruction::Halt);
    let program_bytes = encode_program(&program);
    let steps = n + 1;

    let total_start = Instant::now();
    let mut run = Rv32TraceWiring::from_rom(0, &program_bytes)
        .min_trace_len(steps)
        .max_steps(steps)
        .chunk_rows(chunk_rows)
        .prove()
        .expect("trace prove");
    let prove_time = run.prove_duration();
    run.verify().expect("trace verify");
    let verify_time = run.verify_duration().expect("trace verify duration");
    let total_time = total_start.elapsed();
    let phases = run.prove_phase_durations();

    println!(
        "TRACE n={} chunk_rows={} ccs_n={} ccs_m={} n_p2={} m_p2={} trace_len={} folds={} prove={} verify={} total={} phases(setup={}, chunk_commit={}, fold={})",
        n,
        chunk_rows,
        run.ccs_num_constraints(),
        run.ccs_num_variables(),
        run.ccs_num_constraints().next_power_of_two(),
        run.ccs_num_variables().next_power_of_two(),
        run.trace_len(),
        run.fold_count(),
        fmt_duration(prove_time),
        fmt_duration(verify_time),
        fmt_duration(total_time),
        fmt_duration(phases.setup),
        fmt_duration(phases.chunk_build_commit),
        fmt_duration(phases.fold_and_prove),
    );
}

#[test]
#[ignore = "perf-style test: NS_DEBUG_N=256 cargo test -p neo-fold --release --test perf -- --ignored --nocapture debug_chunked_single_n_addi_only"]
fn debug_chunked_single_n_addi_only() {
    let n = env_usize("NS_DEBUG_N", 256);
    assert!(n > 0);

    let mut program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 1,
            imm: 1,
        };
        n
    ];
    program.push(RiscvInstruction::Halt);
    let program_bytes = encode_program(&program);
    let steps = n + 1;

    let total_start = Instant::now();
    let mut run = Rv32B1::from_rom(0, &program_bytes)
        .chunk_size(steps)
        .ram_bytes(4)
        .max_steps(steps)
        .prove()
        .expect("chunked prove");
    let prove_time = run.prove_duration();
    run.verify().expect("chunked verify");
    let verify_time = run.verify_duration().expect("chunked verify duration");
    let total_time = total_start.elapsed();
    let trace_len = run.riscv_trace_len().expect("trace len");

    println!(
        "CHUNKED n={} ccs_n={} ccs_m={} n_p2={} m_p2={} trace_len={} folds={} prove={} verify={} total={}",
        n,
        run.ccs_num_constraints(),
        run.ccs_num_variables(),
        run.ccs_num_constraints().next_power_of_two(),
        run.ccs_num_variables().next_power_of_two(),
        trace_len,
        run.fold_count(),
        fmt_duration(prove_time),
        fmt_duration(verify_time),
        fmt_duration(total_time),
    );
}

#[test]
#[ignore = "perf-style test: NS_DEBUG_N=256 cargo test -p neo-fold --release --test perf -- --ignored --nocapture debug_trace_vs_chunked_single_n_mixed_ops"]
fn debug_trace_vs_chunked_single_n_mixed_ops() {
    let n = env_usize("NS_DEBUG_N", 256);
    let chunk_rows = env_usize("NS_TRACE_CHUNK_ROWS", n + 1);
    assert!(n > 0);
    assert!(chunk_rows > 0);
    let base = mixed_instruction_sequence();
    assert_eq!(base.len(), 10);

    let mut program: Vec<RiscvInstruction> = (0..n).map(|i| base[i % base.len()].clone()).collect();
    program.push(RiscvInstruction::Halt);
    let program_bytes = encode_program(&program);
    let steps = n + 1;

    let chunk_total_start = Instant::now();
    let mut chunk_run = Rv32B1::from_rom(0, &program_bytes)
        .chunk_size(steps)
        .ram_bytes(4)
        .max_steps(steps)
        .prove()
        .expect("chunked prove (mixed)");
    let chunk_prove = chunk_run.prove_duration();
    chunk_run.verify().expect("chunked verify (mixed)");
    let chunk_verify = chunk_run
        .verify_duration()
        .expect("chunked verify duration");
    let chunk_total = chunk_total_start.elapsed();

    let trace_total_start = Instant::now();
    let trace_res = Rv32TraceWiring::from_rom(0, &program_bytes)
        .min_trace_len(steps)
        .max_steps(steps)
        .chunk_rows(chunk_rows)
        .prove();
    match trace_res {
        Ok(mut trace_run) => {
            let trace_prove = trace_run.prove_duration();
            trace_run.verify().expect("trace verify (mixed)");
            let trace_verify = trace_run.verify_duration().expect("trace verify duration");
            let trace_total = trace_total_start.elapsed();
            println!(
                "MIXED n={} TRACE(prove={}, verify={}, total={}, n_p2={}, m_p2={}) CHUNKED(prove={}, verify={}, total={}, n_p2={}, m_p2={}) ratio_prove={:.2}x",
                n,
                fmt_duration(trace_prove),
                fmt_duration(trace_verify),
                fmt_duration(trace_total),
                trace_run.ccs_num_constraints().next_power_of_two(),
                trace_run.ccs_num_variables().next_power_of_two(),
                fmt_duration(chunk_prove),
                fmt_duration(chunk_verify),
                fmt_duration(chunk_total),
                chunk_run.ccs_num_constraints().next_power_of_two(),
                chunk_run.ccs_num_variables().next_power_of_two(),
                trace_prove.as_secs_f64() / chunk_prove.as_secs_f64(),
            );
        }
        Err(e) => {
            println!(
                "MIXED n={} TRACE(prove=ERROR:{}) CHUNKED(prove={}, verify={}, total={}, n_p2={}, m_p2={})",
                n,
                e,
                fmt_duration(chunk_prove),
                fmt_duration(chunk_verify),
                fmt_duration(chunk_total),
                chunk_run.ccs_num_constraints().next_power_of_two(),
                chunk_run.ccs_num_variables().next_power_of_two(),
            );
        }
    }
}
