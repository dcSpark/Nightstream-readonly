#![allow(non_snake_case)]

use std::time::Duration;

use neo_fold::riscv_shard::Rv32B1;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvMemOp, RiscvOpcode};

#[derive(Clone, Copy, Debug)]
struct Stats {
    min: Duration,
    median: Duration,
    mean: Duration,
    max: Duration,
}

#[test]
#[ignore = "perf-style test: run with `cargo test -p neo-fold --release --test riscv_b1_ab_perf -- --ignored --nocapture`"]
fn rv32_b1_ab_perf_single_chunk() {
    let repeats = env_usize("AB_REPEATS", 64);
    let warmups = env_usize("AB_WARMUPS", 1);
    let samples = env_usize("AB_SAMPLES", 7);
    assert!(repeats > 0, "AB_REPEATS must be > 0");
    assert!(samples > 0, "AB_SAMPLES must be > 0");

    let mut program = Vec::<RiscvInstruction>::new();
    for _ in 0..repeats {
        program.extend([
            // x1 = 3; x2 = 4
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
            // x3 = x1 * x2
            RiscvInstruction::RAlu {
                op: RiscvOpcode::Mul,
                rd: 3,
                rs1: 1,
                rs2: 2,
            },
            // mem[0] = x3; x4 = mem[0]
            RiscvInstruction::Store {
                op: RiscvMemOp::Sw,
                rs1: 0,
                rs2: 3,
                imm: 0,
            },
            RiscvInstruction::Load {
                op: RiscvMemOp::Lw,
                rd: 4,
                rs1: 0,
                imm: 0,
            },
        ]);
    }
    program.push(RiscvInstruction::Halt);

    let program_bytes = encode_program(&program);
    let max_steps = program.len();

    for _ in 0..warmups {
        let mut run = run_once(&program_bytes, max_steps).expect("warmup prove");
        run.verify().expect("warmup verify");
    }

    let mut prove_times = Vec::with_capacity(samples);
    let mut verify_times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut run = run_once(&program_bytes, max_steps).expect("prove");
        run.verify().expect("verify");
        prove_times.push(run.prove_duration());
        verify_times.push(run.verify_duration().unwrap_or(Duration::ZERO));
    }

    let prove = summarize(&prove_times);
    let verify = summarize(&verify_times);

    println!();
    println!("{:=<96}", "");
    println!("RV32 B1 A/B PERF (single chunk, fixed program)");
    println!("{:=<96}", "");
    println!(
        "config: repeats={} instructions={} warmups={} samples={}",
        repeats, max_steps, warmups, samples
    );
    println!("{:-<96}", "");
    println!(
        "{:>10}  {:>10} {:>10} {:>10} {:>10}",
        "phase", "min", "median", "mean", "max"
    );
    println!("{:-<96}", "");
    println!(
        "{:>10}  {:>10} {:>10} {:>10} {:>10}",
        "prove",
        fmt_duration(prove.min),
        fmt_duration(prove.median),
        fmt_duration(prove.mean),
        fmt_duration(prove.max),
    );
    println!(
        "{:>10}  {:>10} {:>10} {:>10} {:>10}",
        "verify",
        fmt_duration(verify.min),
        fmt_duration(verify.median),
        fmt_duration(verify.mean),
        fmt_duration(verify.max),
    );
    println!("{:-<96}", "");
    println!();
}

fn run_once(program_bytes: &[u8], max_steps: usize) -> Result<neo_fold::riscv_shard::Rv32B1Run, neo_fold::PiCcsError> {
    Rv32B1::from_rom(/*program_base=*/ 0, program_bytes)
        .xlen(32)
        .ram_bytes(0x40)
        .chunk_size(max_steps)
        .max_steps(max_steps)
        .shout_auto_minimal()
        .prove()
}

fn summarize(samples: &[Duration]) -> Stats {
    assert!(!samples.is_empty());
    let mut v = samples.to_vec();
    v.sort_unstable();
    let min = v[0];
    let max = v[v.len() - 1];
    let median = v[v.len() / 2];
    let mean_secs = v.iter().map(Duration::as_secs_f64).sum::<f64>() / v.len() as f64;
    let mean = Duration::from_secs_f64(mean_secs);
    Stats { min, median, mean, max }
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
