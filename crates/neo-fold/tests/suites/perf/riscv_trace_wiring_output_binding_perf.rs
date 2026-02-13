#![allow(non_snake_case)]

use std::time::Duration;

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_math::F;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;

#[derive(Clone, Copy, Debug)]
enum ClaimMode {
    None,
    Reg,
}

#[derive(Clone, Copy, Debug)]
struct Stats {
    min: Duration,
    median: Duration,
    mean: Duration,
    max: Duration,
}

#[test]
#[ignore = "perf-style test: run with `cargo test -p neo-fold --release --test riscv_trace_wiring_output_binding_perf -- --ignored --nocapture`"]
fn rv32_trace_wiring_output_binding_overhead_perf() {
    let n_adds = env_usize("TW_N_ADDS", 512);
    let samples = env_usize("TW_SAMPLES", 7);
    let warmups = env_usize("TW_WARMUPS", 1);
    assert!(n_adds > 0, "TW_N_ADDS must be > 0");
    assert!(samples > 0, "TW_SAMPLES must be > 0");
    let expected_x1 = F::from_u64(n_adds as u64);

    let mut program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 1,
            imm: 1,
        };
        n_adds
    ];
    program.push(RiscvInstruction::Halt);
    let program_bytes = encode_program(&program);

    let (none, none_shape) = run_samples(
        &program_bytes,
        n_adds + 1,
        expected_x1,
        ClaimMode::None,
        warmups,
        samples,
    );
    let (reg, _reg_shape) = run_samples(
        &program_bytes,
        n_adds + 1,
        expected_x1,
        ClaimMode::Reg,
        warmups,
        samples,
    );

    let none_stats = summarize(&none);
    let reg_stats = summarize(&reg);
    let median_ratio = ratio(reg_stats.median, none_stats.median);
    let mean_ratio = ratio(reg_stats.mean, none_stats.mean);

    println!();
    println!("{:=<96}", "");
    println!("RV32 TRACE WIRING PERF — NO OUTPUT vs REG OUTPUT BINDING");
    println!("{:=<96}", "");
    println!(
        "config: n_adds={} trace_len={} warmups={} samples={}",
        n_adds,
        n_adds + 1,
        warmups,
        samples
    );
    if let Some((ccs_n, ccs_m, trace_len)) = none_shape {
        println!("shape: ccs_n={} ccs_m={} trace_len={}", ccs_n, ccs_m, trace_len);
    }
    println!("{:-<96}", "");
    println!(
        "{:>12}  {:>10} {:>10} {:>10} {:>10}",
        "mode", "min", "median", "mean", "max"
    );
    println!("{:-<96}", "");
    println!(
        "{:>12}  {:>10} {:>10} {:>10} {:>10}",
        "no-output",
        fmt_duration(none_stats.min),
        fmt_duration(none_stats.median),
        fmt_duration(none_stats.mean),
        fmt_duration(none_stats.max),
    );
    println!(
        "{:>12}  {:>10} {:>10} {:>10} {:>10}",
        "reg-output",
        fmt_duration(reg_stats.min),
        fmt_duration(reg_stats.median),
        fmt_duration(reg_stats.mean),
        fmt_duration(reg_stats.max),
    );
    println!("{:-<96}", "");
    println!(
        "ratio reg/no-output: median={:.3}x mean={:.3}x",
        median_ratio, mean_ratio
    );
    let trace_len = n_adds + 1;
    let none_khz = trace_len as f64 / none_stats.median.as_secs_f64() / 1_000.0;
    let reg_khz = trace_len as f64 / reg_stats.median.as_secs_f64() / 1_000.0;
    println!(
        "throughput (median): no-output={:.3} kHz reg-output={:.3} kHz",
        none_khz, reg_khz
    );
    println!("{:-<96}", "");
    println!();
}

fn run_samples(
    program_bytes: &[u8],
    max_steps: usize,
    expected_x1: F,
    claim_mode: ClaimMode,
    warmups: usize,
    samples: usize,
) -> (Vec<Duration>, Option<(usize, usize, usize)>) {
    for _ in 0..warmups {
        let mut run = build_runner(program_bytes, max_steps, expected_x1, claim_mode)
            .prove()
            .expect("warmup prove");
        run.verify().expect("warmup verify");
    }

    let mut out = Vec::with_capacity(samples);
    let mut shape: Option<(usize, usize, usize)> = None;
    for _ in 0..samples {
        let mut run = build_runner(program_bytes, max_steps, expected_x1, claim_mode)
            .prove()
            .expect("prove");
        run.verify().expect("verify");
        if shape.is_none() {
            shape = Some((run.ccs_num_constraints(), run.ccs_num_variables(), run.trace_len()));
        }
        out.push(run.prove_duration());
    }
    (out, shape)
}

fn build_runner(program_bytes: &[u8], max_steps: usize, expected_x1: F, claim_mode: ClaimMode) -> Rv32TraceWiring {
    let runner = Rv32TraceWiring::from_rom(/*program_base=*/ 0, program_bytes)
        .min_trace_len(max_steps)
        .max_steps(max_steps);

    match claim_mode {
        ClaimMode::None => runner,
        ClaimMode::Reg => runner.reg_output_claim(/*reg=*/ 1, expected_x1),
    }
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

fn ratio(numer: Duration, denom: Duration) -> f64 {
    if denom.is_zero() {
        return f64::INFINITY;
    }
    numer.as_secs_f64() / denom.as_secs_f64()
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
