#![allow(non_snake_case)]

use std::time::Duration;

use neo_fold::pi_ccs::FoldingMode;
use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_math::F;
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode};
use p3_field::PrimeCharacteristicRing;

#[derive(Clone, Copy, Debug)]
enum Variant {
    AddFiveTimes,
    MulByFive,
}

#[derive(Clone, Copy, Debug)]
struct Stats {
    min: Duration,
    median: Duration,
    mean: Duration,
    max: Duration,
}

#[derive(Clone, Copy, Debug)]
struct VariantStats {
    prove: Stats,
    verify: Stats,
    trace_len: usize,
}

#[test]
#[ignore = "perf-style test: run with `cargo test -p neo-fold --release --test perf rv32_trace_perf_add5_vs_mul5_large_value -- --ignored --nocapture`"]
fn rv32_trace_perf_add5_vs_mul5_large_value() {
    let repeats = env_usize("ADD5_MUL5_REPEATS", 1024);
    let warmups = env_usize("ADD5_MUL5_WARMUPS", 1);
    let samples = env_usize("ADD5_MUL5_SAMPLES", 7);
    let large_value = env_u32("ADD5_MUL5_VALUE", 0x7F12_6789);
    assert!(repeats > 0, "ADD5_MUL5_REPEATS must be > 0");
    assert!(samples > 0, "ADD5_MUL5_SAMPLES must be > 0");

    let add_program = build_program(Variant::AddFiveTimes, repeats, large_value);
    let mul_program = build_program(Variant::MulByFive, repeats, large_value);
    let expected = expected_result(large_value, repeats);

    let add_stats = run_samples(&add_program, expected, warmups, samples);
    let mul_stats = run_samples(&mul_program, expected, warmups, samples);

    let prove_ratio = ratio(mul_stats.prove.median, add_stats.prove.median);
    let verify_ratio = ratio(mul_stats.verify.median, add_stats.verify.median);

    let add_prove_per_iter = add_stats.prove.median.as_secs_f64() / repeats as f64;
    let mul_prove_per_iter = mul_stats.prove.median.as_secs_f64() / repeats as f64;
    let add_prove_per_inst = add_stats.prove.median.as_secs_f64() / add_stats.trace_len as f64;
    let mul_prove_per_inst = mul_stats.prove.median.as_secs_f64() / mul_stats.trace_len as f64;

    println!();
    println!("{:=<112}", "");
    println!("RV32 TRACE PERF - (ADD value) x5 vs (MUL value*5), same final accumulator");
    println!("{:=<112}", "");
    println!(
        "config: value={:#010x} repeats={} warmups={} samples={}",
        large_value, repeats, warmups, samples
    );
    println!("expected x10 final value: {:#010x} (mod 2^32)", expected);
    println!(
        "trace_len: add5={} instructions, mul5={} instructions",
        add_stats.trace_len, mul_stats.trace_len
    );
    println!("{:-<112}", "");
    println!(
        "{:>12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "variant", "phase", "min", "median", "mean", "max"
    );
    println!("{:-<112}", "");
    print_row("add5", "prove", add_stats.prove);
    print_row("mul5", "prove", mul_stats.prove);
    print_row("add5", "verify", add_stats.verify);
    print_row("mul5", "verify", mul_stats.verify);
    println!("{:-<112}", "");
    println!(
        "median ratio (mul5/add5): prove={:.3}x verify={:.3}x",
        prove_ratio, verify_ratio
    );
    println!(
        "prove median per iteration: add5={:.6}ms mul5={:.6}ms",
        add_prove_per_iter * 1000.0,
        mul_prove_per_iter * 1000.0
    );
    println!(
        "prove median per instruction: add5={:.6}ms mul5={:.6}ms",
        add_prove_per_inst * 1000.0,
        mul_prove_per_inst * 1000.0
    );
    println!("{:-<112}", "");
    println!();
}

fn print_row(variant: &str, phase: &str, stats: Stats) {
    println!(
        "{:>12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        variant,
        phase,
        fmt_duration(stats.min),
        fmt_duration(stats.median),
        fmt_duration(stats.mean),
        fmt_duration(stats.max),
    );
}

fn build_program(variant: Variant, repeats: usize, value: u32) -> Vec<RiscvInstruction> {
    let (value_hi, value_lo) = split_u32_for_lui_addi(value);
    let mut program = Vec::new();
    // x5 = large value
    program.push(RiscvInstruction::Lui { rd: 5, imm: value_hi });
    program.push(RiscvInstruction::IAlu {
        op: RiscvOpcode::Add,
        rd: 5,
        rs1: 5,
        imm: value_lo,
    });
    // x10 accumulator defaults to 0 at reset.
    match variant {
        Variant::AddFiveTimes => {
            for _ in 0..repeats {
                for _ in 0..5usize {
                    program.push(RiscvInstruction::RAlu {
                        op: RiscvOpcode::Add,
                        rd: 10,
                        rs1: 10,
                        rs2: 5,
                    });
                }
            }
        }
        Variant::MulByFive => {
            // x6 = 5
            program.push(RiscvInstruction::IAlu {
                op: RiscvOpcode::Add,
                rd: 6,
                rs1: 0,
                imm: 5,
            });
            for _ in 0..repeats {
                // x7 = x5 * 5
                program.push(RiscvInstruction::RAlu {
                    op: RiscvOpcode::Mul,
                    rd: 7,
                    rs1: 5,
                    rs2: 6,
                });
                // x10 += x7
                program.push(RiscvInstruction::RAlu {
                    op: RiscvOpcode::Add,
                    rd: 10,
                    rs1: 10,
                    rs2: 7,
                });
            }
        }
    }
    program.push(RiscvInstruction::Halt);
    program
}

fn expected_result(value: u32, repeats: usize) -> u32 {
    (value as u64).wrapping_mul(5).wrapping_mul(repeats as u64) as u32
}

fn split_u32_for_lui_addi(value: u32) -> (i32, i32) {
    // Standard RV32 materialization split: value = (hi << 12) + lo, with lo in signed 12-bit range.
    let value_i64 = value as i64;
    let hi = ((value_i64 + 0x800) >> 12) as i32;
    let lo = (value_i64 - ((hi as i64) << 12)) as i32;
    debug_assert!((-2048..=2047).contains(&lo));
    (hi, lo)
}

fn run_samples(program: &[RiscvInstruction], expected_x10: u32, warmups: usize, samples: usize) -> VariantStats {
    let program_bytes = encode_program(program);
    let max_steps = program.len();

    for _ in 0..warmups {
        let mut run = run_once(&program_bytes, max_steps, expected_x10).expect("warmup prove");
        run.verify().expect("warmup verify");
    }

    let mut prove_times = Vec::with_capacity(samples);
    let mut verify_times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut run = run_once(&program_bytes, max_steps, expected_x10).expect("prove");
        run.verify().expect("verify");
        prove_times.push(run.prove_duration());
        verify_times.push(run.verify_duration().unwrap_or(Duration::ZERO));
    }

    VariantStats {
        prove: summarize(&prove_times),
        verify: summarize(&verify_times),
        trace_len: max_steps,
    }
}

fn run_once(
    program_bytes: &[u8],
    max_steps: usize,
    expected_x10: u32,
) -> Result<Rv32TraceWiringRun, neo_fold::PiCcsError> {
    Rv32TraceWiring::from_rom(/*program_base=*/ 0, program_bytes)
        .xlen(32)
        .mode(FoldingMode::Optimized)
        .min_trace_len(max_steps)
        .chunk_rows(max_steps)
        .max_steps(max_steps)
        .shout_auto_minimal()
        .reg_output_claim(/*reg=*/ 10, F::from_u64(expected_x10 as u64))
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

fn env_u32(name: &str, default: u32) -> u32 {
    match std::env::var(name) {
        Ok(v) => {
            if let Some(hex) = v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
                u32::from_str_radix(hex, 16).unwrap_or(default)
            } else {
                v.parse::<u32>().unwrap_or(default)
            }
        }
        Err(_) => default,
    }
}
