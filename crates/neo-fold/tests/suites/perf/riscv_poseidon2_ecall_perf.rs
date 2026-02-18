#![allow(non_snake_case)]

use std::time::Duration;

use neo_fold::riscv_shard::Rv32B1;
use neo_memory::riscv::lookups::{
    encode_program, RiscvInstruction, RiscvOpcode, POSEIDON2_ECALL_NUM, POSEIDON2_READ_ECALL_NUM,
};

fn load_u32_imm(rd: u8, value: u32) -> Vec<RiscvInstruction> {
    let upper = ((value as i64 + 0x800) >> 12) as i32;
    let lower = (value as i32) - (upper << 12);
    vec![
        RiscvInstruction::Lui { rd, imm: upper },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd,
            rs1: rd,
            imm: lower,
        },
    ]
}

fn poseidon2_ecall_program(n_hashes: usize) -> Vec<RiscvInstruction> {
    let mut program = Vec::new();

    for _ in 0..n_hashes {
        // a1 = 0 (n_elements = 0 for empty-input Poseidon2 hash).
        program.push(RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 11,
            rs1: 0,
            imm: 0,
        });

        // a0 = POSEIDON2_ECALL_NUM -> compute ECALL.
        program.extend(load_u32_imm(10, POSEIDON2_ECALL_NUM));
        program.push(RiscvInstruction::Halt);

        // Read all 8 digest words via read ECALLs.
        for _ in 0..8 {
            program.extend(load_u32_imm(10, POSEIDON2_READ_ECALL_NUM));
            program.push(RiscvInstruction::Halt);
        }
    }

    // Clear a0 -> final halt.
    program.push(RiscvInstruction::IAlu {
        op: RiscvOpcode::Add,
        rd: 10,
        rs1: 0,
        imm: 0,
    });
    program.push(RiscvInstruction::Halt);

    program
}

fn run_once(
    program_bytes: &[u8],
    max_steps: usize,
) -> Result<neo_fold::riscv_shard::Rv32B1Run, neo_fold::PiCcsError> {
    Rv32B1::from_rom(0, program_bytes)
        .chunk_size(max_steps)
        .max_steps(max_steps)
        .prove()
}

#[derive(Clone, Copy, Debug)]
struct Stats {
    min: Duration,
    median: Duration,
    mean: Duration,
    max: Duration,
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
    Stats {
        min,
        median,
        mean,
        max,
    }
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

#[test]
#[ignore = "perf-style test: run with `P2_HASHES=1 cargo test -p neo-fold --release --test perf -- --ignored --nocapture rv32_b1_poseidon2_ecall_perf`"]
fn rv32_b1_poseidon2_ecall_perf() {
    let n_hashes = env_usize("P2_HASHES", 1);
    let warmups = env_usize("P2_WARMUPS", 1);
    let samples = env_usize("P2_SAMPLES", 5);
    assert!(n_hashes > 0, "P2_HASHES must be > 0");
    assert!(samples > 0, "P2_SAMPLES must be > 0");

    let program = poseidon2_ecall_program(n_hashes);
    let program_bytes = encode_program(&program);
    let max_steps = program.len() + 64;

    for _ in 0..warmups {
        let mut run = run_once(&program_bytes, max_steps).expect("warmup prove");
        run.verify().expect("warmup verify");
    }

    let mut prove_times = Vec::with_capacity(samples);
    let mut verify_times = Vec::with_capacity(samples);
    let mut end_to_end_times = Vec::with_capacity(samples);
    let mut ccs_n = 0;
    let mut ccs_m = 0;
    let mut fold_count = 0;
    let mut trace_len = 0;

    for _ in 0..samples {
        let total_start = std::time::Instant::now();
        let mut run = run_once(&program_bytes, max_steps).expect("prove");
        run.verify().expect("verify");
        end_to_end_times.push(total_start.elapsed());
        prove_times.push(run.prove_duration());
        verify_times.push(run.verify_duration().unwrap_or(Duration::ZERO));

        ccs_n = run.ccs_num_constraints();
        ccs_m = run.ccs_num_variables();
        fold_count = run.fold_count();
        trace_len = run.riscv_trace_len().unwrap_or(0);
    }

    let prove = summarize(&prove_times);
    let verify = summarize(&verify_times);
    let e2e = summarize(&end_to_end_times);

    let sep = "=".repeat(96);
    let thin = "-".repeat(96);

    println!();
    println!("{sep}");
    println!("POSEIDON2 ECALL BENCHMARK (RV32 B1)");
    println!("{sep}");
    println!(
        "config: hashes={} instructions={} warmups={} samples={}",
        n_hashes,
        program.len(),
        warmups,
        samples
    );
    println!(
        "CCS: n={} (pow2={}) m={} (pow2={}) folds={} trace_len={}",
        ccs_n,
        ccs_n.next_power_of_two(),
        ccs_m,
        ccs_m.next_power_of_two(),
        fold_count,
        trace_len
    );
    println!("{thin}");
    println!(
        "{:>12}  {:>10} {:>10} {:>10} {:>10}",
        "phase", "min", "median", "mean", "max"
    );
    println!("{thin}");
    println!(
        "{:>12}  {:>10} {:>10} {:>10} {:>10}",
        "prove",
        fmt_duration(prove.min),
        fmt_duration(prove.median),
        fmt_duration(prove.mean),
        fmt_duration(prove.max),
    );
    println!(
        "{:>12}  {:>10} {:>10} {:>10} {:>10}",
        "verify",
        fmt_duration(verify.min),
        fmt_duration(verify.median),
        fmt_duration(verify.mean),
        fmt_duration(verify.max),
    );
    println!(
        "{:>12}  {:>10} {:>10} {:>10} {:>10}",
        "end-to-end",
        fmt_duration(e2e.min),
        fmt_duration(e2e.median),
        fmt_duration(e2e.mean),
        fmt_duration(e2e.max),
    );
    println!("{thin}");
    if n_hashes > 0 {
        let per_hash_prove = Duration::from_secs_f64(prove.median.as_secs_f64() / n_hashes as f64);
        let per_hash_e2e = Duration::from_secs_f64(e2e.median.as_secs_f64() / n_hashes as f64);
        println!(
            "per-hash (median): prove={} end-to-end={}",
            fmt_duration(per_hash_prove),
            fmt_duration(per_hash_e2e),
        );
    }
    println!("{sep}");
    println!();
}

#[test]
#[ignore = "perf-style test: run with `cargo test -p neo-fold --release --test perf -- --ignored --nocapture rv32_b1_poseidon2_ecall_scaling`"]
fn rv32_b1_poseidon2_ecall_scaling() {
    let samples = env_usize("P2_SAMPLES", 3);
    let warmups = env_usize("P2_WARMUPS", 1);
    let hash_counts = [1, 2, 4, 8];

    let sep = "=".repeat(110);
    let thin = "-".repeat(110);

    println!();
    println!("{sep}");
    println!("POSEIDON2 ECALL SCALING (RV32 B1) — warmups={} samples={}", warmups, samples);
    println!("{sep}");
    println!(
        "{:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "hashes", "instrs", "ccs_n", "ccs_m", "prove_med", "verify_med", "e2e_med", "per_hash", "throughput"
    );
    println!("{thin}");

    for &n_hashes in &hash_counts {
        let program = poseidon2_ecall_program(n_hashes);
        let program_bytes = encode_program(&program);
        let max_steps = program.len() + 64;

        for _ in 0..warmups {
            let mut run = run_once(&program_bytes, max_steps).expect("warmup prove");
            run.verify().expect("warmup verify");
        }

        let mut prove_times = Vec::with_capacity(samples);
        let mut verify_times = Vec::with_capacity(samples);
        let mut e2e_times = Vec::with_capacity(samples);
        let mut ccs_n = 0;
        let mut ccs_m = 0;

        for _ in 0..samples {
            let total_start = std::time::Instant::now();
            let mut run = run_once(&program_bytes, max_steps).expect("prove");
            run.verify().expect("verify");
            e2e_times.push(total_start.elapsed());
            prove_times.push(run.prove_duration());
            verify_times.push(run.verify_duration().unwrap_or(Duration::ZERO));
            ccs_n = run.ccs_num_constraints();
            ccs_m = run.ccs_num_variables();
        }

        let prove_med = summarize(&prove_times).median;
        let verify_med = summarize(&verify_times).median;
        let e2e_med = summarize(&e2e_times).median;
        let per_hash = Duration::from_secs_f64(e2e_med.as_secs_f64() / n_hashes as f64);
        let hashes_per_sec = n_hashes as f64 / e2e_med.as_secs_f64();

        println!(
            "{:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10.1} h/s",
            n_hashes,
            program.len(),
            ccs_n,
            ccs_m,
            fmt_duration(prove_med),
            fmt_duration(verify_med),
            fmt_duration(e2e_med),
            fmt_duration(per_hash),
            hashes_per_sec,
        );
    }

    println!("{sep}");
    println!();
}
