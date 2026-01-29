#![allow(non_snake_case)]

mod riscv_wasm_demo;

use neo_fold::riscv_shard::Rv32B1;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) => v.trim().parse::<usize>().unwrap_or(default),
        Err(_) => default,
    }
}

fn env_u32(name: &str, default: u32) -> u32 {
    match std::env::var(name) {
        Ok(v) => v.trim().parse::<u32>().unwrap_or(default),
        Err(_) => default,
    }
}

/// Native reproduction of the wasm demo's "RV32 Fibonacci (mini-asm)" circuit.
///
/// Run with:
/// - `cargo test --workspace --release -p neo-fold --test test_riscv_wasm_demo_memory -- --ignored --nocapture`
/// - `./scripts/profile_memory_deep.sh neo-fold test_riscv_wasm_demo_memory test_rv32_fibonacci_mini_asm_peak_rss --ignored`
#[test]
#[ignore]
fn test_rv32_fibonacci_mini_asm_peak_rss() {
    let n = env_u32("NEO_RV32_N", 5);
    let ram_bytes = env_usize("NEO_RV32_RAM_BYTES", 2048);
    let chunk_size = env_usize("NEO_RV32_CHUNK_SIZE", 128);
    let max_steps = env_usize("NEO_RV32_MAX_STEPS", 0);

    let asm = include_str!("riscv_wasm_demo/rv32_fibonacci.asm");
    let program_bytes = riscv_wasm_demo::mini_asm::assemble_rv32_mini_asm(asm).expect("assemble");

    let expected = riscv_wasm_demo::fib_u32(n);
    let expected_f = F::from_u64(expected as u64);

    let rss0 = riscv_wasm_demo::max_rss_bytes();
    println!(
        "rv32-fib: start rss_peak={} pid={}",
        rss0.map(riscv_wasm_demo::fmt_bytes).unwrap_or_else(|| "?".into()),
        std::process::id()
    );

    let mut run = {
        let mut b = Rv32B1::from_rom(/*program_base=*/ 0, &program_bytes)
            .xlen(32)
            .ram_bytes(ram_bytes)
            .ram_init_u32(/*addr=*/ 0x104, n)
            .chunk_size(chunk_size)
            .shout_auto_minimal()
            .output(/*output_addr=*/ 0x100, /*expected_output=*/ expected_f);
        if max_steps > 0 {
            b = b.max_steps(max_steps);
        }
        b.prove().expect("prove")
    };

    let rss1 = riscv_wasm_demo::max_rss_bytes();
    println!(
        "rv32-fib: after prove rss_peak={} prove_ms={:.1}",
        rss1.map(riscv_wasm_demo::fmt_bytes).unwrap_or_else(|| "?".into()),
        run.prove_duration().as_secs_f64() * 1000.0
    );

    run.verify().expect("verify");

    let rss2 = riscv_wasm_demo::max_rss_bytes();
    println!(
        "rv32-fib: after verify rss_peak={} verify_ms={:.1}",
        rss2.map(riscv_wasm_demo::fmt_bytes).unwrap_or_else(|| "?".into()),
        run.verify_duration().map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0)
    );

    let trace_len = run.riscv_trace_len().ok();
    println!(
        "rv32-fib: n={n} expected={expected} trace_len={:?} folds={} chunk_size={} ram_bytes={}",
        trace_len,
        run.fold_count(),
        chunk_size,
        ram_bytes
    );
    println!(
        "rv32-fib: ccs_constraints={} ccs_variables={} shout_lookups={:?}",
        run.ccs_num_constraints(),
        run.ccs_num_variables(),
        run.shout_lookup_count().ok()
    );

    assert!(run.verify_default_output_claim().expect("verify output claim"));
}

