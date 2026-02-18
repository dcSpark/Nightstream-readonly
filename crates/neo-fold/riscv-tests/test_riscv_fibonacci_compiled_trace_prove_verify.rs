//! End-to-end prove+verify for a compiled RV32 Fibonacci guest program under the trace-mode runner.
//!
//! The guest is authored in Rust under `riscv-tests/guests/rv32-fibonacci/` and its ROM bytes are committed in
//! `riscv-tests/binaries/rv32_fibonacci_rom.rs` so this test doesn't need to cross-compile at runtime.

#[path = "binaries/rv32_fibonacci_rom.rs"]
mod rv32_fibonacci_rom;

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_riscv_fibonacci_compiled_trace_prove_verify() {
    // The guest reads n from RAM[0x104], computes fib(n), and writes the result to RAM[0x100].
    let n = 10u32;
    let expected = F::from_u64(55);

    let program_base = rv32_fibonacci_rom::RV32_FIBONACCI_ROM_BASE;
    let program_bytes: &[u8] = &rv32_fibonacci_rom::RV32_FIBONACCI_ROM;

    let mut run = Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .max_steps(64)
        .ram_init_u32(/*addr=*/ 0x104, n)
        .shout_auto_minimal()
        .output(/*output_addr=*/ 0x100, /*expected_output=*/ expected)
        .prove()
        .expect("trace-mode prove fibonacci");

    run.verify().expect("trace-mode verify fibonacci");

    // Wrong output must fail: prove with wrong expected value should fail at verify.
    let wrong_run = Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .max_steps(64)
        .ram_init_u32(/*addr=*/ 0x104, n)
        .shout_auto_minimal()
        .output(/*output_addr=*/ 0x100, /*expected_output=*/ F::from_u64(56))
        .prove();

    match wrong_run {
        Ok(mut run_bad) => {
            assert!(
                run_bad.verify().is_err(),
                "wrong output claim must not verify"
            );
        }
        Err(_) => {
            // Prove itself failed, which is also acceptable.
        }
    }
}
