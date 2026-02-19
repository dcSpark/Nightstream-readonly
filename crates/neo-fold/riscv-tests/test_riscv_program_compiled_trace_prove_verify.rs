//! End-to-end prove+verify for a compiled RV32 smoke-test guest program under the trace-mode runner.
//!
//! The guest is authored in Rust under `riscv-tests/guests/rv32-smoke/` and its ROM bytes are committed in
//! `riscv-tests/binaries/rv32_smoke_rom.rs` so this test doesn't need to cross-compile at runtime.

#[path = "binaries/rv32_smoke_rom.rs"]
mod rv32_smoke_rom;

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_riscv_program_compiled_trace_prove_verify() {
    let program_base = rv32_smoke_rom::RV32_SMOKE_ROM_BASE;
    let program_bytes: &[u8] = &rv32_smoke_rom::RV32_SMOKE_ROM;

    let mut run = Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .shout_auto_minimal()
        .output(
            /*output_addr=*/ 0x100,
            /*expected_output=*/ F::from_u64(0x100c),
        )
        .prove()
        .expect("trace-mode prove compiled smoke");

    run.verify().expect("trace-mode verify compiled smoke");

    // Wrong output must fail.
    let wrong_run = Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .shout_auto_minimal()
        .output(
            /*output_addr=*/ 0x100,
            /*expected_output=*/ F::from_u64(0x100d),
        )
        .prove();

    match wrong_run {
        Ok(mut run_bad) => {
            assert!(
                run_bad.verify().is_err(),
                "wrong output claim must not verify"
            );
        }
        Err(_) => {}
    }
}
