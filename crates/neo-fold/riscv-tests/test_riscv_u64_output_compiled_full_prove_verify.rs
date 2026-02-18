//! End-to-end prove+verify for a small *compiled* RV32 guest program under the trace wiring circuit.
//!
//! The guest is authored in Rust under `riscv-tests/guests/rv32-u64-output/` and its ROM bytes are committed in
//! `riscv-tests/binaries/rv32_u64_output_rom.rs` so this test doesn't need to cross-compile at runtime.

#![allow(non_snake_case)]

#[path = "binaries/rv32_u64_output_rom.rs"]
mod rv32_u64_output_rom;

use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_riscv_u64_output_compiled_full_prove_verify() {
    let output = 0x1122_3344_5566_7788u64;
    let out_lo = F::from_u64(output as u32 as u64);
    let out_hi = F::from_u64((output >> 32) as u32 as u64);

    let program_base = rv32_u64_output_rom::RV32_U64_OUTPUT_ROM_BASE;
    let program_bytes: &[u8] = &rv32_u64_output_rom::RV32_U64_OUTPUT_ROM;

    let mut run = Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .chunk_rows(4)
        .shout_auto_minimal()
        .output_claim(/*addr=*/ 0x100, /*value=*/ out_lo)
        .output_claim(/*addr=*/ 0x104, /*value=*/ out_hi)
        .prove()
        .expect("prove");

    println!("Prove duration: {:?}", run.prove_duration());
    run.verify().expect("verify");
    println!("Verify duration: {:?}", run.verify_duration().expect("verify duration"));

    match Rv32TraceWiring::from_rom(program_base, program_bytes)
        .xlen(32)
        .chunk_rows(4)
        .shout_auto_minimal()
        .output_claim(/*addr=*/ 0x100, /*value=*/ out_lo)
        .output_claim(/*addr=*/ 0x104, /*value=*/ F::from_u64(0))
        .prove()
    {
        Ok(mut bad_run) => assert!(bad_run.verify().is_err(), "wrong output claims must fail verification"),
        Err(_) => {}
    }
}
