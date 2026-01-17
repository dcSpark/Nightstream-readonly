//! End-to-end prove+verify for a small *compiled* RV32 guest program under the B1 shared-bus step circuit.
//!
//! The guest is authored in Rust under `riscv-tests/guests/rv32-fibonacci/` and its ROM bytes are committed in
//! `riscv-tests/binaries/rv32_fibonacci_rom.rs` so this test doesn't need to cross-compile at runtime.

#![allow(non_snake_case)]

#[path = "binaries/rv32_fibonacci_rom.rs"]
mod rv32_fibonacci_rom;

use neo_fold::riscv_shard::Rv32B1;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use sha2::{Digest, Sha256};

fn hex32(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

fn sha256_field_slice(values: &[F]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for v in values {
        hasher.update(v.as_canonical_u64().to_le_bytes());
    }
    hasher.finalize().into()
}

fn sha256_fields_concat(x: &[F], w: &[F]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for v in x {
        hasher.update(v.as_canonical_u64().to_le_bytes());
    }
    for v in w {
        hasher.update(v.as_canonical_u64().to_le_bytes());
    }
    hasher.finalize().into()
}

fn nonzero_concat(x: &[F], w: &[F]) -> usize {
    x.iter()
        .chain(w.iter())
        .filter(|v| **v != F::ZERO)
        .count()
}

fn preview_first_last(values: &[F], n: usize) -> (Vec<u64>, Vec<u64>) {
    let n = n.min(values.len());
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let first = values
        .iter()
        .take(n)
        .map(|v| v.as_canonical_u64())
        .collect::<Vec<_>>();
    let last = values
        .iter()
        .skip(values.len().saturating_sub(n))
        .map(|v| v.as_canonical_u64())
        .collect::<Vec<_>>();
    (first, last)
}

fn preview_first_last_concat(x: &[F], w: &[F], n: usize) -> (Vec<u64>, Vec<u64>) {
    let z_len = x.len() + w.len();
    let n = n.min(z_len);
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut first = Vec::with_capacity(n);
    for v in x.iter().chain(w.iter()).take(n) {
        first.push(v.as_canonical_u64());
    }

    let last = if w.len() >= n {
        w.iter()
            .skip(w.len() - n)
            .map(|v| v.as_canonical_u64())
            .collect::<Vec<_>>()
    } else {
        let need_from_x = n - w.len();
        x.iter()
            .skip(x.len().saturating_sub(need_from_x))
            .chain(w.iter())
            .map(|v| v.as_canonical_u64())
            .collect::<Vec<_>>()
    };

    (first, last)
}

#[test]
fn test_riscv_fibonacci_compiled_full_prove_verify() {
    // The guest reads n from RAM[0x104], computes fib(n), and writes the result to RAM[0x100].
    let n = 10u32;
    let expected = F::from_u64(55);

    let program_base = rv32_fibonacci_rom::RV32_FIBONACCI_ROM_BASE;
    let program_bytes: &[u8] = &rv32_fibonacci_rom::RV32_FIBONACCI_ROM;

    println!(
        "RV32 ELF .neo_start size: {} bytes ({} instructions)",
        program_bytes.len(),
        program_bytes.len() / 4
    );

    let mut run = Rv32B1::from_rom(program_base, program_bytes)
        .xlen(32)
        .ram_bytes(0x800)
        .ram_init_u32(/*addr=*/ 0x104, n)
        .chunk_size(16)
        .max_steps(512)
        .shout_auto_minimal()
        .output(/*output_addr=*/ 0x100, /*expected_output=*/ expected)
        .prove()
        .expect("prove");

    println!("RV32 executed steps (trace len): {}", run.riscv_trace_len().expect("trace len"));
    println!(
        "Circuit size (CCS): n_constraints={} m_variables={}",
        run.ccs_num_constraints(),
        run.ccs_num_variables()
    );
    println!(
        "Shout lookups used: {}",
        run.shout_lookup_count().expect("shout lookup count")
    );
    println!("Folds: {}", run.fold_count());

    let preview_len: usize = std::env::var("NIGHTSTREAM_WITNESS_PREVIEW_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let print_full = std::env::var("NIGHTSTREAM_PRINT_WITNESS_FULL").is_ok();

    let steps_witness = run.steps_witness();
    println!(
        "Step witness bundles: {} (expected folds={})",
        steps_witness.len(),
        run.fold_count()
    );

    for (fold_idx, step) in steps_witness.iter().enumerate() {
        let mcs_inst = &step.mcs.0;
        let mcs_wit = &step.mcs.1;
        let x = &mcs_inst.x;
        let w = &mcs_wit.w;
        let z_len = x.len() + w.len();

        let z_debug_sha256 = hex32(sha256_fields_concat(x, w));
        let w_debug_sha256 = hex32(sha256_field_slice(w));
        let Z_debug_sha256 = hex32(sha256_field_slice(mcs_wit.Z.as_slice()));
        let z_nonzero = nonzero_concat(x, w);

        let (z_first, z_last) = preview_first_last_concat(x, w, preview_len);
        let (Z_first, Z_last) = preview_first_last(mcs_wit.Z.as_slice(), preview_len);

        println!(
            "Fold {fold_idx}: m_in={} x_len={} w_len={} z_len={} z_nonzero={} lut_instances={} mem_instances={} z_debug_sha256={} w_debug_sha256={} Z_debug_sha256={}",
            mcs_inst.m_in,
            x.len(),
            w.len(),
            z_len,
            z_nonzero,
            step.lut_instances.len(),
            step.mem_instances.len(),
            z_debug_sha256,
            w_debug_sha256,
            Z_debug_sha256,
        );
        println!("  z_first={z_first:?}");
        println!("  z_last ={z_last:?}");
        println!("  Z_first={Z_first:?} (Z is {}x{})", mcs_wit.Z.rows(), mcs_wit.Z.cols());
        println!("  Z_last ={Z_last:?}");

        if print_full {
            println!("  x_full={:?}", x.iter().map(|v| v.as_canonical_u64()).collect::<Vec<_>>());
            println!("  w_full={:?}", w.iter().map(|v| v.as_canonical_u64()).collect::<Vec<_>>());
        }
    }

    println!("Prove duration: {:?}", run.prove_duration());
    run.verify().expect("verify");
    println!("Verify duration: {:?}", run.verify_duration().expect("verify duration"));

    assert!(
        matches!(
            run.verify_output_claim(/*output_addr=*/ 0x100, /*expected_output=*/ F::from_u64(56)),
            Ok(false) | Err(_)
        ),
        "wrong output claim must not verify"
    );
}
