# `circuit-l2-transfer` guest

RV32IM guest scaffold for an L2 transfer circuit-like workload.

Notes:
- The business logic mirrors the requested program shape.
- Poseidon2 here is a temporary in-guest placeholder mixer so the binary can be built today.
- Once the Poseidon2 precompile path is available in `nightstream-sdk` and VM execution, replace
  the placeholder with the real precompile call and un-ignore the runtime prove/verify test.

## Regenerating the committed ROM bytes

From this directory:

```bash
python3 export_rom_rs.py
```

This updates `crates/neo-fold/riscv-tests/binaries/circuit_l2_transfer_rom.rs`.

Prereq: `rustup target add riscv32im-unknown-none-elf`.
