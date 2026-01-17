# rv32-fibonacci

Tiny no-std RV32IM guest that computes Fibonacci in Rust and writes the result to RAM.

The guest reads `n` from RAM at `0x104` and writes `fib(n)` to RAM at `0x100`.

## Build

```bash
cargo build --release
```

## Export ROM bytes

```bash
./export_rom_rs.py
```

This writes `crates/neo-fold/riscv-tests/binaries/rv32_fibonacci_rom.rs` in the repo.
