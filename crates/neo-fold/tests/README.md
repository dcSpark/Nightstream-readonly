# neo-fold integration tests

This directory is organized by suite to make ownership and intent explicit.

## Top-level test crates

Each file at this level is a thin entrypoint that mounts one suite module:

- `trace_shout.rs`
- `trace_twist.rs`
- `shared_bus.rs`
- `session.rs`
- `rv32m.rs`
- `integration.rs`
- `perf.rs`
- `vm.rs`
- `redteam.rs`
- `redteam_riscv.rs`
- `regression.rs`

## Suite layout

- `suites/trace_shout/`: shout-sidecar e2e + red-team coverage.
- `suites/trace_twist/`: twist-sidecar e2e + linkage hardening tests.
- `suites/shared_bus/`: shared CPU bus coverage and attacks.
- `suites/session/`: folding session/unit behavior.
- `suites/rv32m/`: RV32M-specific tests.
- `suites/integration/`: end-to-end/proving pipeline integration.
- `suites/perf/`: perf and scaling tests (typically `#[ignore]`).
- `suites/vm/`: VM extraction and wasm-demo tests.
- `suites/redteam/`: cross-cutting adversarial tests.
- `suites/redteam_riscv/`: RISC-V-specific adversarial tests.
- `suites/regression/`: regression tests.

## Shared helpers

- `common/fixtures.rs`
- `common/fib_twist_shout_vm.rs`
- `common/riscv_shout_event_table_packed.rs`
- `common/setup.rs`

## Suggested run commands

- `cargo test -p neo-fold --test trace_shout`
- `cargo test -p neo-fold --test trace_twist`
- `cargo test -p neo-fold --test shared_bus`
- `cargo test -p neo-fold --test session`
- `cargo test -p neo-fold --test integration`
- `cargo test -p neo-fold --test vm`
- `cargo test -p neo-fold --tests --no-run`
