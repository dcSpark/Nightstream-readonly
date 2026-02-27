# `circuit-l2-transfer` guest

RV32IM guest implementing the **Midnight privacy module note-spend circuit**
(join-split spend verifier) ported from the Sovereign-Ligero adapter.

The circuit proves:
1. **Ownership** — all inputs belong to the same spend key
2. **Merkle membership** — each input commitment is in the state tree
3. **Nullifiers** — correctly derived, match public values, pairwise distinct
4. **Shape validation** — n_out / withdraw_amount / withdraw_to consistency
5. **Output commitments** — correctly formed
6. **Balance** — `sum(inputs) == withdraw_amount + sum(outputs)`
7. **Enforce product** — values non-zero, rhos distinct
8. **Blacklist non-membership** — sender (and pay recipient) not in deny-map
9. **Viewer attestations (Level-B)** — FVK commitment, ct_hash, MAC binding

Poseidon2 hashing uses the `nightstream-sdk` precompile ECALL path.

## Regenerating the committed ROM bytes

From this directory:

```bash
python3 export_rom_rs.py
```

This updates `crates/neo-fold/riscv-tests/binaries/circuit_l2_transfer_rom.rs`.

Prereq: `rustup target add riscv32im-unknown-none-elf`.

## Running the real transfer test

```bash
cargo test -p neo-fold --release --features poseidon-precompile \
  --test test_riscv_circuit_l2_transfer_compiled_trace_prove_verify \
  -- --ignored --nocapture test_note_spend_1in_1out_transfer_prove_verify
```
