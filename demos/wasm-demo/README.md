# neo-fold wasm demo

Minimal browser demo that runs `neo-fold` prove+verify inside `wasm32-unknown-unknown`.

If you want a native iOS build (Swift/Xcode) alongside wasm, see `docs/ios-native.md`.

This demo supports two input modes:

- **TestExport JSON** (same schema as `crates/neo-fold/poseidon2-tests/*.json`):
  R1CS A/B/C sparse matrices + per-step witnesses.
- **RV32 Fibonacci** (pasteable “mini-asm” / words): assembles a small RV32 program, generates a trace,
  and proves it under the RV32 B1 shared-bus step circuit.

## API surface / extending from JS

The current wasm binding is intentionally minimal: it exposes a single entry point that takes
`TestExport` JSON and runs the standard `neo-fold` pipeline (R1CS → CCS → fold+prove+verify).

- wasm export: `prove_verify_test_export_json(json: string)`
- Rust runner: `neo_fold::test_export::run_test_export(&TestExport)`

For more control, the demo also exports a stateful API:

- `new NeoFoldSession(circuitJson)`
- `session.add_step_witness_json(stepWitnessJson)`
- `session.add_steps_from_test_export_json(testExportJson)`
- `proof = session.fold_and_prove()`
- `ok = session.verify(proof)`
- `spartan = session.spartan_prove(proof)` (optional)
- `ok = session.spartan_verify(spartan)` (optional)

This keeps proofs as an opaque JS handle (`NeoFoldProof`) and exposes structured summaries/timings.
See `demos/wasm-demo/wasm/src/lib.rs`.

## Spartan2 “compression” (experimental)

The UI includes an optional checkbox to compress the folding proof into a Spartan2 proof
(Goldilocks + Merkle-MLE backend) and verify it in the browser.

This is WIP and currently intended for demo/profiling only (expect higher wasm size + longer runs).

When enabled, the UI lets you download the Spartan2 SNARK bytes (without bundling the VK).

## UI responsiveness

Proving/verifying runs in a Web Worker so the UI stays responsive while proofs are generated.

## Quick start

1) Build the wasm bundles (default: both, writes into `demos/wasm-demo/web/pkg/` and `demos/wasm-demo/web/pkg_threads/`):

```bash
./demos/wasm-demo/build_wasm.sh
```

To build only a single-thread bundle (writes into `demos/wasm-demo/web/pkg/`):

```bash
./demos/wasm-demo/build_wasm.sh --no-threads
```

To build only the threads bundle (writes into `demos/wasm-demo/web/pkg_threads/`):

```bash
./demos/wasm-demo/build_wasm.sh --threads
```

2) Serve the static demo:

```bash
./demos/wasm-demo/serve.sh
```

Open `http://127.0.0.1:8000`.

## wasm threads (Rayon + SharedArrayBuffer)

Threaded wasm requires:

- A cross-origin isolated page (COOP/COEP) so the browser enables `SharedArrayBuffer`
- `wasm32` atomics + a stdlib built with atomics, so the build uses nightly + `-Z build-std`

To build a threaded wasm bundle (atomics enabled) into `demos/wasm-demo/web/pkg_threads/`:

```bash
./demos/wasm-demo/build_wasm.sh --threads
```

If this is your first time building threads, install the prerequisites:

```bash
rustup toolchain install nightly
rustup target add wasm32-unknown-unknown --toolchain nightly
rustup component add rust-src --toolchain nightly
```

(You can override the toolchain with `WASM_THREADS_TOOLCHAIN=nightly-YYYY-MM-DD`.)

Serve it with COOP/COEP headers (required for `SharedArrayBuffer`):

```bash
./demos/wasm-demo/serve.sh
```

Then open `http://127.0.0.1:8000`.

The page will automatically use the threaded bundle when supported (cross-origin isolated).
You can override:

- Force single-thread: `?threads=0`
- Force threads: `?threads=1`
- Set thread count (optional): `?threads=1&nthreads=4`

To force a rebuild before serving:

```bash
./demos/wasm-demo/serve.sh --force-refresh
```

If the page shows a `404` for `pkg/neo_fold_demo.js`, the wasm bundle hasn’t been built yet.
Run `./demos/wasm-demo/build_wasm.sh` (or `./demos/wasm-demo/build_wasm.sh --no-threads`), or re-run `serve.sh` (which now auto-builds when missing).

## Using a real circuit export

- Use the file picker to load something like:
  - `crates/neo-fold/poseidon2-tests/poseidon2_ic_circuit_batch_1.json`
- Or paste the JSON into the textarea.

Then click `Prove + Verify`.

## Built-in examples

- `toy_square.json` (tiny sanity check)
- `toy_square_folding_8_steps.json` (same toy circuit, but 8 steps to demonstrate folding)
- `poseidon2_ic_batch_1.json` (from `crates/neo-fold/poseidon2-tests/poseidon2_ic_circuit_batch_1.json`)
- `rv32_fibonacci.asm` (RV32 “mini-asm” Fibonacci program; reads `n` from RAM[0x104], writes fib(n) to RAM[0x100])

## Deploy (Cloudflare Workers)

Cloudflare Workers supports serving static assets with custom headers. This repo includes a
Wrangler config at `demos/wasm-demo/wrangler.toml` that serves `demos/wasm-demo/web/` and sets
COOP/COEP headers so `?threads=1` can work (when the threads bundle is built).

Deploy with Wrangler:

```bash
cd demos/wasm-demo
npx wrangler@latest deploy
```

`wrangler deploy` runs `./build_wasm.sh` (configured in `wrangler.toml`), so your deploy environment
needs Rust (`wasm32-unknown-unknown`) + `wasm-pack`. By default this builds both bundles (including threads),
so you also need a nightly toolchain + `rust-src` (see above). You can also run
`./demos/wasm-demo/build_wasm.sh` manually ahead of time.

In the Cloudflare dashboard (Workers → Create → Import a repository), point the import at the
`demos/wasm-demo` subdirectory so it finds `wrangler.toml`.
