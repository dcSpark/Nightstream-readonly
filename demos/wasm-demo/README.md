# wasm-demo

Minimal browser demo that runs `neo-fold` prove+verify inside `wasm32-unknown-unknown`.

This demo expects a circuit in the same JSON schema as `crates/neo-fold/poseidon2-tests/*.json`
(`TestExport`: R1CS A/B/C sparse matrices + per-step witnesses).

## API surface / extending from JS

The current wasm binding is intentionally minimal: it exposes a single entry point that takes
`TestExport` JSON and runs the standard `neo-fold` pipeline (R1CS → CCS → fold+prove+verify).

- wasm export: `prove_verify_test_export_json(json: string)`
- Rust runner: `neo_fold::test_export::run_test_export(&TestExport)`

If you want more control from JS (different modes, proving without verifying, exporting proof bytes,
incremental “streaming” step addition, etc.), the approach is to add more `wasm-bindgen` exports in
`demos/wasm-demo/wasm/src/lib.rs` that wrap lower-level `neo-fold` APIs (e.g. `neo_fold::session::FoldingSession`).
Most Rust types can stay internal/opaque behind a JS class, and we can expose only the knobs/results
you care about.

## Quick start

1) Build the wasm bundle (writes into `demos/wasm-demo/web/pkg/`):

```bash
./demos/wasm-demo/build_wasm.sh
```

2) Serve the static demo:

```bash
./demos/wasm-demo/serve.sh
```

Open `http://127.0.0.1:8000`.

To force a rebuild before serving:

```bash
./demos/wasm-demo/serve.sh --force-refresh
```

If the page shows a `404` for `pkg/neo_fold_demo.js`, the wasm bundle hasn’t been built yet.
Run `./demos/wasm-demo/build_wasm.sh` (or re-run `serve.sh`, which now auto-builds when missing).

## Using a real circuit export

- Use the file picker to load something like:
  - `crates/neo-fold/poseidon2-tests/poseidon2_ic_circuit_batch_1.json`
- Or paste the JSON into the textarea.

Then click `Prove + Verify`.

## Built-in examples

- `toy_square.json` (tiny sanity check)
- `toy_square_folding_8_steps.json` (same toy circuit, but 8 steps to demonstrate folding)
- `poseidon2_ic_batch_1.json` (from `crates/neo-fold/poseidon2-tests/poseidon2_ic_circuit_batch_1.json`)

## Deploy (GitHub Pages)

This repo includes a Pages workflow at `.github/workflows/wasm-demo-pages.yml`.

To enable it:

1) In GitHub: `Settings` → `Pages`
2) Set `Build and deployment` → `Source` to `GitHub Actions`

After that, pushes to `main` (or manual `workflow_dispatch`) will publish the demo site.
