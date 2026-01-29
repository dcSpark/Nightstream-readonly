#!/usr/bin/env node

import { performance } from "node:perf_hooks";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(SCRIPT_DIR, "..");

const DEFAULT_EXAMPLE = "toy_square_folding_8_steps";
const DEFAULT_RUNS = 3;
const DEFAULT_WARMUP = 1;

function printHelp() {
  console.log(`Usage: node scripts/bench_wasm_node.mjs [options]

Runs the same WKWebView (web/pkg) wasm bundle on your machine and measures prove+verify timings.

Options:
  --example <name>   Example from TestingWasm/Resources/examples (default: ${DEFAULT_EXAMPLE})
  --input <path>     Path to a TestExport JSON file (overrides --example)
  --runs <n>         Measured runs (default: ${DEFAULT_RUNS})
  --warmup <n>       Warmup runs (default: ${DEFAULT_WARMUP})
  --spartan          Also run Spartan2 compress+verify (if exported)
  --json             Emit machine-readable JSON
  --help             Show this help

Notes:
  - This script uses the single-thread bundle (web/pkg). To benchmark wasm-threads,
    run in a browser with COOP/COEP + SharedArrayBuffer enabled.
`);
}

function parseIntArg(value, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.floor(n));
}

function parseArgs(argv) {
  const out = {
    example: DEFAULT_EXAMPLE,
    inputPath: null,
    runs: DEFAULT_RUNS,
    warmup: DEFAULT_WARMUP,
    spartan: false,
    json: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    switch (a) {
      case "--example":
        out.example = String(argv[++i] ?? "");
        break;
      case "--input":
        out.inputPath = String(argv[++i] ?? "");
        break;
      case "--runs":
        out.runs = parseIntArg(argv[++i], DEFAULT_RUNS);
        break;
      case "--warmup":
        out.warmup = parseIntArg(argv[++i], DEFAULT_WARMUP);
        break;
      case "--spartan":
        out.spartan = true;
        break;
      case "--json":
        out.json = true;
        break;
      case "--help":
        printHelp();
        process.exit(0);
      default:
        console.error(`Unknown option: ${a}`);
        printHelp();
        process.exit(1);
    }
  }

  return out;
}

async function fileExists(p) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function avg(values) {
  if (!Array.isArray(values) || values.length === 0) return null;
  const sum = values.reduce((acc, v) => acc + v, 0);
  return sum / values.length;
}

function min(values) {
  if (!Array.isArray(values) || values.length === 0) return null;
  return values.reduce((acc, v) => Math.min(acc, v), Infinity);
}

function max(values) {
  if (!Array.isArray(values) || values.length === 0) return null;
  return values.reduce((acc, v) => Math.max(acc, v), 0);
}

function pickMetrics(runs, key) {
  return runs
    .map((r) => r?.timings_ms?.[key])
    .filter((v) => typeof v === "number" && Number.isFinite(v));
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));

  const examplesDir = path.join(PROJECT_DIR, "TestingWasm", "Resources", "examples");
  const inputPath = opts.inputPath
    ? path.resolve(process.cwd(), opts.inputPath)
    : path.join(examplesDir, `${opts.example}.json`);

  if (!(await fileExists(inputPath))) {
    throw new Error(`Input JSON not found: ${inputPath}`);
  }

  const wasmDir = path.join(PROJECT_DIR, "web", "pkg");
  const wasmJsPath = path.join(wasmDir, "neo_fold_demo.js");
  const wasmWasmPath = path.join(wasmDir, "neo_fold_demo_bg.wasm");

  if (!(await fileExists(wasmJsPath)) || !(await fileExists(wasmWasmPath))) {
    throw new Error(
      `Missing wasm bundle in ${wasmDir}.\nBuild it first:\n  ./scripts/build_wasm.sh --release\n(or add --threads for the iOS WKWebView threads bundle)`,
    );
  }

  const jsonText = await fs.readFile(inputPath, "utf8");
  const parsed = safeJsonParse(jsonText);

  const wasmModule = await import(pathToFileURL(wasmJsPath).href);
  const wasmBytes = await fs.readFile(wasmWasmPath);

  const initStart = performance.now();
  await wasmModule.default({ module_or_path: wasmBytes });
  const wasmInitMs = performance.now() - initStart;
  if (typeof wasmModule.init_panic_hook === "function") wasmModule.init_panic_hook();

  async function runOnce() {
    const totalStart = performance.now();

    const createStart = performance.now();
    const session = new wasmModule.NeoFoldSession(jsonText);
    const sessionCreateMs = performance.now() - createStart;

    const addStart = performance.now();
    session.add_steps_from_test_export_json(jsonText);
    const addStepsMs = performance.now() - addStart;

    const proveStart = performance.now();
    const proof = session.fold_and_prove();
    const proveMs = performance.now() - proveStart;

    const verifyStart = performance.now();
    const verifyOk = session.verify(proof);
    const verifyMs = performance.now() - verifyStart;

    const steps = typeof proof.step_count === "function" ? proof.step_count() : 0;

    let spartan;
    if (opts.spartan && typeof session.spartan_prove === "function") {
      const spStart = performance.now();
      const sp = session.spartan_prove(proof);
      const spartanProveMs = performance.now() - spStart;

      const bytes = typeof sp.bytes === "function" ? sp.bytes() : null;
      const snarkBytes = bytes?.length;

      const spVerifyStart = performance.now();
      const spartanVerifyOk = session.spartan_verify(sp);
      const spartanVerifyMs = performance.now() - spVerifyStart;

      spartan = {
        prove_ms: spartanProveMs,
        verify_ms: spartanVerifyMs,
        verify_ok: spartanVerifyOk,
        snark_bytes: typeof snarkBytes === "number" ? snarkBytes : null,
      };

      if (typeof sp.free === "function") sp.free();
    }

    if (typeof proof.free === "function") proof.free();
    if (typeof session.free === "function") session.free();

    const totalMs = performance.now() - totalStart;

    return {
      verify_ok: verifyOk,
      steps,
      timings_ms: {
        session_create: sessionCreateMs,
        add_steps_total: addStepsMs,
        fold_and_prove: proveMs,
        verify: verifyMs,
        total: totalMs,
      },
      spartan,
    };
  }

  for (let i = 0; i < opts.warmup; i++) {
    await runOnce();
  }

  const runs = [];
  for (let i = 0; i < opts.runs; i++) {
    runs.push(await runOnce());
  }

  const summary = {
    runs: runs.length,
    wasm_init_ms: wasmInitMs,
    avg_ms: {
      session_create: avg(pickMetrics(runs, "session_create")),
      add_steps_total: avg(pickMetrics(runs, "add_steps_total")),
      fold_and_prove: avg(pickMetrics(runs, "fold_and_prove")),
      verify: avg(pickMetrics(runs, "verify")),
      total: avg(pickMetrics(runs, "total")),
    },
    min_ms: {
      total: min(pickMetrics(runs, "total")),
    },
    max_ms: {
      total: max(pickMetrics(runs, "total")),
    },
  };

  const out = {
    env: {
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      cpu: os.cpus()?.[0]?.model ?? null,
      cpu_count: os.cpus()?.length ?? null,
    },
    wasm: {
      bundle: "web/pkg",
      wasm_init_ms: wasmInitMs,
      js: path.relative(PROJECT_DIR, wasmJsPath),
      wasm: path.relative(PROJECT_DIR, wasmWasmPath),
    },
    input: {
      path: inputPath,
      bytes: Buffer.byteLength(jsonText, "utf8"),
      constraints: parsed?.num_constraints ?? null,
      variables: parsed?.num_variables ?? null,
      steps: Array.isArray(parsed?.witness) ? parsed.witness.length : null,
    },
    config: {
      warmup: opts.warmup,
      runs: opts.runs,
      spartan: opts.spartan,
    },
    runs,
    summary,
  };

  if (opts.json) {
    console.log(JSON.stringify(out, null, 2));
    return;
  }

  const fmt = (n) => (typeof n === "number" && Number.isFinite(n) ? `${n.toFixed(1)} ms` : "n/a");
  console.log(`Input: ${out.input.path} (${out.input.bytes} bytes)`);
  if (out.input.constraints != null || out.input.variables != null || out.input.steps != null) {
    console.log(
      `Parsed: constraints=${out.input.constraints ?? "?"} variables=${out.input.variables ?? "?"} steps=${out.input.steps ?? "?"}`,
    );
  }
  console.log(`WASM init: ${fmt(out.wasm.wasm_init_ms)}`);
  console.log("");

  runs.forEach((r, i) => {
    const ok = r.verify_ok ? "ok" : "FAIL";
    console.log(
      `Run ${i + 1}: total=${fmt(r.timings_ms.total)} prove=${fmt(r.timings_ms.fold_and_prove)} verify=${fmt(r.timings_ms.verify)} (${ok}, steps=${r.steps})`,
    );
    if (r.spartan) {
      console.log(
        `        spartan: prove=${fmt(r.spartan.prove_ms)} verify=${fmt(r.spartan.verify_ms)} ok=${String(r.spartan.verify_ok)} snark_bytes=${r.spartan.snark_bytes ?? "?"}`,
      );
    }
  });

  console.log("");
  console.log(
    `Avg: total=${fmt(summary.avg_ms.total)} prove=${fmt(summary.avg_ms.fold_and_prove)} verify=${fmt(summary.avg_ms.verify)} (n=${summary.runs})`,
  );
}

main().catch((err) => {
  console.error(`ERROR: ${err?.message ?? String(err)}`);
  process.exit(1);
});

