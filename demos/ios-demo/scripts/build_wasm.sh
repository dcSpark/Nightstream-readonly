#!/bin/bash

# Build script for the Nightstream neo-fold WASM module (from halo3 wasm-demo)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/TestingWasm/Resources"
WEB_OUTPUT_DIR="$PROJECT_DIR/web"

# Default: use the wasm-demo from this repo checkout.
HALO3_DIR_DEFAULT="$(cd "$PROJECT_DIR/../.." && pwd)"
HALO3_WASM_DEMO_DIR_DEFAULT="$HALO3_DIR_DEFAULT/demos/wasm-demo"
HALO3_WASM_DEMO_DIR="${HALO3_WASM_DEMO_DIR:-$HALO3_WASM_DEMO_DIR_DEFAULT}"
HALO3_WASM_DIR="$HALO3_WASM_DEMO_DIR/wasm"
HALO3_WEB_EXAMPLES_DIR="$HALO3_WASM_DEMO_DIR/web/examples"

PROFILE="release"
TARGET="no-modules"
OUT_NAME="neo_fold_demo"
BUILD_THREADS=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            PROFILE="debug"
            shift
            ;;
        --release)
            PROFILE="release"
            shift
            ;;
        --threads)
            BUILD_THREADS=1
            shift
            ;;
        --halo3-wasm-demo-dir)
            HALO3_WASM_DEMO_DIR="$2"
            HALO3_WASM_DIR="$HALO3_WASM_DEMO_DIR/wasm"
            HALO3_WEB_EXAMPLES_DIR="$HALO3_WASM_DEMO_DIR/web/examples"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug                 Build with Debug profile (default: Release)"
            echo "  --release               Build with Release profile (default: Release)"
            echo "  --threads               Also build a wasm-threads bundle for WKWebView (SharedArrayBuffer)"
            echo "  --halo3-wasm-demo-dir   Path to halo3/demos/wasm-demo (default: $HALO3_WASM_DEMO_DIR_DEFAULT)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ ! -d "$HALO3_WASM_DIR" ]]; then
    echo "halo3 wasm-demo not found at: $HALO3_WASM_DIR"
    echo "Override with: HALO3_WASM_DEMO_DIR=/path/to/halo3/demos/wasm-demo $0"
    echo "  (or pass: --halo3-wasm-demo-dir /path/to/halo3/demos/wasm-demo)"
    exit 1
fi

if ! command -v wasm-pack >/dev/null 2>&1; then
    echo "wasm-pack is required."
    echo "Install with: cargo install wasm-pack"
    exit 1
fi

echo "Building neo-fold WASM bundle..."

echo "  Profile: $PROFILE"
echo "  Source:  $HALO3_WASM_DIR"
echo "  Threads: $([[ $BUILD_THREADS -eq 1 ]] && echo yes || echo no)"

OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUT_DIR"' EXIT

WASM_PACK_ARGS=(build "$HALO3_WASM_DIR" --target "$TARGET" --out-dir "$OUT_DIR" --out-name "$OUT_NAME")
if [[ "$PROFILE" == "release" ]]; then
    WASM_PACK_ARGS+=(--release)
fi

wasm-pack "${WASM_PACK_ARGS[@]}"

mkdir -p "$OUTPUT_DIR/wasm"
cp -f "$OUT_DIR/$OUT_NAME.js" "$OUTPUT_DIR/wasm/$OUT_NAME.js"
cp -f "$OUT_DIR/${OUT_NAME}_bg.wasm" "$OUTPUT_DIR/wasm/${OUT_NAME}_bg.wasm"

echo ""
echo "Building neo-fold WASM bundle for WKWebView (ES modules)…"

rm -rf "$WEB_OUTPUT_DIR/pkg"
mkdir -p "$WEB_OUTPUT_DIR/pkg"

WASM_PACK_WEB_ARGS=(build "$HALO3_WASM_DIR" --target web --out-dir "$WEB_OUTPUT_DIR/pkg" --out-name "$OUT_NAME")
if [[ "$PROFILE" == "release" ]]; then
    WASM_PACK_WEB_ARGS+=(--release)
fi

wasm-pack "${WASM_PACK_WEB_ARGS[@]}"

if [[ $BUILD_THREADS -eq 1 ]]; then
    echo ""
    echo "Building neo-fold WASM threads bundle for WKWebView (SharedArrayBuffer)…"

    if ! command -v rustup >/dev/null 2>&1; then
        echo "rustup is required to build wasm threads (nightly + -Z build-std)."
        echo "Install: https://rustup.rs"
        exit 1
    fi

    TOOLCHAIN="${WASM_THREADS_TOOLCHAIN:-nightly}"

    if ! rustup run "${TOOLCHAIN}" rustc --version >/dev/null 2>&1; then
        echo "Rust toolchain \"${TOOLCHAIN}\" is not installed."
        echo "Install with: rustup toolchain install ${TOOLCHAIN}"
        exit 1
    fi
    if ! rustup target list --installed --toolchain "${TOOLCHAIN}" | grep -Eq "^wasm32-unknown-unknown$"; then
        echo "Target wasm32-unknown-unknown is not installed for toolchain \"${TOOLCHAIN}\"."
        echo "Install with: rustup target add wasm32-unknown-unknown --toolchain ${TOOLCHAIN}"
        exit 1
    fi
    if ! rustup component list --toolchain "${TOOLCHAIN}" | grep -Eq "^rust-src\\s+\\(installed\\)$"; then
        echo "rust-src is required for wasm threads (-Z build-std)."
        echo "Install with: rustup component add rust-src --toolchain ${TOOLCHAIN}"
        exit 1
    fi

    rm -rf "$WEB_OUTPUT_DIR/pkg_threads"
    mkdir -p "$WEB_OUTPUT_DIR/pkg_threads"

    export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+atomics,+bulk-memory,+mutable-globals"

    WASM_PACK_THREADS_ARGS=(build "$HALO3_WASM_DIR" --target web --out-dir "$WEB_OUTPUT_DIR/pkg_threads" --out-name "$OUT_NAME")
    if [[ "$PROFILE" == "release" ]]; then
        WASM_PACK_THREADS_ARGS+=(--release)
    fi

    RUSTUP_TOOLCHAIN="${TOOLCHAIN}" \
        wasm-pack "${WASM_PACK_THREADS_ARGS[@]}" \
        -- \
        --features wasm-threads \
        -Z build-std=std,panic_abort \
        -Z build-std-features=panic_immediate_abort

    # wasm-bindgen-rayon emits a Worker helper that does `import('../../..')`, which relies on
    # bundler-style directory resolution. Patch it to import the actual JS entrypoint so it works
    # when served as plain browser modules.
    python3 - "$WEB_OUTPUT_DIR/pkg_threads" "$OUT_NAME" <<'PY'
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
out_name = sys.argv[2]
paths = list(out_dir.glob("snippets/wasm-bindgen-rayon-*/src/workerHelpers.js"))
if not paths:
    raise SystemExit("ERROR: wasm-bindgen-rayon workerHelpers.js not found; cannot patch for web.")

target = f"../../../{out_name}.js"
patched = 0
for p in paths:
    txt = p.read_text(encoding="utf-8")
    new = txt.replace("import('../../..')", f"import('{target}')")
    if new != txt:
        p.write_text(new, encoding="utf-8")
        patched += 1

if patched == 0:
    raise SystemExit("ERROR: workerHelpers.js found but no replacements made; threads may hang at init.")
print(f"Patched {patched} workerHelpers.js file(s) for plain web module loading.")
PY
fi

mkdir -p "$OUTPUT_DIR/examples"
cp -f "$HALO3_WEB_EXAMPLES_DIR/toy_square.json" "$OUTPUT_DIR/examples/toy_square.json"
cp -f "$HALO3_WEB_EXAMPLES_DIR/toy_square_folding_8_steps.json" "$OUTPUT_DIR/examples/toy_square_folding_8_steps.json"
cp -f "$HALO3_WEB_EXAMPLES_DIR/poseidon2_ic_batch_1.json" "$OUTPUT_DIR/examples/poseidon2_ic_batch_1.json"

echo ""
echo "WASM bundle written to:"
echo "  $OUTPUT_DIR/wasm/$OUT_NAME.js"
echo "  $OUTPUT_DIR/wasm/${OUT_NAME}_bg.wasm"
echo "  $WEB_OUTPUT_DIR/pkg/$OUT_NAME.js"
echo "  $WEB_OUTPUT_DIR/pkg/${OUT_NAME}_bg.wasm"
if [[ $BUILD_THREADS -eq 1 ]]; then
    echo "  $WEB_OUTPUT_DIR/pkg_threads/$OUT_NAME.js"
    echo "  $WEB_OUTPUT_DIR/pkg_threads/${OUT_NAME}_bg.wasm"
fi
echo ""
echo "Examples copied to:"
echo "  $OUTPUT_DIR/examples/"
