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
        -Z build-std=std,panic_abort

    # wasm-bindgen-rayon can emit either workerHelpers.no-bundler.js (preferred) or workerHelpers.js.
    # Patch the default helper if present so it can load as plain browser modules (no bundler),
    # and add a catch handler to avoid unhandled promise rejections if workers fail to start.
    python3 - "$WEB_OUTPUT_DIR/pkg_threads" "$OUT_NAME" <<'PY'
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
out_name = sys.argv[2]
paths_no_bundler = list(out_dir.glob("snippets/wasm-bindgen-rayon-*/src/workerHelpers.no-bundler.js"))
paths_default = list(out_dir.glob("snippets/wasm-bindgen-rayon-*/src/workerHelpers.js"))
if not paths_no_bundler and not paths_default:
    raise SystemExit("ERROR: wasm-bindgen-rayon workerHelpers snippet not found; cannot patch.")

target = f"../../../{out_name}.js"
patched_import = 0
already_import = 0
patched_error = 0
already_error = 0
failed = []

def patch_error_handler(txt, needle):
    if "wasm_bindgen_worker_error" in txt:
        return txt, False, True
    patched = txt.replace(
        needle,
        needle.replace(
            ");\n});\n",
            ");\n}).catch((e) => {\n  console.error(e);\n  postMessage({ type: 'wasm_bindgen_worker_error', error: String(e) });\n  close();\n});\n",
        ),
    )
    if patched == txt:
        return txt, False, False
    return patched, True, False

for p in paths_default:
    original = p.read_text(encoding="utf-8")
    txt = original

    # 1) Fix module resolution for plain (non-bundler) `--target web` usage.
    if "import('../../..')" in txt:
        txt = txt.replace("import('../../..')", f"import('{target}')")
        patched_import += 1
    elif f"import('{target}')" in txt:
        already_import += 1
    else:
        failed.append(f"{p}: unexpected main-module import")

    # 2) Avoid unhandled promise rejections if the worker crashes during start.
    txt2, did_patch, did_already = patch_error_handler(
        txt, "  pkg.wbg_rayon_start_worker(receiver);\n});\n"
    )
    txt = txt2
    if did_patch:
        patched_error += 1
    elif did_already:
        already_error += 1
    else:
        failed.append(f"{p}: could not patch error handler (pattern not found)")

    if txt != original:
        p.write_text(txt, encoding="utf-8")

for p in paths_no_bundler:
    original = p.read_text(encoding="utf-8")
    txt = original

    # Bundlerless helper already resolves the main JS entrypoint dynamically (no import patch needed).
    txt2, did_patch, did_already = patch_error_handler(
        txt, "  pkg.wbg_rayon_start_worker(data.receiver);\n});\n"
    )
    txt = txt2
    if did_patch:
        patched_error += 1
    elif did_already:
        already_error += 1
    else:
        failed.append(f"{p}: could not patch error handler (pattern not found)")

    if txt != original:
        p.write_text(txt, encoding="utf-8")

if failed:
    raise SystemExit("ERROR: failed to patch wasm-bindgen-rayon workerHelpers:\n- " + "\n- ".join(failed))

print(
    f"Patched wasm-bindgen-rayon helpers: import={patched_import} (already {already_import}), "
    f"error_handler={patched_error} (already {already_error})."
)
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
