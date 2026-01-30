#!/usr/bin/env bash

# Build script for the Nightstream neo-fold WASM bundles (from halo3 wasm-demo),
# and copy them into the Android demo app assets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ASSETS_WEB_DIR="$PROJECT_DIR/app/src/main/assets/web"
ASSETS_EXAMPLES_DIR="$PROJECT_DIR/app/src/main/assets/examples"

# Default: use the wasm-demo from this repo checkout.
HALO3_DIR_DEFAULT="$(cd "$PROJECT_DIR/../.." && pwd)"
HALO3_WASM_DEMO_DIR_DEFAULT="$HALO3_DIR_DEFAULT/demos/wasm-demo"
HALO3_WASM_DEMO_DIR="${HALO3_WASM_DEMO_DIR:-$HALO3_WASM_DEMO_DIR_DEFAULT}"
HALO3_WASM_DIR="$HALO3_WASM_DEMO_DIR/wasm"
HALO3_WEB_EXAMPLES_DIR="$HALO3_WASM_DEMO_DIR/web/examples"

PROFILE="release"
BUILD_THREADS=0
OUT_NAME="neo_fold_demo"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --debug                 Build with Debug profile (default: Release)
  --release               Build with Release profile (default: Release)
  --threads               Also build a wasm-threads bundle (SharedArrayBuffer) into assets/web/pkg_threads
  --halo3-wasm-demo-dir   Path to halo3/demos/wasm-demo (default: $HALO3_WASM_DEMO_DIR_DEFAULT)
  --help                  Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$HALO3_WASM_DIR" ]]; then
  echo "halo3 wasm-demo not found at: $HALO3_WASM_DIR" >&2
  echo "Override with: HALO3_WASM_DEMO_DIR=/path/to/halo3/demos/wasm-demo $0" >&2
  exit 1
fi

command -v wasm-pack >/dev/null 2>&1 || {
  echo "wasm-pack is required." >&2
  echo "Install with: cargo install wasm-pack" >&2
  exit 1
}

mkdir -p "$ASSETS_WEB_DIR" "$ASSETS_EXAMPLES_DIR"

build_web_bundle() {
  local out_dir="$1"
  local threads="$2" # 0|1

  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  printf '*\n!.gitignore\n' >"$out_dir/.gitignore"

  local args=(build "$HALO3_WASM_DIR" --target web --out-dir "$out_dir" --out-name "$OUT_NAME")
  if [[ "$PROFILE" == "release" ]]; then
    args+=(--release)
  fi

  if [[ "$threads" == "1" ]]; then
    if ! command -v rustup >/dev/null 2>&1; then
      echo "rustup is required to build wasm threads (nightly + -Z build-std)." >&2
      echo "Install: https://rustup.rs" >&2
      exit 1
    fi

    local toolchain="${WASM_THREADS_TOOLCHAIN:-nightly}"
    if ! rustup run "${toolchain}" rustc --version >/dev/null 2>&1; then
      echo "Rust toolchain \"${toolchain}\" is not installed." >&2
      echo "Install with: rustup toolchain install ${toolchain}" >&2
      exit 1
    fi
    if ! rustup target list --installed --toolchain "${toolchain}" | grep -Eq "^wasm32-unknown-unknown$"; then
      echo "Target wasm32-unknown-unknown is not installed for toolchain \"${toolchain}\"." >&2
      echo "Install with: rustup target add wasm32-unknown-unknown --toolchain ${toolchain}" >&2
      exit 1
    fi
    if ! rustup component list --toolchain "${toolchain}" | grep -Eq "^rust-src\\s+\\(installed\\)$"; then
      echo "rust-src is required for wasm threads (-Z build-std)." >&2
      echo "Install with: rustup component add rust-src --toolchain ${toolchain}" >&2
      exit 1
    fi

    export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+atomics,+bulk-memory,+mutable-globals"

    RUSTUP_TOOLCHAIN="${toolchain}" \
      wasm-pack "${args[@]}" \
      -- \
      --features wasm-threads \
      -Z build-std=std,panic_abort \
      -Z build-std-features=panic_immediate_abort

    # wasm-bindgen-rayon can emit either workerHelpers.no-bundler.js (preferred) or workerHelpers.js.
    # Patch the default helper if present so it can load as plain browser modules (no bundler).
    python3 - "$out_dir" "$OUT_NAME" <<'PY'
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
out_name = sys.argv[2]
paths_no_bundler = list(out_dir.glob("snippets/wasm-bindgen-rayon-*/src/workerHelpers.no-bundler.js"))
paths_default = list(out_dir.glob("snippets/wasm-bindgen-rayon-*/src/workerHelpers.js"))
if not paths_no_bundler and not paths_default:
    raise SystemExit("ERROR: wasm-bindgen-rayon workerHelpers snippet not found; cannot patch.")

target = f"../../../{out_name}.js"
failed = []

def patch_error_handler(txt, needle):
    if "wasm_bindgen_worker_error" in txt:
        return txt, True
    patched = txt.replace(
        needle,
        needle.replace(
            ");\n});\n",
            ");\n}).catch((e) => {\n  console.error(e);\n  postMessage({ type: 'wasm_bindgen_worker_error', error: String(e) });\n  close();\n});\n",
        ),
    )
    return patched, patched != txt

for p in paths_default:
    original = p.read_text(encoding="utf-8")
    txt = original

    if "import('../../..')" in txt:
        txt = txt.replace("import('../../..')", f"import('{target}')")
    elif f"import('{target}')" not in txt:
        failed.append(f"{p}: unexpected main-module import")

    txt, ok = patch_error_handler(txt, "  pkg.wbg_rayon_start_worker(receiver);\n});\n")
    if not ok:
        failed.append(f"{p}: could not patch error handler (pattern not found)")

    if txt != original:
        p.write_text(txt, encoding="utf-8")

for p in paths_no_bundler:
    original = p.read_text(encoding="utf-8")
    txt, ok = patch_error_handler(txt, "  pkg.wbg_rayon_start_worker(data.receiver);\n});\n")
    if not ok:
        failed.append(f"{p}: could not patch error handler (pattern not found)")
    if txt != original:
        p.write_text(txt, encoding="utf-8")

if failed:
    raise SystemExit("ERROR: failed to patch wasm-bindgen-rayon workerHelpers:\n- " + "\n- ".join(failed))
PY
  else
    wasm-pack "${args[@]}"
  fi
}

echo "Building neo-fold WASM bundle for Android demo…"
echo "  Profile:  $PROFILE"
echo "  Source:   $HALO3_WASM_DIR"
echo "  Threads:  $([[ $BUILD_THREADS -eq 1 ]] && echo yes || echo no)"
echo "  Output:   $ASSETS_WEB_DIR"

build_web_bundle "$ASSETS_WEB_DIR/pkg" 0
if [[ $BUILD_THREADS -eq 1 ]]; then
  build_web_bundle "$ASSETS_WEB_DIR/pkg_threads" 1
fi

echo "Copying examples…"
rm -f "$ASSETS_EXAMPLES_DIR"/*.json 2>/dev/null || true
cp -f "$HALO3_WEB_EXAMPLES_DIR/toy_square.json" "$ASSETS_EXAMPLES_DIR/toy_square.json"
cp -f "$HALO3_WEB_EXAMPLES_DIR/toy_square_folding_8_steps.json" "$ASSETS_EXAMPLES_DIR/toy_square_folding_8_steps.json"
cp -f "$HALO3_WEB_EXAMPLES_DIR/poseidon2_ic_batch_1.json" "$ASSETS_EXAMPLES_DIR/poseidon2_ic_batch_1.json"

echo ""
echo "WASM bundles written to:"
echo "  $ASSETS_WEB_DIR/pkg/"
if [[ $BUILD_THREADS -eq 1 ]]; then
  echo "  $ASSETS_WEB_DIR/pkg_threads/"
fi
echo ""
echo "Examples copied to:"
echo "  $ASSETS_EXAMPLES_DIR/"

