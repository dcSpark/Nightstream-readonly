#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WASM_DIR="${DEMO_DIR}/wasm"
WEB_DIR="${DEMO_DIR}/web"

usage() {
  cat <<'EOF'
Usage: ./demos/wasm-demo/build_wasm.sh [--no-threads]

Options:
  --no-threads  Build a single-thread wasm bundle into demos/wasm-demo/web/pkg/.
  --threads     Build a wasm-threads (SharedArrayBuffer) bundle into demos/wasm-demo/web/pkg_threads/ (default).
EOF
}

THREADS=1
for arg in "$@"; do
  case "${arg}" in
    --threads) THREADS=1 ;;
    --no-threads) THREADS=0 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "wasm-pack not found."
  echo "Install with: cargo install wasm-pack"
  exit 1
fi

PKG_DIR="${WEB_DIR}/pkg"
OUT_DIR="${PKG_DIR}"
OUT_NAME="neo_fold_demo"

if [[ "${THREADS}" == "1" ]]; then
  OUT_DIR="${WEB_DIR}/pkg_threads"
  export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+atomics,+bulk-memory,+mutable-globals"
  echo "Building wasm threads bundle (requires COOP/COEP + SharedArrayBuffer)…"
else
  echo "Building single-thread wasm bundle…"
fi

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

if [[ "${THREADS}" == "1" ]]; then
  TOOLCHAIN="${WASM_THREADS_TOOLCHAIN:-nightly}"
  if ! command -v rustup >/dev/null 2>&1; then
    echo "rustup not found (needed to select a nightly toolchain for wasm threads)." >&2
    echo "Install: https://rustup.rs" >&2
    exit 1
  fi
  if ! rustup run "${TOOLCHAIN}" rustc --version >/dev/null 2>&1; then
    echo "Rust toolchain \"${TOOLCHAIN}\" is not installed." >&2
    echo "Install with: rustup toolchain install ${TOOLCHAIN}" >&2
    exit 1
  fi
  if ! rustup target list --installed --toolchain "${TOOLCHAIN}" | grep -Eq "^wasm32-unknown-unknown$"; then
    echo "Target wasm32-unknown-unknown is not installed for toolchain \"${TOOLCHAIN}\"." >&2
    echo "Install with: rustup target add wasm32-unknown-unknown --toolchain ${TOOLCHAIN}" >&2
    exit 1
  fi
  if ! rustup component list --toolchain "${TOOLCHAIN}" | grep -Eq "^rust-src\\s+\\(installed\\)$"; then
    echo "rust-src is required for wasm threads (-Z build-std)." >&2
    echo "Install with: rustup component add rust-src --toolchain ${TOOLCHAIN}" >&2
    exit 1
  fi

  RUSTUP_TOOLCHAIN="${TOOLCHAIN}" \
    wasm-pack build "${WASM_DIR}" \
    --release \
    --target web \
    --out-dir "${OUT_DIR}" \
    --out-name "${OUT_NAME}" \
    -- \
    --features wasm-threads \
    -Z build-std=std,panic_abort \
    -Z build-std-features=panic_immediate_abort

  # wasm-bindgen-rayon has two modes:
  # - "no-bundler" (preferred for this demo): emits workerHelpers.no-bundler.js (no patch needed)
  # - default: emits workerHelpers.js which does `import('../../..')` and needs patching for plain web modules
  python3 - "${OUT_DIR}" "${OUT_NAME}" <<'PY'
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
out_name = sys.argv[2]
paths_no_bundler = list(
    out_dir.glob("snippets/wasm-bindgen-rayon-*/src/workerHelpers.no-bundler.js")
)
if paths_no_bundler:
    print("Found wasm-bindgen-rayon no-bundler helper; no patch needed.")
    raise SystemExit(0)

paths = list(out_dir.glob("snippets/wasm-bindgen-rayon-*/src/workerHelpers.js"))
if not paths:
    raise SystemExit(
        "ERROR: wasm-bindgen-rayon workerHelpers snippet not found; cannot patch for web."
    )

target = f"../../../{out_name}.js"
patched = 0
for p in paths:
    txt = p.read_text(encoding="utf-8")
    new = txt.replace("import('../../..')", f"import('{target}')")
    if new != txt:
        p.write_text(new, encoding="utf-8")
        patched += 1

if patched == 0:
    raise SystemExit(
        "ERROR: workerHelpers.js found but no replacements made; threads may hang at init."
    )
print(f"Patched {patched} workerHelpers.js file(s) for plain web module loading.")
PY
else
  wasm-pack build "${WASM_DIR}" \
    --release \
    --target web \
    --out-dir "${OUT_DIR}" \
    --out-name "${OUT_NAME}"
fi

python3 - "${OUT_DIR}" "${DEMO_DIR}/../.." "${THREADS}" <<'PY'
import datetime
import json
import subprocess
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
repo_root = Path(sys.argv[2]).resolve()
threads = sys.argv[3] == "1"

info = {
    "bundle": "pkg_threads" if threads else "pkg",
    "build_time_utc": datetime.datetime.now(datetime.timezone.utc)
    .replace(microsecond=0)
    .isoformat()
    .replace("+00:00", "Z"),
}

def git(args):
    return (
        subprocess.check_output(["git", "-C", str(repo_root), *args], stderr=subprocess.DEVNULL)
        .decode("utf-8")
        .strip()
    )

try:
    info["git_commit"] = git(["rev-parse", "HEAD"])
    info["git_commit_short"] = git(["rev-parse", "--short=12", "HEAD"])
    dirty_tracked = (
        subprocess.call(
            ["git", "-C", str(repo_root), "diff", "--quiet", "--exit-code"],
            stderr=subprocess.DEVNULL,
        )
        != 0
        or subprocess.call(
            ["git", "-C", str(repo_root), "diff", "--cached", "--quiet", "--exit-code"],
            stderr=subprocess.DEVNULL,
        )
        != 0
    )
    info["git_dirty"] = dirty_tracked
except Exception:
    info["git_commit"] = None
    info["git_commit_short"] = None
    info["git_dirty"] = None

out = out_dir / "build_info.json"
out.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Wrote build info: {out}")
PY

echo "Wrote wasm bundle to: ${OUT_DIR}"
