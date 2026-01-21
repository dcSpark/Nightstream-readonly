#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WASM_DIR="${DEMO_DIR}/wasm"
WEB_DIR="${DEMO_DIR}/web"
PKG_DIR="${WEB_DIR}/pkg"

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "wasm-pack not found."
  echo "Install with: cargo install wasm-pack"
  exit 1
fi

rm -rf "${PKG_DIR}"
mkdir -p "${PKG_DIR}"

wasm-pack build "${WASM_DIR}" \
  --release \
  --target web \
  --out-dir "${PKG_DIR}" \
  --out-name neo_fold_demo

echo "Wrote wasm bundle to: ${PKG_DIR}"
