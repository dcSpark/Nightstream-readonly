#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./demos/wasm-demo/serve.sh [--force-refresh] [--no-threads]

Options:
  --force-refresh  Rebuild the wasm bundle before serving.
  --threads        Serve the demo with wasm threads (default; builds demos/wasm-demo/web/pkg_threads/).
  --no-threads     Serve the demo single-threaded (builds demos/wasm-demo/web/pkg/).
EOF
}

FORCE_REFRESH=0
THREADS=1
for arg in "$@"; do
  case "${arg}" in
    --force-refresh) FORCE_REFRESH=1 ;;
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

if [[ "${FORCE_REFRESH}" == "1" ]]; then
  echo "Force-refresh requested."
  if [[ "${THREADS}" == "1" ]]; then
    "${DEMO_DIR}/build_wasm.sh" --threads
    # Keep a single-thread bundle available for browsers that fail wasm threads at runtime.
    "${DEMO_DIR}/build_wasm.sh" --no-threads
  else
    "${DEMO_DIR}/build_wasm.sh" --no-threads
  fi
else
  if [[ "${THREADS}" == "1" ]]; then
    if [[ ! -f "${DEMO_DIR}/web/pkg_threads/neo_fold_demo.js" ]]; then
      echo "Missing demos/wasm-demo/web/pkg_threads/neo_fold_demo.js"
      echo "Building wasm threads bundle..."
      "${DEMO_DIR}/build_wasm.sh" --threads
    fi
    if [[ ! -f "${DEMO_DIR}/web/pkg/neo_fold_demo.js" ]]; then
      echo "Missing demos/wasm-demo/web/pkg/neo_fold_demo.js (needed for fallback)"
      echo "Building single-thread wasm bundle..."
      "${DEMO_DIR}/build_wasm.sh" --no-threads
    fi
  else
    if [[ ! -f "${DEMO_DIR}/web/pkg/neo_fold_demo.js" ]]; then
      echo "Missing demos/wasm-demo/web/pkg/neo_fold_demo.js"
      echo "Building wasm bundle..."
      "${DEMO_DIR}/build_wasm.sh" --no-threads
    fi
  fi
fi

cd "${DEMO_DIR}/web"

PORT="${PORT:-8000}"
echo "Serving http://127.0.0.1:${PORT}"
python3 "${DEMO_DIR}/serve_with_headers.py" --port "${PORT}" --dir "${DEMO_DIR}/web"
