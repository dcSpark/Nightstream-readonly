#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./demos/wasm-demo/serve.sh [--force-refresh]

Options:
  --force-refresh  Rebuild the wasm bundle before serving.
EOF
}

FORCE_REFRESH=0
for arg in "$@"; do
  case "${arg}" in
    --force-refresh) FORCE_REFRESH=1 ;;
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
  "${DEMO_DIR}/build_wasm.sh"
elif [[ ! -f "${DEMO_DIR}/web/pkg/neo_fold_demo.js" ]]; then
  echo "Missing demos/wasm-demo/web/pkg/neo_fold_demo.js"
  echo "Building wasm bundle..."
  "${DEMO_DIR}/build_wasm.sh"
fi

cd "${DEMO_DIR}/web"

PORT="${PORT:-8000}"
echo "Serving http://127.0.0.1:${PORT}"
python3 -m http.server "${PORT}"
