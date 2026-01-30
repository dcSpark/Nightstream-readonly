#!/usr/bin/env bash
set -euo pipefail

DEMOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DEMOS_DIR}/.." && pwd)"

WASM_DEMO_DIR="${REPO_ROOT}/demos/wasm-demo"
WASM_DIR="${WASM_DEMO_DIR}/wasm"
WASM_WEB_DIR="${WASM_DEMO_DIR}/web"

OUT_NAME="neo_fold_demo"

IOS_DIR="${REPO_ROOT}/demos/ios-demo"
IOS_WEB_DIR="${IOS_DIR}/web"
IOS_RES_DIR="${IOS_DIR}/TestingWasm/Resources"
IOS_RES_WASM_DIR="${IOS_RES_DIR}/wasm"
IOS_RES_EXAMPLES_DIR="${IOS_RES_DIR}/examples"

ANDROID_DIR="${REPO_ROOT}/demos/android-demo"
ANDROID_ASSETS_WEB_DIR="${ANDROID_DIR}/app/src/main/assets/web"
ANDROID_ASSETS_EXAMPLES_DIR="${ANDROID_DIR}/app/src/main/assets/examples"

usage() {
  cat <<'EOF'
Usage: ./demos/build_wasm.sh [options]

Builds wasm-demo bundles and syncs them into the iOS/Android demos.

Options:
  --both         Build/copy both bundles (pkg + pkg_threads) (default).
  --no-threads   Build/copy only the single-thread bundle (pkg).
  --threads      Build/copy only the threads bundle (pkg_threads).

  --release      Release profile (default).
  --debug        Debug profile.

  --skip-ios     Skip iOS outputs (Resources/ + ios-demo/web/).
  --skip-android Skip Android outputs (assets/).
EOF
}

MODE="both"
PROFILE="release"
SKIP_IOS=0
SKIP_ANDROID=0

for arg in "$@"; do
  case "${arg}" in
    --both) MODE="both" ;;
    --no-threads) MODE="no-threads" ;;
    --threads) MODE="threads" ;;
    --release) PROFILE="release" ;;
    --debug) PROFILE="debug" ;;
    --skip-ios) SKIP_IOS=1 ;;
    --skip-android) SKIP_ANDROID=1 ;;
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

if [[ "${PROFILE}" != "release" && "${PROFILE}" != "debug" ]]; then
  echo "ERROR: invalid PROFILE=${PROFILE}" >&2
  exit 2
fi

need_cmd() {
  local bin="$1"
  if ! command -v "${bin}" >/dev/null 2>&1; then
    echo "ERROR: ${bin} not found in PATH" >&2
    exit 1
  fi
}

need_cmd wasm-pack
need_cmd python3
need_cmd rsync

sync_dir() {
  local src="$1"
  local dst="$2"
  mkdir -p "${dst}"
  # Keep the destination .gitignore (these dirs typically ignore generated outputs).
  rsync -a --delete --exclude=".gitignore" "${src}/" "${dst}/"
}

sync_examples_android() {
  mkdir -p "${ANDROID_ASSETS_EXAMPLES_DIR}"
  rm -f "${ANDROID_ASSETS_EXAMPLES_DIR}"/*.json 2>/dev/null || true
  cp -f "${WASM_WEB_DIR}/examples/toy_square.json" "${ANDROID_ASSETS_EXAMPLES_DIR}/toy_square.json"
  cp -f "${WASM_WEB_DIR}/examples/toy_square_folding_8_steps.json" "${ANDROID_ASSETS_EXAMPLES_DIR}/toy_square_folding_8_steps.json"
  cp -f "${WASM_WEB_DIR}/examples/poseidon2_ic_batch_1.json" "${ANDROID_ASSETS_EXAMPLES_DIR}/poseidon2_ic_batch_1.json"
}

sync_examples_ios() {
  mkdir -p "${IOS_RES_EXAMPLES_DIR}"
  cp -f "${WASM_WEB_DIR}/examples/toy_square.json" "${IOS_RES_EXAMPLES_DIR}/toy_square.json"
  cp -f "${WASM_WEB_DIR}/examples/toy_square_folding_8_steps.json" "${IOS_RES_EXAMPLES_DIR}/toy_square_folding_8_steps.json"
  cp -f "${WASM_WEB_DIR}/examples/poseidon2_ic_batch_1.json" "${IOS_RES_EXAMPLES_DIR}/poseidon2_ic_batch_1.json"
}

build_ios_no_modules() {
  mkdir -p "${IOS_RES_WASM_DIR}"

  local tmp
  tmp="$(mktemp -d)"
  trap "rm -rf '${tmp}'" EXIT

  local args=(build "${WASM_DIR}" --target no-modules --out-dir "${tmp}" --out-name "${OUT_NAME}")
  if [[ "${PROFILE}" == "release" ]]; then
    args+=(--release)
  fi

  wasm-pack "${args[@]}"

  cp -f "${tmp}/${OUT_NAME}.js" "${IOS_RES_WASM_DIR}/${OUT_NAME}.js"
  cp -f "${tmp}/${OUT_NAME}_bg.wasm" "${IOS_RES_WASM_DIR}/${OUT_NAME}_bg.wasm"
}

echo "Building wasm-demo (${PROFILE})…"

case "${MODE}" in
  both)
    "${WASM_DEMO_DIR}/build_wasm.sh" --both "--${PROFILE}"
    ;;
  no-threads)
    "${WASM_DEMO_DIR}/build_wasm.sh" --no-threads "--${PROFILE}"
    ;;
  threads)
    "${WASM_DEMO_DIR}/build_wasm.sh" --threads "--${PROFILE}"
    ;;
  *)
    echo "ERROR: unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

if [[ "${SKIP_IOS}" == "0" ]]; then
  echo "Syncing iOS demo…"
  build_ios_no_modules

  if [[ "${MODE}" != "threads" ]]; then
    sync_dir "${WASM_WEB_DIR}/pkg" "${IOS_WEB_DIR}/pkg"
  fi
  if [[ "${MODE}" != "no-threads" ]]; then
    sync_dir "${WASM_WEB_DIR}/pkg_threads" "${IOS_WEB_DIR}/pkg_threads"
  fi

  sync_examples_ios
fi

if [[ "${SKIP_ANDROID}" == "0" ]]; then
  echo "Syncing Android demo…"
  if [[ "${MODE}" != "threads" ]]; then
    sync_dir "${WASM_WEB_DIR}/pkg" "${ANDROID_ASSETS_WEB_DIR}/pkg"
  fi
  if [[ "${MODE}" != "no-threads" ]]; then
    sync_dir "${WASM_WEB_DIR}/pkg_threads" "${ANDROID_ASSETS_WEB_DIR}/pkg_threads"
  fi

  sync_examples_android
fi

echo "Done."
