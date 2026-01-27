#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build an iOS XCFramework for the neo-fold C-ABI wrapper.

Usage:
  ./scripts/build_ios_xcframework.sh [--profile release|profiling] [--out <dir>] [--name <xcframework_name>] [--features <features>] [--no-default-features] [--no-spartan] [--include-x86_64-sim]

Examples:
  ./scripts/build_ios_xcframework.sh
  ./scripts/build_ios_xcframework.sh --profile profiling
  ./scripts/build_ios_xcframework.sh --out dist --name NeoFoldFFI
  ./scripts/build_ios_xcframework.sh --profile profiling --features debug-logs
  ./scripts/build_ios_xcframework.sh --no-spartan
  ./scripts/build_ios_xcframework.sh --include-x86_64-sim
EOF
}

PROFILE="release"
OUT_DIR=""
FRAMEWORK_NAME="NeoFoldFFI"
INCLUDE_X86_64_SIM=0
FEATURES=""
NO_DEFAULT_FEATURES=0
ENABLE_SPARTAN=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --out)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --name)
      FRAMEWORK_NAME="${2:-}"
      shift 2
      ;;
    --features)
      FEATURES="${2:-}"
      shift 2
      ;;
    --no-default-features)
      NO_DEFAULT_FEATURES=1
      shift 1
      ;;
    --no-spartan)
      ENABLE_SPARTAN=0
      shift 1
      ;;
    --include-x86_64-sim)
      INCLUDE_X86_64_SIM=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${ROOT_DIR}/dist"
fi

case "${PROFILE}" in
  release|profiling) ;;
  *)
    echo "Unsupported --profile \"${PROFILE}\" (expected release|profiling)" >&2
    exit 2
    ;;
esac

command -v xcodebuild >/dev/null 2>&1 || {
  echo "Missing xcodebuild; install Xcode Command Line Tools (xcode-select --install)" >&2
  exit 1
}

HEADER_DIR="${ROOT_DIR}/crates/neo-fold-ffi/include"
if [[ ! -f "${HEADER_DIR}/neo_fold_ffi.h" ]]; then
  echo "Missing header: ${HEADER_DIR}/neo_fold_ffi.h" >&2
  exit 1
fi

PACKAGE="neo-fold-ffi"
LIB_BASENAME="libneo_fold_ffi.a"

normalize_features_csv() {
  local input="${1:-}"
  input="${input//[[:space:]]/,}"
  input="$(printf '%s' "${input}" | tr -s ',')"
  input="${input#,}"
  input="${input%,}"
  printf '%s' "${input}"
}

csv_has_feature() {
  local csv="${1:-}"
  local feature="${2:-}"
  [[ ",${csv}," == *",${feature},"* ]]
}

csv_add_feature() {
  local csv="${1:-}"
  local feature="${2:-}"
  if [[ -z "${csv}" ]]; then
    printf '%s' "${feature}"
    return
  fi
  if csv_has_feature "${csv}" "${feature}"; then
    printf '%s' "${csv}"
    return
  fi
  printf '%s,%s' "${csv}" "${feature}"
}

FEATURES="$(normalize_features_csv "${FEATURES}")"
if [[ "${ENABLE_SPARTAN}" == "1" ]]; then
  FEATURES="$(csv_add_feature "${FEATURES}" "spartan")"
else
  if csv_has_feature "${FEATURES}" "spartan" || csv_has_feature "${FEATURES}" "spartan-debug-logs"; then
    echo "Conflicting options: --no-spartan cannot be combined with --features including \"spartan\"." >&2
    exit 2
  fi
fi

TARGETS=("aarch64-apple-ios" "aarch64-apple-ios-sim")
if [[ "${INCLUDE_X86_64_SIM}" == "1" ]]; then
  TARGETS+=("x86_64-apple-ios")
fi

for target in "${TARGETS[@]}"; do
  if ! rustup target list --installed | grep -Eq "^${target}$"; then
    echo "Installing Rust target: ${target}"
    rustup target add "${target}"
  fi

  cargo_args=(build -p "${PACKAGE}" --profile "${PROFILE}" --target "${target}")
  if [[ "${NO_DEFAULT_FEATURES}" == "1" ]]; then
    cargo_args+=(--no-default-features)
  fi
  if [[ -n "${FEATURES}" ]]; then
    cargo_args+=(--features "${FEATURES}")
  fi
  cargo "${cargo_args[@]}"
done

mkdir -p "${OUT_DIR}"
rm -rf "${OUT_DIR:?}/${FRAMEWORK_NAME}.xcframework"

xcframework_args=()
for target in "${TARGETS[@]}"; do
  lib_path="${ROOT_DIR}/target/${target}/${PROFILE}/${LIB_BASENAME}"
  if [[ ! -f "${lib_path}" ]]; then
    echo "Missing built library: ${lib_path}" >&2
    exit 1
  fi
  xcframework_args+=(-library "${lib_path}" -headers "${HEADER_DIR}")
done

xcodebuild -create-xcframework "${xcframework_args[@]}" -output "${OUT_DIR}/${FRAMEWORK_NAME}.xcframework"

echo "Wrote: ${OUT_DIR}/${FRAMEWORK_NAME}.xcframework"
