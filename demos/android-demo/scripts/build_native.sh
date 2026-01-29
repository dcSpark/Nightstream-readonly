#!/usr/bin/env bash

# Build script for the halo3 NeoFold Android JNI library (native prover)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default: use this repo checkout.
HALO3_DIR_DEFAULT="$(cd "$PROJECT_DIR/../.." && pwd)"
HALO3_DIR="${HALO3_DIR:-$HALO3_DIR_DEFAULT}"

PROFILE="release"
FEATURES=""
ALL_ABIS=0

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --release            Build with Cargo release profile (default)
  --profiling          Build with Cargo profiling profile
  --profile <name>     Explicit profile (release|profiling)
  --all-abis           Build armv7 + arm64 + x86 + x86_64 (default: arm64 + x86_64)
  --features <list>    Pass cargo features (e.g. "spartan debug-logs")
  --spartan            Convenience flag for --features spartan
  --help               Show this help message

Environment:
  HALO3_DIR            Path to halo3 repo (default: $HALO3_DIR_DEFAULT)
  ANDROID_NDK_HOME     Path to Android NDK (required by cargo-ndk)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --profiling)
      PROFILE="profiling"
      shift
      ;;
    --release)
      PROFILE="release"
      shift
      ;;
    --all-abis)
      ALL_ABIS=1
      shift
      ;;
    --features)
      FEATURES="$2"
      shift 2
      ;;
    --spartan)
      if [[ -z "$FEATURES" ]]; then
        FEATURES="spartan"
      else
        FEATURES="$FEATURES spartan"
      fi
      shift
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

case "${PROFILE}" in
  release|profiling) ;;
  *)
    echo "Unsupported --profile \"${PROFILE}\" (expected release|profiling)" >&2
    exit 2
    ;;
esac

if [[ ! -d "$HALO3_DIR" ]]; then
  echo "halo3 repo not found at: $HALO3_DIR" >&2
  exit 1
fi

if ! cargo ndk --help >/dev/null 2>&1; then
  echo "cargo-ndk is required." >&2
  echo "Install with: cargo install cargo-ndk" >&2
  exit 1
fi

if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
  echo "ANDROID_NDK_HOME is not set (cargo-ndk needs the Android NDK)." >&2
  echo "Install via Android Studio (SDK Manager) and export ANDROID_NDK_HOME." >&2
  exit 1
fi

OUT_DIR="$PROJECT_DIR/app/src/main/jniLibs"
mkdir -p "$OUT_DIR"

declare -a ABIS
if [[ "$ALL_ABIS" == "1" ]]; then
  ABIS=("armeabi-v7a" "arm64-v8a" "x86" "x86_64")
else
  ABIS=("arm64-v8a" "x86_64")
fi

rust_target_for_abi() {
  case "$1" in
    armeabi-v7a) echo "armv7-linux-androideabi" ;;
    arm64-v8a) echo "aarch64-linux-android" ;;
    x86) echo "i686-linux-android" ;;
    x86_64) echo "x86_64-linux-android" ;;
    *) return 1 ;;
  esac
}

for abi in "${ABIS[@]}"; do
  target="$(rust_target_for_abi "$abi")"
  if ! rustup target list --installed | grep -Eq "^${target}$"; then
    echo "Installing Rust target: ${target}"
    rustup target add "${target}"
  fi
done

echo "Building libneo_fold_jni.so…"
echo "  Profile:  $PROFILE"
echo "  Source:   $HALO3_DIR"
echo "  Output:   $OUT_DIR"
echo "  ABIs:     ${ABIS[*]}"
if [[ -n "$FEATURES" ]]; then
  echo "  Features: $FEATURES"
fi

cd "$HALO3_DIR"

args=(ndk)
for abi in "${ABIS[@]}"; do
  args+=(-t "$abi")
done
args+=(-o "$OUT_DIR" build -p neo-fold-jni --profile "$PROFILE")
if [[ -n "$FEATURES" ]]; then
  args+=(--features "$FEATURES")
fi

cargo "${args[@]}"

echo ""
echo "Wrote JNI libs under:"
echo "  $OUT_DIR/<abi>/libneo_fold_jni.so"
