#!/usr/bin/env bash

# Build helper for the Android demo project.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CONFIGURATION="Debug"
BUILD_WASM=0
BUILD_NATIVE=0
WASM_THREADS=0

usage() {
  cat <<'EOF'
Usage: ./scripts/build.sh [options]

Options:
  --release      Build with Release configuration (default: Debug)
  --wasm         Rebuild wasm bundles before Gradle build
  --wasm-threads Rebuild wasm + wasm-threads bundles before Gradle build
  --native       Rebuild native JNI .so before Gradle build
  --all          Rebuild wasm + native before Gradle build
  --help         Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      CONFIGURATION="Release"
      shift
      ;;
    --wasm)
      BUILD_WASM=1
      shift
      ;;
    --wasm-threads)
      BUILD_WASM=1
      WASM_THREADS=1
      shift
      ;;
    --native)
      BUILD_NATIVE=1
      shift
      ;;
    --all)
      BUILD_WASM=1
      BUILD_NATIVE=1
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

cd "$PROJECT_DIR"

if [[ "$BUILD_WASM" == "1" ]]; then
  if [[ "$CONFIGURATION" == "Release" ]]; then
    if [[ "$WASM_THREADS" == "1" ]]; then
      ./scripts/build_wasm.sh --release --threads
    else
      ./scripts/build_wasm.sh --release
    fi
  else
    if [[ "$WASM_THREADS" == "1" ]]; then
      ./scripts/build_wasm.sh --debug --threads
    else
      ./scripts/build_wasm.sh --debug
    fi
  fi
fi

if [[ "$BUILD_NATIVE" == "1" ]]; then
  if [[ "$CONFIGURATION" == "Release" ]]; then
    ./scripts/build_native.sh --release
  else
    ./scripts/build_native.sh --profiling
  fi
fi

task="assembleDebug"
if [[ "$CONFIGURATION" == "Release" ]]; then
  task="assembleRelease"
fi

./gradlew ":app:${task}"

