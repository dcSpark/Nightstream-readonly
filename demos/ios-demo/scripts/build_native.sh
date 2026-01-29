#!/bin/bash

# Build script for the halo3 NeoFoldFFI iOS XCFramework (native prover)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default: use this repo checkout.
HALO3_DIR_DEFAULT="$(cd "$PROJECT_DIR/../.." && pwd)"
HALO3_DIR="${HALO3_DIR:-$HALO3_DIR_DEFAULT}"

PROFILE="release"
INCLUDE_X86_64_SIM=0
FEATURES=""

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --include-x86_64-sim)
            INCLUDE_X86_64_SIM=1
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
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --release            Build with Cargo release profile (default)"
            echo "  --profiling          Build with Cargo profiling profile"
            echo "  --profile <name>     Explicit profile (release|profiling)"
            echo "  --include-x86_64-sim Include Intel simulator slice"
            echo "  --features <list>    Pass cargo features to halo3 build (e.g. \"spartan debug-logs\")"
            echo "  --spartan            Convenience flag for --features spartan"
            echo "  --help               Show this help message"
            echo ""
            echo "Environment:"
            echo "  HALO3_DIR            Path to halo3 repo (default: $HALO3_DIR_DEFAULT)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ ! -d "$HALO3_DIR" ]]; then
    echo "halo3 repo not found at: $HALO3_DIR"
    echo "Override with: HALO3_DIR=/path/to/halo3 $0"
    exit 1
fi

OUTPUT_DIR="$PROJECT_DIR/TestingWasm/Frameworks"
mkdir -p "$OUTPUT_DIR"

echo "Building NeoFoldFFI.xcframework..."
echo "  Profile:   $PROFILE"
echo "  Source:    $HALO3_DIR"
echo "  Output:    $OUTPUT_DIR"
if [[ -n "$FEATURES" ]]; then
    echo "  Features:  $FEATURES"
fi

ARGS=(--out "$OUTPUT_DIR" --profile "$PROFILE")
if [[ "$INCLUDE_X86_64_SIM" == "1" ]]; then
    ARGS+=(--include-x86_64-sim)
fi
if [[ -n "$FEATURES" ]]; then
    ARGS+=(--features "$FEATURES")
fi

(cd "$HALO3_DIR" && ./scripts/build_ios_xcframework.sh "${ARGS[@]}")

echo ""
echo "Wrote: $OUTPUT_DIR/NeoFoldFFI.xcframework"
