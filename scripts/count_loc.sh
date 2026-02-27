#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CRATES=(
  neo-ajtai
  neo-ccs
  neo-fold
  neo-math
  neo-memory
  neo-params
  neo-reductions
  neo-transcript
  neo-vm-trace
)

count_loc() {
  local dir="$1"
  find "$dir" -name '*.rs' \
    -not -path '*/tests/*' \
    -not -path '*/test/*' \
    -not -name '*_test.rs' \
    -not -name 'test_*.rs' \
    -print0 \
  | xargs -0 awk '
    BEGIN { n = 0 }
    /^\s*$/      { next }
    /^\s*\/\//   { next }
    /^\s*\/\*/   { block=1 }
    block        { if (/\*\//) block=0; next }
    { n++ }
    END { print n }
  ' 2>/dev/null
}

total=0

printf "%-20s %8s\n" "Crate" "LoC"
printf "%-20s %8s\n" "--------------------" "--------"

for crate in "${CRATES[@]}"; do
  crate_dir="$REPO_ROOT/crates/$crate"
  if [ ! -d "$crate_dir/src" ]; then
    printf "%-20s %8s\n" "$crate" "(missing)"
    continue
  fi

  count=$(count_loc "$crate_dir/src")
  count=${count:-0}

  printf "%-20s %8d\n" "$crate" "$count"
  total=$((total + count))
done

printf "%-20s %8s\n" "--------------------" "--------"
printf "%-20s %8d\n" "TOTAL" "$total"
