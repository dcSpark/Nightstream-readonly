#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WASM_DIR="${DEMO_DIR}/wasm"
WEB_DIR="${DEMO_DIR}/web"
REPO_ROOT="$(cd "${DEMO_DIR}/../.." && pwd)"

OUT_NAME="neo_fold_demo"

usage() {
  cat <<'EOF'
Usage: ./demos/wasm-demo/build_wasm.sh [options]

Options:
  --no-threads  Build a single-thread wasm bundle into demos/wasm-demo/web/pkg/.
  --threads     Build a wasm-threads (SharedArrayBuffer) bundle into demos/wasm-demo/web/pkg_threads/.
  --both        Build both bundles (pkg + pkg_threads) (default).

  --release     Build with Release profile (default).
  --debug       Build with Debug profile.
EOF
}

BUILD_THREADS=0
BUILD_NO_THREADS=0
BUILD_BOTH=0
PROFILE_RELEASE=0
PROFILE_DEBUG=0

for arg in "$@"; do
  case "${arg}" in
    --threads) BUILD_THREADS=1 ;;
    --no-threads) BUILD_NO_THREADS=1 ;;
    --both) BUILD_BOTH=1 ;;
    --release) PROFILE_RELEASE=1 ;;
    --debug) PROFILE_DEBUG=1 ;;
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

if [[ "${PROFILE_RELEASE}" == "1" && "${PROFILE_DEBUG}" == "1" ]]; then
  echo "ERROR: --release and --debug are mutually exclusive." >&2
  usage >&2
  exit 2
fi

PROFILE="release"
if [[ "${PROFILE_DEBUG}" == "1" ]]; then
  PROFILE="debug"
fi

if [[ "${BUILD_THREADS}" == "1" && "${BUILD_NO_THREADS}" == "1" ]]; then
  echo "ERROR: --threads and --no-threads are mutually exclusive (use --both)." >&2
  usage >&2
  exit 2
fi

if [[ "${BUILD_BOTH}" == "1" ]] && ([[ "${BUILD_THREADS}" == "1" ]] || [[ "${BUILD_NO_THREADS}" == "1" ]]); then
  echo "ERROR: --both cannot be combined with --threads/--no-threads." >&2
  usage >&2
  exit 2
fi

MODE="both"
if [[ "${BUILD_THREADS}" == "1" ]]; then
  MODE="threads"
elif [[ "${BUILD_NO_THREADS}" == "1" ]]; then
  MODE="no-threads"
elif [[ "${BUILD_BOTH}" == "1" ]]; then
  MODE="both"
fi

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "wasm-pack not found." >&2
  echo "Install with: cargo install wasm-pack" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found." >&2
  exit 1
fi

build_info() {
  local out_dir="$1"
  local threads="$2" # 0|1
  python3 - "${out_dir}" "${REPO_ROOT}" "${threads}" "${PROFILE}" <<'PY'
import datetime
import json
import subprocess
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
repo_root = Path(sys.argv[2]).resolve()
threads = sys.argv[3] == "1"
profile = sys.argv[4]

info = {
    "bundle": "pkg_threads" if threads else "pkg",
    "profile": profile,
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
}

patch_wasm_bindgen_rayon_helpers() {
  local out_dir="$1"

  # wasm-bindgen-rayon has two modes:
  # - "no-bundler" (preferred for this demo): emits workerHelpers.no-bundler.js (no patch needed)
  # - default: emits workerHelpers.js which does `import('../../..')` and needs patching for plain web modules
  python3 - "${out_dir}" "${OUT_NAME}" <<'PY'
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
out_name = sys.argv[2]
paths_no_bundler = list(
    out_dir.glob("snippets/wasm-bindgen-rayon-*/src/workerHelpers.no-bundler.js")
)
paths_default = list(out_dir.glob("snippets/wasm-bindgen-rayon-*/src/workerHelpers.js"))
if not paths_no_bundler and not paths_default:
    raise SystemExit("ERROR: wasm-bindgen-rayon workerHelpers snippet not found; cannot patch.")

target = f"../../../{out_name}.js"
patched_import = 0
already_import = 0
patched_error = 0
already_error = 0
failed = []


def patch_error_handler(txt, needle):
    if "wasm_bindgen_worker_error" in txt:
        return txt, False, True
    patched = txt.replace(
        needle,
        needle.replace(
            ");\n});\n",
            ");\n}).catch((e) => {\n  console.error(e);\n  postMessage({ type: 'wasm_bindgen_worker_error', error: String(e) });\n  close();\n});\n",
        ),
    )
    if patched == txt:
        return txt, False, False
    return patched, True, False


for p in paths_default:
    original = p.read_text(encoding="utf-8")
    txt = original

    # 1) Fix module resolution for plain (non-bundler) `--target web` usage.
    if "import('../../..')" in txt:
        txt = txt.replace("import('../../..')", f"import('{target}')")
        patched_import += 1
    elif f"import('{target}')" in txt:
        already_import += 1
    else:
        failed.append(
            f"{p}: unexpected main-module import (expected import('../../..') or import('{target}'))"
        )

    # 2) Avoid unhandled promise rejections if the worker crashes during start (Safari reports these).
    txt2, did_patch, did_already = patch_error_handler(
        txt, "  pkg.wbg_rayon_start_worker(receiver);\n});\n"
    )
    txt = txt2
    if did_patch:
        patched_error += 1
    elif did_already:
        already_error += 1
    else:
        failed.append(f"{p}: could not patch error handler (pattern not found)")

    if txt != original:
        p.write_text(txt, encoding="utf-8")

for p in paths_no_bundler:
    original = p.read_text(encoding="utf-8")
    txt = original

    # Bundlerless helper already resolves the main JS entrypoint dynamically (no import patch needed),
    # but we still add a catch handler to avoid Safari's "Unhandled Promise Rejection" noise.
    txt2, did_patch, did_already = patch_error_handler(
        txt, "  pkg.wbg_rayon_start_worker(data.receiver);\n});\n"
    )
    txt = txt2
    if did_patch:
        patched_error += 1
    elif did_already:
        already_error += 1
    else:
        failed.append(f"{p}: could not patch error handler (pattern not found)")

    if txt != original:
        p.write_text(txt, encoding="utf-8")

if failed:
    raise SystemExit(
        "ERROR: failed to patch wasm-bindgen-rayon workerHelpers:\n- " + "\n- ".join(failed)
    )

print(
    f"Patched wasm-bindgen-rayon helpers: import={patched_import} (already {already_import}), "
    f"error_handler={patched_error} (already {already_error})."
)
PY
}

build_web_bundle() {
  local threads="$1" # 0|1
  local out_dir="${WEB_DIR}/pkg"
  if [[ "${threads}" == "1" ]]; then
    out_dir="${WEB_DIR}/pkg_threads"
  fi

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  local wasm_pack_args=(build "${WASM_DIR}" --target web --out-dir "${out_dir}" --out-name "${OUT_NAME}")
  if [[ "${PROFILE}" == "release" ]]; then
    wasm_pack_args+=(--release)
  fi

  if [[ "${threads}" == "1" ]]; then
    local toolchain="${WASM_THREADS_TOOLCHAIN:-nightly}"
    if ! command -v rustup >/dev/null 2>&1; then
      echo "rustup not found (needed to select a nightly toolchain for wasm threads)." >&2
      echo "Install: https://rustup.rs" >&2
      exit 1
    fi
    if ! rustup run "${toolchain}" rustc --version >/dev/null 2>&1; then
      echo "Rust toolchain \"${toolchain}\" is not installed." >&2
      echo "Install with: rustup toolchain install ${toolchain}" >&2
      exit 1
    fi
    if ! rustup target list --installed --toolchain "${toolchain}" | grep -Eq "^wasm32-unknown-unknown$"; then
      echo "Target wasm32-unknown-unknown is not installed for toolchain \"${toolchain}\"." >&2
      echo "Install with: rustup target add wasm32-unknown-unknown --toolchain ${toolchain}" >&2
      exit 1
    fi
    if ! rustup component list --toolchain "${toolchain}" | grep -Eq "^rust-src\\s+\\(installed\\)$"; then
      echo "rust-src is required for wasm threads (-Z build-std)." >&2
      echo "Install with: rustup component add rust-src --toolchain ${toolchain}" >&2
      exit 1
    fi

    echo "Building wasm threads bundle (${PROFILE})…"

    # NOTE: avoid `panic_immediate_abort` so panics can surface (e.g. via console_error_panic_hook)
    # instead of trapping with an opaque "unreachable" in the browser.
    (
      export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+atomics,+bulk-memory,+mutable-globals"
      RUSTUP_TOOLCHAIN="${toolchain}" \
        wasm-pack "${wasm_pack_args[@]}" \
        -- \
        --features wasm-threads \
        -Z build-std=std,panic_abort
    )

    patch_wasm_bindgen_rayon_helpers "${out_dir}"
  else
    echo "Building single-thread wasm bundle (${PROFILE})…"
    wasm-pack "${wasm_pack_args[@]}"
  fi

  build_info "${out_dir}" "${threads}"
  echo "Wrote wasm bundle to: ${out_dir}"
}

case "${MODE}" in
  both)
    echo "Building both wasm bundles (${PROFILE})…"
    build_web_bundle 0
    build_web_bundle 1
    ;;
  no-threads)
    build_web_bundle 0
    ;;
  threads)
    build_web_bundle 1
    ;;
  *)
    echo "BUG: unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac
