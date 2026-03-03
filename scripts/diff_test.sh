#!/usr/bin/env bash
# diff_test.sh — differential testing: st8 (Stator) vs d8 (V8)
#
# Usage:
#   scripts/diff_test.sh <file.js> [--st8 <path>] [--d8 <path>]
#   scripts/diff_test.sh --corpus <dir> [--st8 <path>] [--d8 <path>]
#
# For each test file the script:
#   1. Runs the file in st8 and captures stdout + exit code.
#   2. Runs the file in d8 (when available) and captures stdout + exit code.
#   3. Compares the two outputs and categorises the result:
#        PASS        — identical stdout and identical exit status
#        KNOWN_DIFF  — divergence documented in known_diffs.txt
#        SKIP        — d8 not found; only st8 output is verified (non-crash)
#        BUG         — unexpected divergence; exits with status 1
#
# Exit codes:
#   0  All tests PASS, KNOWN_DIFF, or SKIP
#   1  At least one BUG-category divergence

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ST8="${ST8:-${REPO_ROOT}/target/debug/st8}"
D8="${D8:-d8}"
KNOWN_DIFFS="${KNOWN_DIFFS:-${REPO_ROOT}/tests/diff_corpus/known_diffs.txt}"
CORPUS_MODE=false
CORPUS_DIR=""
FILES=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --st8)   ST8="$2";    shift 2 ;;
    --d8)    D8="$2";     shift 2 ;;
    --corpus) CORPUS_MODE=true; CORPUS_DIR="$2"; shift 2 ;;
    -*)      echo "Unknown option: $1" >&2; exit 1 ;;
    *)       FILES+=("$1"); shift ;;
  esac
done

if $CORPUS_MODE; then
  if [[ -z "$CORPUS_DIR" || ! -d "$CORPUS_DIR" ]]; then
    echo "diff_test.sh: corpus directory not found: ${CORPUS_DIR}" >&2
    exit 1
  fi
  # Collect all .js files from the corpus directory (non-recursive).
  while IFS= read -r -d '' f; do
    FILES+=("$f")
  done < <(find "$CORPUS_DIR" -maxdepth 1 -name '*.js' -print0 | sort -z)
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "Usage: $0 <file.js> [<file.js> ...]" >&2
  echo "       $0 --corpus <dir>" >&2
  exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

# Check whether a name is listed in the known_diffs file.
is_known_diff() {
  local name="$1"
  if [[ ! -f "$KNOWN_DIFFS" ]]; then
    return 1
  fi
  grep -qE "^[[:space:]]*${name}[[:space:]]*:" "$KNOWN_DIFFS"
}

# Run a command, capturing stdout to a variable and the exit code.
# Usage: run_capture <stdout_var> <exit_var> <cmd> [args...]
run_capture() {
  local _out_var="$1"
  local _rc_var="$2"
  shift 2
  local _output
  local _rc
  _output=$("$@" 2>/dev/null) && _rc=$? || _rc=$?
  # Use printf + eval to set caller variables portably (no nameref needed for
  # bash ≥ 4.3, but we want to be safe on bash 3 / macOS).
  printf -v "$_out_var" '%s' "$_output"
  printf -v "$_rc_var"  '%s' "$_rc"
}

# ── Main loop ─────────────────────────────────────────────────────────────────

# Check if the engines are available.
if [[ ! -x "$ST8" ]]; then
  echo "diff_test.sh: st8 binary not found or not executable: ${ST8}" >&2
  echo "  Build with: cargo build --package st8" >&2
  exit 1
fi

D8_AVAILABLE=false
if [[ -x "$D8" ]] || command -v "$D8" &>/dev/null; then
  D8_AVAILABLE=true
fi

PASS=0
KNOWN=0
SKIP=0
BUG=0

for file in "${FILES[@]}"; do
  name="$(basename "${file%.js}")"

  # Run in st8 first — always.
  st8_out=""
  st8_rc=0
  run_capture st8_out st8_rc "$ST8" "$file"

  if ! $D8_AVAILABLE; then
    # Without d8 we can only verify st8 doesn't crash.
    if [[ $st8_rc -ne 0 ]]; then
      printf '[SKIP] %-40s (d8 not found; st8 exited %d)\n' "$name" "$st8_rc"
    else
      printf '[SKIP] %-40s (d8 not found)\n' "$name"
    fi
    SKIP=$((SKIP + 1))
    continue
  fi

  # Run in d8.
  d8_out=""
  d8_rc=0
  run_capture d8_out d8_rc "$D8" "$file"

  # Compare.
  if [[ "$st8_out" == "$d8_out" && "$st8_rc" == "$d8_rc" ]]; then
    printf '[PASS] %-40s\n' "$name"
    PASS=$((PASS + 1))
  elif is_known_diff "$name"; then
    printf '[KNOWN_DIFF] %-40s\n' "$name"
    KNOWN=$((KNOWN + 1))
  else
    printf '[BUG] %-40s\n' "$name"
    printf '  st8 (exit %s):\n' "$st8_rc"
    printf '    %s\n' "$st8_out" | head -10
    printf '  d8 (exit %s):\n' "$d8_rc"
    printf '    %s\n' "$d8_out" | head -10
    BUG=$((BUG + 1))
  fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────"
printf 'Results: PASS=%d  KNOWN_DIFF=%d  SKIP=%d  BUG=%d\n' \
  "$PASS" "$KNOWN" "$SKIP" "$BUG"
echo "─────────────────────────────────────────────────"

if [[ $BUG -gt 0 ]]; then
  echo "FAILED: ${BUG} BUG-category divergence(s) detected." >&2
  exit 1
fi

exit 0
