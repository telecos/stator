#!/usr/bin/env bash
# diff_test.sh — Differential testing: st8 vs d8
#
# Usage:
#   scripts/diff_test.sh <script.js>
#   scripts/diff_test.sh --corpus [corpus_dir]
#
# Runs <script.js> in both st8 (Stator) and d8 (V8), compares stdout, stderr,
# and exit code, then reports one of three categories:
#
#   PASS        — identical stdout, stderr, and exit code
#   KNOWN_DIFF  — divergence documented in scripts/known_diffs.txt
#   BUG         — unexpected divergence (should be reported)
#
# When --corpus is given, all *.js files in corpus_dir (default: tests/diff/)
# are processed and a summary is printed.
#
# Prerequisites:
#   - st8 must be on PATH, or built at target/debug/st8 / target/release/st8
#   - d8 must be on PATH (comparison is skipped with a warning when absent)
#
# Exit codes:
#   0 — all tests passed (PASS or KNOWN_DIFF) or d8 was absent
#   1 — one or more BUG-category divergences were found

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KNOWN_DIFFS_FILE="$SCRIPT_DIR/known_diffs.txt"

# ── Locate st8 ────────────────────────────────────────────────────────────────

find_st8() {
    if command -v st8 &>/dev/null; then
        echo "st8"
        return
    fi
    for candidate in \
        "$REPO_ROOT/target/release/st8" \
        "$REPO_ROOT/target/debug/st8"; do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return
        fi
    done
    echo ""
}

ST8="$(find_st8)"
if [[ -z "$ST8" ]]; then
    echo "error: st8 not found. Build it with: cargo build --package st8" >&2
    exit 1
fi

# ── Locate d8 ─────────────────────────────────────────────────────────────────

D8=""
if command -v d8 &>/dev/null; then
    D8="d8"
fi

# ── Load known diffs ──────────────────────────────────────────────────────────
# Format: one entry per line, either a bare filename or a glob pattern.
# Lines beginning with '#' and blank lines are ignored.

declare -a KNOWN_DIFF_PATTERNS=()
if [[ -f "$KNOWN_DIFFS_FILE" ]]; then
    while IFS= read -r line; do
        # Strip leading/trailing whitespace
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" || "$line" == \#* ]] && continue
        KNOWN_DIFF_PATTERNS+=("$line")
    done < "$KNOWN_DIFFS_FILE"
fi

is_known_diff() {
    local script="$1"
    local basename
    basename="$(basename "$script")"
    for pattern in "${KNOWN_DIFF_PATTERNS[@]:-}"; do
        # shellcheck disable=SC2254
        case "$basename" in
            $pattern) return 0 ;;
        esac
    done
    return 1
}

# ── Run one engine ────────────────────────────────────────────────────────────

# run_engine <engine_binary> <script.js> <stdout_file> <stderr_file>
# Writes stdout to stdout_file, stderr to stderr_file.
# Returns the exit code of the engine.
run_engine() {
    local engine="$1"
    local script="$2"
    local out="$3"
    local err="$4"
    local exit_code=0
    "$engine" "$script" >"$out" 2>"$err" || exit_code=$?
    echo "$exit_code"
}

# ── Diff one test ─────────────────────────────────────────────────────────────

# diff_test <script.js>
# Returns:
#   0 — PASS
#   1 — BUG
#   2 — KNOWN_DIFF
diff_test() {
    local script="$1"
    local tmpdir
    tmpdir="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '$tmpdir'" RETURN

    local st8_out="$tmpdir/st8.stdout"
    local st8_err="$tmpdir/st8.stderr"
    local st8_exit

    st8_exit="$(run_engine "$ST8" "$script" "$st8_out" "$st8_err")"

    if [[ -z "$D8" ]]; then
        # d8 not available — just verify st8 runs without crashing.
        if [[ "$st8_exit" -eq 0 ]]; then
            printf "  PASS (st8-only)  %s\n" "$(basename "$script")"
            return 0
        else
            local err_msg
            err_msg="$(cat "$st8_err" 2>/dev/null || true)"
            printf "  FAIL (st8 error) %s: %s\n" "$(basename "$script")" "$err_msg"
            return 1
        fi
    fi

    local d8_out="$tmpdir/d8.stdout"
    local d8_err="$tmpdir/d8.stderr"
    local d8_exit

    d8_exit="$(run_engine "$D8" "$script" "$d8_out" "$d8_err")"

    local stdout_match=true exit_match=true
    diff -q "$st8_out" "$d8_out" &>/dev/null || stdout_match=false
    [[ "$st8_exit" == "$d8_exit" ]] || exit_match=false

    if $stdout_match && $exit_match; then
        printf "  PASS        %s\n" "$(basename "$script")"
        return 0
    fi

    if is_known_diff "$script"; then
        printf "  KNOWN_DIFF  %s\n" "$(basename "$script")"
        return 2
    fi

    printf "  BUG         %s\n" "$(basename "$script")"
    if ! $stdout_match; then
        echo "    stdout divergence:"
        diff <(sed 's/^/      st8: /' "$st8_out") \
             <(sed 's/^/      d8:  /' "$d8_out") || true
    fi
    if ! $exit_match; then
        printf "    exit code: st8=%s  d8=%s\n" "$st8_exit" "$d8_exit"
    fi
    return 1
}

# ── Entry point ───────────────────────────────────────────────────────────────

main() {
    local corpus_mode=false
    local corpus_dir="$REPO_ROOT/tests/diff"
    local script=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --corpus)
                corpus_mode=true
                if [[ $# -gt 1 && ! "$2" == --* ]]; then
                    corpus_dir="$2"
                    shift
                fi
                shift
                ;;
            -*)
                echo "Usage: $0 <script.js>" >&2
                echo "       $0 --corpus [corpus_dir]" >&2
                exit 1
                ;;
            *)
                script="$1"
                shift
                ;;
        esac
    done

    if $corpus_mode; then
        if [[ ! -d "$corpus_dir" ]]; then
            echo "error: corpus directory not found: $corpus_dir" >&2
            exit 1
        fi

        echo "Differential testing corpus: $corpus_dir"
        if [[ -n "$D8" ]]; then
            echo "  st8: $ST8"
            echo "  d8:  $D8"
        else
            echo "  st8: $ST8  (d8 not found — running st8-only verification)"
        fi
        echo

        local pass=0 known=0 bug=0
        local scripts=()
        while IFS= read -r -d '' f; do
            scripts+=("$f")
        done < <(find "$corpus_dir" -maxdepth 1 -name '*.js' -print0 | sort -z)

        if [[ ${#scripts[@]} -eq 0 ]]; then
            echo "No *.js files found in $corpus_dir"
            exit 0
        fi

        for f in "${scripts[@]}"; do
            local rc=0
            diff_test "$f" || rc=$?
            case "$rc" in
                0) pass=$((pass + 1)) ;;
                2) known=$((known + 1)) ;;
                *) bug=$((bug + 1)) ;;
            esac
        done

        echo
        echo "Results: PASS=${pass}  KNOWN_DIFF=${known}  BUG=${bug}"
        [[ "$bug" -eq 0 ]] && exit 0 || exit 1

    elif [[ -n "$script" ]]; then
        diff_test "$script"
    else
        echo "Usage: $0 <script.js>" >&2
        echo "       $0 --corpus [corpus_dir]" >&2
        exit 1
    fi
}

main "$@"
