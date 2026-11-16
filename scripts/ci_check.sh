#!/usr/bin/env bash
# W5 (D8): local CI check. Runs, in order, stopping at the first failure:
#   1. compileall on src, tests and scripts (syntax only; no lint or type checking)
#   2. secret scan of the files git would commit (scripts/secret_scan.py)
#   3. pytest on the tests that need NO real data: -m "not real_data", --strict-markers
#
# Data boundary (R1, R2):
#   * Tests marked real_data (they read the downloaded IBM CSV or the real Phase 1/2
#     outputs) are deselected. Collection still stats their input paths for skipif; no
#     file content is read.
#   * No real release is checked: `check-release` reads the champion model and
#     split_assignments.parquet, needs the guard key, and appends to the registry's audit
#     log on a gate failure. It is a separate, manually authorised step and is never run
#     here. Gate and canary logic is covered by the synthetic tests in test_release.py.
#   * Bytecode goes to a temporary directory; pytest's cache plugin is disabled, so the
#     work tree is not written to.
#
# Exit codes: 0 all steps passed; 1 a step failed (named on stderr); 2 usage error.
# CI_PYTHON overrides the interpreter (default: .venv/bin/python in the repo); the
# tests use it to substitute a stub.
set -u

if [ "$#" -ne 0 ]; then
    echo "usage: scripts/ci_check.sh (no arguments)" >&2
    echo "Real-release checks are not part of CI; run 'churn-pipeline check-release' " \
         "separately, with authorisation." >&2
    exit 2
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 2
PY="${CI_PYTHON:-$REPO/.venv/bin/python}"
if [ ! -x "$PY" ]; then
    echo "ci_check: python interpreter not found: $PY" >&2
    exit 2
fi

PYCACHE="$(mktemp -d)" || exit 2
trap 'rm -rf "$PYCACHE"' EXIT
export PYTHONPYCACHEPREFIX="$PYCACHE"

step() {
    local name="$1"
    shift
    echo "== ci_check: $name"
    "$@"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "ci_check: FAILED at step '$name' (exit $rc)" >&2
        exit 1
    fi
}

step "compileall" "$PY" -m compileall -q src tests scripts
step "secret-scan" "$PY" scripts/secret_scan.py --root "$REPO"
step "pytest" "$PY" -m pytest -p no:cacheprovider -q --strict-markers -m "not real_data"

echo "== ci_check: all steps passed (real-data tests and real release checks not run)"
exit 0
