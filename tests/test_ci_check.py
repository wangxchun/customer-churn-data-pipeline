"""W5: scripts/ci_check.sh orchestration, run against a stub interpreter (CI_PYTHON).

The stub records its arguments and exits with a per-step code, so these tests check the
step order, the exact pytest selection, stop-on-first-failure and the exit codes,
without running compileall, the scan or pytest for real (no recursion into pytest)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "ci_check.sh"

STUB = """#!/bin/sh
echo "$*" >> "$STUB_LOG"
case "$*" in
  *compileall*) exit "${RC_COMPILE:-0}" ;;
  *secret_scan.py*) exit "${RC_SCAN:-0}" ;;
  *pytest*) exit "${RC_PYTEST:-0}" ;;
esac
exit 97
"""


@pytest.fixture
def stub(tmp_path):
    py = tmp_path / "python"
    py.write_text(STUB)
    py.chmod(0o755)
    return py, tmp_path / "calls.log"


def run(stub, *args, **rcs):
    py, log = stub
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "CI_PYTHON": str(py),
           "STUB_LOG": str(log), **{k: str(v) for k, v in rcs.items()}}
    p = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env)
    calls = log.read_text().splitlines() if log.exists() else []
    return p, calls


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK)


def test_success_runs_the_three_steps_in_order_and_exits_0(stub):
    p, calls = run(stub)
    assert p.returncode == 0, p.stderr
    assert len(calls) == 3
    assert calls[0] == "-m compileall -q src tests scripts"
    assert calls[1] == f"scripts/secret_scan.py --root {REPO}"
    assert calls[2] == "-m pytest -p no:cacheprovider -q --strict-markers -m not real_data"
    assert "all steps passed" in p.stdout


def test_ci_never_runs_a_real_release_check(stub):
    _, calls = run(stub)
    assert not any("check-release" in c or "churn_pipeline" in c or "promote" in c
                   for c in calls)


@pytest.mark.parametrize("failing,expected_calls,step", [
    ("RC_COMPILE", 1, "compileall"),
    ("RC_SCAN", 2, "secret-scan"),
    ("RC_PYTEST", 3, "pytest"),
])
def test_a_failing_step_stops_the_run_and_exits_1(stub, failing, expected_calls, step):
    p, calls = run(stub, **{failing: 3})
    assert p.returncode == 1
    assert len(calls) == expected_calls
    assert f"FAILED at step '{step}' (exit 3)" in p.stderr


def test_secret_scan_findings_fail_ci(stub):
    p, _ = run(stub, RC_SCAN=1)
    assert p.returncode == 1 and "secret-scan" in p.stderr


@pytest.mark.parametrize("args", [["--with-release-check"], ["extra"]])
def test_any_argument_is_a_usage_error_and_runs_nothing(stub, args):
    p, calls = run(stub, *args)
    assert p.returncode == 2 and calls == []
    assert "check-release" in p.stderr and "authorisation" in p.stderr


def test_missing_interpreter_exits_2(tmp_path):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "CI_PYTHON": str(tmp_path / "no-such-python")}
    p = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env)
    assert p.returncode == 2 and "not found" in p.stderr


def test_bytecode_goes_to_a_temporary_directory_that_is_removed(stub, tmp_path):
    py, log = stub
    py.write_text(STUB.replace('echo "$*" >> "$STUB_LOG"',
                               'echo "$*" >> "$STUB_LOG"; echo "$PYTHONPYCACHEPREFIX" '
                               '>> "$STUB_LOG.prefix"'))
    p, _ = run(stub)
    prefixes = set(Path(str(log) + ".prefix").read_text().split())
    assert p.returncode == 0 and len(prefixes) == 1
    prefix = Path(prefixes.pop())
    assert REPO not in prefix.parents and not prefix.exists()
