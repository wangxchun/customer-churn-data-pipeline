"""W5: repository hygiene. Reads only git metadata and the files git would commit;
never anything under data/ or docs/."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Every test that reads real data (R1). Adding one means updating this list on purpose.
REAL_DATA_TESTS = {
    "tests/test_features.py::test_real_master_feature_build",
    "tests/test_modeling.py::test_real_data_preprocessor_reproduces_train_parquet_exactly",
    "tests/test_real_dataset.py::test_checksum_and_size",
    "tests/test_real_dataset.py::test_verified_schema_and_counts",
    "tests/test_real_dataset.py::test_synthetic_generation_is_deterministic",
}


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


@pytest.fixture(scope="module")
def in_git_checkout():
    if git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        pytest.fail("repository hygiene checks need a git checkout")


@pytest.mark.parametrize("path", ["data/raw/customers.csv", "data/raw/x/y.parquet",
                                  "data/processed/models/registry/production.json",
                                  "docs/PHASE4_GUIDE_ZH-TW.md", ".venv/bin/python"])
def test_data_docs_and_venv_are_ignored(in_git_checkout, path):
    assert git("check-ignore", "-q", "--no-index", path).returncode == 0, path


@pytest.mark.parametrize("path", ["data/raw/.gitkeep", "data/processed/.gitkeep"])
def test_only_the_data_skeleton_is_tracked(in_git_checkout, path):
    assert git("check-ignore", "-q", "--no-index", path).returncode == 1


def test_no_data_or_docs_file_is_tracked(in_git_checkout):
    tracked = git("ls-files").stdout.splitlines()
    assert [p for p in tracked if p.startswith("data/")] == ["data/processed/.gitkeep",
                                                             "data/raw/.gitkeep"]
    assert not [p for p in tracked if p.startswith("docs/")]


def test_repository_has_no_secret_scan_findings(in_git_checkout, capsys):
    spec = importlib.util.spec_from_file_location("secret_scan",
                                                  REPO / "scripts" / "secret_scan.py")
    ss = importlib.util.module_from_spec(spec)
    sys.modules["secret_scan"] = ss
    spec.loader.exec_module(ss)
    r = ss.scan(REPO)
    # Locations and rule names only; the scanner never exposes matched values.
    assert r.findings == [], [f.render() for f in r.findings]
    assert r.scanned > 0


def _collect(marker_expr: str) -> set[str]:
    p = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
                        "--strict-markers", "--collect-only", "-m", marker_expr],
                       cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-2000:]
    return {line for line in p.stdout.splitlines() if "::" in line}


def test_real_data_marker_selects_exactly_the_known_real_data_tests():
    # Collection only: no test body runs. skipif conditions stat the real paths.
    assert _collect("real_data") == REAL_DATA_TESTS


def test_default_ci_selection_excludes_every_real_data_test():
    ci = _collect("not real_data")
    assert ci and not (ci & REAL_DATA_TESTS)
    assert not any(t.startswith("tests/test_real_dataset.py") for t in ci)
