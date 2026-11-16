"""TD-3: re-running build-features must never silently overwrite an existing split.

Policy (D2, D2-Q1 = iv, D2-Q2): if the new split differs from the existing
``split_assignments.parquet``, refuse before writing anything and report what
would change, as counts only (no raw customer IDs anywhere in the output).
There is deliberately no override path.

All tests use a synthetic master under ``tmp_path``; none touch ``data/``.

Exception types are looked up on the module (``F.SplitDriftError``) inside the
tests, and refusals are first caught as the pre-existing ``DataContractError``,
so that on the unfixed code these tests fail on behaviour ("DID NOT RAISE",
wrong exit code) rather than on an import error.
"""
import hashlib
import json

import pandas as pd
import pytest

import churn_pipeline.features as F
from churn_pipeline import cli
from churn_pipeline.config import Paths
from churn_pipeline.features import ID_COLUMN, DataContractError, SplitConfig, run_features
from test_features import make_master

FEATURE_FILES = {
    "train.parquet", "validation.parquet", "test.parquet", "split_assignments.parquet",
    "preprocessor.joblib", "feature_report.json", "feature_report.md",
}
# Every synthetic ID ends in one of these; none can appear in a hex hash or a count.
ID_MARKERS = ("ABCDE", "ZZZZZ")


def setup_built(tmp_path, master: pd.DataFrame | None = None) -> Paths:
    """A data dir with a synthetic master and one completed feature build."""
    paths = Paths(tmp_path / "data")
    paths.curated.mkdir(parents=True)
    (master if master is not None else make_master()).to_parquet(
        paths.curated / "customer_master.parquet", index=False)
    run_features(paths)
    return paths


def replace_master(paths: Paths, master: pd.DataFrame) -> None:
    master.to_parquet(paths.curated / "customer_master.parquet", index=False)


def snapshot(paths: Paths) -> dict:
    """sha256 and mtime of every file under the data dir (curated + features)."""
    return {
        str(p.relative_to(paths.data_dir)): (hashlib.sha256(p.read_bytes()).hexdigest(),
                                             p.stat().st_mtime_ns)
        for p in sorted(paths.data_dir.rglob("*")) if p.is_file()
    }


def assert_no_raw_ids(*texts: str) -> None:
    for text in texts:
        for marker in ID_MARKERS:
            assert marker not in text, f"raw customer ID leaked into output: ...{marker}"


def drop_one(master: pd.DataFrame) -> pd.DataFrame:
    return master[master[ID_COLUMN] != "0123-ABCDE"].reset_index(drop=True)


def add_one(master: pd.DataFrame) -> pd.DataFrame:
    extra = master.iloc[[0]].copy()
    extra[ID_COLUMN] = pd.array(["9999-ZZZZZ"], dtype="string")
    return pd.concat([master, extra], ignore_index=True)


# --- Refusal: nothing is written -----------------------------------------------------------
def test_removed_customer_is_refused_with_counts_and_nothing_changes(tmp_path):
    master = make_master()
    paths = setup_built(tmp_path, master)
    replace_master(paths, drop_one(master))
    before = snapshot(paths)

    with pytest.raises(DataContractError) as exc:
        run_features(paths)

    assert type(exc.value).__name__ == "SplitDriftError"
    assert snapshot(paths) == before  # same files, same bytes, same mtimes
    r = exc.value.report
    assert r["identical"] is False
    assert r["removed"] == 1 and r["added"] == 0
    assert "membership_changed" in r["change_types"]
    assert r["n_old"] == 400 and r["n_new"] == 399
    assert r["unchanged"] + r["moved"] == 399
    assert sum(r["moves"].values()) == r["moved"]
    if r["moved"]:
        assert "assignments_moved" in r["change_types"]
    assert r["old_split_content_sha256"] != r["new_split_content_sha256"]
    assert {w["path"] for w in r["planned_writes"]} == FEATURE_FILES
    assert {w["action"] for w in r["planned_writes"]} == {"overwrite"}


def test_added_customer_is_refused_with_counts(tmp_path):
    master = make_master()
    paths = setup_built(tmp_path, master)
    replace_master(paths, add_one(master))
    before = snapshot(paths)

    with pytest.raises(DataContractError) as exc:
        run_features(paths)

    assert type(exc.value).__name__ == "SplitDriftError"
    assert snapshot(paths) == before
    r = exc.value.report
    assert r["added"] == 1 and r["removed"] == 0
    assert r["n_new"] == 401
    assert "membership_changed" in r["change_types"]


def test_different_seed_is_refused_and_nothing_changes(tmp_path):
    paths = setup_built(tmp_path)
    before = snapshot(paths)

    with pytest.raises(DataContractError) as exc:
        run_features(paths, SplitConfig(seed=7))

    assert type(exc.value).__name__ == "SplitDriftError"
    assert snapshot(paths) == before
    r = exc.value.report
    assert r["added"] == r["removed"] == 0
    assert r["moved"] > 0 and r["change_types"] == ["assignments_moved"]
    assert r["split_counts_old"] == r["split_counts_new"]  # same sizes, different members


@pytest.mark.parametrize("damage", ["garbage_bytes", "missing_split_column", "duplicate_ids"])
def test_unreadable_existing_split_is_refused_and_nothing_changes(tmp_path, damage):
    paths = setup_built(tmp_path)
    target = paths.features / "split_assignments.parquet"
    if damage == "garbage_bytes":
        target.write_bytes(b"not a parquet file")
    elif damage == "missing_split_column":
        pd.read_parquet(target)[[ID_COLUMN]].to_parquet(target, index=False)
    else:
        old = pd.read_parquet(target)
        pd.concat([old, old.iloc[[0]]], ignore_index=True).to_parquet(target, index=False)
    before = snapshot(paths)

    with pytest.raises(DataContractError, match="split_assignments.parquet") as exc:
        run_features(paths)

    assert type(exc.value).__name__ != "SplitDriftError"  # a different, explicit failure
    assert snapshot(paths) == before
    assert_no_raw_ids(str(exc.value))


# --- No raw customer IDs in any output (D2-Q2) ---------------------------------------------
@pytest.mark.parametrize("change", ["remove", "add"])
def test_refusal_never_outputs_raw_customer_ids(tmp_path, capsys, change):
    master = make_master()
    paths = setup_built(tmp_path, master)
    replace_master(paths, drop_one(master) if change == "remove" else add_one(master))
    files_before = set(snapshot(paths))
    capsys.readouterr()

    with pytest.raises(DataContractError) as exc:
        run_features(paths)
    assert type(exc.value).__name__ == "SplitDriftError"
    assert_no_raw_ids(str(exc.value), repr(exc.value),
                      json.dumps(exc.value.report, default=str))

    rc = cli.main(["--data-dir", str(paths.data_dir), "build-features"])
    out, err = capsys.readouterr()
    assert rc == 2
    assert_no_raw_ids(out, err)
    assert set(snapshot(paths)) == files_before  # no diff-report file (or any file) appeared


# --- CLI ----------------------------------------------------------------------------------
def test_cli_refusal_exits_2_with_report_on_stderr(tmp_path, capsys):
    master = make_master()
    paths = setup_built(tmp_path, master)
    replace_master(paths, drop_one(master))
    before = snapshot(paths)
    capsys.readouterr()

    rc = cli.main(["--data-dir", str(paths.data_dir), "build-features"])
    out, err = capsys.readouterr()

    assert rc == 2
    assert "Features written" not in out
    assert "refused" in err and "removed: 1" in err
    for name in FEATURE_FILES:
        assert name in err  # what would have been overwritten
    assert snapshot(paths) == before


def test_cli_success_lists_create_then_overwrite(tmp_path, capsys):
    paths = Paths(tmp_path / "data")
    paths.curated.mkdir(parents=True)
    make_master().to_parquet(paths.curated / "customer_master.parquet", index=False)

    assert cli.main(["--data-dir", str(paths.data_dir), "build-features"]) == 0
    first, _ = capsys.readouterr()
    assert "create: split_assignments.parquet" in first

    assert cli.main(["--data-dir", str(paths.data_dir), "build-features"]) == 0
    second, _ = capsys.readouterr()
    assert "overwrite: split_assignments.parquet" in second
    assert "Features written" in second


# --- Accepted re-runs ----------------------------------------------------------------------
def test_identical_rerun_succeeds_and_keeps_content_hash(tmp_path):
    paths = setup_built(tmp_path)
    old = pd.read_parquet(paths.features / "split_assignments.parquet")
    run_features(paths)
    new = pd.read_parquet(paths.features / "split_assignments.parquet")
    pd.testing.assert_frame_equal(old, new)


def test_dtype_only_difference_is_not_treated_as_drift(tmp_path):
    """Guard against over-correction: object vs string dtype is not a split change."""
    paths = setup_built(tmp_path)
    target = paths.features / "split_assignments.parquet"
    old = pd.read_parquet(target)
    old.astype({ID_COLUMN: object, "split": object}).to_parquet(target, index=False)
    run_features(paths)  # must not raise


# --- Unit: compare_assignments / plan_feature_write ----------------------------------------
def frame(pairs, dtype="string"):
    return pd.DataFrame({ID_COLUMN: pd.array([p[0] for p in pairs], dtype=dtype),
                         "split": pd.array([p[1] for p in pairs], dtype=dtype)})


def test_compare_assignments_counts_moves_additions_and_removals():
    old = frame([("A", "train"), ("B", "train"), ("C", "test"), ("D", "validation")])
    new = frame([("A", "train"), ("B", "test"), ("C", "train"), ("E", "train")])
    r = F.compare_assignments(old, new)
    assert r["identical"] is False
    assert r["unchanged"] == 1 and r["moved"] == 2
    assert r["moves"] == {"test->train": 1, "train->test": 1}
    assert r["added"] == 1 and r["removed"] == 1
    assert r["change_types"] == ["membership_changed", "assignments_moved"]
    assert r["split_counts_old"] == {"train": 2, "validation": 1, "test": 1}
    assert r["split_counts_new"] == {"train": 3, "validation": 0, "test": 1}
    assert_no_raw_ids(json.dumps(r))
    for raw in "ABCDE":
        assert raw not in json.dumps(r["moves"]) + str(r["added"]) + str(r["removed"])


def test_compare_assignments_ignores_dtype_and_row_order():
    pairs = [("A", "train"), ("B", "test"), ("C", "validation")]
    r = F.compare_assignments(frame(pairs, dtype=object), frame(pairs[::-1], dtype="string"))
    assert r["identical"] is True and r["moved"] == 0 and r["change_types"] == []


def test_plan_feature_write_reports_create_and_overwrite_without_touching_disk(tmp_path):
    out = tmp_path / "features"
    plan = F.plan_feature_write(out)
    assert not out.exists()  # planning never creates the directory
    assert {w["path"] for w in plan} == FEATURE_FILES
    assert {w["action"] for w in plan} == {"create"}

    out.mkdir()
    (out / "train.parquet").write_bytes(b"x")
    actions = {w["path"]: w["action"] for w in F.plan_feature_write(out)}
    assert actions["train.parquet"] == "overwrite" and actions["test.parquet"] == "create"
