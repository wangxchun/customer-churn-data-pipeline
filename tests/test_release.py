"""W1 (Stage B1): holdout guard, release registry, gates, promote and rollback.

Every test here is a synthetic-fixture test. The master, split assignments,
model, run directory, registry and HMAC key are all created under pytest tmp
directories. Every customer ID is a synthetic ``NNNN-ABCDE`` ID. Nothing reads
or writes ``data/``, the real champion, or the real registry (W1-Q10).
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pandas as pd
import pytest

from churn_pipeline import cli, inference
from churn_pipeline import release as rel
from churn_pipeline.config import PROJECT_ROOT, Paths
from churn_pipeline.features import FEATURE_COLUMNS, ID_COLUMN, build_features, split_content_hash
from test_inference import INPUT_COLUMNS, edit_manifest, fit_model, synthetic_master, write_run

KEY = bytes(range(32))                 # synthetic HMAC key, never a real one
OTHER_KEY = bytes(range(100, 132))
RUN_ID = "20990101T000000Z-syn001"
MASTER_SHA = "c" * 64                  # a recorded value; W1 copies it, never recomputes it


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def master() -> pd.DataFrame:
    return synthetic_master()


@pytest.fixture(scope="module")
def model(master):
    return fit_model(master)


@pytest.fixture(scope="module")
def assignments(master) -> pd.DataFrame:
    return build_features(master).assignments


@pytest.fixture(scope="module")
def test_ids(assignments) -> list[str]:
    return sorted(assignments.loc[assignments["split"] == "test", ID_COLUMN].astype(str))


def make_env(root: Path, model, assignments, *, dirty: bool = False, access_log=None) -> Paths:
    """A data dir with a synthetic split file and one synthetic Phase 3 run."""
    paths = Paths(root / "data")
    paths.features.mkdir(parents=True)
    assignments.to_parquet(paths.features / "split_assignments.parquet", index=False)
    run_dir = write_run(paths.models, model, run_id=RUN_ID)
    edit_manifest(run_dir,
                  data={"split_content_sha256": split_content_hash(assignments),
                        "customer_master_sha256": MASTER_SHA,
                        "split_assignments_sha256": "s" * 64,
                        "feature_report_sha256": "f" * 64},
                  code={"commit": "0" * 40, "dirty": dirty, "n_changed_paths": int(dirty)},
                  params={"fn_cost": 5.0, "fp_cost": 1.0, "seed": 42})
    if access_log is not None:
        (run_dir / "test_access.json").write_text(
            access_log if isinstance(access_log, str) else json.dumps(access_log))
    return paths


@pytest.fixture
def env(tmp_path, model, assignments) -> Paths:
    return make_env(tmp_path, model, assignments)


@pytest.fixture
def released(env) -> tuple[Paths, str]:
    manifest = rel.create_release(env, RUN_ID, KEY, release_id="rel-syn-1")
    return env, manifest["release_id"]


def release_dir(paths: Paths, release_id: str) -> Path:
    return rel.registry_dir(paths) / "releases" / release_id


def tree_hashes(directory: Path) -> dict:
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


@pytest.fixture
def parquet_spy(monkeypatch):
    calls = []
    original = pd.read_parquet

    def spy(path, *args, **kwargs):
        calls.append((Path(path).name, kwargs.get("columns")))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", spy)
    return calls


@pytest.fixture
def no_data_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("this operation must not read data files")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)


@pytest.fixture
def scoring_spy(monkeypatch):
    calls = []
    original = inference.predict_batch

    def spy(m, df):
        calls.append(len(df))
        return original(m, df)
    monkeypatch.setattr(inference, "predict_batch", spy)
    return calls


def assert_no_ids_or_hmacs(text: str, ids: list[str], *, hmacs: bool = True) -> None:
    for cid in ids:
        assert cid not in text, "plaintext holdout ID leaked"
        if hmacs:
            assert rel.id_hmac(KEY, cid) not in text, "holdout ID HMAC leaked"


def write_key_file(path: Path, content: str, mode: int = 0o600) -> Path:
    path.write_text(content)
    os.chmod(path, mode)
    return path


# ---------------------------------------------------------------------------
# HMAC key management (W1-Q4)
# ---------------------------------------------------------------------------
def test_guard_key_loads_from_env_file_outside_repo(tmp_path):
    f = write_key_file(tmp_path / "guard.key", KEY.hex() + "\n")
    assert rel.load_guard_key({rel.GUARD_KEY_ENV: str(f)}) == KEY


@pytest.mark.parametrize("case", ["unset", "empty", "missing", "inside_repo", "directory",
                                  "group_readable", "not_hex", "too_short"])
def test_unusable_guard_key_fails_closed(tmp_path, case):
    env = {rel.GUARD_KEY_ENV: str(tmp_path / "guard.key")}
    if case == "unset":
        env = {}
    elif case == "empty":
        env = {rel.GUARD_KEY_ENV: ""}
    elif case == "inside_repo":
        env = {rel.GUARD_KEY_ENV: str(PROJECT_ROOT / "never-created.key")}  # path check only
    elif case == "directory":
        (tmp_path / "guard.key").mkdir()
    elif case == "group_readable":
        write_key_file(tmp_path / "guard.key", KEY.hex(), mode=0o640)
    elif case == "not_hex":
        write_key_file(tmp_path / "guard.key", "zz" * 32)
    elif case == "too_short":
        write_key_file(tmp_path / "guard.key", KEY.hex()[:62])
    with pytest.raises(rel.GuardKeyError) as exc:
        rel.load_guard_key(env)
    assert KEY.hex() not in str(exc.value)


def test_key_id_identifies_the_key_without_revealing_it():
    assert rel.key_id(KEY) != rel.key_id(OTHER_KEY)
    assert len(rel.key_id(KEY)) == 16 and KEY.hex() not in rel.key_id(KEY)
    assert rel.id_hmac(KEY, "0001-ABCDE") != rel.id_hmac(OTHER_KEY, "0001-ABCDE")
    assert rel.id_hmac(KEY, "0001-ABCDE") != hashlib.sha256(b"0001-ABCDE").hexdigest()


# ---------------------------------------------------------------------------
# Release creation
# ---------------------------------------------------------------------------
def test_release_creates_guard_and_manifest_and_passes_every_gate(released, test_ids):
    paths, release_id = released
    d = release_dir(paths, release_id)
    assert sorted(p.name for p in d.iterdir()) == ["holdout_guard.json", "release.json"]

    m = json.loads((d / "release.json").read_text())
    assert m["release_id"] == release_id and m["run_id"] == RUN_ID
    assert set(m["gates"]) == {"G1", "G2", "G3", "G4", "G5", "G7", "G8", "G9"}  # G6 removed
    assert all(g["status"] == "pass" for g in m["gates"].values())
    assert m["decision"] == {"threshold": 0.3, "fn_cost": 5.0, "fp_cost": 1.0}
    assert m["data"]["customer_master_sha256"] == MASTER_SHA
    assert m["data"]["customer_master_revalidated_at_release"] is False
    assert m["code"] == {"commit": "0" * 40, "dirty": False, "n_changed_paths": 0}
    assert m["provenance_waiver"] is None
    assert m["holdout_guard"]["n_ids"] == len(test_ids)
    assert m["holdout_guard"]["sha256"] == hashlib.sha256(
        (d / "holdout_guard.json").read_bytes()).hexdigest()
    assert m["historical_test_evaluation"] == {"attempts": 0, "output_dirs": [],
                                               "source": "test_access.json (absent)"}
    assert_no_ids_or_hmacs((d / "release.json").read_text(), test_ids)
    assert [e["event"] for e in rel.read_audit_log(paths)] == ["release_created"]


def test_guard_holds_keyed_hmacs_only_with_0600_permissions(released, test_ids):
    paths, release_id = released
    path = release_dir(paths, release_id) / "holdout_guard.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    g = json.loads(path.read_text())
    assert g["hash_scheme"] == "hmac-sha256/v1" and g["key_id"] == rel.key_id(KEY)
    assert g["id_hmacs"] == sorted(rel.id_hmac(KEY, i) for i in test_ids)
    assert g["n_ids"] == len(test_ids)
    assert_no_ids_or_hmacs(path.read_text(), test_ids, hmacs=False)
    assert KEY.hex() not in path.read_text()


def test_release_reads_only_the_two_split_columns_once(env, parquet_spy):
    rel.create_release(env, RUN_ID, KEY, release_id="rel-syn-1")
    assert parquet_spy == [("split_assignments.parquet", ["customer_id", "split"])]


def test_release_does_not_modify_the_run_directory(env):
    run_dir = env.models / RUN_ID
    before = tree_hashes(run_dir)
    rel.create_release(env, RUN_ID, KEY, release_id="rel-syn-1")
    assert tree_hashes(run_dir) == before


def test_release_directory_cannot_be_overwritten(released):
    paths, release_id = released
    before = tree_hashes(release_dir(paths, release_id))
    with pytest.raises(FileExistsError):
        rel.create_release(paths, RUN_ID, KEY, release_id=release_id)
    assert tree_hashes(release_dir(paths, release_id)) == before


# --- Gate failures at release time ---------------------------------------------------------
def assert_refused(paths, gate, fn, *args, **kwargs):
    with pytest.raises(rel.GateFailedError) as exc:
        fn(*args, **kwargs)
    assert exc.value.gate == gate
    releases = rel.registry_dir(paths) / "releases"
    assert not releases.exists() or not any(p.name.startswith("rel-") for p in releases.iterdir())
    assert rel.read_audit_log(paths)[-1]["event"] == "gate_failed"
    assert rel.read_audit_log(paths)[-1]["gate"] == gate
    return exc.value


def test_g1_tampered_model_is_refused_before_the_split_is_read(env, parquet_spy):
    with (env.models / RUN_ID / "model.joblib").open("ab") as f:
        f.write(b"tampered")
    assert_refused(env, "G1", rel.create_release, env, RUN_ID, KEY)
    assert parquet_spy == []


def test_g2_sklearn_version_mismatch_is_refused(env):
    run_dir = env.models / RUN_ID
    m = json.loads((run_dir / "manifest.json").read_text())
    m["environment"]["packages"]["scikit-learn"] = "0.0.1"
    (run_dir / "manifest.json").write_text(json.dumps(m))
    assert_refused(env, "G2", rel.create_release, env, RUN_ID, KEY)


def test_g3_incompatible_schema_is_refused(tmp_path, master, assignments):
    paths = make_env(tmp_path, fit_model(master, columns=FEATURE_COLUMNS[::-1]), assignments)
    err = assert_refused(paths, "G3", rel.create_release, paths, RUN_ID, KEY)
    assert "schema" in err.reason


def test_g4_split_mismatch_is_refused(env):
    edit_manifest(env.models / RUN_ID, data={"split_content_sha256": "0" * 64,
                                             "customer_master_sha256": MASTER_SHA})
    assert_refused(env, "G4", rel.create_release, env, RUN_ID, KEY)


def test_g4_missing_split_hash_in_manifest_is_refused(env):
    edit_manifest(env.models / RUN_ID, data={"customer_master_sha256": MASTER_SHA})
    assert_refused(env, "G4", rel.create_release, env, RUN_ID, KEY)


def test_g5_canary_contract_violation_is_refused(env, monkeypatch):
    original = inference.predict_batch

    def drops_a_row(m, df):
        r = original(m, df)
        return inference.InferenceResult(r.predictions.iloc[1:], r.rejects, r.ignored_columns)
    monkeypatch.setattr(inference, "predict_batch", drops_a_row)
    assert_refused(env, "G5", rel.create_release, env, RUN_ID, KEY)


def test_g7_latency_budget_is_enforced(env):
    assert_refused(env, "G7", rel.create_release, env, RUN_ID, KEY, latency_budget_s=0.0)


def test_g8_dirty_code_without_waiver_is_refused(tmp_path, model, assignments):
    paths = make_env(tmp_path, model, assignments, dirty=True)
    assert_refused(paths, "G8", rel.create_release, paths, RUN_ID, KEY)


def test_g8_waiver_is_recorded_and_dirty_flag_is_kept(tmp_path, model, assignments):
    paths = make_env(tmp_path, model, assignments, dirty=True)
    run_manifest_before = (paths.models / RUN_ID / "manifest.json").read_text()
    m = rel.create_release(paths, RUN_ID, KEY, waiver_reason="synthetic test waiver",
                           release_id="rel-syn-w")
    assert m["code"]["dirty"] is True                       # never rewritten to clean
    w = m["provenance_waiver"]
    assert w["reason"] == "synthetic test waiver" and w["status"] == "active"
    assert {"scope", "limitations", "date", "completion_condition"} <= set(w)
    assert m["gates"]["G8"]["status"] == "pass (waived)"
    assert (paths.models / RUN_ID / "manifest.json").read_text() == run_manifest_before


def test_blank_waiver_reason_is_rejected(tmp_path, model, assignments):
    paths = make_env(tmp_path, model, assignments, dirty=True)
    with pytest.raises(rel.ReleaseError):
        rel.create_release(paths, RUN_ID, KEY, waiver_reason="   ")


def test_missing_run_is_refused(env):
    with pytest.raises(rel.ReleaseError):
        rel.create_release(env, "20990101T000000Z-nope00", KEY)


# --- historical_test_evaluation (W1-Q8) -----------------------------------------------------
def test_access_log_metadata_is_recorded_without_opening_output_dir(
        tmp_path, model, assignments, monkeypatch):
    log = {"reads": [{"attempt": 1, "force": False, "at": "2099-01-01T00:00:00+00:00",
                      "output_dir": "evaluation-01"}]}
    paths = make_env(tmp_path, model, assignments, access_log=log)
    output_dir = paths.models / RUN_ID / "evaluation-01"
    output_dir.mkdir()
    (output_dir / "test_metrics.json").write_text("{}")
    real_listdir, real_scandir = os.listdir, os.scandir

    def guarded(fn):
        def wrapper(p=".", *a, **k):
            assert "evaluation-01" not in str(p), "output_dir must not be enumerated"
            return fn(p, *a, **k)
        return wrapper
    monkeypatch.setattr(os, "listdir", guarded(real_listdir))
    monkeypatch.setattr(os, "scandir", guarded(real_scandir))
    os.chmod(output_dir, 0)  # any attempt to open it would fail
    try:
        m = rel.create_release(paths, RUN_ID, KEY, release_id="rel-syn-h")
    finally:
        os.chmod(output_dir, 0o700)
    assert m["historical_test_evaluation"] == {"attempts": 1, "output_dirs": ["evaluation-01"],
                                               "source": "test_access.json"}


def test_crashed_read_without_output_dir_is_recorded_as_none(tmp_path, model, assignments):
    """evaluation.py logs a read *before* touching test rows and adds output_dir only
    on success, so a crashed read legitimately has no output_dir."""
    paths = make_env(tmp_path, model, assignments,
                     access_log={"reads": [{"attempt": 1, "force": False, "at": "x"}]})
    m = rel.create_release(paths, RUN_ID, KEY, release_id="rel-syn-c")
    assert m["historical_test_evaluation"]["attempts"] == 1
    assert m["historical_test_evaluation"]["output_dirs"] == [None]


@pytest.mark.parametrize("bad", ["not json", json.dumps({"reads": "x"}), json.dumps([]),
                                 json.dumps({"reads": [{"attempt": 1, "output_dir": 5}]})])
def test_malformed_access_log_refuses_the_release(tmp_path, model, assignments, bad):
    paths = make_env(tmp_path, model, assignments, access_log=bad)
    with pytest.raises(rel.ReleaseError, match="test_access.json"):
        rel.create_release(paths, RUN_ID, KEY)


# ---------------------------------------------------------------------------
# Loading a release: guard verification fails closed (G9)
# ---------------------------------------------------------------------------
def rewrite_guard(paths, release_id, **changes):
    """Edit the guard and re-record its sha in release.json, so only the edited field fails."""
    d = release_dir(paths, release_id)
    g = json.loads((d / "holdout_guard.json").read_text())
    g.update(changes)
    (d / "holdout_guard.json").write_text(json.dumps(g))
    m = json.loads((d / "release.json").read_text())
    m["holdout_guard"]["sha256"] = hashlib.sha256((d / "holdout_guard.json").read_bytes()).hexdigest()
    (d / "release.json").write_text(json.dumps(m))


@pytest.mark.parametrize("damage", [
    "guard_missing", "guard_bytes_tampered", "guard_schema_version", "hash_scheme",
    "model_sha256", "split_content_sha256", "n_ids", "id_hmacs_not_hex",
    "release_json_missing", "release_id_mismatch", "feature_schema_sha256",
    "run_split_hash_changed",
])
def test_invalid_release_or_guard_fails_closed_without_scoring(released, damage, scoring_spy):
    paths, release_id = released
    d = release_dir(paths, release_id)
    if damage == "guard_missing":
        (d / "holdout_guard.json").unlink()
    elif damage == "guard_bytes_tampered":
        with (d / "holdout_guard.json").open("a") as f:
            f.write(" ")
    elif damage == "guard_schema_version":
        rewrite_guard(paths, release_id, guard_schema_version="999")
    elif damage == "hash_scheme":
        rewrite_guard(paths, release_id, hash_scheme="sha256/plain")
    elif damage == "model_sha256":
        rewrite_guard(paths, release_id, model_sha256="0" * 64)
    elif damage == "split_content_sha256":
        rewrite_guard(paths, release_id, split_content_sha256="0" * 64)
    elif damage == "n_ids":
        rewrite_guard(paths, release_id, n_ids=1)
    elif damage == "id_hmacs_not_hex":
        g = json.loads((d / "holdout_guard.json").read_text())
        rewrite_guard(paths, release_id, id_hmacs=["not-a-hash"] * g["n_ids"])
    elif damage == "release_json_missing":
        (d / "release.json").unlink()
    elif damage == "release_id_mismatch":
        m = json.loads((d / "release.json").read_text())
        m["release_id"] = "rel-other"
        (d / "release.json").write_text(json.dumps(m))
    elif damage == "feature_schema_sha256":
        m = json.loads((d / "release.json").read_text())
        m["feature_schema_sha256"] = "0" * 64
        (d / "release.json").write_text(json.dumps(m))
    else:
        edit_manifest(paths.models / RUN_ID, data={"split_content_sha256": "0" * 64,
                                                   "customer_master_sha256": MASTER_SHA})
    with pytest.raises(rel.ReleaseIntegrityError):
        rel.load_release(paths, release_id, KEY)
    assert scoring_spy == []


def test_wrong_key_fails_closed(released, scoring_spy):
    paths, release_id = released
    with pytest.raises(rel.GuardKeyError):
        rel.load_release(paths, release_id, OTHER_KEY)
    assert scoring_spy == []


def test_tampered_model_after_release_fails_closed(released):
    paths, release_id = released
    with (paths.models / RUN_ID / "model.joblib").open("ab") as f:
        f.write(b"x")
    with pytest.raises(inference.ModelIntegrityError):
        rel.load_release(paths, release_id, KEY)


# ---------------------------------------------------------------------------
# Release prediction: holdout IDs reject the whole batch (W1-Q1)
# ---------------------------------------------------------------------------
def non_holdout_rows(master, test_ids, n=20) -> pd.DataFrame:
    rows = master[~master[ID_COLUMN].astype(str).isin(test_ids)]
    return rows[INPUT_COLUMNS].head(n).reset_index(drop=True)


def test_one_holdout_id_rejects_the_whole_batch_without_scoring(
        released, master, test_ids, scoring_spy):
    paths, release_id = released
    r = rel.load_release(paths, release_id, KEY)
    model_calls = []
    r.model.model.predict_proba = lambda X: model_calls.append(1)  # instance-level spy
    batch = pd.concat([non_holdout_rows(master, test_ids),
                       master[master[ID_COLUMN] == test_ids[0]][INPUT_COLUMNS]],
                      ignore_index=True)

    with pytest.raises(inference.InferenceInputError) as exc:
        rel.predict_batch(r, batch)
    assert exc.value.level == "batch"
    assert exc.value.reasons == [("customer_id", "holdout_customer_id")]
    assert "1 input row" in str(exc.value)
    assert_no_ids_or_hmacs(str(exc.value) + repr(exc.value), test_ids)
    assert scoring_spy == [] and model_calls == []


def test_several_holdout_ids_are_counted_not_listed(released, master, test_ids):
    paths, release_id = released
    r = rel.load_release(paths, release_id, KEY)
    batch = master[master[ID_COLUMN].isin(test_ids[:3])][INPUT_COLUMNS]
    with pytest.raises(inference.InferenceInputError) as exc:
        rel.predict_batch(r, batch)
    assert "3 input rows" in str(exc.value)
    assert_no_ids_or_hmacs(str(exc.value), test_ids)


def test_holdout_check_runs_even_when_the_row_is_otherwise_invalid(released, master, test_ids):
    paths, release_id = released
    r = rel.load_release(paths, release_id, KEY)
    row = master[master[ID_COLUMN] == test_ids[0]][INPUT_COLUMNS].copy()
    row["contract"] = "Three year"
    with pytest.raises(inference.InferenceInputError) as exc:
        rel.predict_batch(r, row)
    assert exc.value.reasons == [("customer_id", "holdout_customer_id")]


def test_clean_batch_is_scored_exactly_like_td5(released, master, test_ids):
    paths, release_id = released
    r = rel.load_release(paths, release_id, KEY)
    batch = non_holdout_rows(master, test_ids)
    got = rel.predict_batch(r, batch)
    want = inference.predict_batch(inference.load_model(paths.models / RUN_ID), batch)
    pd.testing.assert_frame_equal(got.predictions, want.predictions)
    pd.testing.assert_frame_equal(got.rejects, want.rejects)


def test_predict_one_rejects_a_holdout_record_and_scores_others(released, master, test_ids):
    paths, release_id = released
    r = rel.load_release(paths, release_id, KEY)
    holdout = master[master[ID_COLUMN] == test_ids[0]][INPUT_COLUMNS].iloc[0].to_dict()
    with pytest.raises(inference.InferenceInputError) as exc:
        rel.predict_one(r, holdout)
    assert exc.value.level == "record"
    assert_no_ids_or_hmacs(str(exc.value), test_ids)
    clean = non_holdout_rows(master, test_ids, 1).iloc[0].to_dict()
    assert rel.predict_one(r, clean)["customer_id"] == clean[ID_COLUMN]


def test_td5_batch_errors_still_surface_through_the_release(released, master, test_ids):
    paths, release_id = released
    r = rel.load_release(paths, release_id, KEY)
    with pytest.raises(inference.InferenceInputError) as exc:
        rel.predict_batch(r, non_holdout_rows(master, test_ids).drop(columns=["contract"]))
    assert exc.value.level == "batch"


def test_loading_and_predicting_read_no_data_files(released, master, test_ids, no_data_reads):
    paths, release_id = released
    r = rel.load_release(paths, release_id, KEY)
    rel.predict_batch(r, non_holdout_rows(master, test_ids))


def test_loaded_release_does_not_expose_the_key_in_repr(released):
    paths, release_id = released
    assert KEY.hex() not in repr(rel.load_release(paths, release_id, KEY))


# ---------------------------------------------------------------------------
# check-release
# ---------------------------------------------------------------------------
def test_check_release_passes_without_reading_split_or_changing_the_release(
        released, no_data_reads):
    paths, release_id = released
    before = tree_hashes(release_dir(paths, release_id))
    gates = rel.check_release(paths, release_id, KEY)
    assert set(gates) == {"G1", "G2", "G3", "G5", "G7", "G8", "G9"}  # no G4 (W1-Q3)
    assert tree_hashes(release_dir(paths, release_id)) == before


def test_check_release_fails_on_a_model_changed_after_release(released):
    paths, release_id = released
    with (paths.models / RUN_ID / "model.joblib").open("ab") as f:
        f.write(b"x")
    with pytest.raises(rel.GateFailedError) as exc:
        rel.check_release(paths, release_id, KEY)
    assert exc.value.gate == "G1"
    assert rel.read_audit_log(paths)[-1]["event"] == "gate_failed"


def test_check_release_rechecks_the_waiver(tmp_path, model, assignments):
    paths = make_env(tmp_path, model, assignments, dirty=True)
    rel.create_release(paths, RUN_ID, KEY, waiver_reason="synthetic", release_id="rel-syn-w")
    d = release_dir(paths, "rel-syn-w")
    m = json.loads((d / "release.json").read_text())
    m["provenance_waiver"] = None
    (d / "release.json").write_text(json.dumps(m))
    with pytest.raises(rel.GateFailedError) as exc:
        rel.check_release(paths, "rel-syn-w", KEY)
    assert exc.value.gate == "G8"


# ---------------------------------------------------------------------------
# promote / rollback
# ---------------------------------------------------------------------------
def pointer(paths):
    p = rel.registry_dir(paths) / "production.json"
    return json.loads(p.read_text()) if p.exists() else None


def two_releases(paths):
    rel.create_release(paths, RUN_ID, KEY, release_id="rel-a")
    rel.create_release(paths, RUN_ID, KEY, release_id="rel-b")


def test_promote_then_promote_again_tracks_previous(env):
    two_releases(env)
    rel.promote(env, "rel-a", KEY)
    assert pointer(env)["release_id"] == "rel-a" and pointer(env)["previous_release_id"] is None
    rel.promote(env, "rel-b", KEY)
    assert pointer(env)["release_id"] == "rel-b" and pointer(env)["previous_release_id"] == "rel-a"
    assert rel.load_production(env, KEY).release_id == "rel-b"
    assert [e["event"] for e in rel.read_audit_log(env)][-2:] == ["promoted", "promoted"]


def test_promote_reads_no_data_files(env, monkeypatch):
    two_releases(env)

    def forbidden(*args, **kwargs):
        raise AssertionError("promote must not read data files")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    rel.promote(env, "rel-a", KEY)


def test_failed_post_deploy_canary_rolls_the_pointer_back(env, monkeypatch):
    two_releases(env)
    rel.promote(env, "rel-a", KEY)
    before = pointer(env)
    calls = {"n": 0}
    original = rel.run_canary

    def fails_after_deploy(r, latency_budget_s):
        calls["n"] += 1
        if calls["n"] == 2:  # 1st call: pre-promotion gates; 2nd: post-deploy smoke
            raise rel.GateFailedError("G5", "synthetic post-deploy failure")
        return original(r, latency_budget_s)
    monkeypatch.setattr(rel, "run_canary", fails_after_deploy)

    with pytest.raises(rel.PromotionError):
        rel.promote(env, "rel-b", KEY)
    assert pointer(env) == before
    events = rel.read_audit_log(env)
    assert events[-1]["event"] == "rollback" and events[-1]["release_id"] == "rel-a"
    assert events[-1]["reason"] == "post_deploy_smoke_failed"


def test_failed_post_deploy_canary_on_first_promotion_removes_the_pointer(env, monkeypatch):
    two_releases(env)
    calls = {"n": 0}
    original = rel.run_canary

    def fails_after_deploy(r, latency_budget_s):
        calls["n"] += 1
        if calls["n"] == 2:
            raise rel.GateFailedError("G5", "synthetic post-deploy failure")
        return original(r, latency_budget_s)
    monkeypatch.setattr(rel, "run_canary", fails_after_deploy)
    with pytest.raises(rel.PromotionError):
        rel.promote(env, "rel-a", KEY)
    assert pointer(env) is None


def test_promote_refuses_a_release_that_fails_its_gates(env):
    two_releases(env)
    rel.promote(env, "rel-a", KEY)
    before = pointer(env)
    with pytest.raises(rel.GateFailedError):
        rel.promote(env, "rel-b", KEY, latency_budget_s=0.0)
    assert pointer(env) == before


def test_rollback_restores_the_previous_release_one_level(env):
    two_releases(env)
    rel.promote(env, "rel-a", KEY)
    rel.promote(env, "rel-b", KEY)
    rel.rollback(env, KEY)
    assert pointer(env)["release_id"] == "rel-a"
    assert pointer(env)["previous_release_id"] is None
    assert rel.read_audit_log(env)[-1]["event"] == "rollback"
    with pytest.raises(rel.RollbackError):
        rel.rollback(env, KEY)  # only one level is kept


def test_rollback_without_a_pointer_is_refused(env):
    with pytest.raises(rel.RollbackError):
        rel.rollback(env, KEY)


def test_rollback_refuses_a_broken_previous_release(env):
    two_releases(env)
    rel.promote(env, "rel-a", KEY)
    rel.promote(env, "rel-b", KEY)
    (release_dir(env, "rel-a") / "holdout_guard.json").unlink()
    before = pointer(env)
    with pytest.raises(rel.RollbackError):
        rel.rollback(env, KEY)
    assert pointer(env) == before


def test_audit_log_is_append_only_and_free_of_ids(env, test_ids):
    log = rel.registry_dir(env) / "audit_log.jsonl"
    snapshots = []
    rel.create_release(env, RUN_ID, KEY, release_id="rel-a")
    snapshots.append(log.read_text())
    rel.create_release(env, RUN_ID, KEY, release_id="rel-b")
    snapshots.append(log.read_text())
    rel.promote(env, "rel-a", KEY)
    snapshots.append(log.read_text())
    with pytest.raises(rel.GateFailedError):
        rel.check_release(env, "rel-b", KEY, latency_budget_s=0.0)
    snapshots.append(log.read_text())
    rel.promote(env, "rel-b", KEY)
    rel.rollback(env, KEY)
    snapshots.append(log.read_text())
    for earlier, later in zip(snapshots, snapshots[1:]):
        assert later.startswith(earlier) and len(later) > len(earlier)
    assert_no_ids_or_hmacs(log.read_text(), test_ids)


# ---------------------------------------------------------------------------
# CLI (always with --data-dir under tmp_path; key via env var to a tmp key file)
# ---------------------------------------------------------------------------
@pytest.fixture
def cli_key(tmp_path, monkeypatch):
    f = write_key_file(tmp_path / "guard.key", KEY.hex())
    monkeypatch.setenv(rel.GUARD_KEY_ENV, str(f))
    return f


def run_cli(paths, *args):
    return cli.main(["--data-dir", str(paths.data_dir), *args])


def only_release_id(paths) -> str:
    (d,) = [p for p in (rel.registry_dir(paths) / "releases").iterdir() if p.name.startswith("rel-")]
    return d.name


def test_cli_release_check_promote_rollback(env, cli_key, capsys, test_ids):
    assert run_cli(env, "release", "--run", RUN_ID) == 0
    rid = only_release_id(env)
    assert rid in capsys.readouterr().out
    assert run_cli(env, "check-release", "--release", rid) == 0
    assert run_cli(env, "promote", "--release", rid) == 0
    assert pointer(env)["release_id"] == rid
    capsys.readouterr()
    assert run_cli(env, "rollback") == 2  # no previous release
    out, err = capsys.readouterr()
    assert err.startswith("refused:")
    assert_no_ids_or_hmacs(out + err, test_ids)


def test_cli_without_key_refuses(env, monkeypatch, capsys):
    monkeypatch.delenv(rel.GUARD_KEY_ENV, raising=False)
    assert run_cli(env, "release", "--run", RUN_ID) == 2
    assert "refused:" in capsys.readouterr().err
    assert not (rel.registry_dir(env) / "releases").exists()


def test_cli_dirty_run_needs_waiver(tmp_path, model, assignments, cli_key, capsys):
    paths = make_env(tmp_path, model, assignments, dirty=True)
    assert run_cli(paths, "release", "--run", RUN_ID) == 2
    assert "G8" in capsys.readouterr().err
    assert run_cli(paths, "release", "--run", RUN_ID,
                   "--waive-dirty-code", "synthetic reason") == 0
    m = json.loads((release_dir(paths, only_release_id(paths)) / "release.json").read_text())
    assert m["provenance_waiver"]["reason"] == "synthetic reason"


def test_cli_gate_failure_exits_2(env, cli_key, capsys):
    with (env.models / RUN_ID / "model.joblib").open("ab") as f:
        f.write(b"x")
    assert run_cli(env, "release", "--run", RUN_ID) == 2
    assert "G1" in capsys.readouterr().err
