"""W2: batch predict through a release.

Every test here is a synthetic-fixture test. The master, split, run, release,
registry and key are the W1 synthetic fixtures, created under tmp_path. Input
files are generated here. Nothing reads ``data/``, D7 outputs or a real champion.
"""
from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from churn_pipeline import cli, inference
from churn_pipeline import predict as pr
from churn_pipeline import release as rel
from churn_pipeline.evaluation import feature_baseline, score_baseline
from churn_pipeline.features import ID_COLUMN, _to_model_frame, build_features
from test_inference import INPUT_COLUMNS, fit_model, synthetic_master
from test_release import KEY, OTHER_KEY, RUN_ID, make_env, non_holdout_rows, write_key_file

RELEASE_ID = "rel-syn-pred"
SYN_THRESHOLD = 0.3  # test_inference.write_run's default threshold


def add_baseline(paths, model, master, assignments, run_id=RUN_ID) -> dict:
    """A synthetic baseline.json, built like Phase 3 builds it: feature baseline from
    TRAIN rows, score baseline and flag rate from VALIDATION scores. Its sha256 is
    registered in the synthetic run manifest, so W3's integrity check applies."""
    data = master.assign(**{ID_COLUMN: master[ID_COLUMN].astype("string")}).merge(
        assignments, on=ID_COLUMN)
    x_train = _to_model_frame(data[data["split"] == "train"])
    p_val = model.predict_proba(_to_model_frame(data[data["split"] == "validation"]))[:, 1]
    baseline = {"features": feature_baseline(x_train),
                "features_source": "train rows (synthetic fixture)",
                "score": score_baseline(p_val),
                "score_source": "validation scores (synthetic fixture)",
                "performance": {"at_threshold": {
                    "predicted_positive_rate": float((p_val >= SYN_THRESHOLD).mean())}}}
    run_dir = paths.models / run_id
    (run_dir / "baseline.json").write_text(json.dumps(baseline))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["artifacts"]["baseline.json"] = hashlib.sha256(
        (run_dir / "baseline.json").read_bytes()).hexdigest()
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return baseline


@pytest.fixture(scope="module")
def master():
    return synthetic_master()


@pytest.fixture(scope="module")
def model(master):
    return fit_model(master)


@pytest.fixture(scope="module")
def assignments(master):
    return build_features(master).assignments


@pytest.fixture(scope="module")
def test_ids(assignments):
    return sorted(assignments.loc[assignments["split"] == "test", ID_COLUMN].astype(str))


@pytest.fixture
def env(tmp_path, model, assignments, master):
    paths = make_env(tmp_path, model, assignments)
    add_baseline(paths, model, master, assignments)
    rel.create_release(paths, RUN_ID, KEY, release_id=RELEASE_ID)
    rel.promote(paths, RELEASE_ID, KEY)
    return paths


@pytest.fixture
def rows(master, test_ids):
    return non_holdout_rows(master, test_ids, n=30)


def write_parquet(path: Path, df: pd.DataFrame) -> Path:
    df.to_parquet(path, index=False)
    return path


def write_csv(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, index=False)
    return path


def batch_dir(paths, batch_id):
    return pr.inference_dir(paths) / batch_id


def inference_entries(paths):
    d = pr.inference_dir(paths)
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


# ---------------------------------------------------------------------------
# Successful batches: files, schema, contract
# ---------------------------------------------------------------------------
def test_parquet_batch_writes_three_files_with_the_contract(env, rows, tmp_path):
    inp = write_parquet(tmp_path / "in.parquet", rows)
    m = pr.run_predict(env, inp, "production", KEY, batch_id="b-1")
    d = batch_dir(env, "b-1")
    assert sorted(p.name for p in d.iterdir()) == sorted(
        [pr.PREDICTIONS_FILE, pr.REJECTS_FILE, pr.MANIFEST_FILE])

    preds = pd.read_parquet(d / pr.PREDICTIONS_FILE)
    assert list(preds.columns) == inference.OUTPUT_COLUMNS
    assert {c: str(t) for c, t in preds.dtypes.items()} == {
        "customer_id": "string", "churn_probability": "float64", "churn_flag": "bool",
        "decision_threshold": "float64", "model_run_id": "string", "model_sha256": "string",
        "inference_schema_version": "string"}
    assert list(preds[ID_COLUMN]) == list(rows[ID_COLUMN])
    assert list(pd.read_parquet(d / pr.REJECTS_FILE).columns) == inference.REJECT_COLUMNS

    saved = json.loads((d / pr.MANIFEST_FILE).read_text())
    assert saved["batch_id"] == "b-1" and saved["release_id"] == RELEASE_ID
    assert saved["run_id"] == RUN_ID and saved["n_input"] == 30 and saved["n_scored"] == 30
    assert saved["n_rejected_rows"] == 0 and saved["reject_counts_by_reason"] == {}
    assert saved["input"] == {"format": "parquet", "file_name": "in.parquet",
                              "size": inp.stat().st_size,
                              "sha256": hashlib.sha256(inp.read_bytes()).hexdigest()}
    assert saved["synthetic"] is None and saved["synthetic_source"] == "no metadata sidecar"
    assert set(saved["timings_s"]) == {"load_release", "read_input", "predict_batch", "write"}
    assert saved["model_sha256"] == preds["model_sha256"].iloc[0]
    assert m["output_dir"] == str(d)


def test_csv_and_parquet_inputs_give_identical_predictions(env, rows, tmp_path):
    a = pr.run_predict(env, write_parquet(tmp_path / "in.parquet", rows), "production", KEY,
                       batch_id="b-parquet")
    b = pr.run_predict(env, write_csv(tmp_path / "in.csv", rows), "production", KEY,
                       batch_id="b-csv")
    assert b["input"]["format"] == "csv"
    pd.testing.assert_frame_equal(
        pd.read_parquet(batch_dir(env, "b-parquet") / pr.PREDICTIONS_FILE),
        pd.read_parquet(batch_dir(env, "b-csv") / pr.PREDICTIONS_FILE))
    assert a["n_scored"] == b["n_scored"] == 30


def test_rejected_rows_are_written_with_reasons_and_input_positions(env, rows, tmp_path):
    bad = rows.copy()
    bad.loc[3, "contract"] = "Three year"
    bad.loc[7, "monthly_charges"] = pd.NA
    m = pr.run_predict(env, write_parquet(tmp_path / "in.parquet", bad), "production", KEY,
                       batch_id="b-rej")
    assert m["n_input"] == 30 and m["n_scored"] == 28 and m["n_rejected_rows"] == 2
    assert m["reject_counts_by_reason"] == {"not_allowed": 1, "null_not_allowed": 1}
    assert "once per problem" in m["reject_counts_unit"]       # W3 must not read it as rows
    r = pd.read_parquet(batch_dir(env, "b-rej") / pr.REJECTS_FILE)
    assert sorted(r.index) == [3, 7]                       # input row positions preserved
    assert set(zip(r["column"], r["reason_code"])) == {("contract", "not_allowed"),
                                                       ("monthly_charges", "null_not_allowed")}
    assert "Three year" not in " ".join(r["message"])     # no raw values (TD-5 Q4)


def test_empty_batch_writes_empty_contract_files(env, rows, tmp_path):
    m = pr.run_predict(env, write_parquet(tmp_path / "in.parquet", rows.iloc[0:0]),
                       "production", KEY, batch_id="b-empty")
    d = batch_dir(env, "b-empty")
    assert m["n_input"] == m["n_scored"] == m["n_rejected_rows"] == 0
    assert list(pd.read_parquet(d / pr.PREDICTIONS_FILE).columns) == inference.OUTPUT_COLUMNS
    assert pd.read_parquet(d / pr.PREDICTIONS_FILE).empty


def test_explicit_release_id_is_supported(env, rows, tmp_path):
    m = pr.run_predict(env, write_parquet(tmp_path / "in.parquet", rows), RELEASE_ID, KEY,
                       batch_id="b-named")
    assert m["release_id"] == RELEASE_ID


# ---------------------------------------------------------------------------
# P3 synthetic flag from the D7 sidecar (never from IDs)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sidecar,expected", [({"synthetic": True}, True),
                                              ({"synthetic": False}, False),
                                              ({"scenario": "none"}, None),
                                              ("not json", None)])
def test_synthetic_flag_comes_from_the_sidecar(env, rows, tmp_path, sidecar, expected):
    inp = write_parquet(tmp_path / "batch.parquet", rows)
    (tmp_path / "batch.json").write_text(sidecar if isinstance(sidecar, str)
                                         else json.dumps(sidecar))
    m = pr.run_predict(env, inp, "production", KEY, batch_id="b-syn")
    assert m["synthetic"] is expected


# ---------------------------------------------------------------------------
# Fail closed: holdout, key, release problems -> nothing written
# ---------------------------------------------------------------------------
def test_one_holdout_id_rejects_the_whole_batch_and_writes_nothing(env, rows, master, test_ids,
                                                                   tmp_path):
    holdout = master[master[ID_COLUMN] == test_ids[0]][INPUT_COLUMNS]
    inp = write_parquet(tmp_path / "in.parquet", pd.concat([rows, holdout], ignore_index=True))
    with pytest.raises(inference.InferenceInputError) as exc:
        pr.run_predict(env, inp, "production", KEY, batch_id="b-hold")
    assert exc.value.reasons == [("customer_id", "holdout_customer_id")]
    assert test_ids[0] not in str(exc.value)
    assert inference_entries(env) == []


def test_non_canonical_holdout_id_is_never_scored(env, rows, test_ids, master, tmp_path):
    """M3 characterization: the W1 guard matches exact strings, so ' <holdout id>' is not
    a guard hit. TD-5's format check still rejects the row, so the holdout customer is
    never scored. Whether such variants should reject the whole batch is an open policy
    question (release.py is out of scope this round)."""
    variant = master[master[ID_COLUMN] == test_ids[0]][INPUT_COLUMNS].copy()
    variant[ID_COLUMN] = " " + test_ids[0]
    inp = write_parquet(tmp_path / "in.parquet", pd.concat([rows, variant], ignore_index=True))
    m = pr.run_predict(env, inp, "production", KEY, batch_id="b-variant")
    preds = pd.read_parquet(batch_dir(env, "b-variant") / pr.PREDICTIONS_FILE)
    assert test_ids[0] not in set(preds[ID_COLUMN].str.strip())
    assert m["n_scored"] == len(rows) and m["reject_counts_by_reason"] == {"invalid_format": 1}


@pytest.fixture
def input_read_spy(monkeypatch):
    calls = []
    real = Path.read_bytes

    def spy(self):
        calls.append(self.name)
        return real(self)
    monkeypatch.setattr(Path, "read_bytes", spy)
    return calls


def test_wrong_key_fails_closed_before_the_input_is_read(env, rows, tmp_path, input_read_spy):
    inp = write_parquet(tmp_path / "in.parquet", rows)
    with pytest.raises(rel.GuardKeyError):
        pr.run_predict(env, inp, "production", OTHER_KEY, batch_id="b-key")
    assert "in.parquet" not in input_read_spy and inference_entries(env) == []


def test_tampered_guard_fails_closed(env, rows, tmp_path):
    guard = rel.registry_dir(env) / "releases" / RELEASE_ID / "holdout_guard.json"
    with guard.open("a") as f:
        f.write(" ")
    with pytest.raises(rel.ReleaseIntegrityError):
        pr.run_predict(env, write_parquet(tmp_path / "in.parquet", rows), "production", KEY,
                       batch_id="b-guard")
    assert inference_entries(env) == []


def test_no_production_pointer_fails_closed(tmp_path, model, assignments, rows):
    paths = make_env(tmp_path, model, assignments)       # no release, no pointer
    with pytest.raises(rel.ReleaseIntegrityError, match="no production release"):
        pr.run_predict(paths, write_parquet(tmp_path / "in.parquet", rows), "production", KEY)
    assert inference_entries(paths) == []


def test_unsupported_format_and_bad_batch_id_are_refused_before_any_access(
        env, rows, tmp_path, input_read_spy):
    with pytest.raises(ValueError, match="unsupported input format"):
        pr.run_predict(env, tmp_path / "in.txt", "production", KEY)
    with pytest.raises(ValueError, match="invalid batch id"):
        pr.run_predict(env, write_parquet(tmp_path / "in.parquet", rows), "production", KEY,
                       batch_id="../escape")
    assert input_read_spy == [] and inference_entries(env) == []


def test_missing_input_file_writes_nothing(env, tmp_path):
    with pytest.raises(FileNotFoundError):
        pr.run_predict(env, tmp_path / "absent.parquet", "production", KEY, batch_id="b-miss")
    assert inference_entries(env) == []


# ---------------------------------------------------------------------------
# Atomic writes, no overwrite, reproducibility
# ---------------------------------------------------------------------------
def test_failure_while_writing_leaves_no_partial_batch(env, rows, tmp_path, monkeypatch):
    real = pd.DataFrame.to_parquet

    def fail_on_rejects(self, path, *a, **kw):
        if Path(path).name == pr.REJECTS_FILE:
            raise OSError("simulated failure while writing rejects")
        return real(self, path, *a, **kw)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_on_rejects)
    inp = tmp_path / "in.parquet"
    real(rows, inp, index=False)
    with pytest.raises(OSError):
        pr.run_predict(env, inp, "production", KEY, batch_id="b-fail")
    assert inference_entries(env) == []                 # no batch dir, no .tmp-* dir


def test_concurrent_batch_creation_leaves_no_temp_dir_and_keeps_theirs(env, rows, tmp_path,
                                                                     monkeypatch):
    """H1 regression: another run creates a NON-empty batch dir of the same name while
    we write. The final rename fails, so the temp dir (holding predictions and rejects
    with customer IDs) must be removed, and the other run's files left untouched."""
    inp = write_parquet(tmp_path / "in.parquet", rows)
    final = batch_dir(env, "b-race")
    real_write_json = pr.write_json

    def concurrent_creator(path, obj):
        real_write_json(path, obj)
        final.mkdir(parents=True)
        (final / "theirs.txt").write_text("other run")
    monkeypatch.setattr(pr, "write_json", concurrent_creator)
    with pytest.raises(OSError):
        pr.run_predict(env, inp, "production", KEY, batch_id="b-race")
    assert inference_entries(env) == ["b-race"]                  # no .tmp-b-race-* left
    assert sorted(p.name for p in final.iterdir()) == ["theirs.txt"]
    assert (final / "theirs.txt").read_text() == "other run"


def test_existing_batch_is_never_overwritten(env, rows, tmp_path, input_read_spy):
    inp = write_parquet(tmp_path / "in.parquet", rows)
    pr.run_predict(env, inp, "production", KEY, batch_id="b-dup")
    d = batch_dir(env, "b-dup")
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    input_read_spy.clear()
    with pytest.raises(FileExistsError):
        pr.run_predict(env, inp, "production", KEY, batch_id="b-dup")
    assert input_read_spy == []                          # refused before reading anything
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before


def test_predictions_are_reproducible_across_batches(env, rows, tmp_path):
    inp = write_parquet(tmp_path / "in.parquet", rows)
    a = pr.run_predict(env, inp, "production", KEY)
    b = pr.run_predict(env, inp, "production", KEY)
    assert a["batch_id"] != b["batch_id"]                # metadata differs by design
    for f in (pr.PREDICTIONS_FILE, pr.REJECTS_FILE):
        pd.testing.assert_frame_equal(pd.read_parquet(batch_dir(env, a["batch_id"]) / f),
                                      pd.read_parquet(batch_dir(env, b["batch_id"]) / f))
    volatile = {"batch_id", "created_at", "timings_s", "output_dir"}
    assert {k: v for k, v in a.items() if k not in volatile} == \
        {k: v for k, v in b.items() if k not in volatile}


# ---------------------------------------------------------------------------
# Data access: only the input file (plus release artefacts); never split or master
# ---------------------------------------------------------------------------
def test_predict_reads_only_the_input_and_never_the_split(env, rows, tmp_path, monkeypatch):
    inp = write_parquet(tmp_path / "in.parquet", rows)
    opened, parquet_reads, csv_reads = [], [], []
    real_open, real_rp = builtins.open, pd.read_parquet

    def open_spy(file, *a, **kw):
        opened.append(str(file))
        return real_open(file, *a, **kw)

    def rp_spy(src, *a, **kw):
        parquet_reads.append(type(src).__name__)
        return real_rp(src, *a, **kw)
    # pathlib (read_bytes/read_text) goes through io.open, joblib through builtins.open:
    # spy on both so no Python-level file open escapes. pyarrow's native reads are
    # covered by the read_parquet spy.
    import io as _io
    monkeypatch.setattr(builtins, "open", open_spy)
    monkeypatch.setattr(_io, "open", open_spy)
    monkeypatch.setattr(pd, "read_parquet", rp_spy)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: csv_reads.append(1))
    pr.run_predict(env, inp, "production", KEY, batch_id="b-spy")
    assert parquet_reads == ["BytesIO"] and csv_reads == []   # one parse, of the input bytes
    assert not [p for p in opened if "split_assignments" in p or "customer_master" in p]
    assert sum(p.endswith("in.parquet") for p in opened) == 1  # input file opened once


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@pytest.fixture
def cli_key(tmp_path, monkeypatch):
    f = write_key_file(tmp_path / "guard.key", KEY.hex())
    monkeypatch.setenv(rel.GUARD_KEY_ENV, str(f))
    return f


def test_cli_predict_defaults_to_production(env, rows, tmp_path, cli_key, capsys):
    inp = write_parquet(tmp_path / "in.parquet", rows)
    assert cli.main(["--data-dir", str(env.data_dir), "predict", "--input", str(inp)]) == 0
    out = capsys.readouterr().out
    assert f"release {RELEASE_ID}" in out and "30 scored" in out
    (bid,) = inference_entries(env)
    m = json.loads((batch_dir(env, bid) / pr.MANIFEST_FILE).read_text())
    assert m["release_id"] == RELEASE_ID


def test_cli_predict_accepts_data_dir_after_the_subcommand_and_release_id(
        env, rows, tmp_path, cli_key):
    inp = write_parquet(tmp_path / "in.parquet", rows)
    assert cli.main(["predict", "--data-dir", str(env.data_dir), "--input", str(inp),
                     "--release", RELEASE_ID]) == 0
    assert len(inference_entries(env)) == 1


@pytest.mark.parametrize("argv", [["predict"],                         # no --input
                                  ["predict", "--input", "x.parquet"]])  # no --data-dir
def test_cli_predict_required_arguments(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_cli_predict_invalid_request_is_refused_before_the_key_is_read(env, tmp_path,
                                                                       monkeypatch, capsys):
    """M1 regression: semantic argument checks run before any side effect."""
    called = []
    monkeypatch.setattr(rel, "load_guard_key", lambda *a, **k: called.append(1))
    assert cli.main(["--data-dir", str(env.data_dir), "predict",
                     "--input", str(tmp_path / "in.txt")]) == 2
    assert "unsupported input format" in capsys.readouterr().err
    assert called == [] and inference_entries(env) == []


def test_cli_predict_conflicting_data_dirs_are_refused(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(rel, "load_guard_key", lambda *a, **k: called.append(1))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--data-dir", str(tmp_path / "a"), "predict", "--data-dir",
                  str(tmp_path / "b"), "--input", str(tmp_path / "in.parquet")])
    assert exc.value.code == 2 and called == []


def test_cli_predict_without_data_dir_touches_nothing(tmp_path, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(rel, "load_guard_key", lambda *a, **k: called.append(1))
    with pytest.raises(SystemExit) as exc:
        cli.main(["predict", "--input", str(tmp_path / "in.parquet")])
    assert exc.value.code == 2 and called == []
    assert "--data-dir is required for predict" in capsys.readouterr().err


@pytest.mark.parametrize("problem", ["no_key", "holdout", "missing_input"])
def test_cli_predict_failures_exit_2_and_write_nothing(env, rows, master, test_ids, tmp_path,
                                                       monkeypatch, cli_key, capsys, problem):
    inp = tmp_path / "in.parquet"
    if problem == "no_key":
        monkeypatch.delenv(rel.GUARD_KEY_ENV)
        write_parquet(inp, rows)
    elif problem == "holdout":
        holdout = master[master[ID_COLUMN] == test_ids[0]][INPUT_COLUMNS]
        write_parquet(inp, pd.concat([rows, holdout], ignore_index=True))
    assert cli.main(["--data-dir", str(env.data_dir), "predict", "--input", str(inp)]) == 2
    out_err = capsys.readouterr()
    assert out_err.err.startswith("refused:") and test_ids[0] not in out_err.err
    assert inference_entries(env) == []
