"""TD-5: a guarded inference entry point (load_model / predict_batch / predict_one).

Every test here is a synthetic-fixture test: data comes from ``make_signal_master()``
(randomly generated, not derived from any real customer) and every model and run
directory is created under pytest's tmp directories. Nothing reads ``data/``.
(D4 + TD-5 PM decision Q1: synthetic only; three synthetic splits for parity.)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from churn_pipeline import inference as inf
from churn_pipeline.features import (
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    LABEL_MAP,
    SPLIT_NAMES,
    TARGET_COLUMN,
    _to_model_frame,
    build_features,
)
from churn_pipeline.modeling import fitted_preprocessor, make_pipeline
from churn_pipeline.sources import IBM_COLUMNS
from churn_pipeline.validation import Column, _parse_value, check_value
from phase3_data import make_signal_master
from test_features import with_new_customers

INPUT_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS]
NEW_CUSTOMERS = ["0003-ABCDE", "0017-ABCDE", "0042-ABCDE"]  # tenure 0, blank total_charges


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
def synthetic_master(n: int = 600, seed: int = 0) -> pd.DataFrame:
    return with_new_customers(make_signal_master(n, seed), NEW_CUSTOMERS)


def fit_model(master: pd.DataFrame, estimator=None, columns: list[str] | None = None):
    """Fit a Phase-3-style Pipeline on the synthetic train split only."""
    r = build_features(master)
    data = master.assign(**{ID_COLUMN: master[ID_COLUMN].astype("string")}).merge(
        r.assignments, on=ID_COLUMN)
    train = data[data["split"] == "train"]
    X = _to_model_frame(train)
    if columns is not None:
        X = X[columns]
    y = train[TARGET_COLUMN].map(LABEL_MAP).astype("int64")
    if estimator is None:  # not `or`: an unfitted ensemble's __len__ raises
        estimator = LogisticRegression(max_iter=2000)
    return make_pipeline(estimator).fit(X, y)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_run(root: Path, model, threshold: float = 0.3, run_id: str = "20990101T000000Z-syn001",
              sklearn_version: str | None = None) -> Path:
    """A run directory with the same manifest fields Phase 3 writes."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    joblib.dump(model, run_dir / "model.joblib")
    (run_dir / "selection.json").write_text(json.dumps({"threshold": {"chosen": threshold}}))
    manifest = {
        "run_id": run_id,
        "model_sha256": sha(run_dir / "model.joblib"),
        "artifacts": {"model.joblib": sha(run_dir / "model.joblib"),
                      "selection.json": sha(run_dir / "selection.json")},
        "metrics": {"threshold": threshold},
        "environment": {"packages": {"scikit-learn": sklearn_version or sklearn.__version__}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


def edit_manifest(run_dir: Path, **changes) -> None:
    m = json.loads((run_dir / "manifest.json").read_text())
    m.update(changes)
    (run_dir / "manifest.json").write_text(json.dumps(m))


@pytest.fixture(scope="module")
def master() -> pd.DataFrame:
    return synthetic_master()


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, master) -> Path:
    return write_run(tmp_path_factory.mktemp("models"), fit_model(master))


@pytest.fixture(scope="module")
def loaded(run_dir) -> inf.LoadedModel:
    return inf.load_model(run_dir)


@pytest.fixture
def batch(master) -> pd.DataFrame:
    """20 valid inference rows (input columns only), including one new customer."""
    rows = master[master[ID_COLUMN].isin(NEW_CUSTOMERS[:1]) | (master.index < 20)]
    return rows[INPUT_COLUMNS].head(20).reset_index(drop=True)


def reasons_for(rejects: pd.DataFrame, position: int) -> set[tuple[str, str]]:
    r = rejects.loc[[position]] if position in rejects.index else rejects.iloc[0:0]
    return set(zip(r["column"], r["reason_code"]))


def no_raw_values(text: str, *values) -> None:
    for v in values:
        assert str(v) not in text, f"raw value {v!r} leaked into {text!r}"


# ---------------------------------------------------------------------------
# 0. Phase 1 reason codes (validation.check_value) agree with _parse_value
# ---------------------------------------------------------------------------
CODE_CASES = [
    (Column("x", "str"), "a", None),
    (Column("x", "str"), None, "null_not_allowed"),
    (Column("x", "str"), "", "null_not_allowed"),
    (Column("x", "str", nullable=True), None, None),
    (Column("x", "str"), 5, "invalid_type"),
    (Column("x", "int"), "7", None),
    (Column("x", "int"), True, "invalid_type"),
    (Column("x", "int"), 1.5, "invalid_type"),
    (Column("x", "int"), "1.5", "invalid_type"),
    (Column("x", "float"), "abc", "invalid_type"),
    (Column("x", "float"), True, "invalid_type"),
    (Column("x", "float"), float("inf"), "non_finite"),
    (Column("x", "float"), float("nan"), "null_not_allowed"),
    (Column("x", "float", nullable=True, blank_as_null=True), " ", None),
    (Column("x", "timestamp"), 5, "invalid_type"),
    (Column("x", "timestamp"), "nope", "invalid_type"),
    (Column("x", "timestamp"), "2024-01-01T00:00:00", "missing_timezone"),
    (Column("x", "timestamp"), "2024-01-01T00:00:00+00:00", None),
    (Column("x", "str", allowed=frozenset({"a"})), "b", "not_allowed"),
    (Column("x", "str", pattern=r"\d{4}-[A-Z]{5}"), "12-AB", "invalid_format"),
    (Column("x", "int", min_value=0), -1, "below_minimum"),
]


@pytest.mark.parametrize("col,value,code", CODE_CASES)
def test_check_value_codes_match_parse_value_messages(col, value, code):
    parsed, got_code, message = check_value(value, col)
    assert got_code == code
    assert _parse_value(value, col) == (parsed, message)  # Phase 1 behaviour unchanged
    assert (got_code is None) == (message is None)


@pytest.mark.parametrize("col", IBM_COLUMNS, ids=lambda c: c.name)
def test_check_value_agrees_with_parse_value_on_every_ibm_column(col):
    for value in (None, "", " ", "Yes", "No", "0", "1", "-3", "12.5", "x", 0, 1, 2, -1, 3.5,
                  "1111-AAAAA", "Month-to-month", "Three year"):
        parsed, code, message = check_value(value, col)
        assert _parse_value(value, col) == (parsed, message)
        assert (code is None) == (message is None)


# ---------------------------------------------------------------------------
# 1. The three TD-5 scenarios: batch rejects the row, single-record raises
# ---------------------------------------------------------------------------
TD5_SCENARIOS = [
    ("total_charges", pd.NA, "null_not_allowed"),     # tenure > 0, so the blank is illegal
    ("monthly_charges", pd.NA, "null_not_allowed"),
    ("contract", "Three year", "not_allowed"),
]


@pytest.mark.parametrize("column,value,code", TD5_SCENARIOS)
def test_td5_scenarios_are_rejected_in_batch_and_raise_for_single(loaded, batch, column, value, code):
    k = int(batch.index[batch["tenure_months"] > 0][0])
    bad = batch.copy()
    bad.loc[k, column] = value

    res = inf.predict_batch(loaded, bad)
    assert reasons_for(res.rejects, k) == {(column, code)}
    assert len(res.predictions) == len(batch) - 1
    assert bad.loc[k, ID_COLUMN] not in set(res.predictions[ID_COLUMN])
    no_raw_values(" ".join(res.rejects["message"]), "Three year")

    record = {c: (None if v is pd.NA else v) for c, v in bad.loc[k].items()}
    with pytest.raises(inf.InferenceInputError) as exc:
        inf.predict_one(loaded, record)
    assert exc.value.level == "record"
    assert (column, code) in exc.value.reasons
    no_raw_values(str(exc.value), bad.loc[k, ID_COLUMN], "Three year")


# ---------------------------------------------------------------------------
# 2. Further invalid inputs, valid nulls, unseen categories
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("column,value,code", [
    ("gender", "Other", "not_allowed"),           # binary: would encode like the dropped level
    ("senior_citizen", 2, "not_allowed"),         # passthrough column: would reach the model as-is
    ("tenure_months", -1, "below_minimum"),
    ("tenure_months", pd.NA, "null_not_allowed"),  # pd.NA must count as missing, not a bad type
    ("customer_id", "BAD-ID", "invalid_format"),
    ("monthly_charges", float("inf"), "non_finite"),
])
def test_invalid_values_get_reason_codes_and_valid_rows_are_still_scored(
        loaded, batch, column, value, code):
    bad = batch.copy()
    bad.loc[5, column] = value
    res = inf.predict_batch(loaded, bad)
    assert (column, code) in reasons_for(res.rejects, 5)
    assert len(res.predictions) == len(batch) - 1
    no_raw_values(" ".join(res.rejects["message"]), "Other", "BAD-ID")


def test_wrong_python_type_is_invalid_type(loaded, batch):
    bad = batch.astype({"monthly_charges": object})
    bad.loc[2, "monthly_charges"] = "abc"
    res = inf.predict_batch(loaded, bad)
    assert reasons_for(res.rejects, 2) == {("monthly_charges", "invalid_type")}


def test_every_row_with_a_duplicated_id_is_rejected(loaded, batch):
    dup = pd.concat([batch, batch.iloc[[3]]], ignore_index=True)
    res = inf.predict_batch(loaded, dup)
    assert set(res.rejects.index) == {3, len(batch)}
    assert set(res.rejects["reason_code"]) == {"duplicate_customer_id"}
    assert batch.loc[3, ID_COLUMN] not in set(res.predictions[ID_COLUMN])
    assert len(res.predictions) == len(batch) - 1


def test_multiple_problems_on_one_row_are_all_reported(loaded, batch):
    bad = batch.copy()
    bad.loc[1, "gender"] = "Other"
    bad.loc[1, "tenure_months"] = -4
    res = inf.predict_batch(loaded, bad)
    assert reasons_for(res.rejects, 1) == {("gender", "not_allowed"),
                                           ("tenure_months", "below_minimum")}


def test_blank_total_charges_is_valid_for_new_customers(loaded, batch):
    new = batch[batch[ID_COLUMN] == NEW_CUSTOMERS[0]]
    assert len(new) == 1 and pd.isna(new["total_charges"].iloc[0])
    res = inf.predict_batch(loaded, batch)
    assert res.rejects.empty
    assert NEW_CUSTOMERS[0] in set(res.predictions[ID_COLUMN])


def test_unseen_category_is_rejected_not_encoded_as_zeros(tmp_path, master, batch):
    """A Phase-1-legal value the encoder never saw: rejected before scoring (D3)."""
    no_card = master.assign(payment_method=master["payment_method"].replace(
        "Credit card (automatic)", "Mailed check"))
    m = inf.load_model(write_run(tmp_path, fit_model(no_card)))
    assert "Credit card (automatic)" not in m.categories["payment_method"]

    bad = batch.copy()
    bad.loc[4, "payment_method"] = "Credit card (automatic)"
    res = inf.predict_batch(m, bad)
    assert ("payment_method", "unseen_category") in reasons_for(res.rejects, 4)
    no_raw_values(" ".join(res.rejects["message"]), "Credit card")
    with pytest.raises(inf.InferenceInputError) as exc:
        inf.predict_one(m, bad.loc[4].to_dict())
    assert ("payment_method", "unseen_category") in exc.value.reasons


def test_characterization_raw_random_forest_scores_nan_silently(tmp_path, master, batch):
    """Why the guard matters: the unguarded RF Pipeline returns a normal-looking
    probability for a missing monthly_charges; the guarded path rejects the row."""
    rf = fit_model(master, RandomForestClassifier(n_estimators=20, random_state=0))
    bad = batch.copy()
    bad.loc[0, "monthly_charges"] = pd.NA
    p = rf.predict_proba(_to_model_frame(bad))[:, 1]
    assert np.isfinite(p).all()  # no error, no NaN: silently wrong

    m = inf.load_model(write_run(tmp_path, rf))
    res = inf.predict_batch(m, bad)
    assert reasons_for(res.rejects, 0) == {("monthly_charges", "null_not_allowed")}


# ---------------------------------------------------------------------------
# 3. Feature parity with the Phase 2 path, on three synthetic splits
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("split", SPLIT_NAMES)
def test_guarded_features_equal_phase2_features_for_each_synthetic_split(master, split):
    r = build_features(master)
    ids = set(r.assignments.loc[r.assignments["split"] == split, ID_COLUMN])
    rows = master[master[ID_COLUMN].isin(ids)].sort_values(ID_COLUMN)

    v = inf.validate_inference_input(rows[INPUT_COLUMNS])
    assert v.rejects.empty and len(v.valid) == len(rows)
    got = r.preprocessor.transform(_to_model_frame(v.valid))
    expected = r.splits[split].drop(columns=[ID_COLUMN, TARGET_COLUMN])
    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected.reset_index(drop=True),
                                  check_exact=True)


# ---------------------------------------------------------------------------
# 4. Prediction parity with the Phase 3 scoring path
# ---------------------------------------------------------------------------
def test_guarded_predictions_equal_direct_model_scores(loaded, master):
    rows = master.sort_values(ID_COLUMN).head(200)
    res = inf.predict_batch(loaded, rows[INPUT_COLUMNS])
    direct = loaded.model.predict_proba(_to_model_frame(rows))[:, 1]
    assert list(res.predictions[ID_COLUMN]) == list(rows[ID_COLUMN])
    np.testing.assert_array_equal(res.predictions["churn_probability"].to_numpy(), direct)


# ---------------------------------------------------------------------------
# 5. Model integrity: checks run before unpickling
# ---------------------------------------------------------------------------
@pytest.fixture
def load_spy(monkeypatch):
    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        raise AssertionError("joblib.load must not be called")
    monkeypatch.setattr(inf.joblib, "load", spy)
    return calls


def fresh_run(tmp_path, master, **kw) -> Path:
    return write_run(tmp_path, fit_model(master), **kw)


def test_model_hash_mismatch_refuses_before_unpickling(tmp_path, master, load_spy):
    run = fresh_run(tmp_path, master)
    with (run / "model.joblib").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(inf.ModelIntegrityError, match="model.joblib"):
        inf.load_model(run)
    assert load_spy == []


def test_selection_hash_mismatch_refuses_before_unpickling(tmp_path, master, load_spy):
    run = fresh_run(tmp_path, master)
    (run / "selection.json").write_text(json.dumps({"threshold": {"chosen": 0.01}}))
    with pytest.raises(inf.ModelIntegrityError, match="selection.json"):
        inf.load_model(run)
    assert load_spy == []


def test_sklearn_version_mismatch_refuses_before_unpickling(tmp_path, master, load_spy):
    run = fresh_run(tmp_path, master, sklearn_version="0.0.1")
    with pytest.raises(inf.ModelIntegrityError, match="scikit-learn"):
        inf.load_model(run)
    assert load_spy == []


@pytest.mark.parametrize("damage", ["no_manifest", "no_model_sha", "no_selection_hash",
                                    "model_sha_disagrees_with_artifacts", "run_id_mismatch"])
def test_incomplete_or_inconsistent_manifest_is_refused(tmp_path, master, load_spy, damage):
    run = fresh_run(tmp_path, master)
    m = json.loads((run / "manifest.json").read_text())
    if damage == "no_manifest":
        (run / "manifest.json").unlink()
    elif damage == "no_model_sha":
        del m["model_sha256"]
    elif damage == "no_selection_hash":
        del m["artifacts"]["selection.json"]
    elif damage == "model_sha_disagrees_with_artifacts":
        m["artifacts"]["model.joblib"] = "0" * 64
    else:
        m["run_id"] = "someone-else"
    if damage != "no_manifest":
        (run / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(inf.ModelIntegrityError):
        inf.load_model(run)
    assert load_spy == []


def test_threshold_must_match_between_selection_and_manifest(tmp_path, master):
    run = fresh_run(tmp_path, master, threshold=0.3)
    edit_manifest(run, metrics={"threshold": 0.31})
    with pytest.raises(inf.ModelIntegrityError, match="threshold"):
        inf.load_model(run)


def test_feature_schema_mismatch_is_refused(tmp_path, master):
    reordered = fit_model(master, columns=FEATURE_COLUMNS[::-1])
    with pytest.raises(inf.ModelIntegrityError, match="feature"):
        inf.load_model(write_run(tmp_path, reordered))


def test_encoder_categories_outside_phase1_domain_are_refused(tmp_path, master):
    crypto = master.copy()
    crypto.loc[crypto.index[:60], "payment_method"] = "Crypto wallet"
    with pytest.raises(inf.ModelIntegrityError, match="payment_method"):
        inf.load_model(write_run(tmp_path, fit_model(crypto)))


def test_loaded_model_exposes_the_contract(loaded, run_dir):
    assert loaded.run_id == run_dir.name
    assert loaded.model_sha256 == sha(run_dir / "model.joblib")
    assert loaded.threshold == 0.3
    assert set(loaded.categories) == {c for c in FEATURE_COLUMNS
                                      if c in fitted_preprocessor(loaded.model)
                                      .named_transformers_["categorical"].feature_names_in_}
    assert list(loaded.output_feature_columns) == list(
        fitted_preprocessor(loaded.model).get_feature_names_out())


# ---------------------------------------------------------------------------
# 6. Batch-level input errors; ignored columns
# ---------------------------------------------------------------------------
def test_missing_required_column_fails_the_whole_batch(loaded, batch):
    with pytest.raises(inf.InferenceInputError, match="contract") as exc:
        inf.predict_batch(loaded, batch.drop(columns=["contract"]))
    assert exc.value.level == "batch"


def test_unclassified_column_fails_the_whole_batch(loaded, batch):
    with pytest.raises(inf.InferenceInputError, match="favourite_colour") as exc:
        inf.predict_batch(loaded, batch.assign(favourite_colour="blue"))
    assert exc.value.level == "batch"
    no_raw_values(str(exc.value), "blue")


def test_non_dataframe_input_fails_the_whole_batch(loaded, batch):
    with pytest.raises(inf.InferenceInputError) as exc:
        inf.predict_batch(loaded, batch.to_dict("records"))
    assert exc.value.level == "batch"
    with pytest.raises(inf.InferenceInputError) as exc:
        inf.predict_one(loaded, ["not", "a", "dict"])
    assert exc.value.level == "batch"


def test_single_record_missing_a_field_is_a_batch_level_error(loaded, batch):
    record = batch.loc[0].to_dict()
    del record["tenure_months"]
    with pytest.raises(inf.InferenceInputError) as exc:
        inf.predict_one(loaded, record)
    assert exc.value.level == "batch"


def test_excluded_columns_and_label_are_ignored_and_reported(loaded, master):
    rows = master.head(30)
    full = inf.predict_batch(loaded, rows)                # all master columns, incl. churn
    clean = inf.predict_batch(loaded, rows[INPUT_COLUMNS])
    assert full.ignored_columns == sorted([*EXCLUDED_COLUMNS, TARGET_COLUMN])
    assert clean.ignored_columns == []
    pd.testing.assert_frame_equal(full.predictions, clean.predictions)


# ---------------------------------------------------------------------------
# 7. Column-order and row-order invariance
# ---------------------------------------------------------------------------
def test_column_order_does_not_change_output(loaded, batch):
    a = inf.predict_batch(loaded, batch)
    b = inf.predict_batch(loaded, batch[INPUT_COLUMNS[::-1]])
    pd.testing.assert_frame_equal(a.predictions, b.predictions)


def test_row_order_does_not_change_any_customers_output(loaded, batch):
    a = inf.predict_batch(loaded, batch).predictions
    b = inf.predict_batch(loaded, batch.sample(frac=1, random_state=3)).predictions
    key = lambda df: df.sort_values(ID_COLUMN, ignore_index=True)
    pd.testing.assert_frame_equal(key(a), key(b))


# ---------------------------------------------------------------------------
# 8. Output contract
# ---------------------------------------------------------------------------
def test_output_contract_columns_dtypes_and_ranges(loaded, batch):
    p = inf.predict_batch(loaded, batch).predictions
    assert list(p.columns) == inf.OUTPUT_COLUMNS
    assert {c: str(t) for c, t in p.dtypes.items()} == {
        "customer_id": "string", "churn_probability": "float64", "churn_flag": "bool",
        "decision_threshold": "float64", "model_run_id": "string", "model_sha256": "string",
        "inference_schema_version": "string",
    }
    assert p["churn_probability"].between(0, 1).all()
    assert (p["churn_flag"] == (p["churn_probability"] >= p["decision_threshold"])).all()
    assert (p["inference_schema_version"] == inf.INFERENCE_SCHEMA_VERSION).all()
    assert (p["model_sha256"] == loaded.model_sha256).all()
    assert list(inf.predict_batch(loaded, batch).rejects.columns) == inf.REJECT_COLUMNS


def test_flag_is_true_exactly_at_the_threshold(tmp_path, master, batch):
    model = fit_model(master)
    p0 = float(model.predict_proba(_to_model_frame(batch.iloc[[0]]))[0, 1])
    m = inf.load_model(write_run(tmp_path, model, threshold=p0))
    out = inf.predict_one(m, batch.loc[0].to_dict())
    assert out["churn_probability"] == p0 and out["churn_flag"] is True


def test_predict_one_returns_plain_python_values(loaded, batch):
    out = inf.predict_one(loaded, batch.loc[0].to_dict())
    assert list(out) == inf.OUTPUT_COLUMNS
    assert type(out["churn_probability"]) is float and type(out["churn_flag"]) is bool
    assert type(out["customer_id"]) is str


def test_empty_batch_returns_empty_contract_frames(loaded, batch):
    res = inf.predict_batch(loaded, batch.iloc[0:0])
    assert res.predictions.empty and res.rejects.empty
    assert list(res.predictions.columns) == inf.OUTPUT_COLUMNS


# ---------------------------------------------------------------------------
# 9. Scoring failures are ScoringError, never a silent prediction
# ---------------------------------------------------------------------------
def test_nan_after_preprocessing_is_a_scoring_error(loaded, batch, monkeypatch):
    pre = fitted_preprocessor(loaded.model)
    monkeypatch.setattr(pre, "transform", lambda X: pd.DataFrame(
        np.nan, index=X.index, columns=list(loaded.output_feature_columns)))
    with pytest.raises(inf.ScoringError):
        inf.predict_batch(loaded, batch)


def test_nan_probability_is_a_scoring_error(loaded, batch, monkeypatch):
    monkeypatch.setattr(loaded.model, "predict_proba",
                        lambda X: np.full((len(X), 2), np.nan))
    with pytest.raises(inf.ScoringError):
        inf.predict_batch(loaded, batch)


def test_model_exception_is_wrapped_as_scoring_error(loaded, batch, monkeypatch):
    def boom(X):
        raise ValueError("internal failure")
    monkeypatch.setattr(loaded.model, "predict_proba", boom)
    with pytest.raises(inf.ScoringError):
        inf.predict_batch(loaded, batch)


# ---------------------------------------------------------------------------
# 10. Inference reads no data files (no parquet, no CSV)
# ---------------------------------------------------------------------------
def test_inference_never_reads_data_files(run_dir, batch, monkeypatch):
    import pyarrow.parquet as pq

    def forbidden(*args, **kwargs):
        raise AssertionError("inference must not read data files")
    for target, name in ((pd, "read_parquet"), (pd, "read_csv"), (pq, "read_table")):
        monkeypatch.setattr(target, name, forbidden)
    m = inf.load_model(run_dir)
    inf.predict_batch(m, batch)
    inf.predict_one(m, batch.loc[0].to_dict())
