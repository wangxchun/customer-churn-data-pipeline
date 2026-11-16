"""TD-5: guarded inference on new, unlabeled rows.

``model.joblib`` is a full sklearn Pipeline. Called directly, it silently accepts
invalid input: an unseen category becomes all zeros, a blank ``total_charges``
becomes 0, and a Random Forest scores a missing value without complaint. This
module is the only supported way to score new rows. It applies the same input
rules as training, and it refuses instead of guessing.

Three failure classes (decision D3):

* ``InferenceInputError``: the input breaks the contract. ``level="batch"``
  means the frame as a whole is unusable (missing or unclassified column, not a
  DataFrame). ``level="record"`` comes from ``predict_one`` when its record is
  rejected. In a batch, bad rows are rejected one by one and the rest are scored.
* ``ModelIntegrityError``: the run directory cannot be trusted or does not match
  the expected schema. Hashes and versions are checked *before* unpickling.
* ``ScoringError``: the model or preprocessor misbehaved on input that passed
  validation (NaN after preprocessing, invalid probabilities, an exception).

Privacy (decision Q4): exception messages carry column names and reason codes
only, never raw values or ``customer_id``. The in-memory rejects table keeps
``customer_id`` for traceability, but its ``message`` holds no raw values.
Nothing here writes files.

Known limitation: the manifest itself is not signed. Anyone who can rewrite both
``manifest.json`` and ``model.joblib`` can defeat the hash check.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from .download import sha256_file
from .features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    DataContractError,
    _to_model_frame,  # private on purpose: same dtype conversion as training
    classify_columns,
)
from .modeling import fitted_preprocessor
from .sources import IBM_COLUMNS, IBM_RENAME, IBM_ROW_RULES
from .validation import NULL_NOT_ALLOWED, Column, _to_series, check_value

INFERENCE_SCHEMA_VERSION = "1"

UNSEEN_CATEGORY = "unseen_category"
DUPLICATE_CUSTOMER_ID = "duplicate_customer_id"
MISSING_COLUMN = "missing_column"
UNCLASSIFIED_COLUMN = "unclassified_column"

REQUIRED_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS]
REJECT_COLUMNS = ["customer_id", "reason_code", "column", "message"]
OUTPUT_COLUMNS = [
    "customer_id", "churn_probability", "churn_flag", "decision_threshold",
    "model_run_id", "model_sha256", "inference_schema_version",
]

# Phase 1's per-column rules, renamed to the curated column names. Reusing them
# (instead of restating them here) keeps training and inference on one contract.
_TO_IBM = {curated: ibm for ibm, curated in IBM_RENAME.items()}
COLUMN_SPECS: dict[str, Column] = {
    IBM_RENAME[c.name]: replace(c, name=IBM_RENAME[c.name])
    for c in IBM_COLUMNS if IBM_RENAME[c.name] in REQUIRED_COLUMNS
}
# Phase 1 row rules are named in prose; map each to the column and code it reports.
_ROW_RULE_CODES = {
    "TotalCharges blank but tenure > 0": ("total_charges", NULL_NOT_ALLOWED),
}
if set(COLUMN_SPECS) != set(REQUIRED_COLUMNS):  # pragma: no cover - import-time contract
    raise RuntimeError("Phase 1 IBM_COLUMNS no longer cover every inference column")
if {name for name, _ in IBM_ROW_RULES} != set(_ROW_RULE_CODES):  # pragma: no cover
    raise RuntimeError("a Phase 1 row rule has no inference reason code; add it to _ROW_RULE_CODES")


class InferenceInputError(DataContractError):
    """Inference input violates the contract. Messages never contain raw values or IDs."""

    def __init__(self, message: str, *, level: str,
                 reasons: list[tuple[str, str]] | tuple = ()):
        if level not in ("batch", "record"):
            raise ValueError(level)
        self.level = level
        self.reasons = list(reasons)  # (column, reason_code) pairs
        super().__init__(message)


class ModelIntegrityError(RuntimeError):
    """The model run directory is missing, tampered with, or incompatible."""


class ScoringError(RuntimeError):
    """Scoring failed on rows that passed input validation."""


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    run_id: str
    model_sha256: str
    threshold: float
    categories: dict[str, tuple]          # learned by the model's encoder
    output_feature_columns: tuple[str, ...]


@dataclass
class InputValidation:
    valid: pd.DataFrame        # typed rows that passed, index = input row position
    rejects: pd.DataFrame      # REJECT_COLUMNS, index = input row position
    ignored_columns: list[str]


@dataclass
class InferenceResult:
    predictions: pd.DataFrame  # OUTPUT_COLUMNS, in input order
    rejects: pd.DataFrame      # REJECT_COLUMNS, index = input row position
    ignored_columns: list[str]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _manifest_value(manifest: dict, *keys: str):
    value: Any = manifest
    for k in keys:
        if not isinstance(value, dict) or k not in value:
            raise ModelIntegrityError(f"manifest.json is missing {'.'.join(keys)}")
        value = value[k]
    return value


def load_model(run_dir: Path) -> LoadedModel:
    """Load a Phase 3 run for inference. Reads only manifest, selection and model.

    Order matters: every check that can be done without unpickling is done first,
    because unpickling runs code from the file.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ModelIntegrityError(f"no manifest.json in run directory {run_dir.name}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as e:
        raise ModelIntegrityError("manifest.json is unreadable") from e

    run_id = _manifest_value(manifest, "run_id")
    model_sha = _manifest_value(manifest, "model_sha256")
    hashes = {"model.joblib": _manifest_value(manifest, "artifacts", "model.joblib"),
              "selection.json": _manifest_value(manifest, "artifacts", "selection.json")}
    trained_with = _manifest_value(manifest, "environment", "packages", "scikit-learn")
    manifest_threshold = _manifest_value(manifest, "metrics", "threshold")

    if run_id != run_dir.name:
        raise ModelIntegrityError("manifest run_id does not match the run directory name")
    if model_sha != hashes["model.joblib"]:
        raise ModelIntegrityError("manifest model_sha256 disagrees with artifacts['model.joblib']")
    for name, expected in hashes.items():
        path = run_dir / name
        if not path.is_file():
            raise ModelIntegrityError(f"{name} is missing from the run directory")
        if sha256_file(path) != expected:
            raise ModelIntegrityError(f"{name} sha256 does not match the manifest; refusing to load")
    if trained_with != sklearn.__version__:
        raise ModelIntegrityError(f"model was pickled with scikit-learn {trained_with}; installed "
                                  f"is {sklearn.__version__}. Refusing to unpickle.")

    try:
        threshold = json.loads((run_dir / "selection.json").read_text())["threshold"]["chosen"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ModelIntegrityError("selection.json has no threshold.chosen") from e
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not 0 <= threshold <= 1 or threshold != manifest_threshold):
        raise ModelIntegrityError("decision threshold is invalid or differs between "
                                  "selection.json and manifest.json")

    model = joblib.load(run_dir / "model.joblib")  # only after every check above passed

    try:
        pre = fitted_preprocessor(model)
        encoder = pre.named_transformers_["categorical"]
    except (AttributeError, KeyError, IndexError, TypeError) as e:
        raise ModelIntegrityError("model has no fitted preprocessor with a categorical encoder") from e
    if list(getattr(pre, "feature_names_in_", [])) != FEATURE_COLUMNS:
        raise ModelIntegrityError("model input feature schema does not match FEATURE_COLUMNS")
    if [int(c) for c in getattr(model, "classes_", [])] != [0, 1]:
        raise ModelIntegrityError("model classes are not [0, 1]")
    if list(encoder.feature_names_in_) != CATEGORICAL_FEATURES:
        raise ModelIntegrityError("encoder feature schema does not match CATEGORICAL_FEATURES")

    categories: dict[str, tuple] = {}
    for column, learned in zip(encoder.feature_names_in_, encoder.categories_):
        outside = set(learned) - set(COLUMN_SPECS[column].allowed or ())
        if outside:
            raise ModelIntegrityError(f"encoder categories for {column} include {len(outside)} "
                                      "value(s) outside the Phase 1 allowed domain")
        categories[str(column)] = tuple(learned)

    return LoadedModel(model=model, run_id=str(run_id), model_sha256=str(model_sha),
                       threshold=float(threshold), categories=categories,
                       output_feature_columns=tuple(pre.get_feature_names_out()))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _normalize(value: Any) -> Any:
    """Plain Python scalar; any missing marker (None, NaN, pd.NA, NaT) becomes None."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # non-scalar: leave it for check_value to reject
        pass
    return value


def _empty_rejects() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="string") for c in REJECT_COLUMNS})


def validate_inference_input(df: pd.DataFrame, categories: dict[str, tuple] | None = None
                             ) -> InputValidation:
    """Split ``df`` into typed valid rows and rejects (one reject row per problem).

    ``categories``: the model encoder's learned categories. A Phase-1-legal value
    outside them is rejected as ``unseen_category`` rather than encoded as zeros.
    """
    if not isinstance(df, pd.DataFrame):
        raise InferenceInputError("inference input must be a pandas DataFrame", level="batch")
    if df.columns.duplicated().any():
        raise InferenceInputError("inference input has duplicate column names", level="batch")
    missing, unclassified = classify_columns(df.columns, REQUIRED_COLUMNS, optional=(TARGET_COLUMN,))
    if missing:
        raise InferenceInputError(f"inference input is missing required columns {missing}",
                                  level="batch", reasons=[(c, MISSING_COLUMN) for c in missing])
    if unclassified:
        raise InferenceInputError(
            f"inference input has unclassified columns {unclassified}; only the model inputs, "
            f"{ID_COLUMN}, {TARGET_COLUMN} and EXCLUDED_COLUMNS are accepted",
            level="batch", reasons=[(c, UNCLASSIFIED_COLUMN) for c in unclassified])
    ignored = sorted(c for c in df.columns if c not in REQUIRED_COLUMNS)

    n = len(df)
    positions = pd.RangeIndex(n)
    problems: list[list[tuple[str, str]]] = [[] for _ in range(n)]
    typed: dict[str, pd.Series] = {}
    for column in REQUIRED_COLUMNS:
        spec = COLUMN_SPECS[column]
        known = None if categories is None else categories.get(column)
        values = []
        for i, raw in enumerate(df[column].tolist()):
            parsed, code, _ = check_value(_normalize(raw), spec)
            if code is None and known is not None and parsed not in known:
                parsed, code = None, UNSEEN_CATEGORY
            if code is not None:
                problems[i].append((column, code))
            values.append(parsed)
        typed[column] = _to_series(values, spec, positions)
    typed_df = pd.DataFrame(typed, index=positions)

    columns_ok = pd.Series([not p for p in problems], index=positions)
    as_ibm = typed_df.rename(columns=_TO_IBM)
    for rule_name, rule in IBM_ROW_RULES:
        violated = rule(as_ibm).fillna(False).astype(bool) & columns_ok
        for i in violated.to_numpy().nonzero()[0]:
            problems[i].append(_ROW_RULE_CODES[rule_name])

    ids = [_normalize(v) for v in df[ID_COLUMN].tolist()]
    id_series = pd.Series(ids, index=positions, dtype=object)
    duplicated = id_series.notna() & id_series.duplicated(keep=False)
    for i in duplicated.to_numpy().nonzero()[0]:
        problems[i].append((ID_COLUMN, DUPLICATE_CUSTOMER_ID))

    reject_rows = [(i, column, code) for i in range(n) for column, code in problems[i]]
    if reject_rows:
        rejects = pd.DataFrame(
            {"customer_id": pd.array([ids[i] if isinstance(ids[i], str) else None
                                      for i, _, _ in reject_rows], dtype="string"),
             "reason_code": pd.array([code for _, _, code in reject_rows], dtype="string"),
             "column": pd.array([column for _, column, _ in reject_rows], dtype="string"),
             "message": pd.array([f"{column}: {code}" for _, column, code in reject_rows],
                                 dtype="string")},
            index=pd.Index([i for i, _, _ in reject_rows]),
        )
    else:
        rejects = _empty_rejects()

    # A boolean Series, not a list: an empty list would select columns, not rows.
    ok = pd.Series([not p for p in problems], index=positions, dtype=bool)
    return InputValidation(valid=typed_df.loc[ok], rejects=rejects, ignored_columns=ignored)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _predictions_frame(m: LoadedModel, ids: pd.Series, p: np.ndarray) -> pd.DataFrame:
    n = len(p)
    return pd.DataFrame({
        "customer_id": pd.array(ids.astype(str).tolist(), dtype="string"),
        "churn_probability": p.astype("float64"),
        "churn_flag": (p >= m.threshold).astype(bool),
        "decision_threshold": np.full(n, m.threshold, dtype="float64"),
        "model_run_id": pd.array([m.run_id] * n, dtype="string"),
        "model_sha256": pd.array([m.model_sha256] * n, dtype="string"),
        "inference_schema_version": pd.array([INFERENCE_SCHEMA_VERSION] * n, dtype="string"),
    }, columns=OUTPUT_COLUMNS)


def _score(m: LoadedModel, valid: pd.DataFrame) -> np.ndarray:
    X = _to_model_frame(valid)
    try:
        transformed = fitted_preprocessor(m.model).transform(X)
    except Exception as e:
        raise ScoringError(f"preprocessor failed on {len(X)} validated rows "
                           f"({type(e).__name__})") from e
    if list(getattr(transformed, "columns", [])) != list(m.output_feature_columns):
        raise ScoringError("preprocessed feature columns differ from the model's output columns")
    if np.isnan(np.asarray(transformed, dtype="float64")).any():
        raise ScoringError("NaN in preprocessed features of validated rows")
    try:
        proba = np.asarray(m.model.predict_proba(X), dtype="float64")
    except Exception as e:
        raise ScoringError(f"model failed on {len(X)} validated rows ({type(e).__name__})") from e
    if proba.shape != (len(X), 2):
        raise ScoringError("model returned probabilities of an unexpected shape")
    p = proba[:, 1]
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ScoringError("model returned non-finite or out-of-range probabilities")
    return p


def predict_batch(m: LoadedModel, df: pd.DataFrame) -> InferenceResult:
    """Validate every row; score the valid ones; return predictions and rejects (D3)."""
    v = validate_inference_input(df, m.categories)
    p = _score(m, v.valid) if len(v.valid) else np.empty(0, dtype="float64")
    return InferenceResult(predictions=_predictions_frame(m, v.valid[ID_COLUMN], p),
                           rejects=v.rejects, ignored_columns=v.ignored_columns)


def predict_one(m: LoadedModel, record: dict) -> dict:
    """Score one record, or raise ``InferenceInputError`` if it is invalid (D3)."""
    if not isinstance(record, dict):
        raise InferenceInputError("single-record input must be a dict", level="batch")
    result = predict_batch(m, pd.DataFrame([record]))
    if not result.rejects.empty:
        reasons = list(zip(result.rejects["column"], result.rejects["reason_code"]))
        raise InferenceInputError(
            "inference input rejected: " + "; ".join(f"{c}: {code}" for c, code in reasons),
            level="record", reasons=reasons)
    row = result.predictions.iloc[0]
    out = {
        "customer_id": str(row["customer_id"]),
        "churn_probability": float(row["churn_probability"]),
        "churn_flag": bool(row["churn_flag"]),
        "decision_threshold": float(row["decision_threshold"]),
        "model_run_id": str(row["model_run_id"]),
        "model_sha256": str(row["model_sha256"]),
        "inference_schema_version": str(row["inference_schema_version"]),
    }
    assert math.isfinite(out["churn_probability"])  # guaranteed by _score
    return out
