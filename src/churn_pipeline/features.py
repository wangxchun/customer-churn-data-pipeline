"""Phase 2: transformation and feature engineering.

Input:  ``curated/customer_master.parquet`` (Phase 1 output, one row per customer).
Output: ``features/{train,validation,test}.parquet``, the split assignments,
        the fitted preprocessor, and a feature report.

Leakage rules this module enforces:

* Every input column is explicitly classified as a feature, the identifier,
  the target, or excluded (with a reason). An unclassified column fails the
  run, so a new column can never become a model input by accident.
* The split happens *before* anything is fitted. The preprocessor (scaler,
  one-hot encoder) is fitted on the training rows only; validation and test
  rows are only transformed.
* Steps that learn nothing from data (the constant ``total_charges`` fill, the
  label mapping) are stateless, so they cannot leak across splits.

What the split does NOT do: the IBM data has no snapshot or churn date, so the
split is random (stratified), not temporal. It estimates performance on unseen
customers drawn from the same snapshot, not on a future period.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Paths
from .download import sha256_file
from .validation import SchemaError

ID_COLUMN = "customer_id"
TARGET_COLUMN = "churn"
LABEL_MAP = {"No": 0, "Yes": 1}

# Continuous, scaled. total_charges additionally gets the constant fill below.
SCALED_NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges"]
# Already 0/1 (validated in Phase 1); scaling would only obscure its meaning.
BINARY_NUMERIC_FEATURES = ["senior_citizen"]
CATEGORICAL_FEATURES = [
    "gender", "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "contract",
    "paperless_billing", "payment_method",
]
FEATURE_COLUMNS = SCALED_NUMERIC_FEATURES + BINARY_NUMERIC_FEATURES + CATEGORICAL_FEATURES

_SYNTHETIC_PROFILE = (
    "synthetic profile field (Source B): generated independently of churn, so any "
    "association with the label is chance"
)
_SYNTHETIC_EVENT = (
    "synthetic event aggregate (Source C): generated independently of churn, and "
    "aggregated over all time with no cutoff date (temporal-leakage risk)"
)
EXCLUDED_COLUMNS: dict[str, str] = {
    "region": _SYNTHETIC_PROFILE,
    "signup_channel": _SYNTHETIC_PROFILE,
    "preferred_contact": _SYNTHETIC_PROFILE,
    "profile_updated_at": _SYNTHETIC_PROFILE,
    "has_profile": _SYNTHETIC_PROFILE,
    "n_events": _SYNTHETIC_EVENT,
    "n_login": _SYNTHETIC_EVENT,
    "n_app_session": _SYNTHETIC_EVENT,
    "n_support_call": _SYNTHETIC_EVENT,
    "n_payment": _SYNTHETIC_EVENT,
    "n_plan_change": _SYNTHETIC_EVENT,
    "total_duration_minutes": _SYNTHETIC_EVENT,
    "first_event_ts": _SYNTHETIC_EVENT,
    "last_event_ts": _SYNTHETIC_EVENT,
}

TOTAL_CHARGES_FILL = 0.0
TOTAL_CHARGES_ASSUMPTION = (
    "A null total_charges is filled with 0.0 and is accepted only when "
    "tenure_months == 0. Assumption, not a confirmed fact: IBM does not document "
    "why these values are blank; 'not billed yet' is inferred from every blank "
    "having zero tenure. 0.0 is a constant, not a statistic learned from data, so "
    "it cannot leak information between splits."
)
SPLIT_NAMES = ("train", "validation", "test")


class DataContractError(ValueError):
    """The data violates an assumption that a Phase 2 decision depends on."""


@dataclass(frozen=True)
class SplitConfig:
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42

    def __post_init__(self) -> None:
        if not (0 < self.validation_fraction < 1 and 0 < self.test_fraction < 1):
            raise ValueError("split fractions must be between 0 and 1")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation + test fractions must leave rows for training")


@dataclass
class FeatureResult:
    splits: dict[str, pd.DataFrame]       # customer_id + feature columns + churn (0/1)
    assignments: pd.DataFrame             # customer_id, split
    preprocessor: ColumnTransformer       # fitted on train only
    feature_names: list[str]
    report: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Input checks (stateless: they learn nothing, so running them on all rows is safe)
# ---------------------------------------------------------------------------
def check_input(master: pd.DataFrame) -> None:
    required = [ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]
    missing = [c for c in required if c not in master.columns]
    if missing:
        raise SchemaError(f"customer_master: missing required columns {missing}")

    known = set(required) | set(EXCLUDED_COLUMNS)
    unclassified = [c for c in master.columns if c not in known]
    if unclassified:
        raise SchemaError(
            f"customer_master: unclassified columns {unclassified}; add each one to "
            "FEATURE_COLUMNS or EXCLUDED_COLUMNS (with a reason) in features.py"
        )

    ids = master[ID_COLUMN]
    if ids.isna().any() or not ids.is_unique:
        raise DataContractError("customer_id must be non-null and unique (one row per customer)")

    bad_labels = ~master[TARGET_COLUMN].isin(list(LABEL_MAP)).fillna(False).astype(bool)
    if bad_labels.any():
        raise DataContractError(
            f"churn must be one of {sorted(LABEL_MAP)}; {int(bad_labels.sum())} rows are not"
        )

    tc_null = master["total_charges"].isna()
    unexplained = tc_null & (master["tenure_months"] != 0).fillna(True).astype(bool)
    if unexplained.any():
        raise DataContractError(
            f"{int(unexplained.sum())} rows have null total_charges with tenure_months != 0; "
            "the constant-0 fill assumes nulls only occur for new customers. "
            "Decide how to handle these explicitly before continuing."
        )

    others = [c for c in FEATURE_COLUMNS if c != "total_charges"]
    with_nulls = [c for c in others if master[c].isna().any()]
    if with_nulls:
        raise DataContractError(
            f"unexpected nulls in {with_nulls}; only total_charges has a missing-value policy"
        )


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split_customers(master: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """Stratified train/validation/test assignment, one row per customer.

    Rows are sorted by customer_id first, so the result depends only on the
    data and the seed, not on the order rows happen to be stored in.
    """
    base = master[[ID_COLUMN, TARGET_COLUMN]].sort_values(ID_COLUMN, ignore_index=True)
    counts = base[TARGET_COLUMN].value_counts()
    if len(counts) < 2 or counts.min() < len(SPLIT_NAMES):
        raise DataContractError(
            f"stratified split needs >= {len(SPLIT_NAMES)} rows of each class; got {counts.to_dict()}"
        )

    rest, test = train_test_split(
        base, test_size=cfg.test_fraction, stratify=base[TARGET_COLUMN],
        random_state=cfg.seed,
    )
    # Re-express the validation share relative to what is left after the test split.
    val_share = cfg.validation_fraction / (1 - cfg.test_fraction)
    train, val = train_test_split(
        rest, test_size=val_share, stratify=rest[TARGET_COLUMN], random_state=cfg.seed,
    )
    parts = [df[[ID_COLUMN]].assign(split=name) for name, df in zip(SPLIT_NAMES, (train, val, test))]
    out = pd.concat(parts, ignore_index=True).sort_values(ID_COLUMN, ignore_index=True)
    out["split"] = out["split"].astype("string")
    return out


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Unfitted preprocessor. Fit it on training rows only."""
    return ColumnTransformer(
        [
            ("total_charges", Pipeline([
                ("fill_zero", SimpleImputer(strategy="constant", fill_value=TOTAL_CHARGES_FILL)),
                ("scale", StandardScaler()),
            ]), ["total_charges"]),
            ("numeric", StandardScaler(), ["tenure_months", "monthly_charges"]),
            ("binary", "passthrough", BINARY_NUMERIC_FEATURES),
            # Categories are learned from train. handle_unknown="ignore": a category
            # never seen in training becomes all zeros instead of crashing inference.
            # drop="if_binary": Yes/No columns become one 0/1 column, not two.
            ("categorical", OneHotEncoder(
                drop="if_binary", handle_unknown="ignore", sparse_output=False,
            ), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    ).set_output(transform="pandas")


def _to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable dtypes to plain numpy dtypes sklearn handles reliably."""
    out = pd.DataFrame(index=df.index)
    for c in SCALED_NUMERIC_FEATURES + BINARY_NUMERIC_FEATURES:
        out[c] = df[c].astype("float64")  # pd.NA -> NaN
    for c in CATEGORICAL_FEATURES:
        out[c] = df[c].astype(object)
    return out


def _transform(pre: ColumnTransformer, rows: pd.DataFrame) -> pd.DataFrame:
    X = pre.transform(_to_model_frame(rows))
    if X.isna().any().any():
        raise DataContractError(f"NaN in transformed features: {X.columns[X.isna().any()].tolist()}")
    return X


def build_features(master: pd.DataFrame, cfg: SplitConfig = SplitConfig()) -> FeatureResult:
    check_input(master)
    # Fix the key dtype at the boundary so output schemas don't depend on how the
    # upstream Parquet file happened to be written (seen: object-backed vs NA-backed strings).
    master = master.assign(**{ID_COLUMN: master[ID_COLUMN].astype(pd.StringDtype())})
    assignments = split_customers(master, cfg)

    data = master.merge(assignments, on=ID_COLUMN, how="left", validate="one_to_one")
    data = data.sort_values(ID_COLUMN, ignore_index=True)
    rows = {name: data[data["split"] == name] for name in SPLIT_NAMES}

    pre = build_preprocessor()
    pre.fit(_to_model_frame(rows["train"]))  # <- the only fit call: training rows only

    splits: dict[str, pd.DataFrame] = {}
    for name, part in rows.items():
        X = _transform(pre, part)
        out = pd.concat([part[[ID_COLUMN]], X], axis=1)
        out[TARGET_COLUMN] = part[TARGET_COLUMN].map(LABEL_MAP).astype("int8")
        splits[name] = out.reset_index(drop=True)

    feature_names = list(pre.get_feature_names_out())
    result = FeatureResult(splits, assignments, pre, feature_names)
    result.report = _report(master, rows, pre, feature_names, cfg)
    return result


# ---------------------------------------------------------------------------
# Report and I/O
# ---------------------------------------------------------------------------
def _report(master, rows, pre, feature_names, cfg) -> dict:
    tc_scaler = pre.named_transformers_["total_charges"].named_steps["scale"]
    num_scaler = pre.named_transformers_["numeric"]
    encoder = pre.named_transformers_["categorical"]
    scaling = {"total_charges": {"mean": float(tc_scaler.mean_[0]), "std": float(tc_scaler.scale_[0])}}
    for name, mean, std in zip(num_scaler.feature_names_in_, num_scaler.mean_, num_scaler.scale_):
        scaling[str(name)] = {"mean": float(mean), "std": float(std)}

    return {
        "input_rows": len(master),
        "split_config": asdict(cfg),
        "split_method": "stratified random on churn (NOT temporal: the data has no dates)",
        "splits": {
            name: {
                "rows": len(part),
                "churn_rate": round(float((part[TARGET_COLUMN] == "Yes").mean()), 4),
                "total_charges_filled": int(part["total_charges"].isna().sum()),
            }
            for name, part in rows.items()
        },
        "id_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "label_mapping": LABEL_MAP,
        "input_feature_columns": FEATURE_COLUMNS,
        "output_feature_columns": feature_names,
        "n_output_features": len(feature_names),
        "excluded_columns": EXCLUDED_COLUMNS,
        "missing_value_policy": {"total_charges": TOTAL_CHARGES_ASSUMPTION},
        "fitted_on": "train",
        "scaling_params_from_train": scaling,
        "categories_from_train": {
            str(c): [str(v) for v in cats]
            for c, cats in zip(encoder.feature_names_in_, encoder.categories_)
        },
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__},
    }


def render_markdown(report: dict, input_sha256: str | None = None) -> str:
    lines = ["# Feature build report (Phase 2)", ""]
    if input_sha256:
        lines += [f"- Input: `customer_master.parquet`, sha256 `{input_sha256[:12]}`"]
    cfg = report["split_config"]
    lines += [
        f"- Input rows: {report['input_rows']}",
        f"- Split: {report['split_method']}; seed {cfg['seed']}, "
        f"validation {cfg['validation_fraction']}, test {cfg['test_fraction']}",
        f"- Preprocessor fitted on: **{report['fitted_on']}** only",
        f"- Output features: {report['n_output_features']}",
        "",
        "| split | rows | churn rate | total_charges filled with 0 |",
        "|---|---|---|---|",
    ]
    for name, s in report["splits"].items():
        lines.append(f"| {name} | {s['rows']} | {s['churn_rate']:.2%} | {s['total_charges_filled']} |")
    lines += ["", "## Missing-value policy", "", report["missing_value_policy"]["total_charges"],
              "", "## Excluded columns", ""]
    lines += [f"- `{c}`: {why}" for c, why in report["excluded_columns"].items()]
    lines += [f"- `{report['id_column']}`: identifier (kept in output files, never a feature)",
              f"- `{report['target_column']}`: target (kept as label, never a feature)", ""]
    return "\n".join(lines)


def run_features(paths: Paths, cfg: SplitConfig = SplitConfig()) -> dict:
    src = paths.curated / "customer_master.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found; run Phase 1 (`churn-pipeline run`) first")
    master = pd.read_parquet(src)
    result = build_features(master, cfg)

    out = paths.features
    out.mkdir(parents=True, exist_ok=True)
    for name, df in result.splits.items():
        df.to_parquet(out / f"{name}.parquet", engine="pyarrow", index=False)
    result.assignments.to_parquet(out / "split_assignments.parquet", engine="pyarrow", index=False)
    # Pickle-based: only load files you produced yourself, with the same sklearn version.
    joblib.dump(result.preprocessor, out / "preprocessor.joblib")

    sha = sha256_file(src)
    report = {"input_file": str(src.relative_to(paths.data_dir)), "input_sha256": sha,
              **result.report}
    (out / "feature_report.json").write_text(json.dumps(report, indent=2))
    (out / "feature_report.md").write_text(render_markdown(report, sha))
    return report


def load_split(path: Path) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Read one split file and separate it into (ids, X, y).

    Use this rather than reading the Parquet directly, so customer_id and churn
    can never end up in the model inputs by accident.
    """
    df = pd.read_parquet(path)
    X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    return df[ID_COLUMN], X, df[TARGET_COLUMN]
