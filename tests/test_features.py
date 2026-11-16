"""Phase 2: feature selection, missing values, split, and leakage prevention.

Uses a generated master-like table, so these tests need no downloaded data.
"""
import json

import numpy as np
import pandas as pd
import pytest

from churn_pipeline.config import DEFAULT_PATHS, Paths
from churn_pipeline.features import (
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    DataContractError,
    SplitConfig,
    build_features,
    load_split,
    run_features,
    split_customers,
)
from churn_pipeline.validation import SchemaError

CATEGORIES = {
    "gender": ["Male", "Female"],
    "partner": ["Yes", "No"],
    "dependents": ["Yes", "No"],
    "phone_service": ["Yes", "No"],
    "multiple_lines": ["Yes", "No", "No phone service"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "online_security": ["Yes", "No", "No internet service"],
    "online_backup": ["Yes", "No", "No internet service"],
    "device_protection": ["Yes", "No", "No internet service"],
    "tech_support": ["Yes", "No", "No internet service"],
    "streaming_tv": ["Yes", "No", "No internet service"],
    "streaming_movies": ["Yes", "No", "No internet service"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperless_billing": ["Yes", "No"],
    "payment_method": ["Electronic check", "Mailed check",
                       "Bank transfer (automatic)", "Credit card (automatic)"],
}


def make_master(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A table with customer_master's columns and dtypes, with random valid values."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    monthly = rng.uniform(18, 120, n).round(2)
    df = pd.DataFrame({ID_COLUMN: pd.array([f"{i:04d}-ABCDE" for i in range(n)], dtype="string")})
    df["senior_citizen"] = pd.array(rng.integers(0, 2, n), dtype="Int64")
    df["tenure_months"] = pd.array(tenure, dtype="Int64")
    df["monthly_charges"] = pd.array(monthly, dtype="Float64")
    df["total_charges"] = pd.array((tenure * monthly).round(2), dtype="Float64")
    for c, values in CATEGORIES.items():
        df[c] = pd.array(rng.choice(values, n), dtype="string")
    df[TARGET_COLUMN] = pd.array(np.where(rng.random(n) < 0.27, "Yes", "No"), dtype="string")
    # Synthetic Phase 1 columns that must never become features.
    df["region"] = pd.array(rng.choice(["North", "South"], n), dtype="string")
    df["signup_channel"] = pd.array(["web"] * n, dtype="string")
    df["preferred_contact"] = pd.array(["email"] * n, dtype="string")
    df["profile_updated_at"] = pd.Timestamp("2024-03-01", tz="UTC")
    df["has_profile"] = True
    for c in ("n_events", "n_login", "n_app_session", "n_support_call", "n_payment", "n_plan_change"):
        df[c] = pd.array(rng.integers(0, 5, n), dtype="Int64")
    df["total_duration_minutes"] = pd.array(rng.uniform(0, 50, n), dtype="Float64")
    df["first_event_ts"] = pd.Timestamp("2024-01-01", tz="UTC")
    df["last_event_ts"] = pd.Timestamp("2024-06-01", tz="UTC")
    return df


def with_new_customers(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """Mimic the IBM quirk: tenure 0 and a blank total_charges."""
    df = df.copy()
    mask = df[ID_COLUMN].isin(ids)
    df.loc[mask, "tenure_months"] = 0
    df.loc[mask, "total_charges"] = pd.NA
    return df


def load_split_from(r, name):
    df = r.splits[name]
    return df[ID_COLUMN], df.drop(columns=[ID_COLUMN, TARGET_COLUMN]), df[TARGET_COLUMN]


# --- Which columns are features --------------------------------------------------------
# Written out here on purpose, NOT imported from features.py: if the test reused
# the module's own EXCLUDED_COLUMNS, moving a column from EXCLUDED_COLUMNS into
# FEATURE_COLUMNS would change both sides at once and the test could never fail.
NEVER_FEATURES = {
    "customer_id", "churn",
    # Synthetic Source B (profiles)
    "region", "signup_channel", "preferred_contact", "profile_updated_at", "has_profile",
    # Synthetic Source C (event aggregates)
    "n_events", "n_login", "n_app_session", "n_support_call", "n_payment",
    "n_plan_change", "total_duration_minutes", "first_event_ts", "last_event_ts",
}


def test_id_target_and_synthetic_columns_are_not_features():
    r = build_features(make_master())
    forbidden = NEVER_FEATURES
    assert not set(FEATURE_COLUMNS) & forbidden
    assert set(EXCLUDED_COLUMNS) | {ID_COLUMN, TARGET_COLUMN} == NEVER_FEATURES
    # Output feature names are prefixed by the transformer; none may derive from a forbidden column.
    for name in r.feature_names:
        assert not any(name.split("__", 1)[1].startswith(f) for f in forbidden), name
    _, X, y = load_split_from(r, "train")
    assert ID_COLUMN not in X.columns and TARGET_COLUMN not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_unclassified_new_column_fails_instead_of_becoming_a_feature():
    m = make_master()
    m["n_events_last_30d"] = 1
    with pytest.raises(SchemaError, match="unclassified columns.*n_events_last_30d"):
        build_features(m)


def test_missing_required_column_fails():
    with pytest.raises(SchemaError, match="contract"):
        build_features(make_master().drop(columns=["contract"]))


def test_duplicate_customer_rows_fail():
    m = make_master()
    with pytest.raises(DataContractError, match="unique"):
        build_features(pd.concat([m, m.iloc[[0]]], ignore_index=True))


def test_invalid_label_fails():
    m = make_master()
    m.loc[0, TARGET_COLUMN] = "Maybe"
    with pytest.raises(DataContractError, match="churn"):
        build_features(m)


# --- total_charges missing-value decision -----------------------------------------------
def test_blank_total_charges_for_new_customers_filled_with_zero_and_counted():
    new_ids = ["0001-ABCDE", "0002-ABCDE", "0003-ABCDE"]
    r = build_features(with_new_customers(make_master(), new_ids))

    assert sum(s["total_charges_filled"] for s in r.report["splits"].values()) == 3
    # After scaling, the filled value is (0 - train_mean) / train_std.
    scaler = r.preprocessor.named_transformers_["total_charges"].named_steps["scale"]
    expected = (0.0 - scaler.mean_[0]) / scaler.scale_[0]
    rows = pd.concat(r.splits.values())
    filled = rows.loc[rows[ID_COLUMN].isin(new_ids), "total_charges__total_charges"]
    assert len(filled) == 3 and np.allclose(filled, expected)
    assert not rows.drop(columns=[ID_COLUMN]).isna().any().any()


def test_blank_total_charges_with_positive_tenure_is_not_silently_imputed():
    m = make_master()
    m.loc[5, "total_charges"] = pd.NA  # tenure_months > 0 for every generated row
    with pytest.raises(DataContractError, match="tenure_months != 0"):
        build_features(m)


def test_nulls_outside_total_charges_fail_rather_than_pass_through_as_nan():
    m = make_master()
    m.loc[5, "monthly_charges"] = pd.NA
    with pytest.raises(DataContractError, match="monthly_charges"):
        build_features(m)


# --- Split -------------------------------------------------------------------------------
def test_split_is_disjoint_complete_and_stratified():
    m = make_master(1000)
    a = split_customers(m, SplitConfig())
    assert len(a) == len(m) and a[ID_COLUMN].is_unique
    assert set(a[ID_COLUMN]) == set(m[ID_COLUMN])
    sizes = a["split"].value_counts(normalize=True)
    assert sizes["train"] == pytest.approx(0.70, abs=0.01)
    assert sizes["validation"] == pytest.approx(0.15, abs=0.01)
    assert sizes["test"] == pytest.approx(0.15, abs=0.01)

    joined = a.merge(m[[ID_COLUMN, TARGET_COLUMN]], on=ID_COLUMN)
    overall = (m[TARGET_COLUMN] == "Yes").mean()
    for _, part in joined.groupby("split"):
        assert (part[TARGET_COLUMN] == "Yes").mean() == pytest.approx(overall, abs=0.01)


def test_split_is_reproducible_and_independent_of_row_order():
    m = make_master()
    a1 = split_customers(m, SplitConfig(seed=7))
    shuffled = m.sample(frac=1, random_state=123).reset_index(drop=True)
    a2 = split_customers(shuffled, SplitConfig(seed=7))
    pd.testing.assert_frame_equal(a1, a2)
    a3 = split_customers(m, SplitConfig(seed=8))
    assert not a1.equals(a3)


def test_split_config_rejects_impossible_fractions():
    with pytest.raises(ValueError):
        SplitConfig(validation_fraction=0.5, test_fraction=0.5)


# --- Leakage: fit on train only ------------------------------------------------------------
def test_scaler_statistics_come_from_training_rows_only():
    m = make_master()
    r = build_features(m)
    train_ids = r.assignments.loc[r.assignments["split"] == "train", ID_COLUMN]
    train = m[m[ID_COLUMN].isin(train_ids)]
    scaler = r.preprocessor.named_transformers_["numeric"]
    assert scaler.mean_[0] == pytest.approx(train["tenure_months"].astype(float).mean())
    assert scaler.mean_[0] != pytest.approx(m["tenure_months"].astype(float).mean())


def test_changing_holdout_rows_does_not_change_fitted_params_or_train_features():
    m = make_master()
    r1 = build_features(m)
    holdout = r1.assignments.loc[r1.assignments["split"] != "train", ID_COLUMN]

    # Drastically change the feature values of validation/test customers only.
    # The split depends only on customer_id and churn, so assignments stay the same.
    m2 = m.copy()
    mask = m2[ID_COLUMN].isin(holdout)
    m2.loc[mask, "monthly_charges"] = m2.loc[mask, "monthly_charges"] * 100
    m2.loc[mask, "contract"] = "Two year"
    r2 = build_features(m2)

    pd.testing.assert_frame_equal(r1.assignments, r2.assignments)
    pd.testing.assert_frame_equal(r1.splits["train"], r2.splits["train"])
    assert r1.report["scaling_params_from_train"] == r2.report["scaling_params_from_train"]
    assert r1.report["categories_from_train"] == r2.report["categories_from_train"]
    assert not r1.splits["test"].equals(r2.splits["test"])  # holdout output did change


def test_category_seen_only_outside_train_is_not_learned_and_encodes_as_zeros():
    m = make_master()
    # Find a customer that the split puts in the test set, give it a category nobody else has.
    a = split_customers(m, SplitConfig())
    test_id = a.loc[a["split"] == "test", ID_COLUMN].iloc[0]
    m["payment_method"] = m["payment_method"].astype(object)
    m.loc[m[ID_COLUMN] == test_id, "payment_method"] = "Crypto"
    m["payment_method"] = m["payment_method"].astype("string")

    with pytest.warns(UserWarning, match="unknown categories"):
        r = build_features(m)
    assert "Crypto" not in r.report["categories_from_train"]["payment_method"]
    row = r.splits["test"].set_index(ID_COLUMN).loc[test_id]
    pm_cols = [c for c in r.feature_names if c.startswith("categorical__payment_method_")]
    assert (row[pm_cols] == 0).all()


# --- Outputs -----------------------------------------------------------------------------
def test_run_features_writes_reproducible_outputs(tmp_path):
    paths = Paths(tmp_path / "data")
    paths.curated.mkdir(parents=True)
    make_master().to_parquet(paths.curated / "customer_master.parquet", index=False)

    report = run_features(paths)
    first = {n: pd.read_parquet(paths.features / f"{n}.parquet") for n in ("train", "validation", "test")}
    run_features(paths)
    for n, df in first.items():
        pd.testing.assert_frame_equal(df, pd.read_parquet(paths.features / f"{n}.parquet"))

    for f in ("split_assignments.parquet", "preprocessor.joblib",
              "feature_report.json", "feature_report.md"):
        assert (paths.features / f).exists()
    saved = json.loads((paths.features / "feature_report.json").read_text())
    assert saved["fitted_on"] == "train" and saved["input_sha256"] == report["input_sha256"]

    ids, X, y = load_split(paths.features / "train.parquet")
    assert list(X.columns) == report["output_feature_columns"]
    assert len(ids) == len(X) == len(y) == report["splits"]["train"]["rows"]


def test_run_features_requires_phase1_output(tmp_path):
    with pytest.raises(FileNotFoundError, match="churn-pipeline run"):
        run_features(Paths(tmp_path / "data"))


# --- Real data (skipped if Phase 1 has not been run) ------------------------------------
REAL_MASTER = DEFAULT_PATHS.curated / "customer_master.parquet"


@pytest.mark.real_data
@pytest.mark.skipif(not REAL_MASTER.exists(), reason="Phase 1 output not built")
def test_real_master_feature_build():
    r = build_features(pd.read_parquet(REAL_MASTER))
    s = r.report["splits"]
    assert sum(p["rows"] for p in s.values()) == 7043
    assert sum(p["total_charges_filled"] for p in s.values()) == 11
    for p in s.values():
        assert p["churn_rate"] == pytest.approx(1869 / 7043, abs=0.005)
    assert len(r.feature_names) == 40
    assert set(CATEGORICAL_FEATURES) == set(r.report["categories_from_train"])


def test_output_id_dtype_does_not_depend_on_input_string_backing():
    m = make_master()
    a = build_features(m.assign(customer_id=m[ID_COLUMN].astype("str")))      # object/NaN-backed
    b = build_features(m.assign(customer_id=m[ID_COLUMN].astype("string")))   # pd.NA-backed
    for name in a.splits:
        pd.testing.assert_frame_equal(a.splits[name], b.splits[name])
    pd.testing.assert_frame_equal(a.assignments, b.assignments)
