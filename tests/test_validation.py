import pandas as pd
import pytest

from churn_pipeline.sources import load_customers
from churn_pipeline.validation import Column, SchemaError, deduplicate, validate

from conftest import IBM_HEADER, ibm_row, write_ibm_csv


def test_valid_customers_are_typed(tmp_path):
    path = write_ibm_csv(tmp_path / "c.csv", [ibm_row("1111-AAAAA"), ibm_row("2222-BBBBB")])
    r = load_customers(path)
    assert len(r.valid) == 2 and r.rejects.empty
    assert str(r.valid["tenure_months"].dtype) == "Int64"
    assert str(r.valid["monthly_charges"].dtype) == "Float64"
    assert "customer_id" in r.valid.columns  # renamed to snake_case


def test_missing_required_column_fails_whole_source(tmp_path):
    header = [c for c in IBM_HEADER if c != "Churn"]
    path = write_ibm_csv(tmp_path / "c.csv", [ibm_row("1111-AAAAA")], header=header)
    with pytest.raises(SchemaError, match="Churn"):
        load_customers(path)


def test_bad_values_are_rejected_with_reasons_not_dropped(tmp_path):
    rows = [
        ibm_row("1111-AAAAA"),
        ibm_row("2222-BBBBB", tenure="abc"),
        ibm_row("3333-CCCCC", Churn="Maybe"),
        ibm_row("bad-id"),
        ibm_row("4444-DDDDD", MonthlyCharges="-1"),
    ]
    r = load_customers(write_ibm_csv(tmp_path / "c.csv", rows))
    assert len(r.valid) == 1
    assert len(r.rejects) == 4  # nothing silently lost: 1 + 4 == 5 input rows
    reasons = dict(zip(r.rejects["record_key"], r.rejects["reason"]))
    assert "not an integer" in reasons["2222-BBBBB"]
    assert "not in allowed set" in reasons["3333-CCCCC"]
    assert "does not match" in reasons["bad-id"]
    assert "below minimum" in reasons["4444-DDDDD"]
    assert r.rejects["row_ref"].tolist() == ["line 3", "line 4", "line 5", "line 6"]


def test_blank_total_charges_allowed_only_for_new_customers(tmp_path):
    rows = [
        ibm_row("1111-AAAAA", tenure="0", TotalCharges=" "),  # legit: not billed yet
        ibm_row("2222-BBBBB", tenure="12", TotalCharges=" "),  # suspicious
    ]
    r = load_customers(write_ibm_csv(tmp_path / "c.csv", rows))
    assert r.valid["customer_id"].tolist() == ["1111-AAAAA"]
    assert r.valid["total_charges"].isna().all()
    assert r.rejects["reason"].tolist() == ["TotalCharges blank but tenure > 0"]
    assert any("blank TotalCharges" in w for w in r.warnings)


def test_multiple_errors_on_one_row_are_all_reported(tmp_path):
    r = load_customers(write_ibm_csv(
        tmp_path / "c.csv", [ibm_row("1111-AAAAA", gender="X", Partner="Y")]
    ))
    assert "gender" in r.rejects.loc[0, "reason"] and "Partner" in r.rejects.loc[0, "reason"]


def test_duplicate_customer_ids(tmp_path):
    rows = [
        ibm_row("1111-AAAAA"),
        ibm_row("1111-AAAAA"),                # exact duplicate -> keep one
        ibm_row("2222-BBBBB", Churn="No"),
        ibm_row("2222-BBBBB", Churn="Yes"),   # conflicting label -> can't choose
    ]
    r = load_customers(write_ibm_csv(tmp_path / "c.csv", rows))
    assert r.valid["customer_id"].tolist() == ["1111-AAAAA"]
    assert sorted(r.rejects["reason"]) == [
        "conflicting_duplicate_key", "conflicting_duplicate_key", "exact_duplicate",
    ]


@pytest.mark.parametrize("value,ok", [
    ("2024-01-01T00:00:00+00:00", True),
    ("2024-01-01T00:00:00Z", True),
    ("2024-01-01T02:00:00+02:00", True),
    ("2024-01-01T00:00:00", False),   # naive: ambiguous
    ("01/02/2024", False),
    (1704067200, False),              # epoch int: wrong type
])
def test_timestamp_parsing(value, ok):
    raw = pd.DataFrame({"ts": [value]}, dtype="object")
    r = validate(raw, [Column("ts", "timestamp")], "t", "ts", ["row 1"])
    assert (len(r.valid) == 1) is ok
    if ok:
        assert r.valid["ts"].iloc[0] == pd.Timestamp("2024-01-01T00:00:00Z")


def test_dedupe_latest_wins_and_ties_conflict():
    df = pd.DataFrame({
        "k": ["a", "a", "b", "b"],
        "v": [1, 2, 3, 4],
        "ts": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01", "2024-01-01"], utc=True),
        "_row_ref": ["r0", "r1", "r2", "r3"],
    })
    kept, rejects = deduplicate(df, "k", "t", order_by="ts")
    assert kept["v"].tolist() == [2]
    assert dict(zip(rejects["row_ref"], rejects["reason"])) == {
        "r0": "superseded_by_newer_record",
        "r2": "conflicting_duplicate_key",
        "r3": "conflicting_duplicate_key",
    }
