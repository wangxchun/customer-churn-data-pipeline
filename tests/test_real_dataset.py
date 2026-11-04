"""Integration checks against the real downloaded IBM file (skipped if absent)."""
import pytest

from churn_pipeline.config import DEFAULT_PATHS, IBM_CSV_BYTES
from churn_pipeline.download import verify_ibm_csv
from churn_pipeline.sources import load_customers
from churn_pipeline.synthetic import generate_event_batches, generate_profiles

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATHS.ibm_csv.exists(),
    reason="real dataset not downloaded (run `churn-pipeline download`)",
)


def test_checksum_and_size():
    verify_ibm_csv(DEFAULT_PATHS.ibm_csv)
    assert DEFAULT_PATHS.ibm_csv.stat().st_size == IBM_CSV_BYTES


def test_verified_schema_and_counts():
    r = load_customers(DEFAULT_PATHS.ibm_csv)
    assert len(r.valid) == 7043 and r.rejects.empty
    assert r.valid["customer_id"].is_unique
    assert r.valid["churn"].value_counts().to_dict() == {"No": 5174, "Yes": 1869}
    blank = r.valid[r.valid["total_charges"].isna()]
    assert len(blank) == 11 and (blank["tenure_months"] == 0).all()


def test_synthetic_generation_is_deterministic():
    ids = load_customers(DEFAULT_PATHS.ibm_csv).valid["customer_id"].tolist()
    assert generate_profiles(ids, 42) == generate_profiles(ids, 42)
    assert generate_event_batches(ids, 42) == generate_event_batches(ids, 42)
    assert generate_profiles(ids, 42) != generate_profiles(ids, 7)
