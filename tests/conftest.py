"""Small handcrafted fixtures so tests never depend on the downloaded dataset."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from churn_pipeline.config import Paths
from churn_pipeline.sources import IBM_RENAME

IBM_HEADER = list(IBM_RENAME)  # original IBM column order

BASE_ROW = {
    "customerID": "0000-AAAAA", "gender": "Female", "SeniorCitizen": "0",
    "Partner": "Yes", "Dependents": "No", "tenure": "5", "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No",
    "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": "29.85", "TotalCharges": "149.25", "Churn": "No",
}

CUSTOMER_IDS = ["1111-AAAAA", "2222-BBBBB", "3333-CCCCC"]


def ibm_row(cid: str, **overrides: str) -> dict:
    return {**BASE_ROW, "customerID": cid, **overrides}


def write_ibm_csv(path: Path, rows: list[dict], header: list[str] = IBM_HEADER) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def profile(cid: str, updated_at: str = "2024-03-01T00:00:00+00:00", **kw) -> dict:
    return {"customer_id": cid, "region": "North", "signup_channel": "web",
            "preferred_contact": "email", "updated_at": updated_at, **kw}


def write_profiles(path: Path, records: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": {"synthetic": True}, "records": records}))
    return path


def event(eid: str, cid: str = "1111-AAAAA", **kw) -> dict:
    return {"event_id": eid, "customer_id": cid, "event_type": "login",
            "event_ts": "2024-01-01T10:00:00+00:00", "channel": "web",
            "duration_minutes": None, **kw}


def write_jsonl(path: Path, items: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [i if isinstance(i, str) else json.dumps(i) for i in items]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    """A complete, valid mini data directory."""
    p = Paths(tmp_path / "data")
    write_ibm_csv(p.ibm_csv, [ibm_row(c) for c in CUSTOMER_IDS])
    # 2222 has no profile; 9999 is an orphan profile.
    write_profiles(p.profiles_json, [profile("1111-AAAAA"), profile("3333-CCCCC"),
                                     profile("9999-ZZZZZ")])
    write_jsonl(p.events_dir / "batch_001.jsonl", [
        event("evt-1", "1111-AAAAA"),
        event("evt-2", "1111-AAAAA", event_type="support_call", channel="phone",
              duration_minutes=12.5),
        event("evt-3", "9999-ZZZZZ"),  # orphan event
    ])
    return p
