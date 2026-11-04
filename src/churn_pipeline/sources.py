"""Load and validate each input source.

Every loader returns a ``ValidationResult``: typed valid rows, rejects with
reasons, and warnings. Loaders read raw values as strings and let
``validation.validate`` do all typing, so pandas never guesses types silently.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .validation import (
    REJECT_COLUMNS,
    Column,
    SchemaError,
    ValidationResult,
    deduplicate,
    validate,
)

CUSTOMER_ID_PATTERN = r"\d{4}-[A-Z]{5}"
YES_NO = frozenset({"Yes", "No"})
INTERNET_ADDON = frozenset({"Yes", "No", "No internet service"})

# ---------------------------------------------------------------------------
# Source A: IBM Telco CSV. Original column name -> snake_case curated name.
# ---------------------------------------------------------------------------
IBM_RENAME = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure_months",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

IBM_COLUMNS = [
    Column("customerID", "str", pattern=CUSTOMER_ID_PATTERN),
    Column("gender", "str", allowed=frozenset({"Male", "Female"})),
    Column("SeniorCitizen", "int", allowed=frozenset({0, 1})),
    Column("Partner", "str", allowed=YES_NO),
    Column("Dependents", "str", allowed=YES_NO),
    Column("tenure", "int", min_value=0),
    Column("PhoneService", "str", allowed=YES_NO),
    Column("MultipleLines", "str", allowed=frozenset({"Yes", "No", "No phone service"})),
    Column("InternetService", "str", allowed=frozenset({"DSL", "Fiber optic", "No"})),
    Column("OnlineSecurity", "str", allowed=INTERNET_ADDON),
    Column("OnlineBackup", "str", allowed=INTERNET_ADDON),
    Column("DeviceProtection", "str", allowed=INTERNET_ADDON),
    Column("TechSupport", "str", allowed=INTERNET_ADDON),
    Column("StreamingTV", "str", allowed=INTERNET_ADDON),
    Column("StreamingMovies", "str", allowed=INTERNET_ADDON),
    Column("Contract", "str", allowed=frozenset({"Month-to-month", "One year", "Two year"})),
    Column("PaperlessBilling", "str", allowed=YES_NO),
    Column(
        "PaymentMethod",
        "str",
        allowed=frozenset(
            {
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            }
        ),
    ),
    Column("MonthlyCharges", "float", min_value=0),
    # Upstream encodes "not yet billed" as a single space. Keep it as null and
    # let the row rule below decide whether that is legitimate.
    Column("TotalCharges", "float", nullable=True, blank_as_null=True, min_value=0),
    Column("Churn", "str", allowed=YES_NO),
]

IBM_ROW_RULES = [
    (
        "TotalCharges blank but tenure > 0",
        lambda df: df["TotalCharges"].isna() & (df["tenure"] > 0),
    ),
]


def load_customers(path: Path) -> ValidationResult:
    # dtype=str + na_filter=False: every cell arrives exactly as written.
    raw = pd.read_csv(path, dtype=str, na_filter=False)
    row_refs = [f"line {i + 2}" for i in range(len(raw))]  # +1 header, +1 1-based
    result = validate(
        raw, IBM_COLUMNS, "ibm_customers", "customerID", row_refs, IBM_ROW_RULES
    )
    kept, dup_rejects = deduplicate(result.valid, "customerID", "ibm_customers")

    n_null_tc = int(kept["TotalCharges"].isna().sum())
    if n_null_tc:
        result.warnings.append(
            f"ibm_customers: {n_null_tc} rows have blank TotalCharges, all with "
            "tenure == 0; stored as null (imputation deferred to Phase 2)"
        )
    result.valid = kept.rename(columns=IBM_RENAME)
    result.rejects = pd.concat([result.rejects, dup_rejects], ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Source B: synthetic customer profiles (JSON document)
# ---------------------------------------------------------------------------
PROFILE_COLUMNS = [
    Column("customer_id", "str", pattern=CUSTOMER_ID_PATTERN),
    Column("region", "str", allowed=frozenset({"North", "South", "East", "West", "Central"})),
    Column(
        "signup_channel",
        "str",
        allowed=frozenset({"web", "retail_store", "call_center", "partner"}),
    ),
    Column("preferred_contact", "str", allowed=frozenset({"email", "sms", "phone"})),
    Column("updated_at", "timestamp"),
]


def _non_object_rejects(source: str, items: list[tuple[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": source,
            "row_ref": [ref for ref, _ in items],
            "record_key": None,
            "reason": "record is not a JSON object",
            "raw": [json.dumps(obj, default=str) for _, obj in items],
        },
        columns=REJECT_COLUMNS,
    )


def _records_frame(records: list[dict], columns: list[Column]) -> pd.DataFrame:
    # Build with object dtype so JSON types (int vs str vs null) are preserved
    # for validation; absent keys become None.
    names = [c.name for c in columns]
    extra = sorted({k for r in records for k in r} - set(names))
    return pd.DataFrame(
        [[r.get(k) for k in names + extra] for r in records],
        columns=names + extra,
        dtype="object",
    )


def load_profiles(path: Path) -> ValidationResult:
    # A JSON document is all-or-nothing: one syntax error invalidates the file.
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SchemaError(f"profiles: {path.name} is not valid JSON: {e}") from e
    if not isinstance(doc, dict) or not isinstance(doc.get("records"), list):
        raise SchemaError("profiles: expected an object with a 'records' list")

    good, bad = [], []
    for i, rec in enumerate(doc["records"]):
        (good if isinstance(rec, dict) else bad).append((f"record {i}", rec))

    raw = _records_frame([r for _, r in good], PROFILE_COLUMNS)
    result = validate(
        raw, PROFILE_COLUMNS, "profiles", "customer_id", [ref for ref, _ in good]
    )
    kept, dup_rejects = deduplicate(
        result.valid, "customer_id", "profiles", order_by="updated_at"
    )
    result.valid = kept
    result.rejects = pd.concat(
        [_non_object_rejects("profiles", bad), result.rejects, dup_rejects],
        ignore_index=True,
    )
    return result


# ---------------------------------------------------------------------------
# Source C: synthetic usage events (JSONL, one event per line)
# ---------------------------------------------------------------------------
EVENT_COLUMNS = [
    Column("event_id", "str", pattern=r"evt-[0-9a-z]+"),
    Column("customer_id", "str", pattern=CUSTOMER_ID_PATTERN),
    Column(
        "event_type",
        "str",
        allowed=frozenset({"login", "app_session", "support_call", "payment", "plan_change"}),
    ),
    Column("event_ts", "timestamp"),
    Column("channel", "str", allowed=frozenset({"web", "mobile_app", "phone", "store"})),
    Column("duration_minutes", "float", nullable=True, min_value=0),
]


def load_events_file(path: Path) -> ValidationResult:
    """Parse one JSONL file. Unlike JSON, a bad line only costs that line."""
    source = f"events:{path.name}"
    good: list[tuple[str, dict]] = []
    rejects = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            ref = f"line {lineno}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                rejects.append((ref, None, f"malformed JSON: {e.msg} (col {e.colno})", line.rstrip("\n")))
                continue
            if not isinstance(obj, dict):
                rejects.append((ref, None, "record is not a JSON object", line.rstrip("\n")))
                continue
            good.append((ref, obj))

    raw = _records_frame([o for _, o in good], EVENT_COLUMNS)
    result = validate(raw, EVENT_COLUMNS, source, "event_id", [r for r, _ in good])
    parse_rejects = pd.DataFrame(
        [(source, ref, key, reason, raw_line) for ref, key, reason, raw_line in rejects],
        columns=REJECT_COLUMNS,
    )
    result.rejects = pd.concat([parse_rejects, result.rejects], ignore_index=True)
    # NOTE: duplicate event_ids are *not* resolved here. Duplicates can span
    # files and runs, so they are resolved against the store (see events.py).
    return result

