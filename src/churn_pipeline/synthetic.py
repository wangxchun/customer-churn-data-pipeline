"""Deterministic generator for the two SYNTHETIC sources.

Source B (profiles, JSON) and Source C (usage events, JSONL) are invented for
teaching. They are keyed to the IBM customerIDs only so that joins are
meaningful. They are NOT IBM data and NOT real customer data.

Each source contains deliberately injected defects (listed in INJECTED_*), so
the pipeline's reject handling can be exercised and checked.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .config import Paths

NOTICE = (
    "SYNTHETIC DATA generated for educational purposes by churn_pipeline.synthetic. "
    "Not IBM data and not real customer data. Records reuse IBM Telco customerIDs "
    "only to demonstrate joins."
)

REGIONS = ["North", "South", "East", "West", "Central"]
SIGNUP_CHANNELS = ["web", "retail_store", "call_center", "partner"]
CONTACTS = ["email", "sms", "phone"]
EVENT_TYPES = ["login", "app_session", "support_call", "payment", "plan_change"]
EVENT_WEIGHTS = [40, 30, 10, 15, 5]
CHANNELS = {
    "login": ["web", "mobile_app"],
    "app_session": ["web", "mobile_app"],
    "support_call": ["phone"],
    "payment": ["web", "mobile_app", "store"],
    "plan_change": ["web", "phone", "store"],
}

# Customer IDs that match the ID pattern but do not exist in the IBM file.
ORPHAN_IDS = ["9901-ZZSYN", "9902-ZZSYN", "9903-ZZSYN", "9904-ZZSYN", "9905-ZZSYN"]

INJECTED_PROFILE_DEFECTS = {
    "orphan_profiles (valid, but customer not in IBM data)": 5,
    "exact_duplicate": 1,
    "superseded_by_newer_record": 2,
    "missing customer_id": 1,
    "region not in allowed set": 1,
    "timestamp without timezone": 1,
    "record is not a JSON object": 1,
}

INJECTED_EVENT_DEFECTS = {
    "batch 002: replayed events from batch 001 (at-least-once delivery)": 50,
    "batch 003: malformed JSON line": 1,
    "batch 003: line is not a JSON object": 1,
    "batch 003: missing event_type": 1,
    "batch 003: event_type not in allowed set": 1,
    "batch 003: timestamp without timezone": 1,
    "batch 003: negative duration": 1,
    "batch 003: orphan events (valid, customer not in IBM data)": 3,
    "batch 003: event_id reused with different payload": 1,
    "batch 003: exact duplicate within the same file": 1,
    "batch 003: unexpected extra field 'app_version' (schema drift)": 2,
}


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat()


def generate_profiles(customer_ids: list[str], seed: int = 42) -> dict:
    rng = random.Random(seed)
    covered = sorted(rng.sample(customer_ids, k=int(len(customer_ids) * 0.9)))
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def profile(cid: str) -> dict:
        return {
            "customer_id": cid,
            "region": rng.choice(REGIONS),
            "signup_channel": rng.choice(SIGNUP_CHANNELS),
            "preferred_contact": rng.choice(CONTACTS),
            "updated_at": _iso(base + timedelta(days=rng.randrange(300), hours=rng.randrange(24))),
        }

    records: list = [profile(c) for c in covered + ORPHAN_IDS]

    # --- injected defects ---
    records.append(dict(records[0]))  # exact duplicate
    for victim in records[1:3]:  # older, different versions -> superseded
        old = dict(victim, region="Central" if victim["region"] != "Central" else "North")
        old["updated_at"] = _iso(datetime(2023, 6, 1, tzinfo=timezone.utc))
        records.append(old)
    missing_id = profile("placeholder")
    del missing_id["customer_id"]
    records.append(missing_id)
    records.append(dict(profile(covered[10]), region="Atlantis"))
    records.append(dict(profile(covered[11]), updated_at="2024-05-01T12:00:00"))
    records.append("corrupted-record")

    rng.shuffle(records)
    return {
        "metadata": {
            "synthetic": True,
            "notice": NOTICE,
            "seed": seed,
            "schema_version": 1,
            "injected_defects": INJECTED_PROFILE_DEFECTS,
        },
        "records": records,
    }


def _make_events(rng: random.Random, ids: list[str], n: int, start: datetime,
                 days: int, first_seq: int) -> list[dict]:
    events = []
    for i in range(n):
        etype = rng.choices(EVENT_TYPES, weights=EVENT_WEIGHTS)[0]
        duration = None
        if etype == "app_session":
            duration = round(rng.uniform(1, 60), 1)
        elif etype == "support_call":
            duration = round(rng.uniform(2, 45), 1)
        ts = start + timedelta(seconds=rng.randrange(days * 86400))
        events.append({
            "event_id": f"evt-{first_seq + i:06d}",
            "customer_id": rng.choice(ids),
            "event_type": etype,
            "event_ts": _iso(ts),
            "channel": rng.choice(CHANNELS[etype]),
            "duration_minutes": duration,
        })
    return sorted(events, key=lambda e: e["event_ts"])


def generate_event_batches(customer_ids: list[str], seed: int = 42) -> dict[str, list[str]]:
    """Return {file_name: jsonl_lines} for three batches arriving over time."""
    rng = random.Random(seed + 1)
    active = sorted(rng.sample(customer_ids, k=600))
    b1 = _make_events(rng, active, 400, datetime(2024, 1, 1, tzinfo=timezone.utc), 120, 1)
    b2 = _make_events(rng, active, 300, datetime(2024, 5, 1, tzinfo=timezone.utc), 120, 401)
    b3 = _make_events(rng, active, 250, datetime(2024, 9, 1, tzinfo=timezone.utc), 120, 701)

    # Batch 2 re-delivers 50 events from batch 1 (upstream retry).
    b2_lines = [json.dumps(e) for e in b2 + rng.sample(b1, 50)]

    # Batch 3 defects.
    t = "2024-12-15T10:00:00+00:00"
    bad = [
        '{"event_id": "evt-900001", "customer_id": "7590-VHVEG", "event_ty',  # truncated
        "[1, 2, 3]",
        json.dumps({"event_id": "evt-900002", "customer_id": active[0],
                    "event_ts": t, "channel": "web", "duration_minutes": None}),
        json.dumps({"event_id": "evt-900003", "customer_id": active[1], "event_type": "teleport",
                    "event_ts": t, "channel": "web", "duration_minutes": None}),
        json.dumps({"event_id": "evt-900004", "customer_id": active[2], "event_type": "login",
                    "event_ts": "2024-12-15 10:00:00", "channel": "web", "duration_minutes": None}),
        json.dumps({"event_id": "evt-900005", "customer_id": active[3], "event_type": "app_session",
                    "event_ts": t, "channel": "mobile_app", "duration_minutes": -5.0}),
    ]
    orphans = [
        {"event_id": f"evt-90001{i}", "customer_id": ORPHAN_IDS[i], "event_type": "login",
         "event_ts": t, "channel": "web", "duration_minutes": None}
        for i in range(3)
    ]
    conflict = dict(b2[0], event_type="plan_change", channel="store", duration_minutes=None)
    drift = [dict(e, app_version="2.1.0") for e in b3[:2]]
    b3_records = drift + b3[2:] + orphans + [conflict, dict(b3[5])]
    b3_lines = [json.dumps(e) for e in b3_records] + bad
    rng.shuffle(b3_lines)

    return {
        "synthetic_events_batch_001.jsonl": [json.dumps(e) for e in b1],
        "synthetic_events_batch_002.jsonl": b2_lines,
        "synthetic_events_batch_003.jsonl": b3_lines,
    }


def write_synthetic(paths: Paths, seed: int = 42, batches: int = 3) -> list[Path]:
    """Write profiles and the first ``batches`` event files. Returns paths written."""
    ids = pd.read_csv(paths.ibm_csv, usecols=["customerID"], dtype=str)["customerID"].tolist()
    written = []

    paths.profiles_json.parent.mkdir(parents=True, exist_ok=True)
    paths.profiles_json.write_text(json.dumps(generate_profiles(ids, seed), indent=2))
    written.append(paths.profiles_json)

    paths.events_dir.mkdir(parents=True, exist_ok=True)
    notice = paths.events_dir / "SYNTHETIC_DATA_NOTICE.txt"
    notice.write_text(
        NOTICE + "\n\nInjected defects:\n"
        + "\n".join(f"- {k}: {v}" for k, v in INJECTED_EVENT_DEFECTS.items()) + "\n"
    )
    for name, lines in list(generate_event_batches(ids, seed).items())[:batches]:
        out = paths.events_dir / name
        out.write_text("\n".join(lines) + "\n")
        written.append(out)
    return written
