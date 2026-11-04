"""Build the customer master table: one row per IBM customer.

Join design:
* The IBM customers are the *spine*: they carry the label (churn), so every
  labelled customer must survive and no customer may be duplicated.
* Profiles are joined 1:1 (deduplicated upstream). ``validate="one_to_one"``
  makes pandas raise instead of silently multiplying rows if that assumption
  ever breaks.
* Events are many-per-customer, so they are AGGREGATED to customer grain
  before joining. Joining raw events would turn 7,043 rows into thousands of
  customer-event rows.
* Unmatched keys on either side are counted, never silently dropped.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from .storage import drop_meta

EVENT_FEATURES_SQL = """
SELECT e.customer_id,
       COUNT(*)                                      AS n_events,
       SUM(e.event_type = 'login')                   AS n_login,
       SUM(e.event_type = 'app_session')             AS n_app_session,
       SUM(e.event_type = 'support_call')            AS n_support_call,
       SUM(e.event_type = 'payment')                 AS n_payment,
       SUM(e.event_type = 'plan_change')             AS n_plan_change,
       COALESCE(SUM(e.duration_minutes), 0)          AS total_duration_minutes,
       MIN(e.event_ts)                               AS first_event_ts,
       MAX(e.event_ts)                               AS last_event_ts
FROM events e
JOIN customers c ON c.customer_id = e.customer_id
GROUP BY e.customer_id
"""

ORPHAN_EVENTS_SQL = """
SELECT COUNT(*) AS n_events, COUNT(DISTINCT e.customer_id) AS n_customers
FROM events e LEFT JOIN customers c ON c.customer_id = e.customer_id
WHERE c.customer_id IS NULL
"""

COUNT_COLS = [
    "n_events", "n_login", "n_app_session", "n_support_call",
    "n_payment", "n_plan_change",
]


@dataclass
class JoinReport:
    customers: int = 0
    customers_with_profile: int = 0
    customers_without_profile: int = 0
    orphan_profiles: int = 0
    orphan_profile_ids: list[str] = field(default_factory=list)
    customers_with_events: int = 0
    customers_without_events: int = 0
    orphan_events: int = 0
    orphan_event_customers: int = 0
    master_rows: int = 0


def event_features(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(EVENT_FEATURES_SQL, conn)
    # Match the key dtype of the curated tables; mismatched key types are a
    # classic source of silent join failures.
    df["customer_id"] = df["customer_id"].astype("string")
    for c in ("first_event_ts", "last_event_ts"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def build_customer_master(
    customers: pd.DataFrame, profiles: pd.DataFrame, conn: sqlite3.Connection
) -> tuple[pd.DataFrame, JoinReport]:
    customers = drop_meta(customers)
    profiles = drop_meta(profiles).rename(columns={"updated_at": "profile_updated_at"})
    rep = JoinReport(customers=len(customers))

    # Outer join + indicator just to *measure* both sides; then keep the spine.
    probe = customers[["customer_id"]].merge(
        profiles[["customer_id"]], on="customer_id", how="outer",
        indicator=True, validate="one_to_one",
    )
    rep.customers_with_profile = int((probe["_merge"] == "both").sum())
    rep.customers_without_profile = int((probe["_merge"] == "left_only").sum())
    orphans = probe.loc[probe["_merge"] == "right_only", "customer_id"]
    rep.orphan_profiles = len(orphans)
    rep.orphan_profile_ids = sorted(orphans.astype(str).tolist())

    master = customers.merge(profiles, on="customer_id", how="left", validate="one_to_one")
    master["has_profile"] = master["region"].notna()

    feats = event_features(conn)
    master = master.merge(feats, on="customer_id", how="left", validate="one_to_one")
    # A customer with no events genuinely had zero events in the observed window,
    # so counts are 0, not unknown. First/last timestamps stay null.
    master[COUNT_COLS] = master[COUNT_COLS].fillna(0).astype("Int64")
    master["total_duration_minutes"] = (
        master["total_duration_minutes"].fillna(0.0).astype("Float64")
    )

    rep.customers_with_events = int((master["n_events"] > 0).sum())
    rep.customers_without_events = int((master["n_events"] == 0).sum())
    n_ev, n_cust = conn.execute(ORPHAN_EVENTS_SQL).fetchone()
    rep.orphan_events, rep.orphan_event_customers = int(n_ev), int(n_cust)
    rep.master_rows = len(master)

    # Hard invariant: the join must neither multiply nor lose labelled customers.
    if rep.master_rows != rep.customers:
        raise AssertionError(
            f"customer master has {rep.master_rows} rows, expected {rep.customers}"
        )
    return master, rep
