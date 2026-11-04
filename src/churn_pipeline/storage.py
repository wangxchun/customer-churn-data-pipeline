"""SQLite and Parquet storage.

SQLite holds the *stateful* parts of the pipeline: the append-only event
store, the ingestion manifest, and rejected records. Those need keys,
constraints, and atomic transactions. Parquet holds curated analytical
snapshots, which are columnar, typed, and compressed for ML reads.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .validation import REJECT_COLUMNS

EVENT_TYPES_SQL = "('login','app_session','support_call','payment','plan_change')"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS events (
    event_id          TEXT PRIMARY KEY,           -- natural key -> dedup guarantee
    customer_id       TEXT NOT NULL,
    event_type        TEXT NOT NULL CHECK (event_type IN {EVENT_TYPES_SQL}),
    event_ts          TEXT NOT NULL,              -- fixed-width ISO-8601 UTC, sorts correctly
    channel           TEXT NOT NULL,
    duration_minutes  REAL CHECK (duration_minutes IS NULL OR duration_minutes >= 0),
    payload_hash      TEXT NOT NULL,              -- detects same id + different content
    source_file       TEXT NOT NULL,              -- lineage
    ingested_at       TEXT NOT NULL
);
-- Serves the dominant access pattern: "all events for customer X in time order".
CREATE INDEX IF NOT EXISTS idx_events_customer_ts ON events (customer_id, event_ts);

CREATE TABLE IF NOT EXISTS ingested_files (
    file_sha256   TEXT PRIMARY KEY,               -- content identity, not file name
    file_name     TEXT NOT NULL,
    ingested_at   TEXT NOT NULL,
    valid_records INTEGER NOT NULL,
    inserted      INTEGER NOT NULL,
    duplicates    INTEGER NOT NULL,
    conflicts     INTEGER NOT NULL,
    rejected      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS rejected_records (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    source      TEXT NOT NULL,
    row_ref     TEXT,
    record_key  TEXT,
    reason      TEXT NOT NULL,
    raw         TEXT
);
CREATE INDEX IF NOT EXISTS idx_rejects_source ON rejected_records (source);
"""

_SQL_TYPES = {"string": "TEXT", "str": "TEXT", "Int64": "INTEGER", "Float64": "REAL"}


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def _sql_type(dtype) -> str:
    if isinstance(dtype, pd.DatetimeTZDtype):
        return "TEXT"
    return _SQL_TYPES.get(str(dtype), "TEXT")


def to_sql_value(v):
    if v is None or v is pd.NA or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, pd.Timestamp):
        return format_ts(v)
    if hasattr(v, "item"):  # numpy scalar -> Python scalar
        return v.item()
    return v


def format_ts(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def replace_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame, pk: str) -> None:
    """Full refresh of a small dimension table, with an explicit primary key.

    Done in one transaction: readers never see a half-loaded table.
    """
    df = drop_meta(df)
    cols = ", ".join(
        f'"{c}" {_sql_type(t)}' + (" PRIMARY KEY" if c == pk else "")
        for c, t in df.dtypes.items()
    )
    placeholders = ", ".join("?" for _ in df.columns)
    rows = [tuple(to_sql_value(v) for v in r) for r in df.itertuples(index=False)]
    with conn:
        conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        conn.execute(f'CREATE TABLE "{name}" ({cols})')
        conn.executemany(f'INSERT INTO "{name}" VALUES ({placeholders})', rows)


def insert_rejects(conn: sqlite3.Connection, rejects: pd.DataFrame, run_id: str) -> None:
    """Caller controls the transaction (so rejects commit with their batch)."""
    rows = [
        (run_id, *(to_sql_value(v) for v in r))
        for r in rejects[REJECT_COLUMNS].itertuples(index=False)
    ]
    conn.executemany(
        "INSERT INTO rejected_records (run_id, source, row_ref, record_key, reason, raw) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def replace_rejects_for_sources(
    conn: sqlite3.Connection, rejects: pd.DataFrame, sources: list[str], run_id: str
) -> None:
    """Full-refresh sources are re-validated every run, so replace their rejects."""
    with conn:
        conn.executemany(
            "DELETE FROM rejected_records WHERE source = ?", [(s,) for s in sources]
        )
        insert_rejects(conn, rejects, run_id)


def drop_meta(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if not c.startswith("_")]]


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    drop_meta(df).to_parquet(path, engine="pyarrow", index=False, compression="snappy")
    return path
