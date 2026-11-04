"""Idempotent, incremental ingestion of JSONL event files into SQLite.

Two layers of protection:

1. File level: ``ingested_files`` stores each file's SHA-256. A file whose
   exact bytes were already processed is skipped. This is cheap, but it is
   only an optimisation: a renamed copy has the same hash (skipped, correct),
   and a re-exported file with overlapping events has a new hash.
2. Record level: ``events.event_id`` is the PRIMARY KEY. Every insert is
   ``ON CONFLICT DO NOTHING``, so replaying an event can never create a second
   row. This is the actual correctness guarantee.

On a key collision the stored payload hash is compared with the incoming one:
same hash = harmless redelivery (counted); different hash = conflict
(rejected with a reason; first-seen version wins because events are immutable).

Each file is processed in ONE transaction together with its manifest row and
its rejects. A crash mid-file rolls everything back, so the file is simply
retried on the next run: no partial state, no double counting.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .download import sha256_file
from .sources import EVENT_COLUMNS, load_events_file
from .storage import format_ts, insert_rejects, to_sql_value
from .validation import REJECT_COLUMNS

BUSINESS_FIELDS = [c.name for c in EVENT_COLUMNS]


@dataclass
class FileIngestResult:
    file_name: str
    file_sha256: str
    status: str  # "ingested" | "skipped_already_ingested"
    valid_records: int = 0
    inserted: int = 0
    duplicates: int = 0
    conflicts: int = 0
    rejected: int = 0
    warnings: list[str] | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def payload_hash(record: dict) -> str:
    """Hash of the *normalised* business fields (after typing), so formatting
    differences such as 'Z' vs '+00:00' do not count as different content."""
    canonical = json.dumps({k: record[k] for k in BUSINESS_FIELDS}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _normalise(row: pd.Series) -> dict:
    rec = {k: to_sql_value(row[k]) for k in BUSINESS_FIELDS}
    rec["event_ts"] = format_ts(row["event_ts"])
    return rec


def ingest_event_file(
    conn: sqlite3.Connection, path: Path, run_id: str
) -> FileIngestResult:
    sha = sha256_file(path)
    seen = conn.execute(
        "SELECT file_name, ingested_at FROM ingested_files WHERE file_sha256 = ?", (sha,)
    ).fetchone()
    if seen:
        return FileIngestResult(path.name, sha, "skipped_already_ingested")

    result = load_events_file(path)
    res = FileIngestResult(
        path.name, sha, "ingested",
        valid_records=len(result.valid), warnings=result.warnings,
    )
    now = datetime.now(timezone.utc).isoformat()
    conflict_rows = []

    with conn:  # single transaction: events + rejects + manifest, all or nothing
        for _, row in result.valid.iterrows():
            rec = _normalise(row)
            h = payload_hash(rec)
            cur = conn.execute(
                "INSERT INTO events (event_id, customer_id, event_type, event_ts, channel, "
                "duration_minutes, payload_hash, source_file, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(event_id) DO NOTHING",
                (*(rec[k] for k in BUSINESS_FIELDS), h, path.name, now),
            )
            if cur.rowcount == 1:
                res.inserted += 1
                continue
            stored_hash, first_file = conn.execute(
                "SELECT payload_hash, source_file FROM events WHERE event_id = ?",
                (rec["event_id"],),
            ).fetchone()
            if stored_hash == h:
                res.duplicates += 1
            else:
                res.conflicts += 1
                conflict_rows.append({
                    "source": f"events:{path.name}",
                    "row_ref": row["_row_ref"],
                    "record_key": rec["event_id"],
                    "reason": "conflicting_duplicate_event_id: payload differs from "
                              f"version first ingested from {first_file}",
                    "raw": json.dumps(rec, sort_keys=True),
                })

        all_rejects = pd.concat(
            [result.rejects, pd.DataFrame(conflict_rows, columns=REJECT_COLUMNS)],
            ignore_index=True,
        )
        res.rejected = len(all_rejects)
        insert_rejects(conn, all_rejects, run_id)
        conn.execute(
            "INSERT INTO ingested_files VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (sha, path.name, now, res.valid_records, res.inserted,
             res.duplicates, res.conflicts, res.rejected),
        )
    return res


def ingest_pending(
    conn: sqlite3.Connection, events_dir: Path, run_id: str, files: list[Path] | None = None
) -> list[FileIngestResult]:
    """Ingest the given files, or every *.jsonl in the landing directory.

    Raw files are never moved or deleted: the manifest, not the filesystem,
    records what has been processed, so raw data stays an immutable audit trail.
    """
    targets = files if files is not None else sorted(events_dir.glob("*.jsonl"))
    return [ingest_event_file(conn, p, run_id) for p in targets]


def export_events(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT event_id, customer_id, event_type, event_ts, channel, duration_minutes, "
        "source_file, ingested_at FROM events ORDER BY event_ts, event_id",
        conn,
    )
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True)
    return df
