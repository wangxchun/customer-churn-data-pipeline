"""Phase 1 orchestration.

Order matters:
1. Full-refresh sources (customers, profiles): small, re-validated every run.
2. Incremental source (events): only new files are processed.
3. Derived outputs (events snapshot, customer master, summary): rebuilt from
   the stores, so they are always consistent with them.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Paths
from .download import sha256_file, verify_ibm_csv
from .events import export_events, ingest_pending
from .merge import build_customer_master
from .sources import load_customers, load_profiles
from .storage import connect, replace_rejects_for_sources, replace_table, write_parquet
from .summary import write_summary


def _source_stats(path: Path, result, sha: str) -> dict:
    return {
        "file": str(path.name),
        "sha256": sha,
        # Every input record ends up in exactly one of valid / rejected.
        "records_read": len(result.valid) + len(result.rejects),
        "valid": len(result.valid),
        "rejected": len(result.rejects),
        "reject_reasons": dict(Counter(result.rejects["reason"])),
    }


def run(paths: Paths, event_files: list[Path] | None = None, verify_checksum: bool = True) -> dict:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ibm_sha = verify_ibm_csv(paths.ibm_csv) if verify_checksum else sha256_file(paths.ibm_csv)

    customers = load_customers(paths.ibm_csv)
    profiles = load_profiles(paths.profiles_json)
    warnings = customers.warnings + profiles.warnings

    conn = connect(paths.sqlite_db)
    try:
        replace_table(conn, "customers", customers.valid, pk="customer_id")
        replace_table(conn, "customer_profiles", profiles.valid, pk="customer_id")
        replace_rejects_for_sources(
            conn,
            pd.concat([customers.rejects, profiles.rejects], ignore_index=True),
            ["ibm_customers", "profiles"],
            run_id,
        )

        file_results = ingest_pending(conn, paths.events_dir, run_id, event_files)
        for r in file_results:
            warnings += r.warnings or []

        events = export_events(conn)
        master, joins = build_customer_master(customers.valid, profiles.valid, conn)

        store_files = conn.execute("SELECT COUNT(*) FROM ingested_files").fetchone()[0]
        store_rejects = dict(conn.execute(
            "SELECT reason, COUNT(*) FROM rejected_records WHERE source LIKE 'events:%' "
            "GROUP BY reason ORDER BY 2 DESC"
        ).fetchall())
    finally:
        conn.close()

    outputs = {
        "customers": (customers.valid, paths.curated / "customers.parquet"),
        "customer_profiles": (profiles.valid, paths.curated / "customer_profiles.parquet"),
        "events": (events, paths.curated / "events.parquet"),
        "customer_master": (master, paths.curated / "customer_master.parquet"),
    }
    out_stats = {}
    for name, (df, path) in outputs.items():
        write_parquet(df, path)
        out_stats[name] = {"rows": len(df), "bytes": path.stat().st_size,
                           "path": str(path.relative_to(paths.data_dir))}
    out_stats["sqlite_db"] = {"rows": "-", "bytes": paths.sqlite_db.stat().st_size,
                              "path": str(paths.sqlite_db.relative_to(paths.data_dir))}
    out_stats["ibm_csv (raw input, for comparison)"] = {
        "rows": len(customers.valid) + len(customers.rejects),
        "bytes": paths.ibm_csv.stat().st_size,
        "path": str(paths.ibm_csv.relative_to(paths.data_dir)),
    }

    summary = {
        "run_id": run_id,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "ibm_customers": _source_stats(paths.ibm_csv, customers, ibm_sha),
            "profiles": _source_stats(
                paths.profiles_json, profiles, sha256_file(paths.profiles_json)
            ),
        },
        "events": {
            "files": [r.as_dict() for r in file_results],
            "inserted_this_run": sum(r.inserted for r in file_results),
            "store_events": len(events),
            "store_files": store_files,
            "store_reject_reasons": store_rejects,
        },
        "joins": asdict(joins),
        "outputs": out_stats,
        "warnings": warnings,
    }
    write_summary(summary, paths.summary_md, paths.summary_json)
    return summary
