"""End-to-end runs on the mini fixture data."""
import json
import sqlite3

import pandas as pd
import pytest

from churn_pipeline.cli import main
from churn_pipeline.pipeline import run

from conftest import event, write_jsonl


def store_counts(paths):
    with sqlite3.connect(paths.sqlite_db) as c:
        return {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("events", "ingested_files", "rejected_records", "customers")}


def test_end_to_end_outputs(paths):
    s = run(paths, verify_checksum=False)  # fixture CSV is not the real file
    for name in ("customers", "customer_profiles", "events", "customer_master"):
        assert (paths.curated / f"{name}.parquet").exists()
    master = pd.read_parquet(paths.curated / "customer_master.parquet")
    assert len(master) == 3 and not any(c.startswith("_") for c in master.columns)
    assert s["joins"]["master_rows"] == s["joins"]["customers"] == 3
    assert json.loads(paths.summary_json.read_text())["events"]["store_events"] == 3
    assert "Row-count invariant" in paths.summary_md.read_text()


def test_repeated_runs_are_idempotent(paths):
    run(paths, verify_checksum=False)
    before = store_counts(paths)
    s2 = run(paths, verify_checksum=False)
    assert store_counts(paths) == before
    assert s2["events"]["inserted_this_run"] == 0


def test_incremental_arrival(paths):
    run(paths, verify_checksum=False)
    write_jsonl(paths.events_dir / "batch_002.jsonl",
                [event("evt-1"), event("evt-10", "3333-CCCCC"), "{broken"])
    s = run(paths, verify_checksum=False)
    new = [f for f in s["events"]["files"] if f["file_name"] == "batch_002.jsonl"][0]
    assert (new["inserted"], new["duplicates"], new["rejected"]) == (1, 1, 1)
    assert store_counts(paths)["events"] == 4


def test_rejects_are_inspectable(paths):
    write_jsonl(paths.events_dir / "batch_002.jsonl", ["not json"])
    run(paths, verify_checksum=False)
    with sqlite3.connect(paths.sqlite_db) as c:
        rows = c.execute("SELECT source, reason FROM rejected_records").fetchall()
    assert rows and rows[0][0] == "events:batch_002.jsonl"


def test_cli_refuses_unverified_dataset(paths):
    # The CLI always verifies the pinned checksum, so a different CSV fails loudly.
    with pytest.raises(ValueError, match="Checksum mismatch"):
        main(["--data-dir", str(paths.data_dir), "run"])
