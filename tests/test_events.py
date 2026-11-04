"""Idempotent incremental event ingestion."""
import shutil

import pytest

from churn_pipeline import events as events_mod
from churn_pipeline.events import ingest_event_file, ingest_pending
from churn_pipeline.storage import connect

from conftest import event, write_jsonl


def count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    yield c
    c.close()


def test_reingesting_same_file_is_skipped(conn, tmp_path):
    f = write_jsonl(tmp_path / "b1.jsonl", [event("evt-1"), event("evt-2")])
    first = ingest_event_file(conn, f, "run1")
    second = ingest_event_file(conn, f, "run2")
    assert (first.status, first.inserted) == ("ingested", 2)
    assert second.status == "skipped_already_ingested"
    assert count(conn, "events") == 2
    assert count(conn, "ingested_files") == 1


def test_renamed_copy_is_recognised_by_content(conn, tmp_path):
    f = write_jsonl(tmp_path / "b1.jsonl", [event("evt-1")])
    ingest_event_file(conn, f, "run1")
    copy = shutil.copy(f, tmp_path / "b1_retry.jsonl")
    assert ingest_event_file(conn, copy, "run2").status == "skipped_already_ingested"


def test_replayed_events_in_new_file_are_not_duplicated(conn, tmp_path):
    ingest_event_file(conn, write_jsonl(tmp_path / "b1.jsonl",
                                        [event("evt-1"), event("evt-2")]), "r1")
    # New file (different hash) that re-delivers evt-2 plus one new event.
    res = ingest_event_file(conn, write_jsonl(tmp_path / "b2.jsonl",
                                              [event("evt-2"), event("evt-3")]), "r2")
    assert (res.inserted, res.duplicates, res.conflicts) == (1, 1, 0)
    assert count(conn, "events") == 3
    assert count(conn, "rejected_records") == 0  # redelivery is expected, not an error


def test_duplicate_within_one_file(conn, tmp_path):
    res = ingest_event_file(conn, write_jsonl(tmp_path / "b.jsonl",
                                              [event("evt-1"), event("evt-1")]), "r")
    assert (res.inserted, res.duplicates) == (1, 1)
    assert count(conn, "events") == 1


def test_equivalent_timestamp_formats_are_the_same_event(conn, tmp_path):
    ingest_event_file(conn, write_jsonl(tmp_path / "a.jsonl",
                      [event("evt-1", event_ts="2024-01-01T10:00:00+00:00")]), "r1")
    res = ingest_event_file(conn, write_jsonl(tmp_path / "b.jsonl",
                            [event("evt-1", event_ts="2024-01-01T12:00:00+02:00")]), "r2")
    assert (res.duplicates, res.conflicts) == (1, 0)


def test_same_id_different_payload_is_a_rejected_conflict(conn, tmp_path):
    ingest_event_file(conn, write_jsonl(tmp_path / "a.jsonl", [event("evt-1")]), "r1")
    res = ingest_event_file(conn, write_jsonl(tmp_path / "b.jsonl",
                            [event("evt-1", event_type="payment")]), "r2")
    assert (res.inserted, res.conflicts, res.rejected) == (0, 1, 1)
    # First-seen version is kept unchanged.
    assert conn.execute("SELECT event_type FROM events").fetchone()[0] == "login"
    reason = conn.execute("SELECT reason FROM rejected_records").fetchone()[0]
    assert reason.startswith("conflicting_duplicate_event_id") and "a.jsonl" in reason


def test_malformed_lines_are_recorded_and_good_lines_ingested(conn, tmp_path):
    f = write_jsonl(tmp_path / "b.jsonl", [event("evt-1"), "{oops", event("evt-2")])
    res = ingest_event_file(conn, f, "r")
    assert (res.inserted, res.rejected) == (2, 1)
    row = conn.execute("SELECT source, row_ref, raw FROM rejected_records").fetchone()
    assert row == ("events:b.jsonl", "line 2", "{oops")


def test_unknown_customer_events_are_stored(conn, tmp_path):
    """Referential integrity is checked at join time, not ingestion time,
    so late-arriving customers can still be matched later."""
    res = ingest_event_file(conn, write_jsonl(tmp_path / "b.jsonl",
                            [event("evt-1", "9999-ZZZZZ")]), "r")
    assert res.inserted == 1


def test_crash_mid_file_rolls_back_and_retry_succeeds(conn, tmp_path, monkeypatch):
    f = write_jsonl(tmp_path / "b.jsonl", [event("evt-1"), event("evt-2"), event("evt-3")])
    original = events_mod._normalise
    calls = {"n": 0}

    def flaky(row):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        return original(row)

    monkeypatch.setattr(events_mod, "_normalise", flaky)
    with pytest.raises(RuntimeError):
        ingest_event_file(conn, f, "r1")
    # Nothing partial was committed: no events, no manifest row.
    assert count(conn, "events") == 0
    assert count(conn, "ingested_files") == 0

    monkeypatch.setattr(events_mod, "_normalise", original)
    res = ingest_event_file(conn, f, "r2")
    assert res.inserted == 3 and count(conn, "events") == 3


def test_ingest_pending_processes_only_new_files(conn, tmp_path):
    d = tmp_path / "landing"
    write_jsonl(d / "b1.jsonl", [event("evt-1")])
    assert [r.status for r in ingest_pending(conn, d, "r1")] == ["ingested"]
    write_jsonl(d / "b2.jsonl", [event("evt-2")])
    assert [r.status for r in ingest_pending(conn, d, "r2")] == [
        "skipped_already_ingested", "ingested",
    ]
    assert count(conn, "events") == 2
