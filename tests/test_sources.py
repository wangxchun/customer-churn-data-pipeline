"""Malformed input handling for the JSON and JSONL sources."""
import pytest

from churn_pipeline.sources import load_events_file, load_profiles
from churn_pipeline.validation import SchemaError

from conftest import event, profile, write_jsonl, write_profiles


def test_profiles_invalid_json_fails_whole_file(tmp_path):
    p = tmp_path / "profiles.json"
    p.write_text('{"records": [{"customer_id": "1111-AAAAA"},')  # truncated
    with pytest.raises(SchemaError, match="not valid JSON"):
        load_profiles(p)


def test_profiles_wrong_top_level_shape(tmp_path):
    p = tmp_path / "profiles.json"
    p.write_text("[]")
    with pytest.raises(SchemaError, match="records"):
        load_profiles(p)


def test_profiles_bad_records_rejected_rest_kept(tmp_path):
    p = write_profiles(tmp_path / "p.json", [
        profile("1111-AAAAA"),
        "not-an-object",
        {k: v for k, v in profile("2222-BBBBB").items() if k != "region"},  # missing key
        profile(1234),                                                    # wrong type
        profile("3333-CCCCC", signup_channel="carrier_pigeon"),
    ])
    r = load_profiles(p)
    assert r.valid["customer_id"].tolist() == ["1111-AAAAA"]
    reasons = r.rejects["reason"].tolist()
    assert len(reasons) == 4
    assert "record is not a JSON object" in reasons
    assert any("region: missing required value" in x for x in reasons)
    assert any("expected string, got int" in x for x in reasons)
    assert any("signup_channel" in x for x in reasons)


def test_profiles_newer_record_supersedes_older(tmp_path):
    p = write_profiles(tmp_path / "p.json", [
        profile("1111-AAAAA", updated_at="2024-01-01T00:00:00+00:00", region="South"),
        profile("1111-AAAAA", updated_at="2024-06-01T00:00:00+00:00", region="West"),
    ])
    r = load_profiles(p)
    assert r.valid["region"].tolist() == ["West"]
    assert r.rejects["reason"].tolist() == ["superseded_by_newer_record"]


def test_jsonl_bad_line_only_costs_that_line(tmp_path):
    p = write_jsonl(tmp_path / "e.jsonl", [
        event("evt-1"),
        '{"event_id": "evt-2", "customer_id": ',  # truncated line
        "",                                        # blank lines are ignored
        "42",
        event("evt-3", event_type="teleport"),
        event("evt-4", event_ts="2024-01-01 10:00:00"),
        event("evt-5", duration_minutes="ten"),
        event("evt-6"),
    ])
    r = load_events_file(p)
    assert r.valid["event_id"].tolist() == ["evt-1", "evt-6"]
    assert len(r.rejects) == 5
    by_line = dict(zip(r.rejects["row_ref"], r.rejects["reason"]))
    assert by_line["line 2"].startswith("malformed JSON")
    assert by_line["line 4"] == "record is not a JSON object"
    assert "teleport" in by_line["line 5"]
    assert "no timezone" in by_line["line 6"]
    assert "not a number" in by_line["line 7"]


def test_jsonl_unexpected_field_is_a_warning_not_a_reject(tmp_path):
    p = write_jsonl(tmp_path / "e.jsonl", [event("evt-1", app_version="2.0")])
    r = load_events_file(p)
    assert len(r.valid) == 1 and r.rejects.empty
    assert any("app_version" in w for w in r.warnings)
