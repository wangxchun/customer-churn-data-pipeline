"""W2: synthetic-ID suffix certification (check (d)).

All tests are synthetic-fixture tests. IDs are generated here from a seeded RNG,
and parquet files are written under tmp_path. No test reads ``data/`` or any real
customer ID. A passing test here does NOT mean the real-ID check has been done:
that is W2-Q5, which needs separate approval.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import string
from pathlib import Path

import pandas as pd
import pytest

from churn_pipeline import traffic as tr

LETTERS = string.ascii_uppercase


def synthetic_real_ids(n: int, seed: int = 0, avoid=("SYNT",)) -> list[str]:
    """Unique IDs in the Phase 1 format, standing in for the real ones.
    Suffixes starting with SYNT are avoided unless a test adds one on purpose."""
    rng = random.Random(seed)
    ids: set[str] = set()
    while len(ids) < n:
        suffix = "".join(rng.choice(LETTERS) for _ in range(5))
        if suffix.startswith(avoid):
            continue
        ids.add(f"{rng.randrange(10_000):04d}-{suffix}")
    return sorted(ids)


def write_ids(path: Path, ids: list, *, with_split: bool = True) -> Path:
    """A split-assignments-shaped parquet. The split column is a sentinel: it must never be read."""
    df = pd.DataFrame({"customer_id": pd.array(ids, dtype="object")})
    if with_split:
        df["split"] = "SENTINEL-must-not-be-read"
    df.to_parquet(path, index=False)
    return path


def binding_of(path: Path) -> dict:
    st = os.stat(path)
    return {"expected_size": st.st_size, "expected_mtime_ns": st.st_mtime_ns}


def certify_file(path: Path, suffix: str = "SYNTR", expected_count: int | None = None, **kw):
    n = expected_count if expected_count is not None else len(pd.read_parquet(path, columns=["customer_id"]))
    return tr.certify_suffix_from_file(path, suffix, expected_count=n, **{**binding_of(path), **kw})


@pytest.fixture
def ids() -> list[str]:
    return synthetic_real_ids(500)


@pytest.fixture
def id_file(tmp_path, ids) -> Path:
    return write_ids(tmp_path / "split_assignments.parquet", ids)


@pytest.fixture
def parquet_spy(monkeypatch):
    calls = []
    original = pd.read_parquet

    def spy(path, *args, **kwargs):
        calls.append({"file": Path(path).name, "columns": kwargs.get("columns")})
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", spy)
    return calls


def assert_no_ids(text: str, ids: list[str]) -> None:
    for cid in ids:
        assert cid not in text, "a customer ID leaked into the output"


# ---------------------------------------------------------------------------
# 1. Successful certification
# ---------------------------------------------------------------------------
def test_unused_suffix_is_certified_with_counts_only(id_file, ids):
    cert = certify_file(id_file, "SYNTR", expected_count=500)
    s = cert.summary()
    assert s == {"status": "certified", "suffix": "SYNTR", "n_ids": 500, "n_unique": 500,
                 "n_format_violations": 0, "expected_count": 500, "suffix_in_use": False,
                 "input_file": "split_assignments.parquet",
                 "input_size": os.stat(id_file).st_size,
                 "input_mtime_ns": os.stat(id_file).st_mtime_ns}
    assert_no_ids(json.dumps(s) + repr(cert), ids)


def test_certified_suffix_ids_cannot_equal_any_input_id(id_file, ids):
    cert = certify_file(id_file, "SYNTR", expected_count=500)
    assert not set(tr.synthetic_ids(cert, tr.MAX_IDS_PER_SUFFIX)) & set(ids)


# ---------------------------------------------------------------------------
# 2-3. One suffix per call; candidates are independent; no fallback
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("suffix", tr.CANDIDATE_SUFFIXES)
def test_each_candidate_is_certified_independently(tmp_path, ids, suffix):
    others = [s for s in tr.CANDIDATE_SUFFIXES if s != suffix]
    # The other two candidates are in use; this one is not.
    path = write_ids(tmp_path / "ids.parquet", ids + [f"0001-{others[0]}", f"0002-{others[1]}"])
    assert certify_file(path, suffix).suffix == suffix
    for other in others:
        with pytest.raises(tr.SuffixCertificationError, match=f"suffix {other} is used by 1"):
            certify_file(path, other)


def test_a_used_suffix_fails_closed_without_falling_back(tmp_path, ids, parquet_spy):
    path = write_ids(tmp_path / "ids.parquet", ids + ["0042-SYNTR"])
    with pytest.raises(tr.SuffixCertificationError) as exc:
        certify_file(path, "SYNTR")
    msg = str(exc.value)
    assert "SYNTR" in msg and "SYNTS" not in msg and "SYNTT" not in msg
    assert "0042-SYNTR" not in msg
    # one read for the helper's count, one for the certification; never a second suffix
    assert all(c["columns"] == ["customer_id"] for c in parquet_spy)


@pytest.mark.parametrize("bad", [["SYNTR", "SYNTS"], ("SYNTR",), None, 5])
def test_only_a_single_suffix_string_is_accepted(ids, bad):
    with pytest.raises(tr.SuffixCertificationError):
        tr.certify_suffix(ids, bad, expected_count=len(ids),
                          binding=tr.InputBinding("x.parquet", 1, 1))


@pytest.mark.parametrize("suffix,why", [("SYNTX", "not a pre-declared"), ("CANRY", "reserved"),
                                        ("ZZSYN", "reserved"), ("ABCDE", "reserved"),
                                        ("syntr", "five uppercase"), ("SYNT", "five uppercase")])
def test_only_pre_declared_unreserved_candidates_are_allowed(ids, suffix, why):
    with pytest.raises(tr.SuffixCertificationError, match=why):
        tr.certify_suffix(ids, suffix, expected_count=len(ids),
                          binding=tr.InputBinding("x.parquet", 1, 1))


def test_candidates_do_not_overlap_reserved_namespaces():
    assert not set(tr.CANDIDATE_SUFFIXES) & set(tr.RESERVED_SUFFIXES)
    assert len(set(tr.CANDIDATE_SUFFIXES)) == 3


# ---------------------------------------------------------------------------
# 4. The 10,000-ID cap per suffix
# ---------------------------------------------------------------------------
@pytest.fixture
def cert(id_file):
    return certify_file(id_file, "SYNTR", expected_count=500)


def test_cap_allows_exactly_ten_thousand_ids(cert):
    out = tr.synthetic_ids(cert, 10_000)
    assert len(out) == len(set(out)) == 10_000
    assert out[0] == "0000-SYNTR" and out[-1] == "9999-SYNTR"
    assert tr.synthetic_ids(cert, 1, start=9_999) == ["9999-SYNTR"]


@pytest.mark.parametrize("n,start", [(10_001, 0), (2, 9_999), (1, 10_000), (0, 0), (-1, 0),
                                     (1, -1), (True, 0), (1.0, 0)])
def test_cap_and_arguments_are_enforced(cert, n, start):
    with pytest.raises(tr.SuffixCertificationError):
        tr.synthetic_ids(cert, n, start=start)


def test_synthetic_ids_need_a_real_certificate(ids):
    with pytest.raises(tr.SuffixCertificationError):
        tr.synthetic_ids({"suffix": "SYNTR", "status": "certified"}, 1)


# ---------------------------------------------------------------------------
# 5. Duplicates (check (b)) and collisions (check (d))
# ---------------------------------------------------------------------------
def test_duplicate_ids_fail_closed(tmp_path, ids):
    path = write_ids(tmp_path / "ids.parquet", ids + [ids[0], ids[1]])
    with pytest.raises(tr.SuffixCertificationError, match="2 duplicate"):
        certify_file(path)


@pytest.mark.parametrize("colliding", ["0000-SYNTR", "9999-SYNTR", "4321-SYNTR"])
def test_any_real_id_with_the_suffix_fails_closed(tmp_path, ids, colliding):
    path = write_ids(tmp_path / "ids.parquet", ids + [colliding])
    with pytest.raises(tr.SuffixCertificationError, match="used by 1 real ID") as exc:
        certify_file(path)
    assert colliding not in str(exc.value)


def test_expected_count_must_match(id_file):
    with pytest.raises(tr.SuffixCertificationError, match="read 500 IDs, expected 7043"):
        certify_file(id_file, expected_count=7043)


# ---------------------------------------------------------------------------
# 6. Missing column, wrong types, invalid data
# ---------------------------------------------------------------------------
def test_missing_customer_id_column_fails_closed(tmp_path):
    path = tmp_path / "ids.parquet"
    pd.DataFrame({"split": ["train"]}).to_parquet(path, index=False)
    with pytest.raises(tr.SuffixCertificationError, match="cannot read the customer_id column"):
        certify_file(path, expected_count=1)


@pytest.mark.parametrize("bad", [None, float("nan"), "", "1234-ABCD", "12345-ABCDE",
                                 "1234-abcde", " 1234-ABCDE", "1234-SYNTRX"])
def test_invalid_ids_fail_closed(tmp_path, ids, bad):
    path = write_ids(tmp_path / "ids.parquet", ids + [bad])
    with pytest.raises(tr.SuffixCertificationError, match="1 ID\\(s\\) are missing or not in"):
        certify_file(path, expected_count=len(ids) + 1)


def test_non_string_id_column_fails_closed(tmp_path):
    path = tmp_path / "ids.parquet"
    pd.DataFrame({"customer_id": [1, 2, 3]}).to_parquet(path, index=False)
    with pytest.raises(tr.SuffixCertificationError, match="3 ID\\(s\\) are missing"):
        certify_file(path, expected_count=3)


def test_empty_input_fails_closed(tmp_path):
    path = write_ids(tmp_path / "ids.parquet", [])
    with pytest.raises(tr.SuffixCertificationError, match="no customer IDs"):
        certify_file(path, expected_count=1)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "7043"])
def test_expected_count_must_be_a_positive_integer(ids, bad):
    with pytest.raises(tr.SuffixCertificationError, match="expected_count"):
        tr.certify_suffix(ids, "SYNTR", expected_count=bad,
                          binding=tr.InputBinding("x.parquet", 1, 1))


# ---------------------------------------------------------------------------
# 7-8. Input version binding (size + mtime), read errors, changes during the read
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["expected_size", "expected_mtime_ns"])
def test_binding_mismatch_refuses_before_reading(id_file, parquet_spy, field):
    b = binding_of(id_file)
    b[field] += 1
    with pytest.raises(tr.SuffixCertificationError, match="differs from the approved binding"):
        tr.certify_suffix_from_file(id_file, "SYNTR", expected_count=500, **b)
    assert parquet_spy == []


def test_missing_file_fails_closed(tmp_path):
    with pytest.raises(tr.SuffixCertificationError, match="not found"):
        tr.certify_suffix_from_file(tmp_path / "nope.parquet", "SYNTR", expected_count=1,
                                    expected_size=1, expected_mtime_ns=1)


def test_unreadable_file_fails_closed(tmp_path):
    path = tmp_path / "ids.parquet"
    path.write_bytes(b"not a parquet file")
    with pytest.raises(tr.SuffixCertificationError, match="cannot read"):
        tr.certify_suffix_from_file(path, "SYNTR", expected_count=1, **binding_of(path))


def test_file_changed_during_the_read_fails_closed(id_file, monkeypatch):
    original = pd.read_parquet
    b = binding_of(id_file)

    def read_then_touch(path, *args, **kwargs):
        frame = original(path, *args, **kwargs)
        os.utime(path, ns=(b["expected_mtime_ns"] + 10**9, b["expected_mtime_ns"] + 10**9))
        return frame
    monkeypatch.setattr(pd, "read_parquet", read_then_touch)
    with pytest.raises(tr.SuffixCertificationError, match="changed while it was being read"):
        tr.certify_suffix_from_file(id_file, "SYNTR", expected_count=500, **b)


def test_certificate_is_void_once_the_input_changes(cert, id_file):
    tr.verify_binding(cert, id_file)  # unchanged: fine
    st = os.stat(id_file)
    os.utime(id_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    with pytest.raises(tr.SuffixCertificationError, match="no longer valid"):
        tr.verify_binding(cert, id_file)


# ---------------------------------------------------------------------------
# 9-10. Fail closed everywhere, and no IDs in any message, return value or log
# ---------------------------------------------------------------------------
def test_no_failure_path_leaks_an_id_or_returns_a_certificate(tmp_path, ids, caplog):
    caplog.set_level(logging.DEBUG)
    cases = {
        "collision": ids + ["0007-SYNTR"],
        "duplicate": ids + [ids[3]],
        "invalid": ids + ["9-BAD"],
    }
    messages = []
    for name, values in cases.items():
        path = write_ids(tmp_path / f"{name}.parquet", values)
        with pytest.raises(tr.SuffixCertificationError) as exc:
            certify_file(path)
        messages.append(str(exc.value) + repr(exc.value))
        # No exception chain at all: nothing from pandas/pyarrow (which could carry
        # paths or values) travels with the error.
        assert exc.value.__cause__ is None
        assert exc.value.__context__ is None
    every_id = ids + ["0007-SYNTR", "9-BAD"]
    assert_no_ids(" ".join(messages) + caplog.text, every_id)
    assert caplog.records == []  # the module does not log


def test_read_errors_carry_no_exception_chain(tmp_path):
    """A pyarrow error message may contain a path; it must not ride along as __context__."""
    path = tmp_path / "ids.parquet"
    path.write_bytes(b"not a parquet file")
    with pytest.raises(tr.SuffixCertificationError, match="cannot read") as exc:
        tr.certify_suffix_from_file(path, "SYNTR", expected_count=1, **binding_of(path))
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert str(tmp_path) not in str(exc.value)


# ---------------------------------------------------------------------------
# Invalid parameters are refused before ANY file operation (stat, is_file, read)
# ---------------------------------------------------------------------------
@pytest.fixture
def file_access_spy(monkeypatch, id_file):
    """Records every os.stat (also used by Path.is_file/exists) and parquet read of the input."""
    stats, reads = [], []
    real_stat, real_read = os.stat, pd.read_parquet
    target = str(id_file)

    def stat_spy(path, *args, **kwargs):
        if str(path) == target:
            stats.append(str(path))
        return real_stat(path, *args, **kwargs)

    def read_spy(path, *args, **kwargs):
        reads.append(str(path))
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(os, "stat", stat_spy)
    monkeypatch.setattr(pd, "read_parquet", read_spy)
    return {"stats": stats, "reads": reads}


def valid_kwargs(path: Path) -> dict:
    st = os.stat(path)
    return {"suffix": "SYNTR", "expected_count": 500, "expected_size": st.st_size,
            "expected_mtime_ns": st.st_mtime_ns}


INVALID_PARAMETERS = [
    ("expected_count", 0), ("expected_count", -1), ("expected_count", True),
    ("expected_count", 1.0), ("expected_count", "500"), ("expected_count", None),
    ("expected_size", True), ("expected_size", False), ("expected_size", 0),
    ("expected_size", -5), ("expected_size", 1.5), ("expected_size", "85775"),
    ("expected_size", None),
    ("expected_mtime_ns", True), ("expected_mtime_ns", False), ("expected_mtime_ns", 0),
    ("expected_mtime_ns", -1), ("expected_mtime_ns", 1.0), ("expected_mtime_ns", "1"),
    ("expected_mtime_ns", None),
    ("suffix", "SYNTX"), ("suffix", "CANRY"), ("suffix", "syntr"), ("suffix", None),
    ("suffix", ["SYNTR"]),
]


@pytest.mark.parametrize("name,value", INVALID_PARAMETERS,
                         ids=[f"{n}={v!r}" for n, v in INVALID_PARAMETERS])
def test_invalid_parameters_are_refused_before_any_file_access(id_file, file_access_spy,
                                                               name, value):
    kwargs = valid_kwargs(id_file)
    file_access_spy["stats"].clear()          # valid_kwargs itself stat()s the file
    kwargs[name] = value
    suffix = kwargs.pop("suffix")
    with pytest.raises(tr.SuffixCertificationError):
        tr.certify_suffix_from_file(id_file, suffix, **kwargs)
    assert file_access_spy["stats"] == [], "the input was stat()ed before validation"
    assert file_access_spy["reads"] == [], "the input was read before validation"


@pytest.mark.parametrize("bad_path", [None, 5, b"/tmp/x.parquet", "", ["x.parquet"]])
def test_invalid_path_is_refused_before_any_file_access(id_file, file_access_spy, bad_path):
    kwargs = valid_kwargs(id_file)
    file_access_spy["stats"].clear()
    suffix = kwargs.pop("suffix")
    with pytest.raises(tr.SuffixCertificationError, match="path"):
        tr.certify_suffix_from_file(bad_path, suffix, **kwargs)
    assert file_access_spy["reads"] == []


def test_valid_parameters_still_reach_the_single_read(id_file, file_access_spy):
    """Control: the spy does see file access when the parameters are valid."""
    kwargs = valid_kwargs(id_file)
    file_access_spy["stats"].clear()
    suffix = kwargs.pop("suffix")
    tr.certify_suffix_from_file(id_file, suffix, **kwargs)
    assert file_access_spy["stats"] != [] and len(file_access_spy["reads"]) == 1


# ---------------------------------------------------------------------------
# 11. Only customer_id is read, exactly once per certification
# ---------------------------------------------------------------------------
def test_only_the_customer_id_column_is_read_once(id_file, parquet_spy):
    tr.certify_suffix_from_file(id_file, "SYNTR", expected_count=500, **binding_of(id_file))
    assert parquet_spy == [{"file": "split_assignments.parquet", "columns": ["customer_id"]}]


def test_the_split_column_never_reaches_the_certifier(id_file, monkeypatch):
    original = pd.read_parquet
    seen_columns = []

    def recording(path, *args, **kwargs):
        frame = original(path, *args, **kwargs)
        seen_columns.extend(frame.columns)
        return frame
    monkeypatch.setattr(pd, "read_parquet", recording)
    tr.certify_suffix_from_file(id_file, "SYNTR", expected_count=500, **binding_of(id_file))
    assert seen_columns == ["customer_id"]


# ---------------------------------------------------------------------------
# 12. No side effects: nothing is written, the W1 registry is not touched
# ---------------------------------------------------------------------------
def test_certification_writes_nothing_and_leaves_registry_files_unchanged(tmp_path, ids):
    data = tmp_path / "data"
    registry = data / "processed" / "models" / "registry"
    (registry / "releases" / "rel-x").mkdir(parents=True)
    for name, content in {"production.json": "{}", "audit_log.jsonl": "",
                          "releases/rel-x/holdout_guard.json": "{}",
                          "releases/rel-x/release.json": "{}"}.items():
        (registry / name).write_text(content)
    run = data / "processed" / "models" / "run-x"
    run.mkdir(parents=True)
    for name in ("manifest.json", "selection.json", "model.joblib"):
        (run / name).write_text("synthetic")
    features = data / "processed" / "features"
    features.mkdir(parents=True)
    path = write_ids(features / "split_assignments.parquet", ids)

    def snapshot():
        return {str(p.relative_to(tmp_path)): (hashlib.sha256(p.read_bytes()).hexdigest(),
                                               p.stat().st_mtime_ns)
                for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    before = snapshot()
    cert = tr.certify_suffix_from_file(path, "SYNTR", expected_count=500, **binding_of(path))
    tr.synthetic_ids(cert, 100)
    tr.verify_binding(cert, path)
    assert snapshot() == before


def test_traffic_module_does_not_depend_on_the_release_registry():
    import churn_pipeline.traffic as module
    source = Path(module.__file__).read_text()
    for forbidden in ("from .release", "import release", "joblib", "hmac", "registry_dir",
                      "open(", "write_", "DEFAULT_PATHS"):
        assert forbidden not in source, forbidden
