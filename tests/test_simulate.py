"""W2 / D7: synthetic traffic source and generator.

Every test here is a synthetic-fixture test. The master comes from ``make_master()``
(random, never a real customer) and is written under tmp_path, together with a split
file built by ``build_features``. Nothing reads ``data/``. Passing tests do not mean
D7 has been run on real data; that needs separate approval.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from churn_pipeline import inference, traffic
from churn_pipeline import simulate as sim
from churn_pipeline.config import Paths
from churn_pipeline.features import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN, build_features
from test_features import make_master

N_MASTER = 400


@pytest.fixture
def env(tmp_path):
    master = make_master(N_MASTER)
    assignments = build_features(master).assignments
    paths = Paths(tmp_path / "data")
    paths.curated.mkdir(parents=True)
    paths.features.mkdir(parents=True)
    master.to_parquet(paths.curated / "customer_master.parquet", index=False)
    assignments.to_parquet(paths.features / "split_assignments.parquet", index=False)
    split_of = dict(zip(assignments[ID_COLUMN].astype(str), assignments["split"].astype(str)))
    return {"paths": paths, "all_ids": sorted(split_of),
            "train_ids": sorted(i for i, s in split_of.items() if s == "train"),
            "non_train_ids": sorted(i for i, s in split_of.items() if s != "train")}


def load(env, **kw):
    return sim.load_traffic_source(env["paths"], **{"suffix": "SYNTR",
                                                    "expected_count": N_MASTER, **kw})


@pytest.fixture
def spies(monkeypatch):
    rec = {"reads": [], "binding_calls": [], "certify_ids": []}
    real_read, real_binding, real_certify = pd.read_parquet, traffic.input_binding, \
        traffic.certify_suffix

    def read(path, *a, **kw):
        # Copy the filter values now: the implementation clears its ID lists afterwards.
        filters = [(c, o, list(v)) for c, o, v in kw["filters"]] if kw.get("filters") else None
        frame = real_read(path, *a, **kw)
        rec["reads"].append({"file": Path(path).name, "columns": kw.get("columns"),
                             "filters": filters, "returned_ids":
                             sorted(frame[ID_COLUMN].astype(str)) if ID_COLUMN in frame else None,
                             "returned_columns": list(frame.columns)})
        return frame

    def binding(path):
        rec["binding_calls"].append(Path(path).name)
        return real_binding(path)

    def certify(real_ids, suffix, **kw):
        ids = list(real_ids)
        rec["certify_ids"].append({"ids": sorted(map(str, ids)), "suffix": suffix, **kw})
        return real_certify(ids, suffix, **kw)
    monkeypatch.setattr(pd, "read_parquet", read)
    monkeypatch.setattr(traffic, "input_binding", binding)
    monkeypatch.setattr(traffic, "certify_suffix", certify)
    return rec


# ---------------------------------------------------------------------------
# Approved data access (D7-Q1 to Q4)
# ---------------------------------------------------------------------------
def test_exactly_one_stat_and_two_reads_with_approved_columns(env, spies):
    load(env)
    assert spies["binding_calls"] == ["split_assignments.parquet"]          # D7-Q3
    split_read, master_read = spies["reads"]                                 # exactly two
    assert split_read["file"] == "split_assignments.parquet"
    assert split_read["columns"] == ["customer_id", "split"] and split_read["filters"] is None
    assert master_read["file"] == "customer_master.parquet"
    assert master_read["columns"] == [ID_COLUMN, *FEATURE_COLUMNS]
    (col, op, values), = master_read["filters"]
    assert (col, op) == (ID_COLUMN, "in") and sorted(values) == env["train_ids"]


def test_master_read_returns_train_rows_only_without_labels(env, spies):
    load(env)
    master_read = spies["reads"][1]
    assert master_read["returned_ids"] == env["train_ids"]
    assert not set(master_read["returned_ids"]) & set(env["non_train_ids"])  # no test/val rows
    assert TARGET_COLUMN not in master_read["returned_columns"]
    assert not {"region", "n_events", "has_profile"} & set(master_read["returned_columns"])


def test_collision_check_uses_the_full_id_set_before_the_master_read(env, spies):
    src = load(env)
    (call,) = spies["certify_ids"]
    assert call["ids"] == env["all_ids"]  # all splits, not just train (W2-Q7)
    assert call["suffix"] == "SYNTR" and call["expected_count"] == N_MASTER
    assert call["binding"] == traffic.input_binding(env["paths"].features / "split_assignments.parquet")
    assert src.certificate.suffix == "SYNTR" and src.certificate.n_ids == N_MASTER


def test_source_features_carry_no_customer_ids(env):
    src = load(env)
    assert list(src.features.columns) == FEATURE_COLUMNS
    assert len(src.features) == len(env["train_ids"])


# ---------------------------------------------------------------------------
# Fail closed: nothing more is read and nothing is produced
# ---------------------------------------------------------------------------
def test_suffix_collision_stops_before_the_master_is_read(env, spies):
    p = env["paths"].features / "split_assignments.parquet"
    a = pd.read_parquet(p)
    a.loc[0, ID_COLUMN] = "0001-SYNTR"   # a "real" ID using the suffix
    a.to_parquet(p, index=False)
    spies["reads"].clear()
    with pytest.raises(traffic.SuffixCertificationError, match="used by 1") as exc:
        load(env)
    assert [r["file"] for r in spies["reads"]] == ["split_assignments.parquet"]
    assert "0001-SYNTR" not in str(exc.value)


def test_expected_count_mismatch_stops_before_the_master_is_read(env, spies):
    with pytest.raises(traffic.SuffixCertificationError, match="expected 7043"):
        load(env, expected_count=7043)
    assert [r["file"] for r in spies["reads"]] == ["split_assignments.parquet"]


@pytest.mark.parametrize("bad", ["SYNTX", "CANRY", None])
def test_invalid_suffix_is_refused_before_any_file_access(env, spies, bad):
    with pytest.raises(traffic.SuffixCertificationError):
        load(env, suffix=bad)
    assert spies["binding_calls"] == [] and spies["reads"] == []


def test_master_missing_a_train_row_fails_closed(env):
    p = env["paths"].curated / "customer_master.parquet"
    m = pd.read_parquet(p)
    m[m[ID_COLUMN] != env["train_ids"][0]].to_parquet(p, index=False)
    with pytest.raises(sim.TrafficSourceError, match="refusing") as exc:
        load(env)
    assert env["train_ids"][0] not in str(exc.value)


def test_missing_split_file_fails_without_reading_master(env, spies):
    (env["paths"].features / "split_assignments.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        load(env)
    assert spies["reads"] == []


# ---------------------------------------------------------------------------
# Generated traffic
# ---------------------------------------------------------------------------
@pytest.fixture
def src(env):
    return load(env)


def test_traffic_uses_certified_synthetic_ids_only(src, env):
    df = sim.generate_traffic(src, "none", 50, seed=1)
    assert list(df.columns) == [ID_COLUMN, *FEATURE_COLUMNS]
    assert list(df[ID_COLUMN]) == [f"{i:04d}-SYNTR" for i in range(50)]
    assert not set(df[ID_COLUMN]) & set(env["all_ids"])
    assert TARGET_COLUMN not in df.columns


def test_same_seed_same_traffic_different_seed_different(src):
    a = sim.generate_traffic(src, "none", 100, seed=7)
    pd.testing.assert_frame_equal(a, sim.generate_traffic(src, "none", 100, seed=7))
    assert not a.equals(sim.generate_traffic(src, "none", 100, seed=8))


@pytest.mark.parametrize("n", [0, -1, 10_001])
def test_id_cap_and_size_are_enforced(src, n):
    with pytest.raises(traffic.SuffixCertificationError):
        sim.generate_traffic(src, "none", n, seed=0)


def test_unknown_scenario_is_rejected(src):
    with pytest.raises(ValueError):
        sim.generate_traffic(src, "teleport", 10, seed=0)


def codes(df):
    r = inference.validate_inference_input(df).rejects
    return set(zip(r["column"], r["reason_code"])), len(r)


def test_scenarios_produce_the_expected_reject_or_drift_pattern(src):
    n = 2000
    none = sim.generate_traffic(src, "none", n, seed=3)
    assert codes(none) == (set(), 0)

    up = sim.generate_traffic(src, "price_up", n, seed=3)
    pd.testing.assert_series_equal(up["monthly_charges"], (none["monthly_charges"] * 1.15).round(2))
    assert codes(up) == (set(), 0)

    mix = sim.generate_traffic(src, "contract_mix", n, seed=3)
    assert (mix["contract"] == "Month-to-month").mean() == pytest.approx(0.8, abs=1e-9)
    assert codes(mix) == (set(), 0)

    unseen = sim.generate_traffic(src, "unseen_payment", n, seed=3)
    got, k = codes(unseen)
    assert got == {("payment_method", "not_allowed")} and 0.15 < k / n < 0.25

    nulls = sim.generate_traffic(src, "nulls", n, seed=3)
    got, k = codes(nulls)
    assert got == {("monthly_charges", "null_not_allowed")} and 0.05 < k / n < 0.15


# ---------------------------------------------------------------------------
# Writing (only ever to an explicit directory; tests use tmp_path)
# ---------------------------------------------------------------------------
def test_write_is_complete_never_overwrites_and_marks_synthetic(src, tmp_path, env):
    df = sim.generate_traffic(src, "nulls", 30, seed=2)
    out = tmp_path / "out"
    path = sim.write_traffic(df, out, "batch-1", scenario="nulls", seed=2,
                             certificate=src.certificate)
    pd.testing.assert_frame_equal(pd.read_parquet(path), df)
    meta = json.loads((out / "batch-1.json").read_text())
    assert meta["synthetic"] is True and meta["scenario"] == "nulls" and meta["n"] == 30
    assert meta["id_suffix_certification"]["suffix"] == "SYNTR"
    assert sorted(p.name for p in out.iterdir()) == ["batch-1.json", "batch-1.parquet"]
    for cid in env["all_ids"]:
        assert cid not in (out / "batch-1.json").read_text()
    with pytest.raises(FileExistsError):
        sim.write_traffic(df, out, "batch-1", scenario="nulls", seed=2,
                          certificate=src.certificate)


@pytest.mark.parametrize("bad", ["", ".hidden", "a/b"])
def test_write_rejects_unsafe_names(src, tmp_path, bad):
    with pytest.raises(ValueError):
        sim.write_traffic(sim.generate_traffic(src, "none", 1, seed=0), tmp_path, bad,
                          scenario="none", seed=0, certificate=src.certificate)


def test_loading_and_generating_write_nothing(env):
    root = env["paths"].data_dir

    def snapshot():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*")) if p.is_file()}
    before = snapshot()
    sim.generate_traffic(load(env), "unseen_payment", 100, seed=0)
    assert snapshot() == before


def test_module_does_not_use_the_broad_loader_or_other_files():
    source = Path(sim.__file__).read_text()
    code = source.split('"""', 2)[2]  # skip the module docstring
    for forbidden in ("from .tracking", "load_verified_inputs(", "train.parquet",
                      "feature_report", "DEFAULT_PATHS", "os.stat("):
        assert forbidden not in code, forbidden
    assert code.count("read_parquet(") == 2


# ---------------------------------------------------------------------------
# CLI: simulate-traffic (always --data-dir and --out-dir under tmp_path)
# ---------------------------------------------------------------------------
from churn_pipeline import cli  # noqa: E402


def cli_args(env, out_dir, **over):
    args = {"--suffix": "SYNTR", "--expected-count": str(N_MASTER), "--scenario": "none",
            "--n": "25", "--seed": "5", "--out-dir": str(out_dir), "--name": "b1", **over}
    flat = []
    for k, v in args.items():
        if v is not None:
            flat += [k, v]
    return ["--data-dir", str(env["paths"].data_dir), "simulate-traffic", *flat]


def batch_files(out_dir: Path) -> list[str]:
    return sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else []


def test_cli_success_writes_one_complete_batch(env, tmp_path, capsys):
    out = tmp_path / "traffic"
    assert cli.main(cli_args(env, out)) == 0
    assert batch_files(out) == ["b1.json", "b1.parquet"]
    df = pd.read_parquet(out / "b1.parquet")
    assert len(df) == 25 and df[ID_COLUMN].str.endswith("-SYNTR").all()
    assert not set(df[ID_COLUMN]) & set(env["all_ids"])
    assert json.loads((out / "b1.json").read_text())["synthetic"] is True
    assert "Synthetic traffic written" in capsys.readouterr().out


@pytest.mark.parametrize("missing", ["--suffix", "--expected-count", "--out-dir", "--name",
                                     "--scenario", "--n"])
def test_cli_required_arguments(env, tmp_path, spies, missing):
    with pytest.raises(SystemExit) as exc:
        cli.main(cli_args(env, tmp_path / "traffic", **{missing: None}))
    assert exc.value.code == 2
    assert spies["reads"] == [] and spies["binding_calls"] == []   # nothing was touched


def test_cli_rejects_an_unknown_scenario(env, tmp_path, spies):
    with pytest.raises(SystemExit) as exc:
        cli.main(cli_args(env, tmp_path / "traffic", **{"--scenario": "teleport"}))
    assert exc.value.code == 2 and spies["reads"] == []


def test_cli_collision_fails_closed_with_no_output(env, tmp_path, capsys):
    p = env["paths"].features / "split_assignments.parquet"
    a = pd.read_parquet(p)
    a.loc[0, ID_COLUMN] = "0001-SYNTR"
    a.to_parquet(p, index=False)
    out = tmp_path / "traffic"
    assert cli.main(cli_args(env, out)) == 2
    err = capsys.readouterr().err
    assert err.startswith("refused:") and "0001-SYNTR" not in err
    assert batch_files(out) == []


def test_cli_existing_batch_is_refused_before_any_data_is_read(env, tmp_path, spies, capsys):
    out = tmp_path / "traffic"
    out.mkdir()
    (out / "b1.parquet").write_bytes(b"previous batch")
    assert cli.main(cli_args(env, out)) == 2
    assert "already exists" in capsys.readouterr().err
    assert spies["reads"] == [] and spies["binding_calls"] == []
    assert (out / "b1.parquet").read_bytes() == b"previous batch"   # untouched
    assert batch_files(out) == ["b1.parquet"]


@pytest.mark.parametrize("over", [{"--n": "10001"}, {"--n": "0"}, {"--expected-count": "7043"},
                                  {"--suffix": "SYNTX"}, {"--name": "../escape"}])
def test_cli_errors_leave_no_output(env, tmp_path, over, capsys):
    out = tmp_path / "traffic"
    assert cli.main(cli_args(env, out, **over)) == 2
    assert capsys.readouterr().err.startswith("refused:")
    assert batch_files(out) == []


def test_cli_failure_while_writing_leaves_no_partial_batch(env, tmp_path, monkeypatch, capsys):
    import os as _os
    out = tmp_path / "traffic"
    real_replace, calls = _os.replace, []

    def fail_on_metadata(src, dst):
        calls.append(Path(dst).name)
        if str(dst).endswith(".json"):
            raise OSError("simulated failure while placing metadata")
        return real_replace(src, dst)
    monkeypatch.setattr(sim.os, "replace", fail_on_metadata)
    assert cli.main(cli_args(env, out)) == 2
    assert calls == ["b1.parquet", "b1.json"]          # the data file had been placed
    assert batch_files(out) == []                      # ...and was removed again; no temps
