"""W4: benchmark structure and safety. Synthetic-fixture tests only: the release, run,
key and input rows are synthetic and live under pytest tmp directories. Per the W4 plan,
timings are machine-specific, so only the structure is asserted, never the values."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from churn_pipeline import benchmark as bm
from churn_pipeline import cli, inference
from churn_pipeline import release as rel
from churn_pipeline.features import ID_COLUMN, build_features
from test_inference import fit_model, synthetic_master
from test_release import KEY, OTHER_KEY, RUN_ID, make_env, write_key_file

RELEASE_ID = "rel-syn-bench"
FAST = dict(repeats=3, batch_size=40)


@pytest.fixture(scope="module")
def master():
    return synthetic_master()


@pytest.fixture(scope="module")
def model(master):
    return fit_model(master)


@pytest.fixture(scope="module")
def assignments(master):
    return build_features(master).assignments


@pytest.fixture
def env(tmp_path, model, assignments):
    paths = make_env(tmp_path, model, assignments)
    rel.create_release(paths, RUN_ID, KEY, release_id=RELEASE_ID)
    rel.promote(paths, RELEASE_ID, KEY)
    return paths


def entries(paths):
    d = bm.benchmarks_dir(paths)
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def test_benchmark_writes_one_json_with_the_expected_structure(env):
    r = bm.run_benchmark(env, "production", KEY, benchmark_id="bm-1", **FAST)
    d = bm.benchmarks_dir(env) / "bm-1"
    assert sorted(p.name for p in d.iterdir()) == [bm.BENCHMARK_FILE]
    saved = json.loads((d / bm.BENCHMARK_FILE).read_text())
    assert saved["release_id"] == RELEASE_ID and saved["run_id"] == RUN_ID
    assert saved["model_size_bytes"] == (env.models / RUN_ID / "model.joblib").stat().st_size
    assert saved["params"]["repeats"] == 3 and saved["params"]["batch_size"] == 40
    for kind in ("single_row", "batch"):
        for path in ("guarded", "raw"):
            s = saved["latency_ms"][kind][path]
            assert 0 < s["p50"] <= s["p95"] <= s["p99"] and s["n"] == 3
    assert all(v > 0 for v in saved["throughput_rows_per_second"].values())
    assert all(v > 0 for v in saved["load_seconds"].values())
    assert set(saved["tracemalloc_peak_mib"]) == {"guarded_batch", "raw_batch"}
    assert any("this machine" in n for n in saved["notes"])
    assert r["output_dir"] == str(d)


def test_benchmark_outputs_no_ids_or_predictions(env):
    bm.run_benchmark(env, RELEASE_ID, KEY, benchmark_id="bm-2", **FAST)
    text = (bm.benchmarks_dir(env) / "bm-2" / bm.BENCHMARK_FILE).read_text()
    assert not re.search(r"\d{4}-[A-Z]{5}", text)              # no customer-style IDs
    assert "churn_probability" not in text


def test_synthetic_rows_are_valid_unique_and_deterministic(env):
    m = rel.load_release(env, RELEASE_ID, KEY).model
    a = bm.synthetic_rows(m.categories, 200, seed=4)
    pd.testing.assert_frame_equal(a, bm.synthetic_rows(m.categories, 200, seed=4))
    assert a[ID_COLUMN].is_unique and a[ID_COLUMN].str.endswith("-BENCH").all()
    v = inference.validate_inference_input(a, m.categories)
    assert v.rejects.empty and len(v.valid) == 200


def test_benchmark_reads_no_data_files(env, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("benchmark must not read data files")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    bm.run_benchmark(env, "production", KEY, benchmark_id="bm-3", **FAST)


@pytest.mark.parametrize("kw", [dict(repeats=2), dict(repeats=True), dict(batch_size=0),
                                dict(batch_size=10_001), dict(batch_size=1.5),
                                dict(benchmark_id="../x")])
def test_invalid_parameters_are_refused_before_loading(env, monkeypatch, kw):
    called = []
    monkeypatch.setattr(rel, "load_production", lambda *a, **k: called.append(1))
    with pytest.raises(bm.BenchmarkError):
        bm.run_benchmark(env, "production", KEY, **{**FAST, **kw})
    assert called == [] and entries(env) == []


def test_wrong_key_fails_closed_and_writes_nothing(env):
    with pytest.raises(rel.GuardKeyError):
        bm.run_benchmark(env, "production", OTHER_KEY, **FAST)
    assert entries(env) == []


def test_existing_benchmark_is_never_overwritten(env):
    bm.run_benchmark(env, "production", KEY, benchmark_id="bm-dup", **FAST)
    before = (bm.benchmarks_dir(env) / "bm-dup" / bm.BENCHMARK_FILE).read_bytes()
    with pytest.raises(FileExistsError):
        bm.run_benchmark(env, "production", KEY, benchmark_id="bm-dup", **FAST)
    assert (bm.benchmarks_dir(env) / "bm-dup" / bm.BENCHMARK_FILE).read_bytes() == before


def test_incomplete_guarded_batch_is_an_error_and_writes_nothing(env, monkeypatch):
    real = rel.predict_batch

    def drops_a_row(r, df):
        out = real(r, df)
        return inference.InferenceResult(out.predictions.iloc[1:], out.rejects,
                                         out.ignored_columns)
    monkeypatch.setattr(rel, "predict_batch", drops_a_row)
    with pytest.raises(bm.BenchmarkError, match="did not score every"):
        bm.run_benchmark(env, "production", KEY, **FAST)
    assert entries(env) == []


def test_failed_final_rename_leaves_no_temp_dir(env, monkeypatch):
    final = bm.benchmarks_dir(env) / "bm-race"
    real = bm.write_json

    def concurrent_creator(path, obj):
        real(path, obj)
        final.mkdir(parents=True)
        (final / "theirs.txt").write_text("other run")
    monkeypatch.setattr(bm, "write_json", concurrent_creator)
    with pytest.raises(OSError):
        bm.run_benchmark(env, "production", KEY, benchmark_id="bm-race", **FAST)
    assert entries(env) == ["bm-race"]
    assert sorted(p.name for p in final.iterdir()) == ["theirs.txt"]


# --- CLI ------------------------------------------------------------------------------------
@pytest.fixture
def cli_key(tmp_path, monkeypatch):
    monkeypatch.setenv(rel.GUARD_KEY_ENV, str(write_key_file(tmp_path / "guard.key", KEY.hex())))


def test_cli_benchmark_success(env, cli_key, capsys):
    assert cli.main(["--data-dir", str(env.data_dir), "benchmark", "--release", RELEASE_ID,
                     "--repeats", "3", "--batch-size", "40"]) == 0
    assert "this machine only" in capsys.readouterr().out
    assert len(entries(env)) == 1


@pytest.mark.parametrize("argv", [["benchmark", "--release", RELEASE_ID],   # no --data-dir
                                  ["benchmark"]])                            # no --release
def test_cli_benchmark_required_arguments(argv, monkeypatch):
    called = []
    monkeypatch.setattr(rel, "load_guard_key", lambda *a, **k: called.append(1))
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2 and called == []


def test_cli_benchmark_failure_exits_2(env, cli_key, capsys):
    assert cli.main(["--data-dir", str(env.data_dir), "benchmark", "--release",
                     "rel-missing", "--repeats", "3"]) == 2
    assert capsys.readouterr().err.startswith("refused:")
    assert entries(env) == []
