"""W4: latency, load-time and memory benchmark of a release.

It compares two paths on the SAME synthetic rows:

* guarded: ``release.predict_batch`` / ``release.predict_one`` (the holdout guard, then
  TD-5 validation and scoring), which is what ``predict`` runs;
* raw: ``model.predict_proba`` on an already-prepared model frame, i.e. no
  validation or guard (the floor that the guard and validation cost is measured
  against).

Inputs are synthetic rows generated here from the release's own encoder categories
(``BENCH``-suffixed IDs). No data file is read, and no prediction or ID is written.
The only reads are the release artefacts that ``release.load_release`` verifies.

Output: ``<models>/benchmarks/<benchmark_id>/benchmark.json``, written atomically.
The numbers describe THIS machine and THIS run only. ``tracemalloc`` sees only
Python-level allocations, not native memory (numpy and sklearn internals), so it is a
lower bound, not total memory. Tests check the structure, never the timings.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import sys
import time
import tracemalloc
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import inference
from . import release as rel
from .config import Paths
from .features import CATEGORICAL_FEATURES, ID_COLUMN, _to_model_frame
from .tracking import atomic_directory, write_json

BENCHMARK_SCHEMA_VERSION = "1"
BENCHMARK_FILE = "benchmark.json"
ID_SUFFIX = "BENCH"
MAX_BATCH = 10_000


class BenchmarkError(RuntimeError):
    """The benchmark could not run as specified. Nothing is written."""


def benchmarks_dir(paths: Paths) -> Path:
    return paths.models / "benchmarks"


def synthetic_rows(categories: dict, n: int, seed: int = 0) -> pd.DataFrame:
    """``n`` valid synthetic input rows in the raw inference format."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    monthly = np.round(rng.uniform(18, 120, n), 2)
    df = pd.DataFrame({
        ID_COLUMN: [f"{i:04d}-{ID_SUFFIX}" for i in range(n)],
        "senior_citizen": rng.integers(0, 2, n),
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": np.round(tenure * monthly, 2),
    })
    for c in CATEGORICAL_FEATURES:
        values = sorted(str(v) for v in categories[c])
        df[c] = [values[i] for i in rng.integers(0, len(values), n)]
    return df


def _summary_ms(seconds: list[float]) -> dict:
    ms = np.asarray(seconds) * 1000.0
    return {"p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)),
            "p99": float(np.percentile(ms, 99)), "mean": float(ms.mean()), "n": len(ms)}


def _time(fn, repeats: int) -> list[float]:
    out = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


def _peak_mib(fn) -> float:
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 2**20


def _check_int(name, value, lo, hi):
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise BenchmarkError(f"{name} must be an integer in [{lo}, {hi}]")


def run_benchmark(paths: Paths, release: str, key: bytes, *, repeats: int = 20,
                  batch_size: int = 1000, seed: int = 0, benchmark_id: str | None = None) -> dict:
    _check_int("repeats", repeats, 3, 10_000)
    _check_int("batch_size", batch_size, 1, MAX_BATCH)
    if not isinstance(release, str) or not release:
        raise BenchmarkError("release must be 'production' or a release id")
    benchmark_id = benchmark_id or \
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", benchmark_id):
        raise BenchmarkError("invalid benchmark id")
    final = benchmarks_dir(paths) / benchmark_id
    if final.exists():
        raise FileExistsError(f"benchmark {benchmark_id} already exists")

    # Load time: the guarded load (all W1/TD-5 checks) and a raw unpickle of the same file.
    t = time.perf_counter()
    loaded = (rel.load_production(paths, key) if release == "production"
              else rel.load_release(paths, release, key))
    guarded_load = time.perf_counter() - t
    model_path = paths.models / loaded.model.run_id / "model.joblib"
    t = time.perf_counter()
    raw_model = joblib.load(model_path)          # same file; its hash was verified above
    raw_load = time.perf_counter() - t

    rows = synthetic_rows(loaded.model.categories, batch_size, seed)
    valid = inference.validate_inference_input(rows, loaded.model.categories).valid
    if len(valid) != batch_size:
        raise BenchmarkError("synthetic rows did not all validate")
    x = _to_model_frame(valid)                   # prepared once; raw path is not timed on it
    x1, record = x.iloc[[0]], rows.iloc[0].to_dict()

    def guarded_batch():
        r = rel.predict_batch(loaded, rows)
        if len(r.predictions) != batch_size:     # e.g. a holdout hit or a rejected row
            raise BenchmarkError("guarded batch did not score every synthetic row")

    try:
        guarded_batch()                          # warm-up and correctness check
        rel.predict_one(loaded, record)
    except inference.InferenceInputError as e:
        raise BenchmarkError(f"synthetic rows were refused ({e})") from None
    raw_model.predict_proba(x)

    lat = {
        "single_row": {
            "guarded": _summary_ms(_time(lambda: rel.predict_one(loaded, record), repeats)),
            "raw": _summary_ms(_time(lambda: raw_model.predict_proba(x1), repeats))},
        "batch": {
            "guarded": _summary_ms(_time(guarded_batch, repeats)),
            "raw": _summary_ms(_time(lambda: raw_model.predict_proba(x), repeats))},
    }
    result = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release_id": loaded.release_id,
        "run_id": loaded.model.run_id,
        "model_sha256": loaded.model.model_sha256,
        "model_size_bytes": os.stat(model_path).st_size,
        "machine": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "machine": platform.machine(), "cpu_count": os.cpu_count()},
        "params": {"repeats": repeats, "batch_size": batch_size, "seed": seed,
                   "input": f"synthetic rows ({ID_SUFFIX} IDs) from the release's categories"},
        "load_seconds": {"guarded_release_load": guarded_load, "raw_joblib_load": raw_load},
        "latency_ms": lat,
        "throughput_rows_per_second": {
            "guarded": batch_size / (lat["batch"]["guarded"]["p50"] / 1000.0),
            "raw": batch_size / (lat["batch"]["raw"]["p50"] / 1000.0)},
        "tracemalloc_peak_mib": {"guarded_batch": _peak_mib(guarded_batch),
                                 "raw_batch": _peak_mib(lambda: raw_model.predict_proba(x))},
        "notes": ["Numbers are specific to this machine and run; compare only within a run.",
                  "tracemalloc counts Python allocations only (a lower bound on memory).",
                  "raw excludes validation and the holdout guard; guarded includes both.",
                  "In-process latency; no network, serialization or endpoint overhead."],
    }
    tmp_dir = None
    try:
        with atomic_directory(final) as tmp:
            tmp_dir = tmp
            write_json(tmp / BENCHMARK_FILE, result)
    except BaseException:
        # Same guard as predict (H1): a failed final rename must not leave a temp dir.
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    result["output_dir"] = str(final)
    return result
