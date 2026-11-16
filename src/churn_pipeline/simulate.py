"""W2 / D7: synthetic inference traffic resampled from train rows only.

Approved data access (D7-Q1 to Q4, W2-Q7). One D7 run touches exactly:

1. ``traffic.input_binding(split_path)``: one ``stat`` of the split file (D7-Q3).
2. ``split_assignments.parquet``: one read, columns ``customer_id`` and ``split`` only
   (D7-Q2). This yields the FULL real ID set (for the collision check) and the train IDs.
3. ``traffic.certify_suffix(all_ids, suffix, expected_count=..., binding=...)``: the
   pre-output collision check on the full in-memory ID set (W2-Q7). On failure,
   nothing else is read and nothing is produced.
4. ``customer_master.parquet``: one read, columns ``customer_id`` plus
   ``FEATURE_COLUMNS``, with a PyArrow filter that returns train rows only (D7-Q1,
   Q4). No ``churn``, no synthetic columns, no test or validation rows.

It does not use ``tracking.load_verified_inputs``, and it reads no ``train.parquet``,
``feature_report.json``, test data or labels. Real customer IDs are dropped right
after the master read. The traffic rows carry only certified synthetic IDs
(``NNNN-<suffix>``).

Limitation: the binding (file size and ``mtime_ns``) records which file version was
checked. It is not a cryptographic proof of the file's content.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import traffic
from .config import Paths
from .features import FEATURE_COLUMNS, ID_COLUMN, DataContractError

SPLIT_FILE = "split_assignments.parquet"
MASTER_FILE = "customer_master.parquet"
SPLIT_COLUMNS = [ID_COLUMN, "split"]                 # D7-Q2
MASTER_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS]       # D7-Q1: no churn, no synthetic columns
TRAIN_SPLIT = "train"
SCENARIOS = ("none", "price_up", "contract_mix", "unseen_payment", "nulls")
UNSEEN_PAYMENT = "Crypto wallet"                     # outside the Phase 1 domain


class TrafficSourceError(DataContractError):
    """The D7 inputs are unusable. Fail closed. Messages hold counts, never IDs."""


@dataclass(frozen=True)
class TrafficSource:
    features: pd.DataFrame                  # train rows, FEATURE_COLUMNS only (no IDs)
    certificate: traffic.SuffixCertificate  # certified on the full real ID set


def load_traffic_source(paths: Paths, *, suffix: str, expected_count: int) -> TrafficSource:
    """The approved D7 reads, in order. ``suffix`` and ``expected_count`` are required
    (no defaults). The real run's values are part of its execution approval."""
    split_path = paths.features / SPLIT_FILE
    master_path = paths.curated / MASTER_FILE
    traffic.check_candidate(suffix)  # validate before any file operation

    binding = traffic.input_binding(split_path)                          # (1) one stat
    split = pd.read_parquet(split_path, columns=SPLIT_COLUMNS)           # (2) one read
    if list(split.columns) != SPLIT_COLUMNS:
        raise TrafficSourceError("split file returned unexpected columns")
    all_ids = split[ID_COLUMN].tolist()
    train_ids = [i for i, s in zip(all_ids, split["split"].astype(str)) if s == TRAIN_SPLIT]
    del split
    try:
        # (3) pre-output collision check on the FULL real ID set (W2-Q7)
        cert = traffic.certify_suffix(all_ids, suffix, expected_count=expected_count,
                                      binding=binding)
    finally:
        all_ids.clear()
    if not train_ids:
        raise TrafficSourceError("the split has no train rows")

    master = pd.read_parquet(master_path, columns=MASTER_COLUMNS,      # (4) one read
                             filters=[(ID_COLUMN, "in", train_ids)])
    try:
        if list(master.columns) != MASTER_COLUMNS:
            raise TrafficSourceError("master returned unexpected columns")
        loaded = master[ID_COLUMN].astype(str)
        if len(master) != len(train_ids) or not loaded.is_unique \
                or set(loaded) != set(train_ids):
            raise TrafficSourceError(f"master returned {len(master)} rows for "
                                     f"{len(train_ids)} train IDs; refusing to continue")
        features = master[FEATURE_COLUMNS].reset_index(drop=True)  # real IDs dropped here
    finally:
        train_ids.clear()
        del master
    return TrafficSource(features=features, certificate=cert)


def generate_traffic(source: TrafficSource, scenario: str, n: int, seed: int) -> pd.DataFrame:
    """``n`` rows resampled from train rows, with a scenario applied, carrying certified
    synthetic IDs. Deterministic for a given seed. The 10,000-ID cap is enforced by
    ``traffic.synthetic_ids``."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {list(SCENARIOS)}")
    ids = traffic.synthetic_ids(source.certificate, n)  # validates n and the cap first
    rng = np.random.default_rng(seed)
    pool = source.features
    if scenario == "contract_mix":  # 80% month-to-month: a label-correlated mix shift
        m2m = np.flatnonzero(pool["contract"].astype(str).to_numpy() == "Month-to-month")
        other = np.flatnonzero(pool["contract"].astype(str).to_numpy() != "Month-to-month")
        if len(m2m) == 0 or len(other) == 0:
            raise TrafficSourceError("contract_mix needs both contract groups in train rows")
        k = int(round(0.8 * n))
        pick = np.concatenate([rng.choice(m2m, k), rng.choice(other, n - k)])
        rng.shuffle(pick)
    else:
        pick = rng.integers(0, len(pool), n)
    out = pool.iloc[pick].reset_index(drop=True).copy()

    if scenario == "price_up":
        for c in ("monthly_charges", "total_charges"):
            out[c] = (out[c] * 1.15).round(2)
    elif scenario == "unseen_payment":
        hit = rng.random(n) < 0.2
        out.loc[hit, "payment_method"] = UNSEEN_PAYMENT
    elif scenario == "nulls":
        hit = rng.random(n) < 0.1
        out.loc[hit, "monthly_charges"] = pd.NA

    out.insert(0, ID_COLUMN, pd.array(ids, dtype="string"))
    return out


def batch_paths(out_dir: Path, name: str) -> tuple[Path, Path]:
    """``(<out_dir>/<name>.parquet, <out_dir>/<name>.json)``. Validates the name only."""
    if not isinstance(name, str) or not name or "/" in name or name.startswith("."):
        raise ValueError("invalid traffic batch name")
    out_dir = Path(out_dir)
    return out_dir / f"{name}.parquet", out_dir / f"{name}.json"


def ensure_batch_absent(out_dir: Path, name: str) -> None:
    """Refuse an existing batch. Call this BEFORE reading any data, so that a run which
    could never write reads nothing."""
    if any(p.exists() for p in batch_paths(out_dir, name)):
        raise FileExistsError(f"traffic batch {name} already exists; refusing to overwrite")


def write_traffic(df: pd.DataFrame, out_dir: Path, name: str, *, scenario: str, seed: int,
                  certificate: traffic.SuffixCertificate) -> Path:
    """Write ``<out_dir>/<name>.parquet`` plus ``<name>.json`` metadata. Never overwrites.
    The batch is all or nothing: if any step fails, neither file is left behind.
    ``out_dir`` is required: there is no default, and so no real path is used by
    accident."""
    out_dir = Path(out_dir)
    data_path, meta_path = batch_paths(out_dir, name)
    ensure_batch_absent(out_dir, name)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"synthetic": True, "scenario": scenario, "seed": seed, "n": len(df),
            "id_suffix_certification": certificate.summary()}
    tmp = uuid.uuid4().hex[:8]
    tmp_data, tmp_meta = out_dir / f".{name}.{tmp}.parquet", out_dir / f".{name}.{tmp}.json"
    data_placed = False
    try:
        df.to_parquet(tmp_data, index=False)
        tmp_meta.write_text(json.dumps(meta, indent=2))
        os.replace(tmp_data, data_path)
        data_placed = True
        os.replace(tmp_meta, meta_path)
    except BaseException:
        if data_placed:  # never leave a data file without its metadata
            data_path.unlink(missing_ok=True)
        raise
    finally:
        for t in (tmp_data, tmp_meta):
            t.unlink(missing_ok=True)
    return data_path
