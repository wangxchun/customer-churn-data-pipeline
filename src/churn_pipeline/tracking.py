"""Phase 3: input verification, experiment manifests, and atomic run directories.

Experiment tracking here is a local JSON manifest (decision D1), not MLflow.
The manifest is shaped so that each part maps onto an MLflow concept:

    manifest["params"]    -> mlflow.log_params   (seed, n_iter, costs, ...)
    manifest["metrics"]   -> mlflow.log_metrics  (validation headline numbers)
    manifest["tags"]      -> mlflow.set_tags     (phase, champion, git state)
    manifest["artifacts"] -> mlflow.log_artifact (file name -> sha256)

What MLflow would add and this does not: a server/UI, run search across
experiments, a model registry, and concurrent-writer safety.

Input verification guards against stale artifacts (a TD-3 mitigation, not a fix):
Phase 3 refuses to train if ``customer_master.parquet`` no longer matches the
checksum Phase 2 recorded, and every run records a *content* hash of the split
so runs made on different splits can be detected and refused by ``compare``.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn

from .config import PROJECT_ROOT, Paths
from .download import sha256_file
from .features import (
    ID_COLUMN,
    LABEL_MAP,
    SPLIT_NAMES,
    TARGET_COLUMN,
    DataContractError,
    _to_model_frame,  # private on purpose: reused so Phase 3 converts dtypes exactly like Phase 2
    check_input,
    split_content_hash,  # moved to features.py (TD-3); re-exported here for existing callers
)


class StaleArtifactError(RuntimeError):
    """Phase 3 inputs no longer match what an earlier step recorded."""


@dataclass(frozen=True)
class InputFingerprint:
    customer_master_sha256: str
    split_assignments_sha256: str
    feature_report_sha256: str
    split_content_sha256: str  # content hash: independent of Parquet byte layout


@dataclass
class VerifiedInputs:
    data: pd.DataFrame  # customer_master + "split" column, sorted by customer_id
    fingerprint: InputFingerprint


def load_verified_inputs(paths: Paths) -> VerifiedInputs:
    master_path = paths.curated / "customer_master.parquet"
    assign_path = paths.features / "split_assignments.parquet"
    report_path = paths.features / "feature_report.json"
    for p in (master_path, assign_path, report_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found; run Phase 1 (`churn-pipeline run`) and Phase 2 "
                "(`churn-pipeline build-features`) first"
            )

    report = json.loads(report_path.read_text())
    master_sha = sha256_file(master_path)
    if master_sha != report.get("input_sha256"):
        raise StaleArtifactError(
            f"customer_master.parquet sha256 {master_sha[:12]} != feature_report input_sha256 "
            f"{str(report.get('input_sha256'))[:12]}: Phase 1 was re-run without re-running "
            "Phase 2. Run `churn-pipeline build-features` before training."
        )

    master = pd.read_parquet(master_path)
    check_input(master)
    master = master.assign(**{ID_COLUMN: master[ID_COLUMN].astype(pd.StringDtype())})
    assignments = pd.read_parquet(assign_path)
    assignments = assignments.assign(**{ID_COLUMN: assignments[ID_COLUMN].astype(pd.StringDtype()),
                                        "split": assignments["split"].astype(pd.StringDtype())})

    unknown = set(assignments["split"].unique()) - set(SPLIT_NAMES)
    if unknown:
        raise DataContractError(f"split_assignments has unknown split names {sorted(unknown)}")
    if set(master[ID_COLUMN]) != set(assignments[ID_COLUMN]) or not assignments[ID_COLUMN].is_unique:
        raise StaleArtifactError(
            "split_assignments customers do not match customer_master one-to-one; "
            "re-run `churn-pipeline build-features`"
        )

    data = master.merge(assignments, on=ID_COLUMN, how="inner", validate="one_to_one")
    data = data.sort_values(ID_COLUMN, ignore_index=True)
    fp = InputFingerprint(
        customer_master_sha256=master_sha,
        split_assignments_sha256=sha256_file(assign_path),
        feature_report_sha256=sha256_file(report_path),
        split_content_sha256=split_content_hash(assignments),
    )
    return VerifiedInputs(data, fp)


def split_xy(data: pd.DataFrame, split: str) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Raw (untransformed) model inputs for one split, indexed by customer_id.

    The index travels through sklearn's CV slicing and the pandas-output
    preprocessor, which is what lets tests prove which customers a model saw.
    """
    part = data[data["split"] == split]
    index = pd.Index(part[ID_COLUMN].astype(str).to_numpy(dtype=object), name=ID_COLUMN)
    X = _to_model_frame(part)
    X.index = index
    y = pd.Series(part[TARGET_COLUMN].map(LABEL_MAP).astype("int64").to_numpy(), index=index,
                  name=TARGET_COLUMN)
    return pd.Series(index, name=ID_COLUMN), X, y


# ---------------------------------------------------------------------------
# Run directories and JSON
# ---------------------------------------------------------------------------
def new_run_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


@contextmanager
def atomic_directory(final: Path) -> Iterator[Path]:
    """Yield a temporary sibling directory; rename it to ``final`` only on success.

    A failure part-way leaves no ``final`` directory and removes the temp one, so
    readers never see a half-written set of artifacts (contrast TD-4 in Phase 2).
    """
    if final.exists():
        raise FileExistsError(f"{final} already exists")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.parent / f".tmp-{final.name}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    os.rename(tmp, final)  # atomic on one POSIX filesystem


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False))


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    write_json(tmp, obj)
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "joblib": joblib.__version__, "pyarrow": pyarrow.__version__},
    }


def git_state() -> dict:
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                                 text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        # Untracked files count as dirty: uncommitted new source files change behavior too.
        "dirty": None if status is None else bool(status),
        "n_changed_paths": None if status is None else len(status.splitlines()),
    }


def artifact_hashes(directory: Path, exclude: tuple[str, ...] = ("manifest.json",)) -> dict:
    return {
        str(p.relative_to(directory)): sha256_file(p)
        for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in exclude
    }


def build_manifest(*, run_id: str, kind: str, directory: Path, params: dict, metrics: dict,
                   tags: dict, fingerprint: InputFingerprint) -> dict:
    return {
        "run_id": run_id,
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "metrics": metrics,
        "tags": tags,
        "artifacts": artifact_hashes(directory),
        "data": asdict(fingerprint),
        "code": git_state(),
        "environment": environment(),
        "mlflow_mapping": {"params": "log_params", "metrics": "log_metrics",
                           "tags": "set_tags", "artifacts": "log_artifact"},
    }
