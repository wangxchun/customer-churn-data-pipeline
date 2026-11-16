"""W2: batch prediction through a released model, with atomic batch outputs.

``run_predict`` performs these steps, in order:

1. Validate the arguments. With an explicit ``batch_id``, refuse if that batch
   already exists, before anything else is read.
2. Load the release: ``production`` via the pointer, or a named release. This uses
   the W1 checks: release, guard and model integrity, and the HMAC key. Any
   failure fails closed, and the input is not read.
3. Read the input file ONCE as bytes. The sha256 is computed on exactly those bytes,
   and the CSV or Parquet is parsed from the same bytes.
4. ``release.predict_batch``: the holdout guard first (one hit rejects the whole
   batch), then TD-5 validation and scoring.
5. Write ``predictions.parquet``, ``rejects.parquet`` and ``batch_manifest.json``
   into ``<models>/inference/<batch_id>/`` through ``tracking.atomic_directory``.
   The batch is all or nothing, and an existing batch is never overwritten.

Nothing is written unless step 4 succeeds. No split file, customer master,
baseline or other dataset is read. Only the input file, the release artefacts that
W1 already verifies, and the optional D7 metadata sidecar ``<input stem>.json``
(for the ``synthetic`` flag, P3) are read.

Plan decisions used here (Phase 4 W2 plan, P1–P6). They are not yet recorded in
the decision log:
* P1: ``batch_id`` is ``<UTC %Y%m%dT%H%M%SZ>-<6 hex>``, like a run id.
* P2: CSV is read with ``dtype=str`` and ``keep_default_na=False``; all typing is
  left to TD-5's validation, the same rule as Phase 1.
* P3: ``synthetic`` is ``true`` or ``false`` from the sidecar's ``synthetic`` field,
  and ``null`` (unknown) without a sidecar. It is never inferred from ID suffixes.
* P4: on any guard, key or release failure, nothing is written.
* P5: TD-5 does not expose validate, transform and predict timings separately, so
  the timings are ``load_release``, ``read_input``, ``predict_batch`` (the guard
  plus TD-5 validate, transform and score together) and ``write``.
* P6: ``release`` is ``"production"`` or an explicit release id.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import inference
from . import monitoring as mon
from . import release as rel
from .config import Paths
from .features import ID_COLUMN, _to_model_frame
from .tracking import atomic_directory, write_json

BATCH_MANIFEST_SCHEMA_VERSION = "1"
PREDICTIONS_FILE = "predictions.parquet"
REJECTS_FILE = "rejects.parquet"
MANIFEST_FILE = "batch_manifest.json"
SUPPORTED_SUFFIXES = (".csv", ".parquet")
_BATCH_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def inference_dir(paths: Paths) -> Path:
    return paths.models / "inference"


def new_batch_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def _input_format(path: Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported input format {suffix!r}; use .csv or .parquet")
    return suffix.lstrip(".")


def read_input(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read the input file once. Return (frame, {format, file_name, size, sha256}).

    The sha256 is computed on the exact bytes that are parsed (no second read).
    """
    path = Path(path)
    fmt = _input_format(path)
    data = path.read_bytes()
    if fmt == "csv":
        df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    else:
        df = pd.read_parquet(io.BytesIO(data))
    return df, {"format": fmt, "file_name": path.name, "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest()}


def _synthetic_flag(input_path: Path) -> tuple[bool | None, str]:
    """P3: read the D7 sidecar ``<stem>.json`` if present; never guess from IDs."""
    sidecar = Path(input_path).with_suffix(".json")
    if not sidecar.is_file():
        return None, "no metadata sidecar"
    try:
        value = json.loads(sidecar.read_text()).get("synthetic")
    except (ValueError, AttributeError):
        return None, "metadata sidecar unreadable"
    if value is True:
        return True, f"{sidecar.name}: synthetic=true"
    if value is False:
        return False, f"{sidecar.name}: synthetic=false"
    return None, f"{sidecar.name}: no boolean synthetic field"


def validate_request(input_path: Path, release: str, batch_id: str | None = None) -> None:
    """Argument checks that need no file access. The CLI calls this before it loads
    the key, so an invalid request has no side effects at all."""
    _input_format(Path(input_path))
    if not isinstance(release, str) or not release:
        raise ValueError("release must be 'production' or a release id")
    if batch_id is not None and not _BATCH_ID_RE.fullmatch(batch_id):
        raise ValueError("invalid batch id")


def run_predict(paths: Paths, input_path: Path, release: str, key: bytes, *,
                batch_id: str | None = None,
                thresholds: mon.Thresholds = mon.Thresholds()) -> dict:
    """Predict one batch, write it atomically, then append its W3 monitoring record.

    Failure contract:
    * Anything before the batch is committed (release, key, guard, baseline, input,
      scoring, building the monitoring record) raises, and nothing is written.
    * If the batch was committed but appending its monitoring record fails, this
      raises ``monitoring.MonitoringAppendError``. The committed batch is kept, not
      deleted, and the error says so; the run is never reported as fully successful.
    """
    input_path = Path(input_path)
    validate_request(input_path, release, batch_id)        # before any access
    batch_id = batch_id or new_batch_id()
    final = inference_dir(paths) / batch_id
    if final.exists():
        raise FileExistsError(f"batch {batch_id} already exists; refusing to overwrite")

    timings: dict[str, float] = {}
    t = time.perf_counter()
    loaded = (rel.load_production(paths, key) if release == "production"
              else rel.load_release(paths, release, key))   # fails closed on any problem
    timings["load_release"] = time.perf_counter() - t
    # W3: the release's baseline, verified against the run manifest, before any input
    # is read. A missing or tampered baseline fails closed.
    baseline = mon.load_baseline(paths, loaded.model.run_id)

    t = time.perf_counter()
    df, input_info = read_input(input_path)
    timings["read_input"] = time.perf_counter() - t
    synthetic, synthetic_source = _synthetic_flag(input_path)

    t = time.perf_counter()
    result = rel.predict_batch(loaded, df)   # holdout guard first; raises before writing
    timings["predict_batch"] = time.perf_counter() - t

    rejects = result.rejects
    manifest = {
        "batch_manifest_schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release_id": loaded.release_id,
        "run_id": loaded.model.run_id,
        "model_sha256": loaded.model.model_sha256,
        "decision_threshold": loaded.model.threshold,
        "inference_schema_version": inference.INFERENCE_SCHEMA_VERSION,
        "input": input_info,
        "synthetic": synthetic,
        "synthetic_source": synthetic_source,
        "n_input": len(df),
        "n_scored": len(result.predictions),
        "n_rejected_rows": int(rejects.index.nunique()),
        "reject_counts_by_reason": {str(k): int(v) for k, v in
                                    sorted(rejects["reason_code"].value_counts().items())},
        "reject_counts_unit": ("reject records: a row with several problems counts once "
                               "per problem; n_rejected_rows counts rows"),
        "ignored_columns": list(result.ignored_columns),
        "files": [PREDICTIONS_FILE, REJECTS_FILE, MANIFEST_FILE],
    }
    # W3 (P9): build the record now, while the scored rows are in memory, and before
    # anything is written, so that any problem here also leaves no batch behind.
    # TD-5 is deterministic, so re-validating gives exactly the rows it scored.
    valid = inference.validate_inference_input(df, loaded.model.categories).valid
    if list(valid[ID_COLUMN].astype(str)) != list(result.predictions[ID_COLUMN].astype(str)):
        raise mon.MonitoringError("re-validated rows do not match the scored rows")
    record = mon.batch_record(manifest=manifest, input_frame=df, predictions=result.predictions,
                              rejects=rejects, scored_features=_to_model_frame(valid),
                              baseline=baseline, thresholds=thresholds)
    del valid
    t = time.perf_counter()
    tmp_dir: Path | None = None
    try:
        with atomic_directory(final) as tmp:  # all or nothing; FileExistsError if present
            tmp_dir = tmp
            result.predictions.to_parquet(tmp / PREDICTIONS_FILE, index=False)
            # The rejects index is the input row position (TD-5 contract); keep it.
            rejects.to_parquet(tmp / REJECTS_FILE, index=True)
            timings["write"] = time.perf_counter() - t
            manifest["timings_s"] = {k: round(v, 6) for k, v in timings.items()}
            manifest["timings_note"] = ("predict_batch covers the holdout guard plus TD-5 "
                                        "validate, transform and score; TD-5 does not "
                                        "time those stages separately (P5)")
            write_json(tmp / MANIFEST_FILE, manifest)
    except BaseException:
        # atomic_directory removes its temp dir when the block fails, but NOT when the
        # final rename fails (for example, a concurrent run created a non-empty batch
        # dir of the same name). That temp dir holds predictions and rejects with
        # customer IDs, so remove it here. ``final`` is never touched.
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    manifest["output_dir"] = str(final)  # returned to the caller; not part of the file
    record["timings_s"] = manifest["timings_s"]
    try:
        records_file = mon.append_record(paths, record)
    except Exception as e:
        raise mon.MonitoringAppendError(
            f"batch {batch_id} WAS written to {final}, but its monitoring record was NOT "
            f"appended ({type(e).__name__}: {e}); the batch is kept, not deleted",
            batch_id=batch_id, batch_dir=str(final)) from e
    manifest["monitoring_records"] = str(records_file)  # returned only; not in the file
    return manifest
