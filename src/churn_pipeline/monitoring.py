"""W3: batch monitoring records and a static Markdown report.

Each successful W2 ``predict`` batch appends exactly one record to
``<models>/monitoring/<release_id>/batches.jsonl``. A record holds aggregates only:
no ``customer_id``, no feature values, no per-row content. Synthetic IDs can repeat
across batches, so every number is computed per ``batch_id`` and nothing is ever
joined across batches by ``customer_id``.

There are no labels. Every metric here is a proxy on inputs and scores. None of
them measures model performance, and a drift alert does not mean the model got
worse. Alerts only set ``retrain_review_recommended``, which asks a person to
review. Nothing here retrains anything.

Baselines (written by Phase 3 into the run directory, ``baseline.json``):
* feature PSI uses ``baseline["features"]``, built from TRAIN rows (raw inputs);
* score PSI uses ``baseline["score"]``, built from the champion's VALIDATION scores;
* the flag-rate comparison uses
  ``baseline["performance"]["at_threshold"]["predicted_positive_rate"]`` (validation).
``baseline.json`` is verified against the run manifest's sha256 before use, and
any problem fails closed.

Plan decisions used (W3 plan P7–P10; not yet in the decision log):
* P7 unseen share = rows rejected with ``unseen_category`` or ``not_allowed`` on a
  categorical column, divided by ``n_input``. Computed from the rejects, because
  D3/TD-5 reject such rows before scoring, so they never reach the scored data.
* P8 thresholds: PSI > 0.25 (the PSI_RULE_OF_THUMB "large shift"); reject rate
  > 0.05; unseen share > 0; ``min_batch_n`` = 200 scored rows. All configurable.
  With fewer scored rows, the PSIs are not evaluated and no drift alert is raised
  (``insufficient_sample``). Data-quality alerts (reject rate, unseen share) are
  exact counts and are still raised. The flag-rate difference is recorded, but P8
  defines no threshold for it, so it never alerts.
* P9: the record is computed inside ``predict`` while the scored rows are in memory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .config import Paths
from .evaluation import PSI_RULE_OF_THUMB, feature_psi, score_psi
from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from .inference import REQUIRED_COLUMNS

MONITORING_SCHEMA_VERSION = "1"
BASELINE_FILE = "baseline.json"
RECORDS_FILE = "batches.jsonl"
UNSEEN_REASONS = ("unseen_category", "not_allowed")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

NO_LABELS_NOTE = ("No ground-truth labels are available for these batches. These metrics "
                  "describe inputs and scores only. They do not measure model performance, "
                  "and drift does not imply that performance has degraded.")
RETRAIN_NOTE = ("retrain_review_recommended means a person should review whether "
                "retraining is warranted. Nothing is retrained automatically, and it "
                "does not confirm that model performance has degraded.")
PSI_NOTE = (PSI_RULE_OF_THUMB + " Small batches make PSI noisy; batches below "
            "min_batch_n scored rows are marked insufficient_sample and get no drift alert.")


class MonitoringError(RuntimeError):
    """A monitoring input or record is unusable. Fail closed. Messages hold no IDs."""


class MonitoringAppendError(MonitoringError):
    """The prediction batch WAS committed, but its monitoring record was not appended."""

    def __init__(self, message: str, *, batch_id: str, batch_dir: str):
        self.batch_id = batch_id
        self.batch_dir = batch_dir
        super().__init__(message)


@dataclass(frozen=True)
class Thresholds:
    psi: float = 0.25
    reject_rate: float = 0.05
    unseen_share: float = 0.0
    min_batch_n: int = 200


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
def _safe_id(value: str, what: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise MonitoringError(f"invalid {what}")
    return value


def records_path(paths: Paths, release_id: str) -> Path:
    return paths.models / "monitoring" / _safe_id(release_id, "release id") / RECORDS_FILE


# ---------------------------------------------------------------------------
# Baseline (verified against the run manifest before use)
# ---------------------------------------------------------------------------
def load_baseline(paths: Paths, run_id: str) -> dict:
    run_dir = paths.models / _safe_id(run_id, "run id")
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text())
    except (OSError, ValueError):
        raise MonitoringError(f"run {run_id}: manifest.json missing or unreadable") from None
    expected = (manifest.get("artifacts") or {}).get(BASELINE_FILE)
    if not expected:
        raise MonitoringError(f"run {run_id}: manifest records no {BASELINE_FILE} hash")
    path = run_dir / BASELINE_FILE
    try:
        data = path.read_bytes()
    except OSError:
        raise MonitoringError(f"run {run_id}: {BASELINE_FILE} is missing") from None
    if hashlib.sha256(data).hexdigest() != expected:
        raise MonitoringError(f"run {run_id}: {BASELINE_FILE} sha256 does not match the manifest")
    try:
        baseline = json.loads(data)
    except ValueError:
        raise MonitoringError(f"run {run_id}: {BASELINE_FILE} is not valid JSON") from None
    _check_baseline(baseline)
    return baseline


def _check_baseline(b) -> None:
    def bad(msg):
        raise MonitoringError(f"{BASELINE_FILE}: {msg}")
    if not isinstance(b, dict):
        bad("not an object")
    feats = b.get("features")
    if not isinstance(feats, dict) or set(feats) != set(FEATURE_COLUMNS):
        bad("features must cover exactly FEATURE_COLUMNS")
    for col, spec in feats.items():
        kind = spec.get("kind") if isinstance(spec, dict) else None
        props = spec.get("proportions") if isinstance(spec, dict) else None
        if kind == "categorical":
            ok = isinstance(spec.get("categories"), list) and isinstance(props, list) \
                and len(props) == len(spec["categories"]) + 1
        elif kind == "numeric":
            ok = isinstance(spec.get("edges"), list) and isinstance(props, list) \
                and len(props) == len(spec["edges"]) + 2   # bins + missing
        else:
            ok = False
        if not ok:
            bad(f"feature {col} has an invalid spec")
    score = b.get("score")
    if not (isinstance(score, dict) and isinstance(score.get("edges"), list)
            and isinstance(score.get("proportions"), list)
            and len(score["proportions"]) == len(score["edges"]) + 1):
        bad("score baseline is invalid")
    rate = ((b.get("performance") or {}).get("at_threshold") or {}).get("predicted_positive_rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
        bad("performance.at_threshold.predicted_positive_rate is missing or invalid")


# ---------------------------------------------------------------------------
# One batch record
# ---------------------------------------------------------------------------
def _is_missing(series: pd.Series) -> pd.Series:
    """Missing as TD-5 sees it: None/NaN/pd.NA, or an empty or whitespace-only string."""
    values = series.astype(object)          # nullable dtypes (Float64, Int64) included
    missing = values.isna()
    blank = values.map(lambda v: isinstance(v, str) and v.strip() == "")
    return (missing | blank).astype(bool)


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def batch_record(*, manifest: dict, input_frame: pd.DataFrame, predictions: pd.DataFrame,
                 rejects: pd.DataFrame, scored_features: pd.DataFrame, baseline: dict,
                 thresholds: Thresholds = Thresholds()) -> dict:
    """Aggregate-only monitoring record for one batch. Pure: reads and writes nothing."""
    for col in ("churn_probability", "churn_flag"):
        if col not in predictions.columns:
            raise MonitoringError(f"predictions lack column {col}")
    for col in ("reason_code", "column"):
        if col not in rejects.columns:
            raise MonitoringError(f"rejects lack column {col}")
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in input_frame.columns]
    if missing_cols:
        raise MonitoringError(f"input lacks columns {missing_cols}")
    if list(scored_features.columns) != FEATURE_COLUMNS or len(scored_features) != len(predictions):
        raise MonitoringError("scored features do not line up with the predictions")
    for key in ("batch_id", "n_input", "n_scored", "n_rejected_rows", "reject_counts_by_reason",
                "reject_counts_unit", "release_id", "run_id", "model_sha256"):
        if key not in manifest:
            raise MonitoringError(f"batch manifest lacks {key}")
    n_input, n_scored = int(manifest["n_input"]), int(manifest["n_scored"])
    if n_input != len(input_frame) or n_scored != len(predictions):
        raise MonitoringError("batch manifest counts disagree with the batch data")

    reject_rate = _rate(int(manifest["n_rejected_rows"]), n_input)
    unseen = rejects[rejects["reason_code"].isin(UNSEEN_REASONS)
                     & rejects["column"].isin(CATEGORICAL_FEATURES)]
    unseen_rows = int(unseen.index.nunique())
    unseen_share = _rate(unseen_rows, n_input)
    unseen_by_column = {str(c): int(unseen.index[unseen["column"] == c].nunique())
                        for c in sorted(set(unseen["column"]))}
    null_rate = {c: (None if n_input == 0 else float(_is_missing(input_frame[c]).mean()))
                 for c in REQUIRED_COLUMNS}

    insufficient = n_scored < thresholds.min_batch_n
    if insufficient:
        feat_psi, s_psi = None, None   # not evaluated: too few scored rows
    else:
        feat_psi = {c: float(v["psi"]) for c, v in
                    feature_psi(baseline["features"], scored_features).items()}
        s_psi = float(score_psi(baseline["score"], predictions["churn_probability"].to_numpy()))
    flag_rate = None if n_scored == 0 else float(predictions["churn_flag"].astype(bool).mean())
    baseline_flag = float(baseline["performance"]["at_threshold"]["predicted_positive_rate"])

    alerts = []
    if reject_rate is not None and reject_rate > thresholds.reject_rate:
        alerts.append({"type": "data_quality", "metric": "reject_rate",
                       "value": reject_rate, "threshold": thresholds.reject_rate})
    if unseen_share is not None and unseen_share > thresholds.unseen_share:
        alerts.append({"type": "data_quality", "metric": "unseen_share",
                       "value": unseen_share, "threshold": thresholds.unseen_share})
    if not insufficient:
        for col, v in feat_psi.items():
            if v > thresholds.psi:
                alerts.append({"type": "drift", "metric": "feature_psi", "feature": col,
                               "value": v, "threshold": thresholds.psi})
        if s_psi > thresholds.psi:
            alerts.append({"type": "drift", "metric": "score_psi",
                           "value": s_psi, "threshold": thresholds.psi})

    return {
        "monitoring_schema_version": MONITORING_SCHEMA_VERSION,
        "batch_id": manifest["batch_id"],
        "batch_created_at": manifest.get("created_at"),
        "release_id": manifest["release_id"],
        "run_id": manifest["run_id"],
        "model_sha256": manifest["model_sha256"],
        "input_sha256": (manifest.get("input") or {}).get("sha256"),
        "synthetic": manifest.get("synthetic"),
        "n_input": n_input,
        "n_scored": n_scored,
        "n_rejected_rows": int(manifest["n_rejected_rows"]),
        "reject_rate": reject_rate,
        "reject_counts_by_reason": manifest["reject_counts_by_reason"],
        "reject_counts_unit": manifest["reject_counts_unit"],
        "null_rate": null_rate,
        "unseen_share": unseen_share,
        "unseen_rows_by_column": unseen_by_column,
        "feature_psi": feat_psi,
        "score_psi": s_psi,
        "flag_rate": flag_rate,
        "baseline_flag_rate": baseline_flag,
        "flag_rate_diff": None if flag_rate is None else flag_rate - baseline_flag,
        "insufficient_sample": insufficient,
        "thresholds": asdict(thresholds),
        "baseline_sources": {"features": baseline.get("features_source", "train rows"),
                             "score": baseline.get("score_source", "validation scores")},
        "alerts": alerts,
        "retrain_review_recommended": bool(alerts),
        "labels_available": False,
        "timings_s": manifest.get("timings_s"),
        "timings_note": manifest.get("timings_note"),
    }


# ---------------------------------------------------------------------------
# Append-only JSONL
# ---------------------------------------------------------------------------
def read_records(paths: Paths, release_id: str) -> list[dict]:
    """All records for a release. A torn or invalid line fails closed."""
    path = records_path(paths, release_id)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise MonitoringError(f"{RECORDS_FILE} ends with an incomplete record")
    records = []
    for i, line in enumerate(raw.splitlines(), 1):
        try:
            rec = json.loads(line)
        except ValueError:
            raise MonitoringError(f"{RECORDS_FILE} line {i} is not valid JSON") from None
        if not isinstance(rec, dict) or "batch_id" not in rec:
            raise MonitoringError(f"{RECORDS_FILE} line {i} is not a batch record")
        records.append(rec)
    return records


def append_record(paths: Paths, record: dict) -> Path:
    """Append one record. Refuses a batch_id that is already recorded. If the write
    fails part-way, the file is truncated back to its previous length, so no partial
    line is left behind. Existing lines are never rewritten."""
    path = records_path(paths, record["release_id"])
    existing = read_records(paths, record["release_id"])  # also fails closed on corruption
    if any(r.get("batch_id") == record["batch_id"] for r in existing):
        raise MonitoringError(f"batch {record['batch_id']} already has a monitoring record")
    data = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        size_before = os.fstat(fd).st_size
        try:
            written = os.write(fd, data)
            if written != len(data):
                raise OSError(f"short write ({written} of {len(data)} bytes)")
            os.fsync(fd)
        except BaseException:
            os.ftruncate(fd, size_before)   # roll back any partial line
            raise
    finally:
        os.close(fd)
    return path


# ---------------------------------------------------------------------------
# Static Markdown report (reads the JSONL only; never appends)
# ---------------------------------------------------------------------------
def _fmt(v, nd=3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "n/a" if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def render_report(release_id: str, records: list[dict]) -> str:
    lines = [f"# Monitoring report: release {release_id}", "",
             "## Read this first", "",
             f"- **No labels, no performance.** {NO_LABELS_NOTE}",
             f"- **PSI.** {PSI_NOTE}",
             f"- **Retrain review.** {RETRAIN_NOTE}",
             "- Every number is computed per batch. Batches are never merged by customer ID "
             "(synthetic IDs can repeat across batches).", "",
             f"## Batches ({len(records)})", ""]
    if not records:
        lines.append("No batches recorded.")
        return "\n".join(lines) + "\n"
    lines += ["| batch | synthetic | n_input | n_scored | reject rate | unseen share | "
              "max feature PSI | score PSI | flag rate (validation) | sample | alerts | "
              "retrain review |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        fp = r.get("feature_psi")
        max_fp = None if not fp else max(fp.values())
        sample = "insufficient_sample" if r.get("insufficient_sample") else "ok"
        alerts = ", ".join(a["metric"] + (f":{a['feature']}" if "feature" in a else "")
                           for a in r.get("alerts", [])) or "none"
        lines.append(
            f"| {r['batch_id']} | {_fmt(r.get('synthetic'))} | {r.get('n_input')} | "
            f"{r.get('n_scored')} | {_fmt(r.get('reject_rate'))} | {_fmt(r.get('unseen_share'))} | "
            f"{_fmt(max_fp)} | {_fmt(r.get('score_psi'))} | {_fmt(r.get('flag_rate'))} "
            f"({_fmt(r.get('baseline_flag_rate'))}) | {sample} | {alerts} | "
            f"{'yes' if r.get('retrain_review_recommended') else 'no'} |")
    lines += ["", "Feature PSI uses the TRAIN-row baseline; score PSI and the validation flag "
              "rate use the VALIDATION baseline. Reject counts are per reject record; reject "
              "rate uses rows (n_rejected_rows / n_input)."]
    return "\n".join(lines) + "\n"
