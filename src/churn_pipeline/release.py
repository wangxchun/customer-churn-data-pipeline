"""W1: holdout guard, release registry, gates, promote and rollback.

A *release* is an immutable directory that pins one Phase 3 run for serving:

    <models>/registry/releases/<release_id>/release.json        release manifest
    <models>/registry/releases/<release_id>/holdout_guard.json  keyed HMACs of holdout IDs (0600)
    <models>/registry/production.json                           pointer (atomic replace)
    <models>/registry/audit_log.jsonl                           append-only events

Holdout guard (decisions D5, D5-Q1, W1-Q1, W1-Q3, W1-Q4)
    ``create_release`` reads the split file once, using only the ``customer_id`` and
    ``split`` columns. It checks that file's content hash against the run manifest
    (gate G4) and stores HMAC-SHA256(key, id) for every test-split ID. Nothing else
    ever reads the split file: not ``check_release``, ``promote``, ``rollback``, nor
    prediction. If one input ID matches, the whole batch is rejected before TD-5
    scoring runs. Messages give a count, never an ID or HMAC.

HMAC key (W1-Q4)
    The key lives outside the repository and outside every release artefact. Library
    functions take it as ``key: bytes``. The CLI reads it with ``load_guard_key()``
    from the file named by ``$CHURN_HOLDOUT_GUARD_KEY_FILE``. The guard stores only
    ``key_id`` (an HMAC of a constant), so a wrong key is detected and fails closed.

    Residual risk: customer IDs have low entropy (``\\d{4}-[A-Z]{5}``, about 1.2e11
    values). Anyone holding the key can test candidate IDs and recover the guard's
    contents. HMAC is not encryption and does not guarantee confidentiality.

Nothing in this module reads ``test.parquet``, evaluation outputs, or customer data.
Every function needs an explicit ``Paths``; there is no default data directory.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd
import sklearn

from . import inference
from .config import PROJECT_ROOT, Paths
from .download import sha256_file
from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, ID_COLUMN, split_content_hash
from .inference import INFERENCE_SCHEMA_VERSION, OUTPUT_COLUMNS, InferenceInputError
from .tracking import atomic_directory, environment, git_state, write_json, write_json_atomic

GUARD_KEY_ENV = "CHURN_HOLDOUT_GUARD_KEY_FILE"
MIN_KEY_BYTES = 32
_KEY_ID_MESSAGE = b"churn-pipeline/holdout-guard/key-id/v1"
_ID_MESSAGE_PREFIX = b"churn-pipeline/holdout-guard/id/v1\x00"  # domain separation

RELEASE_SCHEMA_VERSION = "1"
GUARD_SCHEMA_VERSION = "1"
HASH_SCHEME = "hmac-sha256/v1"
RELEASE_FILE = "release.json"
GUARD_FILE = "holdout_guard.json"
SPLIT_FILE = "split_assignments.parquet"
SPLIT_COLUMNS = [ID_COLUMN, "split"]  # D5-Q1: split metadata only
HOLDOUT_SPLIT = "test"
HOLDOUT_REASON = "holdout_customer_id"
DEFAULT_LATENCY_BUDGET_S = 5.0
RELEASE_GATES = ("G1", "G2", "G3", "G4", "G5", "G7", "G8", "G9")  # G6 removed (W1-Q2)
CHECK_GATES = ("G1", "G2", "G3", "G5", "G7", "G8", "G9")            # never G4 (W1-Q3)

CAVEATS = [
    "Release gates check integrity and the input/output contract. They say nothing "
    "about production performance.",
    "Champion selection on validation data is seed-sensitive (Phase 3: a different "
    "seed selected a different model family).",
    "Prediction parity is proven only on synthetic data (TD-5); G6 (real validation "
    "parity) was removed by decision W1-Q2.",
]

# D6 provenance waiver: the PM-approved text (W1-Q7, B2). The reason comes from the
# CLI (--waive-dirty-code) and the PM approval date from --waiver-approved-on. The
# approval date is stored separately from the release creation timestamp.
WAIVER_SCOPE = ("Applies only to release {release_id} of run {run_id}. It acknowledges that "
                "the run's training code was uncommitted (code.dirty was true at training "
                "time).")
WAIVER_LIMITATIONS = [
    "Does not rewrite code.dirty or modify any file in the run directory.",
    "Does not claim the run is fully reproducible from a clean Git commit.",
    "Does not authorise retraining, champion reselection, or access to test evaluation data.",
    "Does not authorise general access to holdout/test data, real-customer scoring, "
    "repeated split access, or changes to the existing data-access restrictions.",
]
WAIVER_COMPLETION = ("Ends when this release is superseded or rolled back, or when a "
                     "clean-code run replaces this champion under a separate decision.")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ReleaseError(RuntimeError):
    """A release workflow step was refused."""


class GateFailedError(ReleaseError):
    def __init__(self, gate: str, reason: str):
        self.gate = gate
        self.reason = reason
        super().__init__(f"gate {gate} failed: {reason}")


class PromotionError(ReleaseError):
    """Promotion failed after the pointer moved; the pointer was restored."""


class RollbackError(ReleaseError):
    """Rollback was refused; the pointer is unchanged."""


class ReleaseIntegrityError(inference.ModelIntegrityError):
    """A release or its holdout guard cannot be trusted. Fail closed: nothing is scored."""


class GuardKeyError(ReleaseIntegrityError):
    """The HMAC key is unavailable, invalid, or not the key the guard was built with."""


# ---------------------------------------------------------------------------
# Key management (W1-Q4)
# ---------------------------------------------------------------------------
def load_guard_key(environ: Mapping[str, str] | None = None) -> bytes:
    """Read the HMAC key from the file named by ``$CHURN_HOLDOUT_GUARD_KEY_FILE``.

    The file must sit outside the repository, be a regular file that only its
    owner can access (POSIX), and hold at least 32 bytes as hex.
    """
    env = os.environ if environ is None else environ
    raw = env.get(GUARD_KEY_ENV, "")
    if not raw:
        raise GuardKeyError(f"{GUARD_KEY_ENV} is not set; the holdout guard needs its key")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise GuardKeyError(f"{GUARD_KEY_ENV} must be an absolute path")
    resolved = path.resolve()
    if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:  # checked before any access
        raise GuardKeyError("the guard key file must live outside the repository")
    if not resolved.is_file():
        raise GuardKeyError("the guard key file is missing or not a regular file")
    if os.name == "posix" and resolved.stat().st_mode & 0o077:
        raise GuardKeyError("the guard key file must be accessible by its owner only (chmod 600)")
    try:
        key = bytes.fromhex(resolved.read_text().strip())
    except ValueError:
        raise GuardKeyError("the guard key file does not contain hex") from None
    _check_key(key)
    return key


def _check_key(key: bytes) -> None:
    if not isinstance(key, (bytes, bytearray)) or len(key) < MIN_KEY_BYTES:
        raise GuardKeyError(f"the guard key must be at least {MIN_KEY_BYTES} bytes")


def key_id(key: bytes) -> str:
    """Identifies a key without revealing it: HMAC of a fixed message, first 16 hex chars."""
    return hmac.new(bytes(key), _KEY_ID_MESSAGE, hashlib.sha256).hexdigest()[:16]


def id_hmac(key: bytes, customer_id: str) -> str:
    return hmac.new(bytes(key), _ID_MESSAGE_PREFIX + customer_id.encode("utf-8"),
                    hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Registry, audit log
# ---------------------------------------------------------------------------
def registry_dir(paths: Paths) -> Path:
    return paths.models / "registry"


def _releases_dir(paths: Paths) -> Path:
    return registry_dir(paths) / "releases"


def _pointer_path(paths: Paths) -> Path:
    return registry_dir(paths) / "production.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_name(name: str, what: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ReleaseError(f"invalid {what}")
    return name


def _audit(paths: Paths, event: str, **fields) -> None:
    """Append one event. Callers pass gate names, IDs of runs/releases and reasons only;
    reasons come from this module's messages, which never contain customer IDs."""
    d = registry_dir(paths)
    d.mkdir(parents=True, exist_ok=True)
    record = {"at": _now(), "event": event, **fields}
    with open(d / "audit_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_audit_log(paths: Paths) -> list[dict]:
    p = registry_dir(paths) / "audit_log.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _read_pointer(paths: Paths) -> dict | None:
    p = _pointer_path(paths)
    return json.loads(p.read_text()) if p.exists() else None


def _write_pointer(paths: Paths, pointer: dict | None) -> None:
    p = _pointer_path(paths)
    if pointer is None:
        p.unlink(missing_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(p, pointer)


# ---------------------------------------------------------------------------
# Loaded release
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedRelease:
    release_id: str
    manifest: dict = field(repr=False)
    model: inference.LoadedModel = field(repr=False)
    _holdout_hmacs: frozenset = field(repr=False)
    _key: bytes = field(repr=False)


def feature_schema_sha256(m: inference.LoadedModel) -> str:
    """Hash of the model's own input/category/output schema (no data is read)."""
    schema = {"input_columns": FEATURE_COLUMNS,
              "categories": {c: [str(v) for v in m.categories[c]] for c in CATEGORICAL_FEATURES},
              "output_columns": list(m.output_feature_columns)}
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def _read_run_manifest(paths: Paths, run_id: str) -> tuple[Path, dict]:
    run_dir = paths.models / _safe_name(run_id, "run id")
    p = run_dir / "manifest.json"
    if run_id == "registry" or not p.is_file():
        raise ReleaseError(f"no completed run {run_id}")
    try:
        return run_dir, json.loads(p.read_text())
    except ValueError:
        raise ReleaseError(f"run {run_id} has an unreadable manifest.json") from None


def _verify_release(paths: Paths, release_dir: Path, manifest: dict, key: bytes,
                    expected_release_id: str) -> LoadedRelease:
    """G9 plus the bindings between release, guard, run manifest and model. Fails closed."""
    _check_key(key)

    def bad(msg: str):
        raise ReleaseIntegrityError(f"release {expected_release_id}: {msg}")

    if manifest.get("release_schema_version") != RELEASE_SCHEMA_VERSION:
        bad("unsupported release schema version")
    if manifest.get("release_id") != expected_release_id:
        bad("release_id does not match its directory")
    recorded = manifest.get("holdout_guard") or {}
    guard_path = release_dir / GUARD_FILE
    if not guard_path.is_file():
        bad("holdout guard is missing")
    if sha256_file(guard_path) != recorded.get("sha256"):
        bad("holdout guard sha256 does not match release.json")
    try:
        guard = json.loads(guard_path.read_text())
    except ValueError:
        bad("holdout guard is unreadable")
    if guard.get("guard_schema_version") != GUARD_SCHEMA_VERSION or \
            recorded.get("guard_schema_version") != GUARD_SCHEMA_VERSION:
        bad("unsupported holdout guard schema version")
    if guard.get("hash_scheme") != HASH_SCHEME:
        bad("unsupported holdout guard hash scheme")
    if guard.get("key_id") != key_id(key):
        raise GuardKeyError(f"release {expected_release_id}: the guard key is not the key "
                            "this holdout guard was built with")
    hmacs = guard.get("id_hmacs")
    if not isinstance(hmacs, list) or not all(
            isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in hmacs):
        bad("holdout guard entries are malformed")
    if len(set(hmacs)) != len(hmacs) or guard.get("n_ids") != len(hmacs) \
            or recorded.get("n_ids") != len(hmacs):
        bad("holdout guard n_ids does not match its contents")
    data = manifest.get("data") or {}
    if guard.get("model_sha256") != manifest.get("model_sha256"):
        bad("holdout guard is bound to a different model")
    if guard.get("split_content_sha256") != data.get("split_content_sha256"):
        bad("holdout guard is bound to a different split")
    if guard.get("customer_master_sha256") != data.get("customer_master_sha256"):
        bad("holdout guard is bound to a different customer_master record")

    try:
        run_dir, run_manifest = _read_run_manifest(paths, str(manifest.get("run_id")))
    except ReleaseError as e:
        bad(str(e))
    if (run_manifest.get("data") or {}).get("split_content_sha256") != data.get("split_content_sha256"):
        bad("the run manifest's split hash differs from the release")
    model = inference.load_model(run_dir)  # TD-5: hashes and version before unpickling
    if model.model_sha256 != manifest.get("model_sha256"):
        bad("the run's model differs from the released model")
    if feature_schema_sha256(model) != manifest.get("feature_schema_sha256"):
        bad("the model's feature schema differs from the release")
    if model.threshold != (manifest.get("decision") or {}).get("threshold"):
        bad("the model's threshold differs from the release")
    return LoadedRelease(expected_release_id, manifest, model, frozenset(hmacs), bytes(key))


def load_release(paths: Paths, release_id: str, key: bytes) -> LoadedRelease:
    d = _releases_dir(paths) / _safe_name(release_id, "release id")
    p = d / RELEASE_FILE
    if not p.is_file():
        raise ReleaseIntegrityError(f"release {release_id}: release.json is missing")
    try:
        manifest = json.loads(p.read_text())
    except ValueError:
        raise ReleaseIntegrityError(f"release {release_id}: release.json is unreadable") from None
    return _verify_release(paths, d, manifest, key, release_id)


def load_production(paths: Paths, key: bytes) -> LoadedRelease:
    pointer = _read_pointer(paths)
    if not pointer or not pointer.get("release_id"):
        raise ReleaseIntegrityError("no production release is set")
    return load_release(paths, pointer["release_id"], key)


# ---------------------------------------------------------------------------
# Release prediction (guard first, then TD-5)
# ---------------------------------------------------------------------------
def _holdout_hits(r: LoadedRelease, df) -> int:
    if not isinstance(df, pd.DataFrame) or ID_COLUMN not in df.columns:
        return 0  # TD-5 raises its own batch-level error before scoring
    ids = df[ID_COLUMN]
    if isinstance(ids, pd.DataFrame):  # duplicate column names: TD-5 rejects the batch
        return 0
    return sum(1 for v in ids.tolist()
               if isinstance(v, str) and id_hmac(r._key, v) in r._holdout_hmacs)


def predict_batch(r: LoadedRelease, df: pd.DataFrame) -> inference.InferenceResult:
    """Reject the whole batch if any row is a holdout customer (W1-Q1); else TD-5 scoring."""
    hits = _holdout_hits(r, df)
    if hits:
        rows = "row belongs" if hits == 1 else "rows belong"
        raise InferenceInputError(
            f"{hits} input {rows} to the protected holdout set; the whole batch was "
            "rejected and nothing was scored", level="batch", reasons=[(ID_COLUMN, HOLDOUT_REASON)])
    return inference.predict_batch(r.model, df)


def predict_one(r: LoadedRelease, record: dict) -> dict:
    if isinstance(record, dict) and isinstance(record.get(ID_COLUMN), str) \
            and id_hmac(r._key, record[ID_COLUMN]) in r._holdout_hmacs:
        raise InferenceInputError("the input record belongs to the protected holdout set; "
                                  "nothing was scored", level="record",
                                  reasons=[(ID_COLUMN, HOLDOUT_REASON)])
    return inference.predict_one(r.model, record)


# ---------------------------------------------------------------------------
# Canary (G5) and latency (G7): synthetic rows generated here, never real customers
# ---------------------------------------------------------------------------
CANARY_ID_SUFFIX = "CANRY"
_CANARY_VALID_ROWS = 12
_EXPECTED_DTYPES = {
    "customer_id": "string", "churn_probability": "float64", "churn_flag": "bool",
    "decision_threshold": "float64", "model_run_id": "string", "model_sha256": "string",
    "inference_schema_version": "string",
}


def make_canary(categories: dict[str, tuple]) -> tuple[pd.DataFrame, dict[int, set]]:
    """Deterministic synthetic batch: valid rows from the model's own categories, plus
    rows that must be rejected. Returns (batch, {row position: expected reasons})."""
    rows = []
    for i in range(_CANARY_VALID_ROWS):
        tenure = 0 if i == 0 else 1 + (i * 7) % 72        # row 0: a valid blank total_charges
        monthly = round(20 + (i * 9.25) % 90, 2)
        row = {ID_COLUMN: f"{9900 + i:04d}-{CANARY_ID_SUFFIX}", "senior_citizen": i % 2,
               "tenure_months": tenure, "monthly_charges": monthly,
               "total_charges": None if tenure == 0 else round(tenure * monthly, 2)}
        for c in CATEGORICAL_FEATURES:
            values = sorted(str(v) for v in categories[c])
            row[c] = values[i % len(values)]
        rows.append(row)
    not_allowed = {**rows[1], ID_COLUMN: f"9950-{CANARY_ID_SUFFIX}", "contract": "Three year"}
    missing = {**rows[2], ID_COLUMN: f"9951-{CANARY_ID_SUFFIX}", "monthly_charges": None}
    batch = pd.DataFrame(rows + [not_allowed, missing])
    expected = {_CANARY_VALID_ROWS: {("contract", "not_allowed")},
                _CANARY_VALID_ROWS + 1: {("monthly_charges", "null_not_allowed")}}
    return batch, expected


def _canary_problems(r: LoadedRelease, batch: pd.DataFrame, expected: dict,
                     result: inference.InferenceResult) -> list[str]:
    p, rejects = result.predictions, result.rejects
    problems = []
    if list(p.columns) != OUTPUT_COLUMNS:
        return ["prediction columns differ from the output contract"]
    if {c: str(t) for c, t in p.dtypes.items()} != _EXPECTED_DTYPES:
        problems.append("prediction dtypes differ from the output contract")
    valid_ids = [batch.loc[i, ID_COLUMN] for i in range(len(batch)) if i not in expected]
    if list(p["customer_id"]) != valid_ids:
        problems.append("valid canary rows were not all scored in input order")
    prob = p["churn_probability"].to_numpy()
    if not ((prob >= 0) & (prob <= 1)).all():
        problems.append("probabilities outside [0, 1]")
    if not (p["churn_flag"] == (p["churn_probability"] >= r.model.threshold)).all():
        problems.append("churn_flag disagrees with the threshold")
    if not ((p["decision_threshold"] == r.model.threshold).all()
            and (p["model_sha256"] == r.model.model_sha256).all()
            and (p["model_run_id"] == r.model.run_id).all()
            and (p["inference_schema_version"] == INFERENCE_SCHEMA_VERSION).all()):
        problems.append("prediction metadata differs from the loaded model")
    got = {}
    for idx, col, code in zip(rejects.index, rejects["column"], rejects["reason_code"]):
        got.setdefault(int(idx), set()).add((col, code))
    if got != expected:
        problems.append("invalid canary rows were not rejected with the expected reasons")
    return problems


def run_canary(r: LoadedRelease, latency_budget_s: float) -> dict:
    """G5 (contract) and G7 (latency) on a synthetic batch through the release path."""
    batch, expected = make_canary(r.model.categories)
    start = time.perf_counter()
    try:
        result = predict_batch(r, batch)
    except (InferenceInputError, inference.ScoringError) as e:
        raise GateFailedError("G5", f"canary was not scored ({type(e).__name__}: {e})") from e
    elapsed = time.perf_counter() - start
    problems = _canary_problems(r, batch, expected, result)
    if problems:
        raise GateFailedError("G5", "; ".join(problems))
    if elapsed > latency_budget_s:
        raise GateFailedError("G7", f"canary took {elapsed:.3f}s, budget {latency_budget_s}s")
    return {"G5": {"status": "pass", "detail": f"{len(result.predictions)} scored, "
                                               f"{len(expected)} rejected as expected"},
            "G7": {"status": "pass", "detail": f"{elapsed:.3f}s within {latency_budget_s}s"}}


# ---------------------------------------------------------------------------
# Gates G1, G2, G3, G8 and the one-time split read (G4)
# ---------------------------------------------------------------------------
def _gate_g1(run_dir: Path, run_manifest: dict) -> dict:
    expected = run_manifest.get("model_sha256")
    path = run_dir / "model.joblib"
    if not expected or not path.is_file() or sha256_file(path) != expected:
        raise GateFailedError("G1", "model.joblib sha256 does not match the run manifest")
    return {"status": "pass", "detail": "model.joblib sha256 matches the run manifest"}


def _gate_g2(run_manifest: dict) -> dict:
    trained = ((run_manifest.get("environment") or {}).get("packages") or {}).get("scikit-learn")
    if trained != sklearn.__version__:
        raise GateFailedError("G2", f"model was pickled with scikit-learn {trained}; "
                                    f"installed is {sklearn.__version__}")
    return {"status": "pass", "detail": f"scikit-learn {trained}"}


def _gate_g3(run_dir: Path) -> inference.LoadedModel:
    try:
        return inference.load_model(run_dir)
    except inference.ModelIntegrityError as e:
        raise GateFailedError("G3", f"model contract or schema check failed: {e}") from e


def _gate_g8(code, waiver: dict | None) -> dict:
    if not isinstance(code, dict) or "dirty" not in code:
        raise GateFailedError("G8", "the run manifest has no code provenance")
    if code["dirty"] is False:
        return {"status": "pass", "detail": "training code was committed"}
    if waiver and waiver.get("status") == "active" and str(waiver.get("reason", "")).strip() \
            and _is_iso_date(waiver.get("approved_on")):
        return {"status": "pass (waived)", "detail": "training code was dirty; D6 waiver recorded"}
    raise GateFailedError("G8", "training code was dirty and no provenance waiver is recorded (D6)")


def _read_holdout_ids(paths: Paths, run_manifest: dict) -> tuple[str, list[str]]:
    """G4. The ONLY read of the split file in W1: two columns, once, at release time."""
    expected = (run_manifest.get("data") or {}).get("split_content_sha256")
    if not expected:
        raise GateFailedError("G4", "the run manifest has no split_content_sha256")
    path = paths.features / SPLIT_FILE
    if not path.is_file():
        raise GateFailedError("G4", f"{SPLIT_FILE} not found")
    try:
        assignments = pd.read_parquet(path, columns=SPLIT_COLUMNS)
    except Exception as e:
        raise GateFailedError("G4", f"cannot read {SPLIT_FILE} ({type(e).__name__})") from e
    if split_content_hash(assignments) != expected:
        raise GateFailedError("G4", "the split's content hash differs from the run manifest")
    ids = assignments.loc[assignments["split"].astype(str) == HOLDOUT_SPLIT, ID_COLUMN]
    ids = ids.astype(str).tolist()
    if not ids:
        raise GateFailedError("G4", "the split has no holdout rows")
    return expected, ids


def _historical_test_evaluation(run_dir: Path) -> dict:
    """W1-Q8: the read count and output_dir *values* only. The directories they name
    are never opened, listed or hashed."""
    p = run_dir / "test_access.json"
    if not p.exists():
        return {"attempts": 0, "output_dirs": [], "source": "test_access.json (absent)"}
    try:
        log = json.loads(p.read_text())
    except ValueError:
        raise ReleaseError("test_access.json is malformed") from None
    reads = log.get("reads") if isinstance(log, dict) else None
    # A read is logged before test rows are touched; output_dir is added only on success.
    if not isinstance(reads, list) or not all(
            isinstance(e, dict) and isinstance(e.get("output_dir", ""), str) for e in reads):
        raise ReleaseError("test_access.json is malformed")
    return {"attempts": len(reads), "output_dirs": [e.get("output_dir") for e in reads],
            "source": "test_access.json"}


def _is_iso_date(value) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _check_approval_date(value) -> str:
    """The PM approval date: a real YYYY-MM-DD calendar date, not in the future (UTC)."""
    if not _is_iso_date(value):
        raise ReleaseError("a provenance waiver needs a PM approval date in YYYY-MM-DD form")
    if date.fromisoformat(value) > datetime.now(timezone.utc).date():
        raise ReleaseError("the waiver approval date is in the future")
    return value


def _make_waiver(reason: str, approved_on: str, recorded_at: str,
                 release_id: str, run_id: str) -> dict:
    return {"reason": reason.strip(),
            "scope": WAIVER_SCOPE.format(release_id=release_id, run_id=run_id),
            "limitations": WAIVER_LIMITATIONS,
            "completion_condition": WAIVER_COMPLETION,
            "status": "active",
            "approved_on": approved_on,      # PM approval date (given by the caller)
            "recorded_at": recorded_at,      # release creation timestamp (UTC)
            "text_status": f"approved by PM on {approved_on}"}


# W5 (plan 6.5): a new release directory is owner-only (0700) and every file in it 0600.
# Applies to directories created by create_release from now on; existing releases, the
# registry's production pointer and audit log are not changed. Not enforced on Windows,
# where POSIX mode bits do not apply.
RELEASE_DIR_MODE = 0o700
RELEASE_FILE_MODE = 0o600


def _check_private_tree(d: Path) -> None:
    if os.name == "nt":
        return
    bad = [p.name for p in [d, *d.iterdir()]
           if stat.S_IMODE(p.lstat().st_mode) != (RELEASE_DIR_MODE if p == d
                                                  else RELEASE_FILE_MODE)
           or (p != d and not p.is_file())]
    if bad:
        raise ReleaseError(f"release permissions are not owner-only for: {', '.join(bad)}")


def _write_private_json(path: Path, obj) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.chmod(path, 0o600)  # independent of the process umask


def _new_release_id() -> str:
    return f"rel-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Workflow: release, check-release, promote, rollback
# ---------------------------------------------------------------------------
def create_release(paths: Paths, run_id: str, key: bytes, *, waiver_reason: str | None = None,
                   waiver_approved_on: str | None = None, release_id: str | None = None,
                   latency_budget_s: float = DEFAULT_LATENCY_BUDGET_S) -> dict:
    """Run G1, G2, G3, G8, G4, G9, G5 and G7, then write an immutable release directory."""
    _check_key(key)
    if waiver_reason is not None and not waiver_reason.strip():
        raise ReleaseError("a provenance waiver needs a non-empty reason")
    if waiver_reason is not None:
        _check_approval_date(waiver_approved_on)
    elif waiver_approved_on is not None:
        raise ReleaseError("a waiver approval date was given without a waiver reason")
    run_dir, run_manifest = _read_run_manifest(paths, run_id)
    release_id = _safe_name(release_id or _new_release_id(), "release id")
    final = _releases_dir(paths) / release_id
    if final.exists():
        raise FileExistsError(f"release {release_id} already exists")

    gates: dict[str, dict] = {}
    try:
        gates["G1"] = _gate_g1(run_dir, run_manifest)
        gates["G2"] = _gate_g2(run_manifest)
        model = _gate_g3(run_dir)
        gates["G3"] = {"status": "pass", "detail": "input columns, encoder columns, categories "
                                                   "within the Phase 1 domain, classes [0, 1]"}
        created_at = _now()
        waiver = (_make_waiver(waiver_reason, waiver_approved_on, created_at, release_id, run_id)
                  if waiver_reason else None)
        gates["G8"] = _gate_g8(run_manifest.get("code"), waiver)
        history = _historical_test_evaluation(run_dir)
        split_sha, holdout_ids = _read_holdout_ids(paths, run_manifest)
        gates["G4"] = {"status": "pass",
                       "detail": "split content hash matches the run manifest (read once)"}

        run_data = run_manifest.get("data") or {}
        master_sha = run_data.get("customer_master_sha256")
        guard = {
            "guard_schema_version": GUARD_SCHEMA_VERSION, "hash_scheme": HASH_SCHEME,
            "key_id": key_id(key), "split_content_sha256": split_sha,
            "customer_master_sha256": master_sha, "model_sha256": run_manifest["model_sha256"],
            "n_ids": len(holdout_ids),
            "id_hmacs": sorted(id_hmac(key, i) for i in holdout_ids),
        }
        del holdout_ids  # plaintext IDs are not kept beyond this point
        params = run_manifest.get("params") or {}
        with atomic_directory(final, mode=RELEASE_DIR_MODE) as tmp:
            _write_private_json(tmp / GUARD_FILE, guard)
            manifest = {
                "release_schema_version": RELEASE_SCHEMA_VERSION,
                "release_id": release_id,
                "created_at": created_at,
                "run_id": run_id,
                "model_sha256": run_manifest["model_sha256"],
                "decision": {"threshold": model.threshold, "fn_cost": params.get("fn_cost"),
                             "fp_cost": params.get("fp_cost")},
                "feature_schema_sha256": feature_schema_sha256(model),
                "data": {
                    "split_content_sha256": split_sha,
                    "split_verified_at_release": True,
                    "customer_master_sha256": master_sha,
                    "customer_master_revalidated_at_release": False,
                    "customer_master_note": "copied from the run manifest; the master dataset "
                                            "was not re-read or re-hashed at release time, so "
                                            "this value does not verify the current master (W1-Q5)",
                },
                "training_environment": run_manifest.get("environment"),
                "code": run_manifest.get("code"),        # kept as recorded, including dirty: true
                "release_environment": environment(),
                "release_code": git_state(),
                "provenance_waiver": waiver,
                "holdout_guard": {"file": GUARD_FILE, "sha256": sha256_file(tmp / GUARD_FILE),
                                  "guard_schema_version": GUARD_SCHEMA_VERSION,
                                  "n_ids": guard["n_ids"]},
                "inference_schema_version": INFERENCE_SCHEMA_VERSION,
                "caveats": CAVEATS,
                "historical_test_evaluation": history,
            }
            try:
                loaded = _verify_release(paths, tmp, manifest, key, release_id)
            except inference.ModelIntegrityError as e:
                raise GateFailedError("G9", str(e)) from e
            gates["G9"] = {"status": "pass", "detail": "guard verified and bound to model and split"}
            gates.update(run_canary(loaded, latency_budget_s))
            manifest["gates"] = {g: gates[g] for g in RELEASE_GATES}
            write_json(tmp / RELEASE_FILE, manifest, mode=RELEASE_FILE_MODE)
            _check_private_tree(tmp)             # before the rename: never publish it looser
    except GateFailedError as e:
        _audit(paths, "gate_failed", stage="release", gate=e.gate, reason=e.reason,
               run_id=run_id, release_id=release_id)
        raise
    _audit(paths, "release_created", release_id=release_id, run_id=run_id,
           waiver=waiver is not None)
    return manifest


def _check_gates(paths: Paths, release_id: str, key: bytes, latency_budget_s: float) -> dict:
    """All gates except G4. Reads no split file, no data file; never modifies the release."""
    _check_key(key)
    d = _releases_dir(paths) / _safe_name(release_id, "release id")
    if not (d / RELEASE_FILE).is_file():
        raise GateFailedError("G9", f"release {release_id} not found")
    try:
        manifest = json.loads((d / RELEASE_FILE).read_text())
    except ValueError:
        raise GateFailedError("G9", "release.json is unreadable") from None
    try:
        run_dir, run_manifest = _read_run_manifest(paths, str(manifest.get("run_id")))
    except ReleaseError as e:
        raise GateFailedError("G1", str(e)) from e
    gates = {"G1": _gate_g1(run_dir, run_manifest), "G2": _gate_g2(run_manifest)}
    if run_manifest.get("model_sha256") != manifest.get("model_sha256"):
        raise GateFailedError("G1", "the run's model is not the released model")
    model = _gate_g3(run_dir)
    if feature_schema_sha256(model) != manifest.get("feature_schema_sha256"):
        raise GateFailedError("G3", "the model's feature schema differs from the release")
    gates["G3"] = {"status": "pass", "detail": "schema matches code constants and the release"}
    gates["G8"] = _gate_g8(manifest.get("code"), manifest.get("provenance_waiver"))
    try:
        loaded = _verify_release(paths, d, manifest, key, release_id)
    except inference.ModelIntegrityError as e:
        raise GateFailedError("G9", str(e)) from e
    gates["G9"] = {"status": "pass", "detail": "guard verified and bound to model and split"}
    gates.update(run_canary(loaded, latency_budget_s))
    return {g: gates[g] for g in CHECK_GATES}


def check_release(paths: Paths, release_id: str, key: bytes, *,
                  latency_budget_s: float = DEFAULT_LATENCY_BUDGET_S) -> dict:
    try:
        return _check_gates(paths, release_id, key, latency_budget_s)
    except GateFailedError as e:
        _audit(paths, "gate_failed", stage="check-release", gate=e.gate, reason=e.reason,
               release_id=release_id)
        raise


def promote(paths: Paths, release_id: str, key: bytes, *,
            latency_budget_s: float = DEFAULT_LATENCY_BUDGET_S) -> dict:
    """Re-check gates, move the pointer atomically, smoke-test; restore the pointer on failure."""
    try:
        _check_gates(paths, release_id, key, latency_budget_s)
    except GateFailedError as e:
        _audit(paths, "gate_failed", stage="promote", gate=e.gate, reason=e.reason,
               release_id=release_id)
        raise
    old = _read_pointer(paths)
    previous = old.get("release_id") if old else None
    new = {"release_id": release_id, "previous_release_id": previous, "updated_at": _now()}
    _write_pointer(paths, new)
    _audit(paths, "promoted", release_id=release_id, previous_release_id=previous)
    try:
        run_canary(load_production(paths, key), latency_budget_s)  # post-deploy smoke
    except (GateFailedError, inference.ModelIntegrityError) as e:
        _write_pointer(paths, old)
        _audit(paths, "rollback", reason="post_deploy_smoke_failed", release_id=previous,
               failed_release_id=release_id)
        raise PromotionError(f"post-deploy smoke failed for {release_id}; production pointer "
                             f"restored to {previous or 'none'} ({e})") from e
    return new


def rollback(paths: Paths, key: bytes) -> dict:
    """Point production back at the previous release (one level only)."""
    pointer = _read_pointer(paths)
    if not pointer or not pointer.get("release_id"):
        raise RollbackError("no production release is set")
    previous = pointer.get("previous_release_id")
    if not previous:
        raise RollbackError("no previous release to roll back to (only one level is kept)")
    try:
        load_release(paths, previous, key)
    except (ReleaseError, inference.ModelIntegrityError) as e:
        raise RollbackError(f"previous release {previous} failed verification: {e}") from e
    new = {"release_id": previous, "previous_release_id": None, "updated_at": _now()}
    _write_pointer(paths, new)
    _audit(paths, "rollback", reason="manual", release_id=previous,
           failed_release_id=pointer["release_id"])
    return new
