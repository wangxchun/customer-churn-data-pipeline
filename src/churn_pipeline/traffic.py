"""W2: certify a synthetic customer-ID suffix that no real customer ID uses.

Decision D7 requires synthetic inference traffic to use IDs in the Phase 1 format
``\\d{4}-[A-Z]{5}`` that provably do not collide with any real customer ID. The
approved strategy (W2-Q1 to W2-Q3) is check (d), certified in one read:

* One pre-declared candidate suffix is certified per call. There is no automatic
  fallback to another candidate (W2-Q2). A suffix is *certified* only if every
  real ID matches the format, there are no duplicates (check (b), done in the same
  read), the count equals the expected count, and **no** real ID ends with
  ``-<suffix>``. Synthetic IDs ``NNNN-<suffix>`` then cannot equal any real ID.
* Each suffix yields at most 10,000 IDs (``0000`` to ``9999``).
* Only the ``customer_id`` column is read (W2-Q3, option i). Nothing else is read:
  no ``split`` column, features, labels, guard or HMAC key. The input version is
  bound by file size and mtime. A change before or during the read fails closed.
* Every outcome reports counts only. No message, return value or log contains a
  real customer ID.

Checks (a) and (c) need no separate step (W2-Q1). A real HMAC collision is
negligible, and the W1 guard already rejects duplicate entries. W2 creates no
guard. Synthetic namespaces already in use (``CANRY``, ``ZZSYN``, ``ABCDE``) are
excluded statically.

This module never writes files. It does not touch the W1 registry.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .features import ID_COLUMN, DataContractError
from .sources import CUSTOMER_ID_PATTERN

CANDIDATE_SUFFIXES = ("SYNTR", "SYNTS", "SYNTT")      # W2-Q2, pre-declared
RESERVED_SUFFIXES = {                                 # existing synthetic namespaces
    "CANRY": "W1 release canary IDs",
    "ZZSYN": "Phase 1 synthetic orphan IDs",
    "ABCDE": "synthetic test-fixture IDs",
}
MAX_IDS_PER_SUFFIX = 10_000                           # NNNN = 0000..9999
ID_COLUMNS = [ID_COLUMN]                              # W2-Q3 (i): the only column read

_ID_RE = re.compile(CUSTOMER_ID_PATTERN)
_SUFFIX_RE = re.compile(r"[A-Z]{5}")


class SuffixCertificationError(DataContractError):
    """The suffix could not be certified. Fail closed: no synthetic IDs may be made.
    Messages contain counts and the candidate suffix only, never a customer ID."""


@dataclass(frozen=True)
class InputBinding:
    """The identity of the input file, taken from metadata (stat) only."""
    file_name: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class SuffixCertificate:
    suffix: str
    n_ids: int
    n_unique: int
    n_format_violations: int
    expected_count: int
    binding: InputBinding

    def summary(self) -> dict:
        """The only output of a certification: counts, the suffix, the binding."""
        return {"status": "certified", "suffix": self.suffix, "n_ids": self.n_ids,
                "n_unique": self.n_unique, "n_format_violations": self.n_format_violations,
                "expected_count": self.expected_count, "suffix_in_use": False,
                "input_file": self.binding.file_name, "input_size": self.binding.size,
                "input_mtime_ns": self.binding.mtime_ns}


def check_candidate(suffix: str) -> str:
    """Only a pre-declared candidate may be certified (W2-Q2)."""
    if not isinstance(suffix, str) or not _SUFFIX_RE.fullmatch(suffix):
        raise SuffixCertificationError("a suffix must be exactly five uppercase letters")
    if suffix in RESERVED_SUFFIXES:
        raise SuffixCertificationError(f"suffix {suffix} is reserved for "
                                       f"{RESERVED_SUFFIXES[suffix]}")
    if suffix not in CANDIDATE_SUFFIXES:
        raise SuffixCertificationError(f"suffix {suffix} is not a pre-declared candidate "
                                       f"{list(CANDIDATE_SUFFIXES)}")
    return suffix


def _positive_int(name: str, value) -> int:
    """A positive int. ``bool`` is rejected even though it subclasses ``int``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SuffixCertificationError(f"{name} must be a positive integer")
    return value


def _input_path(path) -> Path:
    """A non-empty str or os.PathLike (not bytes). Checked without touching the file."""
    if isinstance(path, bytes) or not isinstance(path, (str, os.PathLike)) or path == "":
        raise SuffixCertificationError("input path must be a non-empty str or os.PathLike")
    return Path(path)


def certify_suffix(real_ids: Iterable, suffix: str, *, expected_count: int,
                   binding: InputBinding) -> SuffixCertificate:
    """Pure check of one suffix against an in-memory ID collection. Fails closed."""
    check_candidate(suffix)
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) \
            or expected_count < 1:
        raise SuffixCertificationError("expected_count must be a positive integer")
    ending = "-" + suffix
    n = n_bad = n_in_use = 0
    seen: set = set()
    for value in real_ids:
        n += 1
        if not isinstance(value, str) or not _ID_RE.fullmatch(value):
            n_bad += 1
            continue
        seen.add(value)
        if value.endswith(ending):
            n_in_use += 1
    n_unique = len(seen)
    seen.clear()  # do not keep real IDs beyond this point

    if n == 0:
        raise SuffixCertificationError("no customer IDs were read")
    problems = []
    if n_bad:
        problems.append(f"{n_bad} ID(s) are missing or not in the Phase 1 format")
    if n_unique != n - n_bad:
        problems.append(f"{n - n_bad - n_unique} duplicate ID(s)")
    if n != expected_count:
        problems.append(f"read {n} IDs, expected {expected_count}")
    if n_in_use:
        problems.append(f"suffix {suffix} is used by {n_in_use} real ID(s)")
    if problems:
        raise SuffixCertificationError(f"suffix {suffix} not certified: " + "; ".join(problems))
    return SuffixCertificate(suffix=suffix, n_ids=n, n_unique=n_unique, n_format_violations=0,
                             expected_count=expected_count, binding=binding)


def input_binding(path: Path) -> InputBinding:
    st = os.stat(path)
    return InputBinding(file_name=Path(path).name, size=st.st_size, mtime_ns=st.st_mtime_ns)


def certify_suffix_from_file(path: Path, suffix: str, *, expected_count: int,
                             expected_size: int, expected_mtime_ns: int) -> SuffixCertificate:
    """Read only ``customer_id`` from one parquet file, once, and certify one suffix.

    ``expected_size`` and ``expected_mtime_ns`` pin the input version (W2-Q3). They
    are checked before the read, and the file is re-checked after it.

    Every parameter is validated BEFORE any file operation (no stat, no is_file, no
    read). An invalid call therefore never touches the input file.
    """
    check_candidate(suffix)
    _positive_int("expected_count", expected_count)
    _positive_int("expected_size", expected_size)
    _positive_int("expected_mtime_ns", expected_mtime_ns)
    path = _input_path(path)
    # ---- file operations start here ----
    if not path.is_file():
        raise SuffixCertificationError("input file not found")
    before = input_binding(path)
    if (before.size, before.mtime_ns) != (expected_size, expected_mtime_ns):
        raise SuffixCertificationError("input file size or mtime differs from the approved "
                                       "binding; refusing to read it")
    read_error = None
    try:
        frame = pd.read_parquet(path, columns=ID_COLUMNS)
    except Exception as e:
        read_error = type(e).__name__  # keep only the type: messages may hold paths/values
    if read_error is not None:
        # Raised outside the except block, so neither __cause__ nor __context__ carries
        # the original exception.
        raise SuffixCertificationError(f"cannot read the {ID_COLUMN} column ({read_error})")
    if input_binding(path) != before:
        raise SuffixCertificationError("input file changed while it was being read")
    if list(frame.columns) != ID_COLUMNS:
        raise SuffixCertificationError("unexpected columns were returned")
    ids = frame[ID_COLUMN].tolist()
    del frame
    try:
        return certify_suffix(ids, suffix, expected_count=expected_count, binding=before)
    finally:
        ids.clear()


def verify_binding(cert: SuffixCertificate, path: Path) -> None:
    """Before reusing a certificate: the input must still be the certified version."""
    if input_binding(path) != cert.binding:
        raise SuffixCertificationError("the certified input file has changed; the suffix "
                                       "certificate is no longer valid")


def synthetic_ids(cert: SuffixCertificate, n: int, start: int = 0) -> list[str]:
    """``n`` synthetic IDs ``NNNN-<suffix>`` from a certified suffix. Enforces the cap."""
    if not isinstance(cert, SuffixCertificate):
        raise SuffixCertificationError("synthetic IDs need a suffix certificate")
    check_candidate(cert.suffix)
    for name, v in (("n", n), ("start", start)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise SuffixCertificationError(f"{name} must be an integer")
    if n < 1 or start < 0 or start + n > MAX_IDS_PER_SUFFIX:
        raise SuffixCertificationError(f"a suffix yields at most {MAX_IDS_PER_SUFFIX} IDs "
                                       f"(0000-9999); requested start={start}, n={n}")
    ids = [f"{i:04d}-{cert.suffix}" for i in range(start, start + n)]
    assert all(_ID_RE.fullmatch(i) for i in ids)
    return ids
