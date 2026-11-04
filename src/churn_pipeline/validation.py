"""Declarative, row-level schema validation.

Two classes of problems are handled differently on purpose:

* Contract errors (a required *column* is missing, the file is not parseable)
  -> raise ``SchemaError``. The whole source is unusable; continuing would
  produce silently wrong data.
* Record errors (a *value* is bad) -> the row goes to a rejects table with a
  reason, and the rest of the batch proceeds. Nothing is dropped silently.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

import pandas as pd

Kind = Literal["str", "int", "float", "timestamp"]

REJECT_COLUMNS = ["source", "row_ref", "record_key", "reason", "raw"]


class SchemaError(Exception):
    """The source as a whole violates its contract."""


@dataclass(frozen=True)
class Column:
    name: str
    kind: Kind
    nullable: bool = False
    allowed: frozenset[Any] | None = None
    pattern: str | None = None
    min_value: float | None = None
    blank_as_null: bool = False  # treat whitespace-only strings as missing


# A row rule takes the typed frame and returns a boolean mask of *violations*.
RowRule = tuple[str, Callable[[pd.DataFrame], pd.Series]]


@dataclass
class ValidationResult:
    valid: pd.DataFrame
    rejects: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def empty_rejects() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in REJECT_COLUMNS})


def _is_missing(value: Any, col: Column) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value == "":
        return True
    if col.blank_as_null and isinstance(value, str) and value.strip() == "":
        return True
    return False


def _parse_value(value: Any, col: Column) -> tuple[Any, str | None]:
    """Return (parsed_value, error_message_or_None)."""
    if _is_missing(value, col):
        return None, None if col.nullable else f"{col.name}: missing required value"

    if col.kind == "str":
        if not isinstance(value, str):
            return None, f"{col.name}: expected string, got {type(value).__name__}"
        parsed: Any = value
    elif col.kind == "int":
        if isinstance(value, bool):
            return None, f"{col.name}: expected integer, got bool"
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            parsed = int(value)
        else:
            return None, f"{col.name}: not an integer: {value!r}"
    elif col.kind == "float":
        if isinstance(value, bool):
            return None, f"{col.name}: expected number, got bool"
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None, f"{col.name}: not a number: {value!r}"
        if not math.isfinite(parsed):
            return None, f"{col.name}: non-finite number: {value!r}"
    elif col.kind == "timestamp":
        if not isinstance(value, str):
            return None, f"{col.name}: expected ISO-8601 string"
        try:
            ts = datetime.fromisoformat(value)
        except ValueError:
            return None, f"{col.name}: not ISO-8601: {value!r}"
        if ts.tzinfo is None:
            # Naive timestamps are ambiguous across time zones; refuse to guess.
            return None, f"{col.name}: timestamp has no timezone: {value!r}"
        parsed = ts.astimezone(timezone.utc)
    else:  # pragma: no cover - guarded by Kind
        raise ValueError(col.kind)

    if col.allowed is not None and parsed not in col.allowed:
        return None, f"{col.name}: value {parsed!r} not in allowed set"
    if col.pattern is not None and not re.fullmatch(col.pattern, str(parsed)):
        return None, f"{col.name}: value {parsed!r} does not match {col.pattern}"
    if col.min_value is not None and parsed < col.min_value:
        return None, f"{col.name}: value {parsed!r} below minimum {col.min_value}"
    return parsed, None


_DTYPES = {"str": "string", "int": "Int64", "float": "Float64"}


def _to_series(values: list[Any], col: Column, index: pd.Index) -> pd.Series:
    if col.kind == "timestamp":
        return pd.Series(pd.to_datetime(values, utc=True), index=index, name=col.name)
    return pd.Series(values, index=index, name=col.name, dtype=_DTYPES[col.kind])


def raw_json(record: dict[str, Any]) -> str:
    return json.dumps(record, default=str, sort_keys=True)


def validate(
    raw: pd.DataFrame,
    columns: list[Column],
    source: str,
    key: str,
    row_refs: list[str],
    row_rules: list[RowRule] | None = None,
) -> ValidationResult:
    """Validate ``raw`` (untyped values) against ``columns``.

    ``row_refs`` gives a human-readable location per row (e.g. "line 12") so
    every reject can be traced back to the input file.
    """
    expected = [c.name for c in columns]
    missing = [c for c in expected if c not in raw.columns]
    if missing:
        raise SchemaError(f"{source}: missing required columns {missing}")

    warnings: list[str] = []
    extra = [c for c in raw.columns if c not in expected and not c.startswith("_")]
    if extra:
        warnings.append(f"{source}: ignoring unexpected columns {extra}")

    errors: list[list[str]] = [[] for _ in range(len(raw))]
    typed: dict[str, pd.Series] = {}
    for col in columns:
        parsed_values = []
        for i, value in enumerate(raw[col.name].tolist()):
            parsed, err = _parse_value(value, col)
            parsed_values.append(parsed)
            if err:
                errors[i].append(err)
        typed[col.name] = _to_series(parsed_values, col, raw.index)
    typed_df = pd.DataFrame(typed, index=raw.index)

    col_ok = pd.Series([not e for e in errors], index=raw.index)
    for rule_name, rule in row_rules or []:
        # Only meaningful on rows whose columns parsed; NA comparisons -> False.
        violated = rule(typed_df).fillna(False).astype(bool) & col_ok
        for i in violated.to_numpy().nonzero()[0]:
            errors[i].append(rule_name)

    ok = pd.Series([not e for e in errors], index=raw.index)
    rejects = pd.DataFrame(
        {
            "source": source,
            "row_ref": [row_refs[i] for i in range(len(raw)) if errors[i]],
            "record_key": [
                None if _is_missing(v, Column(key, "str")) else str(v)
                for v, bad in zip(raw[key].tolist(), errors)
                if bad
            ],
            "reason": ["; ".join(e) for e in errors if e],
            "raw": [
                raw_json(rec)
                for rec, bad in zip(raw[expected].to_dict("records"), errors)
                if bad
            ],
        },
        columns=REJECT_COLUMNS,
    )
    valid = typed_df[ok].copy()
    valid["_row_ref"] = [r for r, good in zip(row_refs, ok) if good]
    return ValidationResult(valid=valid, rejects=rejects, warnings=warnings)


def deduplicate(
    df: pd.DataFrame,
    key: str,
    source: str,
    order_by: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve duplicate keys. Returns (kept, rejects).

    1. Exact duplicates (identical content): keep the first, reject the rest.
    2. Same key, different content:
       - with ``order_by``: latest wins; older rows are rejected as superseded.
         A tie on the latest timestamp is a genuine conflict -> reject all tied.
       - without ``order_by``: no safe way to choose -> reject all of them.
    """
    content_cols = [c for c in df.columns if not c.startswith("_")]
    exact_dup = df.duplicated(subset=content_cols, keep="first")
    rejected = [(idx, "exact_duplicate") for idx in df.index[exact_dup]]
    rest = df[~exact_dup]

    dup_keys = rest[key][rest[key].duplicated(keep=False)].unique()
    for k in dup_keys:
        group = rest[rest[key] == k]
        if order_by is None:
            rejected += [(idx, "conflicting_duplicate_key") for idx in group.index]
            continue
        latest = group[order_by].max()
        winners = group[group[order_by] == latest]
        losers = group[group[order_by] != latest]
        rejected += [(idx, "superseded_by_newer_record") for idx in losers.index]
        if len(winners) > 1:
            rejected += [(idx, "conflicting_duplicate_key") for idx in winners.index]

    reject_idx = [idx for idx, _ in rejected]
    rej_rows = df.loc[reject_idx]
    rejects = pd.DataFrame(
        {
            "source": source,
            "row_ref": rej_rows["_row_ref"].tolist(),
            "record_key": rej_rows[key].astype(str).tolist(),
            "reason": [reason for _, reason in rejected],
            "raw": [raw_json(r) for r in rej_rows[content_cols].to_dict("records")],
        },
        columns=REJECT_COLUMNS,
    )
    return df.drop(index=reject_idx), rejects
