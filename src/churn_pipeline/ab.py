"""W6 (D10): A/B testing helpers, design-time only.

Two pure functions, no I/O:

* ``assign_variant``: deterministic, sticky traffic allocation. The same
  (customer_id, salt, weights) always gives the same variant, on any machine and Python
  process, because it uses sha256 rather than Python's randomised ``hash``.
* ``required_sample_size``: per-group sample size for a two-sided two-proportion z-test.

By decision D10 this module does NOT simulate outcomes, read labels or data files, or
touch the release registry, champion or model-selection policy. It is NOT an online A/B
test: there is no live traffic, no outcome collection and no analysis of results here.

Allocation: ``u = first 8 bytes of sha256(salt + "\\x1f" + customer_id) / 2**64`` is a
uniform number in [0, 1); as a float it can round up to exactly 1.0 for the top ~2**-53
of the range, which still falls to the last variant, where it belongs. Variants are sorted by name, weights are normalised to sum to
1, and the customer gets the first variant whose cumulative weight exceeds ``u``.
Assignment is sticky only while the salt AND the weights stay fixed: changing the
weights moves the bucket boundaries and reassigns some customers. Use a new salt per
experiment, so that assignments in different experiments are independent.

Sample size (Fleiss, without continuity correction), with p1 = p_base, p2 = p_base + mde,
p̄ = (p1 + p2) / 2:

    n = ( z_{1-α/2} · sqrt(2 p̄ (1 - p̄)) + z_{1-β} · sqrt(p1 (1 - p1) + p2 (1 - p2)) )²
        / (p2 - p1)²

rounded up, per group, for two equal-size groups. ``mde`` is an ABSOLUTE difference
in proportions (0.02 means 2 percentage points); it may be negative, e.g. a treatment
meant to lower churn.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

from scipy.stats import norm

SEPARATOR = "\x1f"          # unit separator: keeps ("ab", "c") and ("a", "bc") apart


class ABInputError(ValueError):
    """An input is missing, of the wrong type, or out of range."""


def _nonempty_str(name: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise ABInputError(f"{name} must be a non-empty string")
    if SEPARATOR in value:
        raise ABInputError(f"{name} must not contain the unit separator")
    return value


def _number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise ABInputError(f"{name} must be a finite number")
    return float(value)


def _open_unit(name: str, value) -> float:
    v = _number(name, value)
    if not 0 < v < 1:
        raise ABInputError(f"{name} must be strictly between 0 and 1")
    return v


def _bucket(customer_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}{SEPARATOR}{customer_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_variant(customer_id: str, salt: str, weights: Mapping[str, float]) -> str:
    """The variant for ``customer_id`` in the experiment identified by ``salt``."""
    _nonempty_str("customer_id", customer_id)
    _nonempty_str("salt", salt)
    if not isinstance(weights, Mapping) or len(weights) < 2:
        raise ABInputError("weights must map at least two variant names to weights")
    for name, w in weights.items():
        if not isinstance(name, str) or not name:
            raise ABInputError("variant names must be non-empty strings")
        if _number(f"weight of {name!r}", w) <= 0:
            raise ABInputError(f"weight of {name!r} must be > 0")
    names = sorted(weights)
    total = sum(float(weights[n]) for n in names)
    if not math.isfinite(total):    # e.g. two weights of 1e308: every share would be 0
        raise ABInputError("the sum of the weights must be finite")
    u = _bucket(customer_id, salt)
    cumulative = 0.0
    for name in names[:-1]:
        cumulative += float(weights[name]) / total
        if u < cumulative:
            return name
    return names[-1]            # also absorbs any floating-point shortfall at the top


def required_sample_size(p_base: float, mde: float, alpha: float = 0.05,
                         power: float = 0.8) -> int:
    """Customers needed PER GROUP to detect an absolute change ``mde`` from ``p_base``
    with a two-sided test at level ``alpha`` and the given ``power``."""
    p1 = _open_unit("p_base", p_base)
    d = _number("mde", mde)
    if d == 0:
        raise ABInputError("mde must be non-zero")
    p2 = p1 + d
    if not 0 < p2 < 1:
        raise ABInputError("p_base + mde must be strictly between 0 and 1")
    a = _open_unit("alpha", alpha)
    b = _open_unit("power", power)
    if b < 0.5:     # z_{1-β} < 0: the squared formula would silently give a wrong n
        raise ABInputError("power must be in [0.5, 1)")
    z_alpha = norm.ppf(1 - a / 2)
    z_beta = norm.ppf(b)
    p_bar = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
                 + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    d2 = d * d
    if d2 == 0 or not math.isfinite(numerator / d2):
        raise ABInputError("mde is too small: the required sample size is not finite")
    return math.ceil(numerator / d2)
