"""W6 (D10): A/B helpers. Synthetic IDs only (``NNNN-SYNAB``); no data file, label,
release or model is involved, and no outcome is simulated."""
from __future__ import annotations

import builtins
import hashlib
import io
import math
import subprocess
import sys
from collections import Counter

import pytest

from churn_pipeline import ab

IDS = [f"{i:05d}-SYNAB" for i in range(100_000)]
HALF = {"control": 0.5, "treatment": 0.5}


# --- assign_variant ---------------------------------------------------------------------
def test_same_customer_always_gets_the_same_variant():
    first = [ab.assign_variant(i, "exp-1", HALF) for i in IDS[:2000]]
    assert first == [ab.assign_variant(i, "exp-1", HALF) for i in IDS[:2000]]


def test_assignment_matches_an_independent_sha256_computation():
    # Recomputed here with hashlib, not by calling the module, so the test is not circular.
    for cid in IDS[:500]:
        h = hashlib.sha256(f"exp-gold\x1f{cid}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2**64
        assert ab.assign_variant(cid, "exp-gold", HALF) == ("control" if u < 0.5 else "treatment")


def test_golden_values_are_stable():
    got = [ab.assign_variant(f"{i:04d}-SYNAB", "exp-golden", HALF) for i in range(8)]
    assert got == ["control", "treatment", "treatment", "control",
                   "treatment", "control", "treatment", "control"]


def test_assignment_does_not_depend_on_python_hash_seed():
    code = ("from churn_pipeline import ab; print(''.join(ab.assign_variant("
            "f'{i:04d}-SYNAB', 'exp-seed', {'a': 1, 'b': 1, 'c': 2})[0] for i in range(200)))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           check=True, env={"PYTHONHASHSEED": seed, "PATH": ""}).stdout
            for seed in ("0", "1", "12345")}
    assert len(outs) == 1


def test_weight_dict_order_does_not_matter():
    w1 = {"a": 0.2, "b": 0.3, "c": 0.5}
    w2 = {"c": 0.5, "a": 0.2, "b": 0.3}
    assert all(ab.assign_variant(i, "exp-o", w1) == ab.assign_variant(i, "exp-o", w2)
               for i in IDS[:5000])


def test_weights_are_normalised():
    assert all(ab.assign_variant(i, "exp-n", {"a": 1, "b": 3})
               == ab.assign_variant(i, "exp-n", {"a": 0.25, "b": 0.75}) for i in IDS[:5000])


@pytest.mark.parametrize("weights", [HALF, {"control": 0.9, "treatment": 0.1},
                                     {"a": 1, "b": 1, "c": 2}])
def test_large_sample_proportions_are_close_to_the_weights(weights):
    counts = Counter(ab.assign_variant(i, "exp-prop", weights) for i in IDS)
    n, total = len(IDS), sum(weights.values())
    assert set(counts) == set(weights)
    for name, w in weights.items():
        p = w / total
        sd = math.sqrt(p * (1 - p) / n)
        assert abs(counts[name] / n - p) < 4 * sd          # 4 standard errors


def test_different_salts_give_independent_assignments():
    a = [ab.assign_variant(i, "exp-A", HALF) for i in IDS[:20_000]]
    b = [ab.assign_variant(i, "exp-B", HALF) for i in IDS[:20_000]]
    agree = sum(x == y for x, y in zip(a, b)) / len(a)
    assert abs(agree - 0.5) < 4 * math.sqrt(0.25 / len(a))


def test_changing_weights_reassigns_some_customers():
    # Documented limitation: stickiness holds only for a fixed salt AND fixed weights.
    old = [ab.assign_variant(i, "exp-w", {"a": 0.5, "b": 0.5}) for i in IDS[:5000]]
    new = [ab.assign_variant(i, "exp-w", {"a": 0.4, "b": 0.6}) for i in IDS[:5000]]
    moved = sum(x != y for x, y in zip(old, new))
    assert 0 < moved and all(x == "a" and y == "b" for x, y in zip(old, new) if x != y)


@pytest.mark.parametrize("kwargs,match", [
    (dict(customer_id=""), "customer_id"),
    (dict(customer_id=123), "customer_id"),
    (dict(customer_id=None), "customer_id"),
    (dict(customer_id="a\x1fb"), "customer_id"),
    (dict(salt=""), "salt"),
    (dict(salt=b"exp"), "salt"),
    (dict(weights={"only": 1}), "at least two"),
    (dict(weights=[("a", 1), ("b", 1)]), "at least two"),
    (dict(weights={"a": 0, "b": 1}), "> 0"),
    (dict(weights={"a": -1, "b": 1}), "> 0"),
    (dict(weights={"a": math.nan, "b": 1}), "finite"),
    (dict(weights={"a": math.inf, "b": 1}), "finite"),
    (dict(weights={"a": True, "b": 1}), "finite"),
    (dict(weights={"a": "1", "b": 1}), "finite"),
    (dict(weights={"": 1, "b": 1}), "variant names"),
    (dict(weights={1: 1, "b": 1}), "variant names"),
    (dict(weights={}), "at least two"),
    (dict(weights={"a": 1e308, "b": 1e308}), "sum of the weights"),   # finite each, inf sum
])
def test_invalid_assignment_inputs_are_refused(kwargs, match):
    base = dict(customer_id="0001-SYNAB", salt="exp-1", weights=HALF)
    with pytest.raises(ab.ABInputError, match=match):
        ab.assign_variant(**{**base, **kwargs})


# --- required_sample_size ---------------------------------------------------------------
def test_sample_size_textbook_hand_example():
    # p1 = 0.20, p2 = 0.30, alpha 0.05 two-sided, power 0.80:
    # p̄ = 0.25; z = 1.959964, 0.841621
    # (1.959964·sqrt(0.375) + 0.841621·sqrt(0.16 + 0.21))² / 0.1²
    # = (1.200237 + 0.511938)² / 0.01 = 293.15 → 294 per group
    z_a, z_b = 1.959963984540054, 0.8416212335729143
    by_hand = (z_a * math.sqrt(0.375) + z_b * math.sqrt(0.37)) ** 2 / 0.01
    assert 293 < by_hand < 294
    assert ab.required_sample_size(0.20, 0.10) == 294
    assert ab.required_sample_size(0.20, 0.10, alpha=0.05, power=0.8) == 294


def test_sample_size_is_symmetric_in_direction():
    # Detecting 0.30 → 0.20 needs the same n as 0.20 → 0.30.
    assert ab.required_sample_size(0.30, -0.10) == ab.required_sample_size(0.20, 0.10)


def test_sample_size_moves_in_the_textbook_directions():
    base = ab.required_sample_size(0.26, -0.03)
    assert ab.required_sample_size(0.26, -0.02) > base                 # smaller effect
    assert ab.required_sample_size(0.26, -0.03, power=0.9) > base      # more power
    assert ab.required_sample_size(0.26, -0.03, alpha=0.01) > base     # stricter alpha
    assert ab.required_sample_size(0.26, -0.05) < base                 # bigger effect


def test_sample_size_roughly_quadruples_when_the_effect_halves():
    small, big = ab.required_sample_size(0.5, 0.02), ab.required_sample_size(0.5, 0.04)
    assert 3.8 < small / big < 4.2


@pytest.mark.parametrize("kwargs,match", [
    (dict(p_base=0), "p_base"), (dict(p_base=1), "p_base"), (dict(p_base=-0.1), "p_base"),
    (dict(p_base=math.nan), "p_base"), (dict(p_base=True), "p_base"),
    (dict(p_base="0.2"), "p_base"),
    (dict(mde=0), "non-zero"), (dict(mde=0.85), "p_base \\+ mde"),
    (dict(mde=-0.2), "p_base \\+ mde"), (dict(mde=math.inf), "mde"),
    (dict(alpha=0), "alpha"), (dict(alpha=1), "alpha"), (dict(alpha=None), "alpha"),
    (dict(power=1), "power"), (dict(power=0.3), "power"), (dict(power=False), "power"),
    (dict(mde=1e-170), "too small"), (dict(mde=-1e-200), "too small"),  # n overflows
    (dict(p_base=0.9, mde=0.1), "p_base \\+ mde"),                     # p2 == 1 exactly
])
def test_invalid_sample_size_inputs_are_refused(kwargs, match):
    base = dict(p_base=0.2, mde=0.1, alpha=0.05, power=0.8)
    with pytest.raises(ab.ABInputError, match=match):
        ab.required_sample_size(**{**base, **kwargs})


# --- safety -----------------------------------------------------------------------------
def test_functions_open_no_files(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("ab must not open files")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(io, "open", forbidden)
    ab.assign_variant("0001-SYNAB", "exp-1", HALF)
    ab.required_sample_size(0.2, 0.1)


def test_module_does_not_reach_the_registry_models_or_data():
    source = open(ab.__file__).read()
    for name in ("release", "inference", "evaluation", "monitoring", "joblib",
                 "read_parquet", "predict"):
        assert name not in source.split('"""', 2)[2], name   # code after the docstring
