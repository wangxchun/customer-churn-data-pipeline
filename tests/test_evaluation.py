"""Phase 3 metrics, thresholds, bootstrap, calibration, subgroups, PSI, explanations.

Small hand-computed examples: each expected number can be checked with pen and paper.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn_pipeline.evaluation import (
    bootstrap_ci,
    confusion_counts,
    cost_optimal_threshold,
    f1_optimal_threshold,
    feature_baseline,
    feature_psi,
    lr_decomposition,
    mcnemar_exact,
    paired_bootstrap_diff,
    pr_auc,
    psi,
    reliability_table,
    roc_auc,
    subgroup_metrics,
    threshold_metrics,
)
from churn_pipeline.modeling import make_pipeline
from churn_pipeline.tracking import split_xy
from churn_pipeline.features import SplitConfig, split_customers
from phase3_data import make_signal_master

Y = np.array([1, 1, 0, 0, 1, 0])
P = np.array([0.9, 0.4, 0.35, 0.8, 0.2, 0.1])


def test_confusion_and_threshold_metrics_match_hand_count():
    # threshold 0.35 -> predicted positive: 0.9, 0.4, 0.35, 0.8
    m = threshold_metrics(Y, P, 0.35, fn_cost=5, fp_cost=1)
    assert confusion_counts(Y, P >= 0.35) == {"tp": 2, "fp": 2, "tn": 1, "fn": 1}
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["cost"] == 5 * 1 + 1 * 2


def test_cost_optimal_threshold_hand_example():
    # Candidate thresholds and cost 5*FN + 1*FP:
    #   0.9 -> FN 2, FP 0 -> 10 | 0.8 -> FN 2, FP 1 -> 11 | 0.4 -> FN 1, FP 1 -> 6
    #   0.35 -> FN 1, FP 2 -> 7 | 0.2 -> FN 0, FP 2 -> 2  | 0.1 -> FN 0, FP 3 -> 3
    assert cost_optimal_threshold(Y, P, 5, 1) == pytest.approx(0.2)
    # Equal costs: 0.9 -> 2, 0.4 -> 2, 0.2 -> 2 tie (and 0.8 -> 3); ties go to the highest threshold.
    assert cost_optimal_threshold(Y, P, 1, 1) == pytest.approx(0.9)


def test_f1_optimal_threshold_hand_example():
    # 0.2 -> TP 3, FP 2 -> P 0.6, R 1 -> F1 0.75 (the maximum)
    assert f1_optimal_threshold(Y, P) == pytest.approx(0.2)


def test_fast_metrics_equal_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(3)
    for trial in range(20):
        y = rng.integers(0, 2, 200)
        p = rng.integers(0, 15, 200) / 15 if trial % 2 else rng.random(200)  # with and without ties
        assert pr_auc(y, p) == pytest.approx(average_precision_score(y, p), abs=1e-12)
        assert roc_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_bootstrap_ci_is_deterministic_for_a_seed_and_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    a = bootstrap_ci(y, p, pr_auc, n_boot=300, seed=7)
    b = bootstrap_ci(y, p, pr_auc, n_boot=300, seed=7)
    assert a == b
    assert a["low"] < a["estimate"] < a["high"]
    assert bootstrap_ci(y, p, pr_auc, n_boot=300, seed=8) != a


def test_paired_bootstrap_of_identical_models_is_exactly_zero():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    d = paired_bootstrap_diff(y, p, p, pr_auc, n_boot=200, seed=0)
    assert d["diff"] == d["low"] == d["high"] == 0


def test_reliability_table_hand_example():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.15, 0.95, 0.85])
    t = reliability_table(y, p, n_bins=10)
    bins = {b["bin"]: b for b in t["bins"]}
    assert bins["[0.0, 0.1)"]["n"] == 1 and bins["[0.0, 0.1)"]["observed_rate"] == 0
    assert bins["[0.9, 1.0)"]["mean_predicted"] == pytest.approx(0.95)
    # ECE = mean |observed - predicted| over the four single-row bins = (0.05+0.15+0.05+0.15)/4
    assert t["ece"] == pytest.approx(0.1)


def test_subgroup_metrics_and_disparities_hand_example():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.9, 0.1, 0.2, 0.1])
    groups = pd.DataFrame({"gender": ["Female"] * 4 + ["Male"] * 4})
    s = subgroup_metrics(y, p, 0.5, groups, min_n=5, min_class=1, n_boot=100, seed=0)
    g = s["attributes"]["gender"]
    f, m = g["groups"]["Female"], g["groups"]["Male"]
    assert f["selection_rate"] == pytest.approx(3 / 4) and m["selection_rate"] == pytest.approx(1 / 4)
    assert f["recall"] == 1.0 and m["recall"] == 0.5
    assert f["fpr"] == 0.5 and m["fpr"] == 0.0
    d = g["disparities"]
    assert d["dppl"]["estimate"] == pytest.approx(1 / 4 - 3 / 4)    # q_a - q_d, d = Female
    assert d["di"]["estimate"] == pytest.approx(3.0)                 # q_d / q_a
    assert d["recall_difference"]["estimate"] == pytest.approx(-0.5)
    # 4 rows < min_n=5: reported but flagged
    assert f["reliable"] is False and m["reliable"] is False


def test_psi_is_zero_for_identical_and_large_for_shifted_distributions():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"monthly_charges": rng.uniform(20, 120, 2000)})
    base = feature_baseline(train)
    assert feature_psi(base, train)["monthly_charges"]["psi"] == pytest.approx(0, abs=1e-12)
    shifted = train.assign(monthly_charges=train["monthly_charges"] + 40)
    assert feature_psi(base, shifted)["monthly_charges"]["psi"] > 0.25


def test_psi_handles_empty_bins_and_unseen_categories():
    assert np.isfinite(psi([0.5, 0.5, 0.0], [0.0, 0.5, 0.5]))
    train = pd.DataFrame({"contract": ["Month-to-month"] * 6 + ["One year"] * 4})
    base = feature_baseline(train)
    new = pd.DataFrame({"contract": ["Month-to-month"] * 5 + ["Three year"] * 5})
    out = feature_psi(base, new)["contract"]
    assert out["unseen_share"] == pytest.approx(0.5)
    assert out["psi"] > 0.25


def test_lr_decomposition_sums_exactly_to_the_logit():
    data = split_customers(make_signal_master(300), SplitConfig()).merge(
        make_signal_master(300), on="customer_id")
    _, X, y = split_xy(data, "train")
    pipe = make_pipeline(LogisticRegression(max_iter=2000)).fit(X, y)
    ref = pipe.named_steps["pre"].transform(X).mean()
    rows = lr_decomposition(pipe, X.iloc[:5], ref)
    logits = pipe.decision_function(X.iloc[:5])
    for r, logit in zip(rows, logits):
        total = r["base_logit"] + sum(c["contribution"] for c in r["contributions"])
        assert total == pytest.approx(logit, abs=1e-10)
        assert r["probability"] == pytest.approx(pipe.predict_proba(X.iloc[:5])[rows.index(r), 1])


def test_mcnemar_counts_only_discordant_pairs():
    a = np.array([1, 1, 1, 0, 1, 1])
    b = np.array([1, 0, 0, 0, 1, 1])
    m = mcnemar_exact(a, b)
    assert (m["only_a_correct"], m["only_b_correct"]) == (2, 0)
    assert m["p_value"] == pytest.approx(0.5)  # two-sided binomial, 2 of 2 discordant
