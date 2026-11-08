"""Phase 3: metrics, uncertainty, calibration, subgroup bias, explainability, drift.

Also the only code path that reads the test split: ``run_final_evaluation``.

Conventions
-----------
* ``y`` is 0/1 (1 = churn), ``p`` is the predicted probability of churn.
* A row is predicted positive when ``p >= threshold``.
* "PR AUC" is ``average_precision_score`` (step-wise average precision, not the
  trapezoidal area under the PR curve; the two differ slightly).
* Bootstrap CIs are percentile intervals over resampled rows. They describe
  sampling noise of *this* evaluation set only, not variation from retraining.
"""
from __future__ import annotations

import math
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import binomtest, rankdata
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss

from .config import Paths
from .features import CATEGORICAL_FEATURES, ID_COLUMN
from .tracking import (
    StaleArtifactError,
    atomic_directory,
    load_verified_inputs,
    read_json,
    split_xy,
    write_json,
    write_json_atomic,
)

Metric = Callable[[np.ndarray, np.ndarray], float]

# Sensitive attributes examined for subgroup metrics. Both are also model inputs.
SENSITIVE_ATTRIBUTES = ["gender", "senior_citizen"]
# Clarify-style facet "d" (the group of interest). The choice is a labeling
# convention for the sign of the disparity metrics, not a claim about who is harmed.
FACET_OF_INTEREST = {"gender": "Female", "senior_citizen": 1}


class TestSetAlreadyUsedError(RuntimeError):
    """The test split was already read for this run; re-reading needs --force."""

    __test__ = False  # not a pytest test class, despite the name


class IncomparableRunsError(ValueError):
    """Two runs were trained/evaluated on different data or splits."""


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------
def pr_auc(y, p) -> float:
    """Average precision, numerically identical to sklearn's ``average_precision_score``.

    AP = sum over distinct score thresholds of (R_k - R_{k-1}) * P_k.
    Re-implemented in numpy because the bootstrap calls it thousands of times and
    sklearn's per-call input validation dominated the runtime;
    ``test_fast_metrics_equal_sklearn`` pins the equivalence.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(ps)), ys.size - 1]  # last row of each tied-score group
    tps = np.cumsum(ys)[last]
    precision = tps / (last + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y, p) -> float:
    """ROC AUC via the Mann-Whitney rank formula (ties get average ranks), equal to sklearn's."""
    y = np.asarray(y).astype(bool)
    ranks = rankdata(np.asarray(p, dtype=float))
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _both_classes(y) -> bool:
    y = np.asarray(y)
    return y.size > 0 and 0 < y.sum() < y.size


def confusion_counts(y, y_pred) -> dict:
    y = np.asarray(y).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y == 1) & (y_pred == 1)).sum()),
        "fp": int(((y == 0) & (y_pred == 1)).sum()),
        "tn": int(((y == 0) & (y_pred == 0)).sum()),
        "fn": int(((y == 1) & (y_pred == 0)).sum()),
    }


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def threshold_metrics(y, p, threshold: float, fn_cost: float | None = None,
                      fp_cost: float | None = None) -> dict:
    c = confusion_counts(y, np.asarray(p) >= threshold)
    precision = _ratio(c["tp"], c["tp"] + c["fp"])
    recall = _ratio(c["tp"], c["tp"] + c["fn"])
    f1 = (None if not precision or not recall
          else 2 * precision * recall / (precision + recall))
    out = {
        "threshold": float(threshold), **c,
        "precision": precision, "recall": recall, "f1": f1 if f1 is not None else 0.0,
        "specificity": _ratio(c["tn"], c["tn"] + c["fp"]),
        "fpr": _ratio(c["fp"], c["fp"] + c["tn"]),
        "predicted_positive_rate": (c["tp"] + c["fp"]) / max(len(np.asarray(y)), 1),
    }
    if fn_cost is not None and fp_cost is not None:
        cost = fn_cost * c["fn"] + fp_cost * c["fp"]
        out["cost"] = float(cost)
        out["cost_per_customer"] = float(cost) / max(len(np.asarray(y)), 1)
    return out


def ranking_metrics(y, p) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    out = {"n": int(y.size), "n_positive": int(y.sum()), "base_rate": float(y.mean()),
           "brier": float(brier_score_loss(y, p)),
           "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))}
    out["roc_auc"] = roc_auc(y, p) if _both_classes(y) else None
    out["pr_auc"] = pr_auc(y, p) if _both_classes(y) else None
    return out


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
def candidate_thresholds(p) -> np.ndarray:
    """Every distinct score, plus one value above the max (= predict nobody positive)."""
    p = np.asarray(p, dtype=float)
    u = np.unique(p)
    return np.append(u, np.nextafter(u.max(), np.inf))


def threshold_curve(y, p) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    t = candidate_thresholds(p)
    pred = p[None, :] >= t[:, None]
    tp = (pred & (y == 1)).sum(axis=1)
    fp = (pred & (y == 0)).sum(axis=1)
    fn = (y == 1).sum() - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return pd.DataFrame({"threshold": t, "tp": tp, "fp": fp, "fn": fn,
                         "precision": precision, "recall": recall, "f1": f1})


def cost_optimal_threshold(y, p, fn_cost: float, fp_cost: float) -> float:
    """Minimise fn_cost*FN + fp_cost*FP. Ties go to the highest threshold
    (fewest customers flagged for the same cost)."""
    curve = threshold_curve(y, p)
    cost = fn_cost * curve["fn"] + fp_cost * curve["fp"]
    best = cost[cost == cost.min()].index
    return float(curve.loc[best, "threshold"].max())


def f1_optimal_threshold(y, p) -> float:
    curve = threshold_curve(y, p)
    best = curve.index[curve["f1"] == curve["f1"].max()]
    return float(curve.loc[best, "threshold"].max())


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _percentile_ci(values: list[float], alpha: float) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci(y, p, metric: Metric, *, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05, indices: np.ndarray | None = None) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    idx = bootstrap_indices(len(y), n_boot, seed) if indices is None else indices
    values = [metric(y[i], p[i]) for i in idx if _both_classes(y[i])]
    lo, hi = _percentile_ci(values, alpha)
    return {"estimate": metric(y, p) if _both_classes(y) else None, "low": lo, "high": hi,
            "n_boot_used": len(values), "alpha": alpha}


def paired_bootstrap_diff(y, p_a, p_b, metric: Metric, *, n_boot: int = 2000, seed: int = 0,
                          alpha: float = 0.05, indices: np.ndarray | None = None) -> dict:
    """CI of metric(a) - metric(b), resampling the *same* rows for both models.

    Pairing removes the shared row-level noise, so it is much tighter than
    comparing two independent CIs.
    """
    y = np.asarray(y)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    idx = bootstrap_indices(len(y), n_boot, seed) if indices is None else indices
    diffs = [metric(y[i], p_a[i]) - metric(y[i], p_b[i]) for i in idx if _both_classes(y[i])]
    lo, hi = _percentile_ci(diffs, alpha)
    return {"diff": metric(y, p_a) - metric(y, p_b), "low": lo, "high": hi,
            "share_a_better": float(np.mean(np.asarray(diffs) > 0)) if diffs else None,
            "n_boot_used": len(diffs), "alpha": alpha}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def reliability_table(y, p, n_bins: int = 10) -> dict:
    """Equal-width probability bins: mean prediction vs observed churn rate."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            rows.append({"bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f})", "n": 0,
                         "mean_predicted": None, "observed_rate": None})
            continue
        mean_pred, obs = float(p[m].mean()), float(y[m].mean())
        ece += m.sum() / len(y) * abs(obs - mean_pred)
        rows.append({"bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f})", "n": int(m.sum()),
                     "mean_predicted": mean_pred, "observed_rate": obs})
    return {"bins": rows, "ece": float(ece), "mean_predicted": float(p.mean()),
            "observed_rate": float(y.mean())}


# ---------------------------------------------------------------------------
# Subgroup (bias) metrics
# ---------------------------------------------------------------------------
def _group_stats(y, p, yhat) -> dict:
    c = confusion_counts(y, yhat)
    n = len(y)
    return {
        "n": n, "n_positive": int(np.sum(y)), "n_negative": int(n - np.sum(y)),
        "base_rate": _ratio(float(np.sum(y)), n),
        "selection_rate": _ratio(c["tp"] + c["fp"], n),
        "recall": _ratio(c["tp"], c["tp"] + c["fn"]),
        "fpr": _ratio(c["fp"], c["fp"] + c["tn"]),
        "precision": _ratio(c["tp"], c["tp"] + c["fp"]),
        "pr_auc": pr_auc(y, p) if _both_classes(y) else None,
        "mean_score": float(np.mean(p)) if n else None,
    }


def subgroup_metrics(y, p, threshold: float, groups: pd.DataFrame, *, min_n: int = 200,
                     min_class: int = 30, n_boot: int = 1000, seed: int = 0) -> dict:
    """Per-group metrics plus Clarify-style post-training disparities.

    DPPL (difference in positive proportions in predicted labels) = q_a - q_d,
    DI (disparate impact) = q_d / q_a, recall difference = recall_a - recall_d,
    where d = FACET_OF_INTEREST[attr] and a = everyone else.
    A group is ``reliable`` only with >= min_n rows and >= min_class of each label;
    otherwise its numbers are reported but flagged.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= threshold).astype(int)
    idx = bootstrap_indices(len(y), n_boot, seed)
    out: dict = {"threshold": threshold, "min_n": min_n, "min_class": min_class, "attributes": {}}
    for attr in groups.columns:
        values = groups[attr].to_numpy()
        per_group = {}
        for v in sorted(pd.unique(values), key=str):
            m = values == v
            s = _group_stats(y[m], p[m], yhat[m])
            s["reliable"] = (s["n"] >= min_n and s["n_positive"] >= min_class
                             and s["n_negative"] >= min_class)
            per_group[str(v)] = s
        entry = {"groups": per_group}
        d_value = FACET_OF_INTEREST.get(attr)
        if d_value is not None and (values == d_value).any() and (values != d_value).any():
            d = values == d_value

            def disparities(i):
                yy, hh, dd = y[i], yhat[i], d[i]
                qa, qd = hh[~dd].mean() if (~dd).any() else np.nan, hh[dd].mean() if dd.any() else np.nan
                ra = hh[~dd & (yy == 1)].mean() if (~dd & (yy == 1)).any() else np.nan
                rd = hh[dd & (yy == 1)].mean() if (dd & (yy == 1)).any() else np.nan
                return qa - qd, (qd / qa if qa else np.nan), ra - rd

            point = disparities(np.arange(len(y)))
            boots = np.array([disparities(i) for i in idx])
            ci = {}
            for k, name in enumerate(("dppl", "di", "recall_difference")):
                col = boots[:, k][~np.isnan(boots[:, k])]
                lo, hi = _percentile_ci(list(col), 0.05)
                ci[name] = {"estimate": None if np.isnan(point[k]) else float(point[k]),
                            "low": lo, "high": hi}
            entry["facet_d"] = str(d_value)
            entry["disparities"] = ci
        out["attributes"][attr] = entry
    return out


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def lr_coefficients(pipeline) -> dict:
    """Global LR explanation: coefficient per *transformed* feature (log-odds per unit).

    Scaled features are per standard deviation of train; one-hot features are
    relative to the dropped/absent category. Identical columns (Phase 2 Part 0 #4)
    share weight, so individual coefficients of duplicated columns are not
    separately meaningful.
    """
    clf = pipeline.named_steps["clf"]
    names = list(pipeline.named_steps["pre"].get_feature_names_out())
    coef = clf.coef_.ravel()
    order = np.argsort(-np.abs(coef))
    return {"intercept": float(clf.intercept_[0]),
            "coefficients": [{"feature": names[i], "coef": float(coef[i])} for i in order]}


def lr_decomposition(pipeline, X_rows: pd.DataFrame, reference: pd.Series) -> list[dict]:
    """Exact additive decomposition of the LR logit for each row.

    logit(x) = intercept + sum_j coef_j * x_j
             = [intercept + sum_j coef_j * ref_j] + sum_j coef_j * (x_j - ref_j)
    with ref = mean transformed training row. The contributions sum to the
    logit exactly (up to float rounding); this is a property of linear models,
    not an approximation like SHAP for non-linear models.
    """
    pre, clf = pipeline.named_steps["pre"], pipeline.named_steps["clf"]
    Z = pre.transform(X_rows)
    coef = pd.Series(clf.coef_.ravel(), index=Z.columns)
    ref = reference[Z.columns]
    base = float(clf.intercept_[0] + (coef * ref).sum())
    logits = clf.decision_function(Z)
    out = []
    for (cid, z), logit in zip(Z.iterrows(), logits):
        contrib = coef * (z - ref)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
        out.append({
            "customer_id": str(cid), "base_logit": base, "logit": float(logit),
            "probability": float(1 / (1 + math.exp(-logit))),
            "sum_check": float(base + contrib.sum() - logit),
            "contributions": [{"feature": k, "contribution": float(v)} for k, v in top.items()],
        })
    return out


def permutation_importance_table(model, X: pd.DataFrame, y, *, n_repeats: int, seed: int) -> list[dict]:
    """Drop in validation PR AUC when one *raw input column* is shuffled.

    Measured on validation (not train), so it reflects generalisation, not memorisation.
    Correlated inputs (total_charges ~ tenure x monthly, r=0.9996) share
    importance: shuffling one leaves the information in the other.
    """
    r = permutation_importance(model, X, y, scoring="average_precision", n_repeats=n_repeats,
                               random_state=seed, n_jobs=1)
    order = np.argsort(-r.importances_mean)
    return [{"feature": X.columns[i], "importance_mean": float(r.importances_mean[i]),
             "importance_std": float(r.importances_std[i])} for i in order]


# ---------------------------------------------------------------------------
# Distribution baseline and PSI
# ---------------------------------------------------------------------------
PSI_RULE_OF_THUMB = ("PSI < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift. "
                     "A convention, not a statistical test; it ignores sample size.")
_MISSING, _UNSEEN = "__missing__", "__unseen__"


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    """Population Stability Index. Empty bins are floored at ``eps`` so the log is finite."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_bins(x: np.ndarray, edges: list[float]) -> np.ndarray:
    labels = np.searchsorted(np.asarray(edges), x, side="right").astype(object)
    labels[np.isnan(x)] = _MISSING
    return labels


def feature_baseline(X: pd.DataFrame, n_bins: int = 10) -> dict:
    """Per raw feature: bin definition and training proportions (the drift baseline)."""
    out = {}
    for c in X.columns:
        if c in CATEGORICAL_FEATURES:
            props = X[c].astype(str).value_counts(normalize=True).sort_index()
            out[c] = {"kind": "categorical", "categories": list(props.index),
                      "proportions": props.tolist() + [0.0]}  # last = unseen
        else:
            x = X[c].to_numpy(dtype=float)
            q = np.nanquantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
            edges = sorted(set(float(v) for v in q))
            labels = _numeric_bins(x, edges)
            keys = list(range(len(edges) + 1)) + [_MISSING]
            props = [float(np.mean(labels == k)) for k in keys]
            out[c] = {"kind": "numeric", "edges": edges, "proportions": props}
    return out


def _actual_proportions(spec: dict, values: pd.Series) -> list[float]:
    if spec["kind"] == "categorical":
        s = values.astype(str)
        props = [float((s == cat).mean()) for cat in spec["categories"]]
        return props + [float((~s.isin(spec["categories"])).mean())]
    labels = _numeric_bins(values.to_numpy(dtype=float), spec["edges"])
    keys = list(range(len(spec["edges"]) + 1)) + [_MISSING]
    return [float(np.mean(labels == k)) for k in keys]


def feature_psi(baseline: dict, X: pd.DataFrame) -> dict:
    out = {}
    for c, spec in baseline.items():
        actual = _actual_proportions(spec, X[c])
        out[c] = {"psi": psi(spec["proportions"], actual),
                  "unseen_share": actual[-1] if spec["kind"] == "categorical" else None}
    return out


def score_baseline(p, n_bins: int = 10) -> dict:
    edges = list(np.linspace(0, 1, n_bins + 1)[1:-1])
    labels = np.searchsorted(edges, np.asarray(p, dtype=float), side="right")
    return {"edges": [float(e) for e in edges],
            "proportions": [float(np.mean(labels == k)) for k in range(n_bins)]}


def score_psi(baseline: dict, p) -> float:
    labels = np.searchsorted(baseline["edges"], np.asarray(p, dtype=float), side="right")
    actual = [float(np.mean(labels == k)) for k in range(len(baseline["proportions"]))]
    return psi(baseline["proportions"], actual)


# ---------------------------------------------------------------------------
# Final (test) evaluation: the only code that reads the test split
# ---------------------------------------------------------------------------
def _check_same_data(manifest: dict, fingerprint) -> None:
    rec = manifest["data"]
    if rec["split_content_sha256"] != fingerprint.split_content_sha256:
        raise StaleArtifactError("the split changed since this run was trained (TD-3); "
                                 "its threshold and model are not valid for the current test set")
    if rec["customer_master_sha256"] != fingerprint.customer_master_sha256:
        raise StaleArtifactError("customer_master changed since this run was trained")


def run_final_evaluation(paths: Paths, run_id: str, *, force: bool = False,
                         n_boot: int = 2000, seed: int = 0) -> dict:
    run_dir = paths.models / run_id
    if not (run_dir / "manifest.json").exists():
        raise FileNotFoundError(f"no completed run at {run_dir}")
    manifest = read_json(run_dir / "manifest.json")
    trained_with = manifest["environment"]["packages"]["scikit-learn"]
    if trained_with != sklearn.__version__:
        raise RuntimeError(f"model was pickled with scikit-learn {trained_with}; "
                           f"installed is {sklearn.__version__}. Refusing to unpickle.")
    inputs = load_verified_inputs(paths)
    _check_same_data(manifest, inputs.fingerprint)

    access_path = run_dir / "test_access.json"
    access = read_json(access_path) if access_path.exists() else {"reads": []}
    if access["reads"] and not force:
        raise TestSetAlreadyUsedError(
            f"test split already read {len(access['reads'])} time(s) for run {run_id}. "
            "Repeated test evaluation turns the test set into a validation set. "
            "Use --force only if you accept that; every read is logged."
        )
    attempt = len(access["reads"]) + 1
    # Record the read *before* touching test rows, so a crash still counts as a read.
    access["reads"].append({"attempt": attempt, "force": force,
                            "at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")})
    write_json_atomic(access_path, access)

    ids, X_test, y_test = split_xy(inputs.data, "test")
    model = joblib.load(run_dir / "model.joblib")  # pickle: self-produced files only
    p = model.predict_proba(X_test)[:, 1]
    selection = read_json(run_dir / "selection.json")
    thr = selection["threshold"]["chosen"]
    fn_cost, fp_cost = selection["threshold"]["fn_cost"], selection["threshold"]["fp_cost"]
    y = y_test.to_numpy()
    idx = bootstrap_indices(len(y), n_boot, seed)

    def at_thr(metric_name):
        return lambda yy, pp: threshold_metrics(yy, pp, thr)[metric_name] or 0.0

    metrics = {
        "run_id": run_id, "attempt": attempt, "champion": selection["champion"],
        "threshold": thr, "fn_cost": fn_cost, "fp_cost": fp_cost,
        "ranking": ranking_metrics(y, p),
        "at_threshold": threshold_metrics(y, p, thr, fn_cost, fp_cost),
        "ci95": {
            "pr_auc": bootstrap_ci(y, p, pr_auc, indices=idx),
            "roc_auc": bootstrap_ci(y, p, roc_auc, indices=idx),
            "recall": bootstrap_ci(y, p, at_thr("recall"), indices=idx),
            "precision": bootstrap_ci(y, p, at_thr("precision"), indices=idx),
        },
        "naive_baseline_pr_auc": float(y.mean()),
        "validation_reference": read_json(run_dir / "validation_metrics.json")["champion"],
        "calibration": reliability_table(y, p),
    }
    raw_groups = inputs.data[inputs.data["split"] == "test"][SENSITIVE_ATTRIBUTES]
    raw_groups = raw_groups.assign(senior_citizen=raw_groups["senior_citizen"].astype(int))
    vm = read_json(run_dir / "validation_metrics.json")["subgroup_rules"]
    subgroups = subgroup_metrics(y, p, thr, raw_groups.reset_index(drop=True),
                                 min_n=vm["min_n"], min_class=vm["min_class"], n_boot=n_boot,
                                 seed=seed)
    baseline = read_json(run_dir / "baseline.json")
    drift = {"rule_of_thumb": PSI_RULE_OF_THUMB,
             "feature_psi_train_to_test": feature_psi(baseline["features"], X_test),
             "score_psi_validation_to_test": score_psi(baseline["score"], p)}

    out_dir = run_dir / f"evaluation-{attempt:02d}"
    with atomic_directory(out_dir) as tmp:
        write_json(tmp / "test_metrics.json", metrics)
        write_json(tmp / "test_subgroups.json", subgroups)
        write_json(tmp / "test_drift.json", drift)
        pd.DataFrame({ID_COLUMN: ids.to_numpy(), "y_true": y, "p": p,
                      "y_pred": (p >= thr).astype(int)}).to_parquet(
            tmp / "test_predictions.parquet", index=False)
        (tmp / "test_report.md").write_text(render_test_report(metrics, subgroups, drift))
    access["reads"][-1]["output_dir"] = out_dir.name
    write_json_atomic(access_path, access)
    return {"metrics": metrics, "subgroups": subgroups, "drift": drift, "output_dir": out_dir}


def _fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def _ci(c, nd=3):
    return f"{_fmt(c['estimate'], nd)} [{_fmt(c['low'], nd)}, {_fmt(c['high'], nd)}]"


def subgroup_markdown(subgroups: dict) -> list[str]:
    lines = ["| attribute | group | n | positives | selection rate | recall | FPR | PR AUC | reliable |",
             "|---|---|---|---|---|---|---|---|---|"]
    for attr, e in subgroups["attributes"].items():
        for g, s in e["groups"].items():
            lines.append(f"| {attr} | {g} | {s['n']} | {s['n_positive']} | {_fmt(s['selection_rate'])} "
                         f"| {_fmt(s['recall'])} | {_fmt(s['fpr'])} | {_fmt(s['pr_auc'])} "
                         f"| {'yes' if s['reliable'] else '**no**'} |")
    lines += ["", "| attribute | facet d | DPPL (q_a − q_d) | DI (q_d / q_a) | recall diff (a − d) |",
              "|---|---|---|---|---|"]
    for attr, e in subgroups["attributes"].items():
        if "disparities" in e:
            d = e["disparities"]
            lines.append(f"| {attr} | {e['facet_d']} | {_ci(d['dppl'])} | {_ci(d['di'])} "
                         f"| {_ci(d['recall_difference'])} |")
    return lines


def render_test_report(metrics: dict, subgroups: dict, drift: dict) -> str:
    r, t, ci = metrics["ranking"], metrics["at_threshold"], metrics["ci95"]
    v = metrics["validation_reference"]
    lines = [
        f"# Test evaluation: run `{metrics['run_id']}` (attempt {metrics['attempt']})", "",
        f"- Champion: `{metrics['champion']}`; threshold {metrics['threshold']:.4f} "
        f"(fixed on validation; FN:FP cost = {metrics['fn_cost']:g}:{metrics['fp_cost']:g}, "
        "an illustrative assumption)",
        f"- Test rows: {r['n']} ({r['n_positive']} churners, base rate {r['base_rate']:.3f} = "
        "naive PR AUC)", "",
        "| metric | test (95% bootstrap CI) | validation |", "|---|---|---|",
        f"| PR AUC | {_ci(ci['pr_auc'])} | {_fmt(v['ranking']['pr_auc'])} |",
        f"| ROC AUC | {_ci(ci['roc_auc'])} | {_fmt(v['ranking']['roc_auc'])} |",
        f"| recall @thr | {_ci(ci['recall'])} | {_fmt(v['at_threshold']['recall'])} |",
        f"| precision @thr | {_ci(ci['precision'])} | {_fmt(v['at_threshold']['precision'])} |",
        f"| Brier | {_fmt(r['brier'])} | {_fmt(v['ranking']['brier'])} |",
        f"| log loss | {_fmt(r['log_loss'])} | {_fmt(v['ranking']['log_loss'])} |",
        f"| cost / customer | {_fmt(t['cost_per_customer'])} | {_fmt(v['at_threshold']['cost_per_customer'])} |",
        "", f"Confusion @thr: TP {t['tp']}, FP {t['fp']}, TN {t['tn']}, FN {t['fn']}.",
        f"Calibration: ECE {metrics['calibration']['ece']:.3f}.", "",
        "## Subgroups (test)", "", *subgroup_markdown(subgroups), "",
        "## Drift", "", drift["rule_of_thumb"], "",
        f"Score PSI validation→test: {drift['score_psi_validation_to_test']:.4f}", "",
        "| feature | PSI train→test |", "|---|---|",
    ]
    lines += [f"| {c} | {d['psi']:.4f} |" for c, d in drift["feature_psi_train_to_test"].items()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Offline champion / challenger comparison (validation only)
# ---------------------------------------------------------------------------
def mcnemar_exact(correct_a, correct_b) -> dict:
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p}


def compare_runs(paths: Paths, champion_id: str, challenger_id: str, *, n_boot: int = 2000,
                 seed: int = 0) -> dict:
    runs = {}
    for role, rid in (("champion", champion_id), ("challenger", challenger_id)):
        d = paths.models / rid
        if not (d / "manifest.json").exists():
            raise FileNotFoundError(f"no completed run at {d}")
        runs[role] = {"dir": d, "manifest": read_json(d / "manifest.json"),
                      "selection": read_json(d / "selection.json")}
    dc, dh = runs["champion"]["manifest"]["data"], runs["challenger"]["manifest"]["data"]
    for key in ("split_content_sha256", "customer_master_sha256"):
        if dc[key] != dh[key]:
            raise IncomparableRunsError(
                f"runs differ in {key} ({dc[key][:12]} vs {dh[key][:12]}): their validation "
                "sets are not the same rows, so a paired comparison is invalid (TD-3)")

    preds = []
    for role, r in runs.items():
        df = pd.read_parquet(r["dir"] / "validation_predictions.parquet")
        preds.append(df[[ID_COLUMN, "y_true", "p_champion", "y_pred_champion"]].rename(
            columns={"p_champion": f"p_{role}", "y_pred_champion": f"y_pred_{role}",
                     "y_true": f"y_{role}"}))
    m = preds[0].merge(preds[1], on=ID_COLUMN, how="inner", validate="one_to_one")
    if len(m) != len(preds[0]) or (m["y_champion"] != m["y_challenger"]).any():
        raise IncomparableRunsError("validation rows or labels differ between runs")
    y = m["y_champion"].to_numpy()
    pc, ph = m["p_champion"].to_numpy(), m["p_challenger"].to_numpy()
    yc, yh = m["y_pred_champion"].to_numpy(), m["y_pred_challenger"].to_numpy()
    idx = bootstrap_indices(len(y), n_boot, seed)
    diff_pr = paired_bootstrap_diff(y, ph, pc, pr_auc, indices=idx)
    tc, th = runs["champion"]["selection"]["threshold"], runs["challenger"]["selection"]["threshold"]
    result = {
        "champion": {"run_id": champion_id, "model": runs["champion"]["selection"]["champion"],
                     "threshold": tc["chosen"]},
        "challenger": {"run_id": challenger_id, "model": runs["challenger"]["selection"]["champion"],
                       "threshold": th["chosen"]},
        "n_validation_rows": int(len(y)),
        "pr_auc": {"champion": pr_auc(y, pc), "challenger": pr_auc(y, ph)},
        "pr_auc_diff_challenger_minus_champion": diff_pr,
        "roc_auc_diff_challenger_minus_champion": paired_bootstrap_diff(y, ph, pc, roc_auc, indices=idx),
        "prediction_agreement": float((yc == yh).mean()),
        "score_correlation": float(np.corrcoef(pc, ph)[0, 1]) if pc.std() and ph.std() else None,
        "mcnemar_on_correctness": mcnemar_exact(yc == y, yh == y),
        "cost": {"champion": threshold_metrics(y, pc, tc["chosen"], tc["fn_cost"], tc["fp_cost"])["cost"],
                 "challenger": threshold_metrics(y, ph, th["chosen"], th["fn_cost"], th["fp_cost"])["cost"]},
        "decision_rule": "promote the challenger only if the 95% CI of its PR AUC gain is entirely > 0",
        "limitations": [
            "Offline replay on the same validation rows, not shadow traffic: no live "
            "latency, no traffic mirroring, no feedback loop.",
            "Both runs already used this validation set for their own selection, so it is "
            "not an untouched holdout for either.",
        ],
    }
    result["decision"] = ("promote challenger" if diff_pr["low"] is not None and diff_pr["low"] > 0
                          else "keep champion")
    out = paths.models / "comparisons"
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / f"{champion_id}__vs__{challenger_id}.json", result)
    return result

