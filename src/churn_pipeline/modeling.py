"""Phase 3: model development (train + validation only; the test split is never loaded here).

Data protocol (see docs/PHASE3_DESIGN_REVIEW_ZH-TW.md, decision 2.1 option B)
---------------------------------------------------------------------------
* Input is the raw ``customer_master`` plus Phase 2's ``split_assignments``, not
  the already-transformed ``train.parquet``. Every model is
  ``Pipeline([("pre", build_preprocessor()), ("clf", ...)])``, so inside
  cross-validation the scaler/encoder are re-fitted on each fold's training
  rows only (no fold-level preprocessing leakage).
* train      -> hyperparameter search per model family (RandomizedSearchCV,
                StratifiedKFold; identical folds for every candidate).
* validation -> cross-family selection, threshold, calibration check,
                explanations, subgroup metrics.
* test       -> only ``evaluation.run_final_evaluation`` (``churn-pipeline evaluate``).

Selection rule (D5): highest validation PR AUC is the reference; every candidate
whose paired-bootstrap 95% CI of (reference - candidate) includes 0 is
"within noise"; among those the simplest wins (dummy < LR < RF = HGB < ensemble).
"""
from __future__ import annotations

import io
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .config import Paths
from .evaluation import _fmt  # shared "— for None" number formatting
from .evaluation import (
    SENSITIVE_ATTRIBUTES,
    bootstrap_ci,
    bootstrap_indices,
    cost_optimal_threshold,
    f1_optimal_threshold,
    feature_baseline,
    lr_coefficients,
    lr_decomposition,
    paired_bootstrap_diff,
    permutation_importance_table,
    pr_auc,
    ranking_metrics,
    reliability_table,
    roc_auc,
    score_baseline,
    subgroup_markdown,
    subgroup_metrics,
    threshold_metrics,
)
from .features import ID_COLUMN, build_preprocessor, load_split
from .tracking import (
    InputFingerprint,
    atomic_directory,
    build_manifest,
    load_verified_inputs,
    new_run_id,
    split_xy,
    write_json,
)

COST_RATIO_NOTE = (
    "The FN:FP cost ratio (default 5:1) is an illustrative business assumption, not an "
    "empirically established fact. It encodes 'missing a churner costs about 5x an "
    "unnecessary retention offer'. Change it with --fn-cost/--fp-cost."
)
SELECTION_RULE = (
    "Reference = highest validation PR AUC. A candidate is within noise of the reference when "
    "the paired-bootstrap 95% CI of (reference - candidate) PR AUC includes 0. Among the "
    "reference and all candidates within noise, choose the simplest (dummy < logistic "
    "regression < random forest = gradient boosting < ensemble); ties by higher PR AUC."
)
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "neg_log_loss": "neg_log_loss"}
ENSEMBLE = "soft_voting_lr_hgb"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    n_iter: int = 20                 # random-search trials per tuned family
    cv_folds: int = 5
    fn_cost: float = 5.0
    fp_cost: float = 1.0
    n_boot: int = 2000
    n_jobs: int = -1                 # search-level parallelism; results do not depend on it
    rf_n_estimators: int = 300
    permutation_repeats: int = 10
    latency_repeats: int = 5
    min_subgroup_n: int = 200
    min_subgroup_class: int = 30

    def __post_init__(self) -> None:
        if self.fn_cost <= 0 or self.fp_cost <= 0:
            raise ValueError("fn_cost and fp_cost must be positive")
        if self.n_iter < 1 or self.cv_folds < 2 or self.n_boot < 100:
            raise ValueError("need n_iter >= 1, cv_folds >= 2, n_boot >= 100")


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    simplicity: int
    interpretability: str
    make_estimator: Callable[[TrainConfig], object]
    search_space: dict = field(default_factory=dict)  # keys without the "clf__" prefix


def candidate_specs() -> list[CandidateSpec]:
    cw = [None, "balanced"]
    return [
        CandidateSpec("dummy_prior", 0, "trivial: predicts the training churn rate",
                      lambda c: DummyClassifier(strategy="prior")),
        CandidateSpec("logistic_regression", 1,
                      "high: global coefficients, exact per-customer additive decomposition",
                      lambda c: LogisticRegression(max_iter=2000),
                      {"C": loguniform(1e-3, 1e2), "class_weight": cw}),
        CandidateSpec("random_forest", 2, "low: post-hoc only (permutation importance)",
                      lambda c: RandomForestClassifier(n_estimators=c.rf_n_estimators,
                                                       random_state=c.seed, n_jobs=1),
                      {"max_depth": [None, 4, 6, 8, 12], "min_samples_leaf": [1, 2, 5, 10, 20, 50],
                       "max_features": ["sqrt", 0.3, 0.5], "class_weight": cw}),
        CandidateSpec("hist_gradient_boosting", 2, "low: post-hoc only (permutation importance)",
                      lambda c: HistGradientBoostingClassifier(
                          max_iter=500, early_stopping=True, validation_fraction=0.1,
                          n_iter_no_change=20, random_state=c.seed),
                      {"learning_rate": loguniform(0.01, 0.3), "max_leaf_nodes": [7, 15, 31],
                       "min_samples_leaf": [10, 20, 50, 100],
                       "l2_regularization": loguniform(1e-3, 10.0), "class_weight": cw}),
    ]


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("pre", build_preprocessor()), ("clf", estimator)])


def cv_splitter(cfg: TrainConfig) -> StratifiedKFold:
    """One splitter for every candidate: same folds -> same evaluation protocol."""
    return StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class ModelingData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    groups_val: pd.DataFrame          # raw sensitive attributes for validation rows
    fingerprint: InputFingerprint | None = None


def prepare_modeling_data(data: pd.DataFrame, fingerprint: InputFingerprint | None = None) -> ModelingData:
    """Pure: master+split frame -> train/validation inputs. Test rows are dropped here."""
    _, X_tr, y_tr = split_xy(data, "train")
    _, X_va, y_va = split_xy(data, "validation")
    val_rows = data[data["split"] == "validation"]
    groups = pd.DataFrame({"gender": val_rows["gender"].astype(str).to_numpy(),
                           "senior_citizen": val_rows["senior_citizen"].astype(int).to_numpy()},
                          index=X_va.index)
    return ModelingData(X_tr, y_tr, X_va, y_va, groups, fingerprint)


def load_modeling_data(paths: Paths) -> ModelingData:
    inputs = load_verified_inputs(paths)
    return prepare_modeling_data(inputs.data, inputs.fingerprint)


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
@dataclass
class TuneResult:
    name: str
    model: object                       # fitted on all training rows
    best_params: dict
    cv: dict                            # mean/std of CV metrics for the chosen params
    cv_results: pd.DataFrame
    search_seconds: float
    refit_seconds: float


def _cv_summary(res: dict, prefix: str = "test_") -> dict:
    out = {}
    for k in SCORING:
        v = np.asarray(res[f"{prefix}{k}"])
        out[f"{k}_mean"], out[f"{k}_std"] = float(v.mean()), float(v.std())
    return out


def tune(spec: CandidateSpec, X: pd.DataFrame, y: pd.Series, cfg: TrainConfig) -> TuneResult:
    pipe = make_pipeline(spec.make_estimator(cfg))
    cv = cv_splitter(cfg)
    t0 = time.perf_counter()
    if not spec.search_space:
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True, n_jobs=1)
        search_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        model = clone(pipe).fit(X, y)
        refit_s = time.perf_counter() - t1
        summary = _cv_summary(res)
        row = {"params": "{}", **{f"mean_test_{k}": summary[f"{k}_mean"] for k in SCORING},
               "std_test_pr_auc": summary["pr_auc_std"],
               "mean_train_pr_auc": float(np.mean(res["train_pr_auc"])),
               "rank_test_pr_auc": 1, "mean_fit_time": float(np.mean(res["fit_time"]))}
        return TuneResult(spec.name, model, {}, summary, pd.DataFrame([row]), search_s, refit_s)

    search = RandomizedSearchCV(
        pipe, {f"clf__{k}": v for k, v in spec.search_space.items()}, n_iter=cfg.n_iter,
        scoring=SCORING, refit="pr_auc", cv=cv, random_state=cfg.seed, n_jobs=cfg.n_jobs,
        return_train_score=True, error_score="raise",
    )
    search.fit(X, y)
    search_s = time.perf_counter() - t0
    r = pd.DataFrame(search.cv_results_)
    b = search.best_index_
    summary = {f"{k}_{s}": float(r.loc[b, f"{s}_test_{k}"]) for k in SCORING for s in ("mean", "std")}
    table = pd.DataFrame({
        "params": [str({k.removeprefix("clf__"): v for k, v in p.items()}) for p in r["params"]],
        **{f"mean_test_{k}": r[f"mean_test_{k}"] for k in SCORING},
        "std_test_pr_auc": r["std_test_pr_auc"], "mean_train_pr_auc": r["mean_train_pr_auc"],
        "rank_test_pr_auc": r["rank_test_pr_auc"], "mean_fit_time": r["mean_fit_time"],
    })
    best = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
    return TuneResult(spec.name, search.best_estimator_, best, summary, table, search_s,
                      float(search.refit_time_))


def build_ensemble(lr: TuneResult, hgb: TuneResult, X, y, cfg: TrainConfig) -> TuneResult:
    """Soft voting of the *tuned* LR and HGB pipelines (members not re-tuned).

    Caveat: if one member uses class_weight="balanced" and the other does not,
    their probabilities are on different scales before averaging.
    """
    ens = VotingClassifier([("lr", clone(lr.model)), ("hgb", clone(hgb.model))], voting="soft")
    t0 = time.perf_counter()
    res = cross_validate(ens, X, y, cv=cv_splitter(cfg), scoring=SCORING,
                         return_train_score=True, n_jobs=1)
    search_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    model = clone(ens).fit(X, y)
    refit_s = time.perf_counter() - t1
    summary = _cv_summary(res)
    row = {"params": str({"lr": lr.best_params, "hgb": hgb.best_params}),
           **{f"mean_test_{k}": summary[f"{k}_mean"] for k in SCORING},
           "std_test_pr_auc": summary["pr_auc_std"],
           "mean_train_pr_auc": float(np.mean(res["train_pr_auc"])), "rank_test_pr_auc": 1,
           "mean_fit_time": float(np.mean(res["fit_time"]))}
    return TuneResult(ENSEMBLE, model, {"members": ["logistic_regression", "hist_gradient_boosting"]},
                      summary, pd.DataFrame([row]), search_s, refit_s)


# ---------------------------------------------------------------------------
# Cost / latency
# ---------------------------------------------------------------------------
def serving_cost(model, X: pd.DataFrame, repeats: int = 5) -> dict:
    rows = X.iloc[np.arange(1000) % len(X)]
    batch = []
    for _ in range(repeats):
        t = time.perf_counter()
        model.predict_proba(rows)
        batch.append(time.perf_counter() - t)
    single = []
    one = X.iloc[[0]]
    for _ in range(repeats * 4):
        t = time.perf_counter()
        model.predict_proba(one)
        single.append(time.perf_counter() - t)
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return {"predict_ms_per_1k_rows_median": 1000 * float(np.median(batch)),
            "predict_ms_single_row_median": 1000 * float(np.median(single)),
            "model_bytes": buf.getbuffer().nbytes}


# ---------------------------------------------------------------------------
# Selection and threshold
# ---------------------------------------------------------------------------
def select_champion(y_val, scores: dict[str, np.ndarray], simplicity: dict[str, int], *,
                    n_boot: int, seed: int) -> dict:
    y = np.asarray(y_val)
    pr = {n: pr_auc(y, p) for n, p in scores.items()}
    reference = max(scores, key=lambda n: (pr[n], -simplicity[n]))
    idx = bootstrap_indices(len(y), n_boot, seed)  # same resamples for every comparison
    comparisons = {n: paired_bootstrap_diff(y, scores[reference], scores[n], pr_auc, indices=idx)
                   for n in scores if n != reference}
    within = [reference] + [n for n, c in comparisons.items() if c["low"] is not None and c["low"] <= 0]
    champion = min(within, key=lambda n: (simplicity[n], -pr[n]))
    if champion == reference:
        reason = f"{champion} has the highest validation PR AUC and no simpler candidate is within noise"
    else:
        reason = (f"{reference} has the highest point estimate, but {champion} is within noise "
                  f"(CI of the difference includes 0) and is simpler")
    return {"metric": "validation_pr_auc", "rule": SELECTION_RULE, "reference": reference,
            "champion": champion, "within_noise_of_reference": sorted(within),
            "pr_auc": pr, "comparisons_vs_reference": comparisons, "reason": reason,
            "caveat": "Pairwise CIs without multiple-comparison correction; validation is "
                      "also reused for the threshold, so its numbers are mildly optimistic."}


def select_threshold(y_val, p_val, fn_cost: float, fp_cost: float) -> dict:
    chosen = cost_optimal_threshold(y_val, p_val, fn_cost, fp_cost)
    f1_t = f1_optimal_threshold(y_val, p_val)
    return {
        "rule": "minimise fn_cost*FN + fp_cost*FP on validation; fixed before test",
        "fn_cost": fn_cost, "fp_cost": fp_cost, "cost_ratio_note": COST_RATIO_NOTE,
        "chosen": chosen,
        "at_chosen": threshold_metrics(y_val, p_val, chosen, fn_cost, fp_cost),
        "f1_optimal": {"threshold": f1_t,
                       **threshold_metrics(y_val, p_val, f1_t, fn_cost, fp_cost)},
        "at_0_5": threshold_metrics(y_val, p_val, 0.5, fn_cost, fp_cost),
        "bayes_threshold_if_calibrated": fp_cost / (fp_cost + fn_cost),
    }


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------
@dataclass
class TrainingResult:
    cfg: TrainConfig
    tuned: dict[str, TuneResult]
    val_scores: dict[str, np.ndarray]
    selection: dict
    threshold: dict
    candidate_table: list[dict]
    validation_metrics: dict
    explanations: dict
    baseline: dict

    @property
    def champion(self) -> TuneResult:
        return self.tuned[self.selection["champion"]]


def fitted_preprocessor(model):
    if isinstance(model, VotingClassifier):
        return model.estimators_[0].named_steps["pre"]
    return model.named_steps["pre"]


def train_models(data: ModelingData, cfg: TrainConfig = TrainConfig(),
                 specs: list[CandidateSpec] | None = None) -> TrainingResult:
    specs = candidate_specs() if specs is None else specs
    X, y, Xv, yv = data.X_train, data.y_train, data.X_val, data.y_val
    tuned: dict[str, TuneResult] = {}
    simplicity, interp = {}, {}
    for spec in specs:
        tuned[spec.name] = tune(spec, X, y, cfg)
        simplicity[spec.name], interp[spec.name] = spec.simplicity, spec.interpretability
    if "logistic_regression" in tuned and "hist_gradient_boosting" in tuned:
        tuned[ENSEMBLE] = build_ensemble(tuned["logistic_regression"],
                                         tuned["hist_gradient_boosting"], X, y, cfg)
        simplicity[ENSEMBLE], interp[ENSEMBLE] = 3, "lowest: average of LR and HGB"

    scores = {n: t.model.predict_proba(Xv)[:, 1] for n, t in tuned.items()}
    selection = select_champion(yv, scores, simplicity, n_boot=cfg.n_boot, seed=cfg.seed)
    champ = selection["champion"]
    p_champ = scores[champ]
    threshold = select_threshold(yv, p_champ, cfg.fn_cost, cfg.fp_cost)

    idx = bootstrap_indices(len(yv), cfg.n_boot, cfg.seed)
    table = []
    for n, t in tuned.items():
        table.append({
            "candidate": n, "simplicity": simplicity[n], "interpretability": interp[n],
            "best_params": t.best_params,
            "cv_pr_auc_mean": t.cv["pr_auc_mean"], "cv_pr_auc_std": t.cv["pr_auc_std"],
            "cv_roc_auc_mean": t.cv["roc_auc_mean"], "cv_roc_auc_std": t.cv["roc_auc_std"],
            "val_pr_auc": bootstrap_ci(yv, scores[n], pr_auc, indices=idx),
            "val_roc_auc": roc_auc(yv, scores[n]),
            "val_log_loss": ranking_metrics(yv, scores[n])["log_loss"],
            "search_seconds": t.search_seconds, "refit_seconds": t.refit_seconds,
            **serving_cost(t.model, Xv, repeats=cfg.latency_repeats),
        })

    validation_metrics = {
        "champion": {"name": champ, "ranking": ranking_metrics(yv, p_champ),
                     "at_threshold": threshold["at_chosen"],
                     "ci95": {"pr_auc": bootstrap_ci(yv, p_champ, pr_auc, indices=idx),
                              "roc_auc": bootstrap_ci(yv, p_champ, roc_auc, indices=idx)}},
        "calibration": reliability_table(yv, p_champ),
        "subgroups": subgroup_metrics(yv, p_champ, threshold["chosen"],
                                      data.groups_val[SENSITIVE_ATTRIBUTES].reset_index(drop=True),
                                      min_n=cfg.min_subgroup_n, min_class=cfg.min_subgroup_class,
                                      n_boot=min(cfg.n_boot, 1000), seed=cfg.seed),
        "subgroup_rules": {"min_n": cfg.min_subgroup_n, "min_class": cfg.min_subgroup_class},
    }

    explanations: dict = {"champion": champ}
    if "logistic_regression" in tuned:
        lr = tuned["logistic_regression"].model
        ref = lr.named_steps["pre"].transform(X).mean()
        order = np.argsort(scores["logistic_regression"])
        picks = Xv.iloc[[order[-1], order[len(order) // 2], order[0]]]
        explanations["logistic_regression"] = {
            "is_champion": champ == "logistic_regression",
            **lr_coefficients(lr),
            "reference": "mean transformed training row",
            "examples_highest_median_lowest_score": lr_decomposition(lr, picks, ref),
        }
    explanations["permutation_importance_validation"] = permutation_importance_table(
        tuned[champ].model, Xv, yv, n_repeats=cfg.permutation_repeats, seed=cfg.seed)

    baseline = {
        "features": feature_baseline(X),
        "features_source": "train rows (raw inputs, before preprocessing)",
        "score": score_baseline(p_champ),
        "score_source": "validation scores of the champion (out-of-sample)",
        "performance": validation_metrics["champion"],
    }
    return TrainingResult(cfg, tuned, scores, selection, threshold, table, validation_metrics,
                          explanations, baseline)


def verify_phase2_consistency(model, X_train: pd.DataFrame, paths: Paths) -> dict:
    """The champion's fitted preprocessor must reproduce Phase 2's train.parquet exactly."""
    path = paths.features / "train.parquet"
    if not path.exists():
        return {"status": "not checked", "reason": f"{path} missing"}
    _, X_phase2, _ = load_split(path)
    Z = fitted_preprocessor(model).transform(X_train)
    same = (list(Z.columns) == list(X_phase2.columns)
            and np.array_equal(Z.to_numpy(), X_phase2.to_numpy()))
    if not same:
        raise RuntimeError("Phase 3 preprocessing does not reproduce train.parquet; "
                           "Phase 2 and Phase 3 would disagree about the features")
    return {"status": "identical", "rows": len(Z), "columns": Z.shape[1], "rtol": 0}


def render_training_report(run_id: str, r: TrainingResult, consistency: dict) -> str:
    s, t, vm = r.selection, r.threshold, r.validation_metrics
    lines = [f"# Training run `{run_id}`", "",
             f"- Champion: **{s['champion']}** — {s['reason']}",
             f"- Selection rule: {s['rule']}",
             f"- Phase 2 consistency of the fitted preprocessor: {consistency['status']}",
             "- The test split was not read by this run.", "",
             "## Candidates (same CV folds; validation = same 1 set of rows)", "",
             "| candidate | CV PR AUC (search-best, optimistic) | val PR AUC [95% CI] | val ROC AUC "
             "| val log loss | refit s | ms / 1k rows | size KB | interpretability |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in r.candidate_table:
        v = c["val_pr_auc"]
        lines.append(f"| {c['candidate']} | {c['cv_pr_auc_mean']:.3f} ± {c['cv_pr_auc_std']:.3f} "
                     f"| {v['estimate']:.3f} [{v['low']:.3f}, {v['high']:.3f}] | {c['val_roc_auc']:.3f} "
                     f"| {c['val_log_loss']:.3f} | {c['refit_seconds']:.2f} "
                     f"| {c['predict_ms_per_1k_rows_median']:.1f} | {c['model_bytes'] / 1024:.0f} "
                     f"| {c['interpretability']} |")
    lines += ["", f"## Paired bootstrap vs reference `{s['reference']}` (reference − candidate)", "",
              "| candidate | diff | 95% CI | within noise |", "|---|---|---|---|"]
    for n, c in s["comparisons_vs_reference"].items():
        lines.append(f"| {n} | {c['diff']:.4f} | [{c['low']:.4f}, {c['high']:.4f}] "
                     f"| {'yes' if n in s['within_noise_of_reference'] else 'no'} |")
    a, f1 = t["at_chosen"], t["f1_optimal"]
    lines += ["", "## Threshold (validation)", "", t["cost_ratio_note"], "",
              "| rule | threshold | precision | recall | flagged share | cost |", "|---|---|---|---|---|---|"]
    for name, m in (("cost-optimal (chosen)", a), ("F1-optimal", f1), ("0.5", t["at_0_5"])):
        lines.append(f"| {name} | {m['threshold']:.4f} | {m['precision'] or 0:.3f} | {m['recall'] or 0:.3f} "
                     f"| {m['predicted_positive_rate']:.3f} | {m['cost']:.0f} |")
    lines += ["", f"Bayes threshold if probabilities were perfectly calibrated: "
                  f"{t['bayes_threshold_if_calibrated']:.4f}.", "",
              f"## Calibration (validation): ECE {vm['calibration']['ece']:.3f}", "",
              "| bin | n | mean predicted | observed |", "|---|---|---|---|"]
    for b in vm["calibration"]["bins"]:
        lines.append(f"| {b['bin']} | {b['n']} | {_fmt(b['mean_predicted'])} "
                     f"| {_fmt(b['observed_rate'])} |")
    lines += ["", "## Subgroups (validation)", "", *subgroup_markdown(vm["subgroups"]), "",
              "## Permutation importance (validation, raw inputs, PR AUC drop)", "",
              "| feature | mean | std |", "|---|---|---|"]
    for p in r.explanations["permutation_importance_validation"][:10]:
        lines.append(f"| {p['feature']} | {p['importance_mean']:.4f} | {p['importance_std']:.4f} |")
    if "logistic_regression" in r.explanations:
        lines += ["", "## Logistic regression: largest coefficients (log-odds)", "",
                  "| feature | coef |", "|---|---|"]
        for c in r.explanations["logistic_regression"]["coefficients"][:10]:
            lines.append(f"| {c['feature']} | {c['coef']:+.3f} |")
    return "\n".join(lines) + "\n"


def run_training(paths: Paths, cfg: TrainConfig = TrainConfig()) -> dict:
    data = load_modeling_data(paths)
    result = train_models(data, cfg)
    champion = result.champion
    consistency = verify_phase2_consistency(champion.model, data.X_train, paths)

    run_id = new_run_id()
    final = paths.models / run_id
    with atomic_directory(final) as tmp:
        joblib.dump(champion.model, tmp / "model.joblib")
        pd.concat([t.cv_results.assign(candidate=n) for n, t in result.tuned.items()],
                  ignore_index=True).to_csv(tmp / "cv_results.csv", index=False)
        write_json(tmp / "selection.json", {**result.selection, "threshold": result.threshold,
                                            "candidates": result.candidate_table,
                                            "phase2_consistency": consistency})
        write_json(tmp / "validation_metrics.json", result.validation_metrics)
        preds = pd.DataFrame({ID_COLUMN: data.X_val.index.to_numpy(),
                              "y_true": data.y_val.to_numpy()})
        for n, p in result.val_scores.items():
            preds[f"p_{n}"] = p
        preds["p_champion"] = result.val_scores[champion.name]
        preds["y_pred_champion"] = (preds["p_champion"] >= result.threshold["chosen"]).astype(int)
        preds.to_parquet(tmp / "validation_predictions.parquet", index=False)
        write_json(tmp / "explanations.json", result.explanations)
        write_json(tmp / "baseline.json", result.baseline)
        (tmp / "report.md").write_text(render_training_report(run_id, result, consistency))
        cm = result.validation_metrics["champion"]
        manifest = build_manifest(
            run_id=run_id, kind="train", directory=tmp,
            params={**asdict(cfg), "candidates": [c["candidate"] for c in result.candidate_table],
                    "search_spaces": {s.name: {k: str(v) for k, v in s.search_space.items()}
                                      for s in candidate_specs()}},
            metrics={"val_pr_auc": cm["ranking"]["pr_auc"], "val_roc_auc": cm["ranking"]["roc_auc"],
                     "val_brier": cm["ranking"]["brier"], "threshold": result.threshold["chosen"],
                     "val_cost_per_customer": cm["at_threshold"]["cost_per_customer"]},
            tags={"phase": "3", "champion": champion.name, "test_split_read": "no"},
            fingerprint=data.fingerprint,
        )
        manifest["model_sha256"] = manifest["artifacts"]["model.joblib"]
        write_json(tmp / "manifest.json", manifest)
    return {"run_id": run_id, "run_dir": final, "champion": champion.name,
            "threshold": result.threshold["chosen"], "validation": result.validation_metrics["champion"],
            "phase2_consistency": consistency}
