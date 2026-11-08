"""Phase 3 learning diagnostics (train + validation only; never influence model selection).

Each experiment answers one question that a single "best score" hides:

* learning curves / regularization sweeps -> over- or under-fitting?
* convergence                              -> why does an optimizer warn, and does it matter?
* epochs / batch size / steps (MLP)        -> what these hyperparameters actually control
* early stopping (HGB)                     -> time saved vs score lost
* HPO budget                               -> diminishing returns of more search trials
* sensitive-feature removal (D6)           -> does "fairness through unawareness" work here?
* drift (PSI)                              -> real train->validation shift + simulated shifts

Validation numbers reported here are for learning only. They were not used to
pick the champion; ``modeling.train_models`` does that independently.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, learning_curve, validation_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from .config import Paths
from .evaluation import (
    FACET_OF_INTEREST,
    PSI_RULE_OF_THUMB,
    SENSITIVE_ATTRIBUTES,
    bootstrap_indices,
    cost_optimal_threshold,
    feature_baseline,
    feature_psi,
    paired_bootstrap_diff,
    pr_auc,
    ranking_metrics,
    score_baseline,
    score_psi,
    subgroup_metrics,
)
from .features import (
    BINARY_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    SCALED_NUMERIC_FEATURES,
    build_preprocessor,
)
from .modeling import (
    ModelingData,
    TrainConfig,
    candidate_specs,
    cv_splitter,
    load_modeling_data,
    make_pipeline,
)
from .tracking import atomic_directory, build_manifest, new_run_id, read_json, write_json

# Output columns of build_preprocessor() that carry the sensitive attributes.
SENSITIVE_OUTPUT_COLUMNS = ["binary__senior_citizen", "categorical__gender_Male"]


def _fit_and_watch(model, X, y) -> tuple[object, list[str], float]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        t = time.perf_counter()
        model.fit(X, y)
        seconds = time.perf_counter() - t
    return model, [str(w.message)[:120] for w in caught if issubclass(w.category, ConvergenceWarning)], seconds


def _val(model, data: ModelingData) -> dict:
    p = model.predict_proba(data.X_val)[:, 1]
    m = ranking_metrics(data.y_val, p)
    return {"val_pr_auc": m["pr_auc"], "val_log_loss": m["log_loss"]}


# ---------------------------------------------------------------------------
# Over/underfitting
# ---------------------------------------------------------------------------
def learning_curves(data: ModelingData, cfg: TrainConfig) -> dict:
    sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    out = {}
    for name, est in (("logistic_regression", LogisticRegression(max_iter=2000)),
                      ("random_forest_unregularized", RandomForestClassifier(
                          n_estimators=200, min_samples_leaf=1, random_state=cfg.seed, n_jobs=1))):
        n, tr, cv = learning_curve(make_pipeline(est), data.X_train, data.y_train, train_sizes=sizes,
                                   cv=cv_splitter(cfg), scoring="average_precision", n_jobs=cfg.n_jobs)
        out[name] = [{"train_rows": int(k), "train_pr_auc": float(a.mean()), "cv_pr_auc": float(b.mean()),
                      "cv_std": float(b.std()), "gap": float(a.mean() - b.mean())}
                     for k, a, b in zip(n, tr, cv)]
    return out


def regularization_sweeps(data: ModelingData, cfg: TrainConfig) -> dict:
    out = {}
    for name, est, param, values in (
        ("logistic_regression_C", LogisticRegression(max_iter=5000), "clf__C",
         [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]),
        ("random_forest_min_samples_leaf", RandomForestClassifier(
            n_estimators=200, random_state=cfg.seed, n_jobs=1), "clf__min_samples_leaf",
         [1, 5, 20, 50, 100, 300]),
    ):
        tr, cv = validation_curve(make_pipeline(est), data.X_train, data.y_train, param_name=param,
                                  param_range=values, cv=cv_splitter(cfg), scoring="average_precision",
                                  n_jobs=cfg.n_jobs)
        out[name] = [{"value": v, "train_pr_auc": float(a.mean()), "cv_pr_auc": float(b.mean()),
                      "gap": float(a.mean() - b.mean())} for v, a, b in zip(values, tr, cv)]
    return out


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------
def unscaled_preprocessor() -> ColumnTransformer:
    """Same one-hot encoding as Phase 2 but numeric columns left in raw units.

    total_charges ranges to ~8,700 while one-hot columns are 0/1: an
    ill-conditioned problem for gradient-based solvers.
    """
    return ColumnTransformer([
        ("total_charges", SimpleImputer(strategy="constant", fill_value=0.0), ["total_charges"]),
        ("numeric", "passthrough", [c for c in SCALED_NUMERIC_FEATURES if c != "total_charges"]
         + BINARY_NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ]).set_output(transform="pandas")


def convergence_experiment(data: ModelingData, cfg: TrainConfig,
                           max_iters=(1, 3, 10, 30, 100, 1000),
                           mlp_learning_rates=(1e-5, 1e-3, 1e-1, 1.0)) -> dict:
    lr_rows = []
    for scaling in ("scaled", "unscaled"):
        for mi in max_iters:
            pre = build_preprocessor() if scaling == "scaled" else unscaled_preprocessor()
            model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=mi))])
            model, warns, sec = _fit_and_watch(model, data.X_train, data.y_train)
            lr_rows.append({"inputs": scaling, "max_iter": mi,
                            "n_iter": int(model.named_steps["clf"].n_iter_[0]),
                            "convergence_warning": bool(warns), "fit_seconds": sec, **_val(model, data)})
    mlp_rows = []
    for lr in mlp_learning_rates:
        model = make_pipeline(MLPClassifier(hidden_layer_sizes=(16,), learning_rate_init=lr, max_iter=200,
                                            random_state=cfg.seed))
        model, warns, sec = _fit_and_watch(model, data.X_train, data.y_train)
        clf = model.named_steps["clf"]
        curve = clf.loss_curve_
        mlp_rows.append({"learning_rate_init": lr, "epochs_run": int(clf.n_iter_),
                         "convergence_warning": bool(warns), "first_loss": float(curve[0]),
                         "final_loss": float(curve[-1]), "min_loss": float(min(curve)),
                         "loss_increases": int(np.sum(np.diff(curve) > 0)), "fit_seconds": sec,
                         **_val(model, data)})
    return {"logistic_regression": lr_rows, "mlp_learning_rate": mlp_rows}


def epoch_batch_experiment(data: ModelingData, cfg: TrainConfig, epochs: int = 30,
                           batch_sizes=(16, 64, 256, None)) -> list[dict]:
    """epoch = one pass over the data; step = one weight update on one mini-batch.

    steps per epoch = ceil(n_rows / batch_size). Smaller batches = more, noisier steps.
    n_iter_no_change is set to `epochs` so every run trains for the same number of epochs.
    """
    n = len(data.X_train)
    rows = []
    for bs in batch_sizes:
        size = n if bs is None else bs
        model = make_pipeline(MLPClassifier(hidden_layer_sizes=(16,), batch_size=size, max_iter=epochs,
                                            n_iter_no_change=epochs, random_state=cfg.seed))
        model, warns, sec = _fit_and_watch(model, data.X_train, data.y_train)
        clf = model.named_steps["clf"]
        rows.append({"batch_size": size, "epochs": int(clf.n_iter_),
                     "steps_per_epoch": int(np.ceil(n / size)),
                     "total_steps": int(clf.n_iter_ * np.ceil(n / size)),
                     "samples_seen": int(clf.t_), "fit_seconds": sec,
                     "loss_by_epoch": [round(float(v), 4) for v in clf.loss_curve_],
                     "convergence_warning": bool(warns), **_val(model, data)})
    return rows


def early_stopping_experiment(data: ModelingData, cfg: TrainConfig, max_iter: int = 1000) -> list[dict]:
    rows = []
    for es in (False, True):
        model = make_pipeline(HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=max_iter, early_stopping=es, validation_fraction=0.1,
            n_iter_no_change=20, random_state=cfg.seed))
        model, _, sec = _fit_and_watch(model, data.X_train, data.y_train)
        rows.append({"early_stopping": es, "max_iter": max_iter,
                     "boosting_rounds_used": int(model.named_steps["clf"].n_iter_),
                     "fit_seconds": sec, **_val(model, data)})
    return rows


# ---------------------------------------------------------------------------
# HPO budget
# ---------------------------------------------------------------------------
def hpo_budget_experiment(data: ModelingData, cfg: TrainConfig, max_trials: int = 30,
                          budgets=(1, 3, 5, 10, 20, 30)) -> list[dict]:
    """One random search of ``max_trials``; a budget of k = its first k trials.

    Random-search trials are i.i.d. draws, so the first k trials *are* a
    k-trial random search; no need to re-run the search per budget.
    """
    spec = next(s for s in candidate_specs() if s.name == "hist_gradient_boosting")
    search = RandomizedSearchCV(make_pipeline(spec.make_estimator(cfg)),
                                {f"clf__{k}": v for k, v in spec.search_space.items()},
                                n_iter=max_trials, scoring="average_precision", cv=cv_splitter(cfg),
                                random_state=cfg.seed, n_jobs=cfg.n_jobs, refit=False)
    t = time.perf_counter()
    search.fit(data.X_train, data.y_train)
    wall = time.perf_counter() - t
    r = pd.DataFrame(search.cv_results_)
    fit_cost = (r["mean_fit_time"] * cfg.cv_folds).cumsum()
    rows = []
    for k in [b for b in budgets if b <= max_trials]:
        best = int(r["mean_test_score"].iloc[:k].idxmax())
        model = make_pipeline(spec.make_estimator(cfg)).set_params(**r.loc[best, "params"])
        model.fit(data.X_train, data.y_train)
        rows.append({"trials": k, "best_cv_pr_auc": float(r.loc[best, "mean_test_score"]),
                     "best_cv_std": float(r.loc[best, "std_test_score"]),
                     "cumulative_fit_seconds": float(fit_cost.iloc[k - 1]), **_val(model, data)})
    return {"family": "hist_gradient_boosting", "wall_seconds_all_trials": wall, "rows": rows}


# ---------------------------------------------------------------------------
# D6: removing sensitive features
# ---------------------------------------------------------------------------
def drop_columns(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return X.drop(columns=columns)


def without_sensitive(estimator) -> Pipeline:
    pipe = make_pipeline(estimator)
    pipe.steps.insert(1, ("drop_sensitive", FunctionTransformer(
        drop_columns, kw_args={"columns": SENSITIVE_OUTPUT_COLUMNS})))
    return pipe


def sensitive_feature_experiment(data: ModelingData, cfg: TrainConfig) -> dict:
    """Same family, same search, same folds, same seed; only the inputs differ."""
    spec = next(s for s in candidate_specs() if s.name == "logistic_regression")
    space = {f"clf__{k}": v for k, v in spec.search_space.items()}
    yv = data.y_val.to_numpy()
    groups = data.groups_val[SENSITIVE_ATTRIBUTES].reset_index(drop=True)
    variants, scores = {}, {}
    for name, pipe in (("with_sensitive", make_pipeline(spec.make_estimator(cfg))),
                       ("without_sensitive", without_sensitive(spec.make_estimator(cfg)))):
        search = RandomizedSearchCV(pipe, space, n_iter=cfg.n_iter, scoring="average_precision",
                                    cv=cv_splitter(cfg), random_state=cfg.seed, n_jobs=cfg.n_jobs)
        search.fit(data.X_train, data.y_train)
        p = search.best_estimator_.predict_proba(data.X_val)[:, 1]
        thr = cost_optimal_threshold(yv, p, cfg.fn_cost, cfg.fp_cost)
        scores[name] = p
        variants[name] = {
            "best_params": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
            "cv_pr_auc": float(search.best_score_), "val_pr_auc": pr_auc(yv, p), "threshold": thr,
            "subgroups": subgroup_metrics(yv, p, thr, groups, min_n=cfg.min_subgroup_n,
                                          min_class=cfg.min_subgroup_class,
                                          n_boot=min(cfg.n_boot, 1000), seed=cfg.seed),
        }
    idx = bootstrap_indices(len(yv), cfg.n_boot, cfg.seed)

    # Proxy check: can the remaining features reconstruct each sensitive attribute?
    proxies = {}
    rest = [c for c in data.X_train.columns if c not in SENSITIVE_ATTRIBUTES]
    for attr in SENSITIVE_ATTRIBUTES:
        target = (data.X_train[attr] == FACET_OF_INTEREST[attr]).astype(int)
        pre = build_preprocessor()
        pre.transformers = [(n, t, [c for c in cols if c in rest]) for n, t, cols in pre.transformers]
        pre.transformers = [tr for tr in pre.transformers if tr[2]]
        proxy = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
        auc = cross_val_score(proxy, data.X_train[rest], target, cv=cv_splitter(cfg), scoring="roc_auc")
        proxies[attr] = {"cv_roc_auc_mean": float(auc.mean()), "cv_roc_auc_std": float(auc.std()),
                         "positive_share": float(target.mean())}
    g = groups
    defensibility = {
        f"{a}={v}": {"n": int((g[a] == v).sum()), "positives": int(yv[(g[a] == v).to_numpy()].sum())}
        for a in SENSITIVE_ATTRIBUTES for v in sorted(g[a].unique(), key=str)
    }
    return {
        "variants": variants,
        "pr_auc_diff_without_minus_with": paired_bootstrap_diff(
            yv, scores["without_sensitive"], scores["with_sensitive"], pr_auc, indices=idx),
        "prediction_correlation": float(np.corrcoef(scores["with_sensitive"],
                                                    scores["without_sensitive"])[0, 1]),
        "proxy_reconstruction_from_other_features": proxies,
        "validation_group_sizes": defensibility,
        "dropped_output_columns": SENSITIVE_OUTPUT_COLUMNS,
    }


# ---------------------------------------------------------------------------
# Drift (PSI): real train->validation, plus simulated shifts of validation
# ---------------------------------------------------------------------------
def simulated_shifts(X: pd.DataFrame, y: pd.Series, seed: int) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    rng = np.random.default_rng(seed)
    out = {"none (validation as is)": (X, y)}
    up = X.copy()
    up["monthly_charges"] = up["monthly_charges"] * 1.15
    up["total_charges"] = up["total_charges"] * 1.15
    out["price increase +15%"] = (up, y)
    m2m = np.flatnonzero(X["contract"].to_numpy() == "Month-to-month")
    other = np.flatnonzero(X["contract"].to_numpy() != "Month-to-month")
    n = len(X)
    pick = np.concatenate([rng.choice(m2m, int(0.8 * n)), rng.choice(other, n - int(0.8 * n))])
    out["contract mix: 80% month-to-month (label shift too)"] = (X.iloc[pick], y.iloc[pick])
    unseen = X.copy()
    rows = rng.random(n) < 0.2
    unseen["payment_method"] = np.where(rows, "Crypto wallet", unseen["payment_method"].to_numpy())
    out["unseen category: 20% 'Crypto wallet' payment"] = (unseen, y)
    return out


def drift_experiment(data: ModelingData, cfg: TrainConfig, model=None) -> dict:
    if model is None:
        model = make_pipeline(LogisticRegression(max_iter=2000)).fit(data.X_train, data.y_train)
        model_name = "logistic_regression (default params, fitted here)"
    else:
        model_name = "champion from --run"
    base = feature_baseline(data.X_train)
    score_base = score_baseline(model.predict_proba(data.X_val)[:, 1])
    rows = []
    for name, (X, y) in simulated_shifts(data.X_val, data.y_val, cfg.seed).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # unknown-category warning is the point here
            p = model.predict_proba(X)[:, 1]
        f = feature_psi(base, X)
        top = sorted(f.items(), key=lambda kv: -kv[1]["psi"])[:3]
        rows.append({"scenario": name, "top_feature_psi": {k: v["psi"] for k, v in top},
                     "max_feature_psi": top[0][1]["psi"],
                     "score_psi_vs_validation": score_psi(score_base, p),
                     "base_rate": float(np.mean(y)), "pr_auc": pr_auc(y, p)})
    return {"rule_of_thumb": PSI_RULE_OF_THUMB, "model": model_name,
            "feature_psi_train_to_validation": feature_psi(base, data.X_val), "scenarios": rows,
            "note": "No time column exists, so real drift over time cannot be measured; "
                    "train->validation is a same-snapshot random split and should be ~0."}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_diagnostics(paths: Paths, cfg: TrainConfig = TrainConfig(), run_id: str | None = None) -> dict:
    data = load_modeling_data(paths)
    model = None
    if run_id is not None:
        run_dir = paths.models / run_id
        if read_json(run_dir / "manifest.json")["data"] != asdict(data.fingerprint):
            raise ValueError(f"run {run_id} was trained on different inputs")
        model = joblib.load(run_dir / "model.joblib")
    timings = {}
    results = {}
    for name, fn in (("learning_curves", lambda: learning_curves(data, cfg)),
                     ("regularization_sweeps", lambda: regularization_sweeps(data, cfg)),
                     ("convergence", lambda: convergence_experiment(data, cfg)),
                     ("epoch_batch", lambda: epoch_batch_experiment(data, cfg)),
                     ("early_stopping", lambda: early_stopping_experiment(data, cfg)),
                     ("hpo_budget", lambda: hpo_budget_experiment(data, cfg)),
                     ("sensitive_features", lambda: sensitive_feature_experiment(data, cfg)),
                     ("drift", lambda: drift_experiment(data, cfg, model))):
        t = time.perf_counter()
        results[name] = fn()
        timings[name] = time.perf_counter() - t
    results["seconds"] = timings
    did = new_run_id()
    final = paths.models / "diagnostics" / did
    with atomic_directory(final) as tmp:
        write_json(tmp / "diagnostics.json", results)
        (tmp / "diagnostics.md").write_text(render_diagnostics(results))
        write_json(tmp / "manifest.json", build_manifest(
            run_id=did, kind="diagnose", directory=tmp, params=asdict(cfg), metrics={},
            tags={"phase": "3", "affects_selection": "no", "test_split_read": "no",
                  "model_from_run": run_id or "none"}, fingerprint=data.fingerprint))
    return {"diagnostics_id": did, "dir": final, "results": results}


def _table(rows: list[dict], cols: list[str]) -> list[str]:
    def f(v):
        if isinstance(v, float):
            if v != 0 and abs(v) < 1e-3:
                return f"{v:.0e}"
            return f"{v:.4f}" if abs(v) < 1000 else f"{v:.0f}"
        return str(v)
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + [
        "| " + " | ".join(f(r.get(c)) for c in cols) + " |" for r in rows]


def render_diagnostics(r: dict) -> str:
    L = ["# Phase 3 diagnostics (learning only; not used for model selection)", ""]
    for name, rows in r["learning_curves"].items():
        L += [f"## Learning curve: {name}", "", *_table(rows, ["train_rows", "train_pr_auc", "cv_pr_auc", "gap"]), ""]
    for name, rows in r["regularization_sweeps"].items():
        L += [f"## Regularization sweep: {name}", "", *_table(rows, ["value", "train_pr_auc", "cv_pr_auc", "gap"]), ""]
    L += ["## Convergence: logistic regression", "",
          *_table(r["convergence"]["logistic_regression"],
                  ["inputs", "max_iter", "n_iter", "convergence_warning", "val_pr_auc", "val_log_loss"]), "",
          "## Convergence: MLP learning rate", "",
          *_table(r["convergence"]["mlp_learning_rate"],
                  ["learning_rate_init", "epochs_run", "convergence_warning", "first_loss", "final_loss",
                   "loss_increases", "val_pr_auc"]), "",
          "## Epochs, batch size, steps (MLP)", "",
          *_table(r["epoch_batch"], ["batch_size", "epochs", "steps_per_epoch", "total_steps", "fit_seconds",
                                     "val_pr_auc", "val_log_loss"]), "",
          "## Early stopping (HGB)", "",
          *_table(r["early_stopping"], ["early_stopping", "boosting_rounds_used", "fit_seconds", "val_pr_auc",
                                        "val_log_loss"]), "",
          "## HPO budget (HGB random search)", "",
          *_table(r["hpo_budget"]["rows"], ["trials", "best_cv_pr_auc", "best_cv_std", "cumulative_fit_seconds",
                                            "val_pr_auc"]), ""]
    s = r["sensitive_features"]
    d = s["pr_auc_diff_without_minus_with"]
    L += ["## D6: removing gender and senior_citizen (logistic regression)", "",
          f"- Validation PR AUC with: {s['variants']['with_sensitive']['val_pr_auc']:.4f}; without: "
          f"{s['variants']['without_sensitive']['val_pr_auc']:.4f}; diff (without − with) "
          f"{d['diff']:+.4f}, 95% CI [{d['low']:+.4f}, {d['high']:+.4f}]",
          f"- Prediction correlation: {s['prediction_correlation']:.4f}", ""]
    rows = []
    for v, e in s["variants"].items():
        for attr, a in e["subgroups"]["attributes"].items():
            dd = a.get("disparities", {})
            rows.append({"variant": v, "attribute": attr,
                         "dppl": dd.get("dppl", {}).get("estimate"),
                         "dppl_ci": f"[{dd['dppl']['low']:.3f}, {dd['dppl']['high']:.3f}]" if dd else "",
                         "recall_diff": dd.get("recall_difference", {}).get("estimate")})
    L += _table(rows, ["variant", "attribute", "dppl", "dppl_ci", "recall_diff"])
    L += ["", "Proxy check (CV ROC AUC of predicting the attribute from the other 17 inputs):", ""]
    L += [f"- {a}: {p['cv_roc_auc_mean']:.3f} ± {p['cv_roc_auc_std']:.3f}"
          for a, p in s["proxy_reconstruction_from_other_features"].items()]
    L += ["", "Validation group sizes: " + ", ".join(
        f"{k} n={v['n']} (positives {v['positives']})" for k, v in s["validation_group_sizes"].items()), ""]
    dr = r["drift"]
    L += ["## Drift (PSI)", "", dr["rule_of_thumb"], "", dr["note"], "",
          f"Max feature PSI train→validation: "
          f"{max(v['psi'] for v in dr['feature_psi_train_to_validation'].values()):.4f}", "",
          *_table(dr["scenarios"], ["scenario", "max_feature_psi", "score_psi_vs_validation", "base_rate",
                                    "pr_auc"]), "",
          "## Seconds per experiment", "", *[f"- {k}: {v:.1f}" for k, v in r["seconds"].items()], ""]
    return "\n".join(L)
