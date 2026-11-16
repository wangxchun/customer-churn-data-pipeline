"""Phase 3: test isolation, CV leakage, Phase 2 consistency, stale-data guards,
selection rule, reproducibility, atomic outputs, and the one-time test lock."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from churn_pipeline import modeling
from churn_pipeline.cli import main
from churn_pipeline.config import DEFAULT_PATHS, Paths
from churn_pipeline.evaluation import (
    IncomparableRunsError,
    TestSetAlreadyUsedError,
    compare_runs,
    cost_optimal_threshold,
    run_final_evaluation,
)
from churn_pipeline.features import SplitConfig, split_customers
from churn_pipeline.modeling import (
    CandidateSpec,
    TrainConfig,
    cv_splitter,
    load_modeling_data,
    make_pipeline,
    prepare_modeling_data,
    run_training,
    select_champion,
    train_models,
    verify_phase2_consistency,
)
from churn_pipeline.tracking import StaleArtifactError, load_verified_inputs, read_json
from phase3_data import FAST, make_signal_master, write_phase2


def merged(master: pd.DataFrame) -> pd.DataFrame:
    master = master.assign(customer_id=master["customer_id"].astype("string"))
    return master.merge(split_customers(master, SplitConfig()), on="customer_id").sort_values(
        "customer_id", ignore_index=True)


class SpyLR(LogisticRegression):
    """Records which customers (index values) reach fit / predict_proba."""

    seen_fit: list = []
    seen_predict: list = []

    def fit(self, X, y, sample_weight=None):
        SpyLR.seen_fit.append(frozenset(X.index))
        return super().fit(X, y, sample_weight)

    def predict_proba(self, X):
        SpyLR.seen_predict.append(frozenset(X.index))
        return super().predict_proba(X)


def spy_specs(*names):
    return [CandidateSpec(n, i + 1, "spy", lambda c: SpyLR(max_iter=500), {"C": [0.1, 1.0]})
            for i, n in enumerate(names)]


@pytest.fixture(scope="module")
def trained_template(tmp_path_factory):
    """One real run directory, trained once per module."""
    root = tmp_path_factory.mktemp("trained")
    paths = write_phase2(root)
    return paths, run_training(paths, FAST)["run_id"]


@pytest.fixture
def trained(tmp_path, trained_template):
    """A private copy of the trained data dir, so tests can mutate it freely."""
    src, run_id = trained_template
    dst = Paths(tmp_path / "data")
    shutil.copytree(src.data_dir, dst.data_dir)
    return dst, run_id


def clone_run(paths, run_id, new_id):
    shutil.copytree(paths.models / run_id, paths.models / new_id)
    m = read_json(paths.models / new_id / "manifest.json")
    m["run_id"] = new_id
    (paths.models / new_id / "manifest.json").write_text(json.dumps(m))
    return new_id


@pytest.fixture(scope="module")
def baseline_run():
    data = prepare_modeling_data(merged(make_signal_master()))
    return data, train_models(data, FAST)


# ---------------------------------------------------------------------------
# Test-set isolation
# ---------------------------------------------------------------------------
def test_modeling_data_contains_no_test_customers():
    data = merged(make_signal_master())
    md = prepare_modeling_data(data)
    test_ids = set(data.loc[data["split"] == "test", "customer_id"].astype(str))
    seen = set(md.X_train.index) | set(md.X_val.index)
    assert test_ids and not (seen & test_ids)
    assert len(seen) == (data["split"] != "test").sum()


def test_scrambling_test_rows_changes_nothing_in_training(baseline_run):
    _, ref = baseline_run
    master = make_signal_master()
    data = merged(master)
    test = data["split"] == "test"
    rng = np.random.default_rng(99)
    data.loc[test, "churn"] = rng.permutation(data.loc[test, "churn"].to_numpy())
    data.loc[test, "monthly_charges"] = rng.uniform(500, 900, test.sum())
    data.loc[test, "contract"] = "Two year"
    out = train_models(prepare_modeling_data(data), FAST)
    assert out.selection["champion"] == ref.selection["champion"]
    assert out.selection["pr_auc"] == ref.selection["pr_auc"]
    assert out.threshold["chosen"] == ref.threshold["chosen"]
    assert {n: t.best_params for n, t in out.tuned.items()} == {n: t.best_params for n, t in ref.tuned.items()}
    for n in ref.val_scores:
        np.testing.assert_array_equal(out.val_scores[n], ref.val_scores[n])


def test_spy_model_never_sees_a_test_customer():
    SpyLR.seen_fit.clear(), SpyLR.seen_predict.clear()
    data = merged(make_signal_master())
    test_ids = set(data.loc[data["split"] == "test", "customer_id"].astype(str))
    val_ids = set(data.loc[data["split"] == "validation", "customer_id"].astype(str))
    train_models(prepare_modeling_data(data), FAST, specs=spy_specs("spy"))
    seen = set().union(*SpyLR.seen_fit, *SpyLR.seen_predict)
    assert not (seen & test_ids)
    assert val_ids <= set().union(*SpyLR.seen_predict)  # the spy works: validation was scored
    assert not (set().union(*SpyLR.seen_fit) & val_ids)  # and validation was never fitted on


def test_every_candidate_is_cross_validated_on_identical_folds():
    SpyLR.seen_fit.clear()
    data = prepare_modeling_data(merged(make_signal_master()))
    modeling.tune(spy_specs("a")[0], data.X_train, data.y_train, FAST)
    folds_a = set(SpyLR.seen_fit[:-1])  # last fit = refit on all training rows
    SpyLR.seen_fit.clear()
    modeling.tune(spy_specs("b")[0], data.X_train, data.y_train, FAST)
    assert folds_a == set(SpyLR.seen_fit[:-1])
    assert len(folds_a) == FAST.cv_folds


# ---------------------------------------------------------------------------
# Leakage inside CV, and agreement with Phase 2
# ---------------------------------------------------------------------------
def test_preprocessor_is_refitted_on_each_folds_training_rows_only():
    data = prepare_modeling_data(merged(make_signal_master()))
    X, y = data.X_train, data.y_train
    res = cross_validate(make_pipeline(LogisticRegression(max_iter=500)), X, y, cv=cv_splitter(FAST),
                         return_estimator=True)
    full_mean = X[["tenure_months", "monthly_charges"]].mean().to_numpy()
    for est, (tr, _) in zip(res["estimator"], cv_splitter(FAST).split(X, y)):
        fold_mean = X.iloc[tr][["tenure_months", "monthly_charges"]].mean().to_numpy()
        scaler = est.named_steps["pre"].named_transformers_["numeric"]
        np.testing.assert_allclose(scaler.mean_, fold_mean, rtol=1e-12)
        assert not np.allclose(scaler.mean_, full_mean, rtol=1e-6)


def test_fitted_preprocessor_reproduces_phase2_train_parquet(tmp_path):
    paths = write_phase2(tmp_path)
    data = load_modeling_data(paths)
    pipe = make_pipeline(LogisticRegression(max_iter=500)).fit(data.X_train, data.y_train)
    assert verify_phase2_consistency(pipe, data.X_train, paths)["status"] == "identical"


@pytest.mark.real_data
@pytest.mark.skipif(not (DEFAULT_PATHS.features / "train.parquet").exists(),
                    reason="real Phase 2 outputs not built")
def test_real_data_preprocessor_reproduces_train_parquet_exactly():
    data = load_modeling_data(DEFAULT_PATHS)
    pipe = make_pipeline(LogisticRegression(max_iter=2000)).fit(data.X_train, data.y_train)
    out = verify_phase2_consistency(pipe, data.X_train, DEFAULT_PATHS)
    assert out == {"status": "identical", "rows": 4929, "columns": 40, "rtol": 0}


# ---------------------------------------------------------------------------
# Stale-data guards (TD-3 mitigation)
# ---------------------------------------------------------------------------
def test_training_refuses_a_master_that_phase2_did_not_build_from(tmp_path):
    paths = write_phase2(tmp_path)
    master = pd.read_parquet(paths.curated / "customer_master.parquet")
    master.loc[0, "monthly_charges"] = 99.99
    master.to_parquet(paths.curated / "customer_master.parquet", index=False)
    with pytest.raises(StaleArtifactError, match="build-features"):
        load_modeling_data(paths)


def test_compare_refuses_runs_made_on_different_splits(trained):
    paths, a = trained
    b = clone_run(paths, a, "challenger")
    m = paths.models / b / "manifest.json"
    manifest = read_json(m)
    manifest["data"]["split_content_sha256"] = "0" * 64
    m.write_text(json.dumps(manifest))
    with pytest.raises(IncomparableRunsError, match="split_content_sha256"):
        compare_runs(paths, a, b, n_boot=200)


# ---------------------------------------------------------------------------
# Selection rule (D5), threshold (D4)
# ---------------------------------------------------------------------------
def _scores(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    signal = y + rng.normal(0, 1.0, n)
    return y, signal


def test_selection_prefers_the_simpler_model_when_the_gap_is_noise():
    y, s = _scores()
    rng = np.random.default_rng(1)
    complex_ = s + rng.normal(0, 0.05, len(y))  # same signal, tiny jitter
    out = select_champion(y, {"simple": s, "complex": complex_}, {"simple": 1, "complex": 3},
                          n_boot=300, seed=0)
    assert "complex" in out["within_noise_of_reference"] or out["reference"] == "simple"
    assert out["champion"] == "simple"


def test_selection_keeps_the_better_model_when_the_gap_is_significant():
    y, s = _scores()
    weak = s + np.random.default_rng(2).normal(0, 3.0, len(y))  # much noisier
    out = select_champion(y, {"simple": weak, "complex": s}, {"simple": 1, "complex": 3},
                          n_boot=300, seed=0)
    assert out["reference"] == "complex"
    assert out["comparisons_vs_reference"]["simple"]["low"] > 0
    assert out["champion"] == "complex"


def test_threshold_is_the_cost_optimum_on_validation_scores(baseline_run):
    data, r = baseline_run
    p = r.val_scores[r.selection["champion"]]
    assert r.threshold["chosen"] == cost_optimal_threshold(data.y_val, p, 5.0, 1.0)
    assert r.threshold["bayes_threshold_if_calibrated"] == pytest.approx(1 / 6)


def test_train_config_rejects_non_positive_costs():
    with pytest.raises(ValueError):
        TrainConfig(fn_cost=0)


# ---------------------------------------------------------------------------
# Reproducibility: metrics and predictions asserted; model bytes only measured
# ---------------------------------------------------------------------------
def test_same_seed_gives_identical_metrics_and_predictions(baseline_run):
    data, ref = baseline_run
    again = train_models(data, FAST)
    assert again.selection["pr_auc"] == ref.selection["pr_auc"]
    for n in ref.val_scores:
        np.testing.assert_array_equal(again.val_scores[n], ref.val_scores[n])
    # Byte-level identity of the pickled model is NOT guaranteed; it is measured in the
    # real run's manifest (model_sha256) and documented, not asserted.


# ---------------------------------------------------------------------------
# Run directory, test lock, CLI
# ---------------------------------------------------------------------------
def test_run_directory_is_all_or_nothing(tmp_path, monkeypatch):
    paths = write_phase2(tmp_path)
    real = modeling.write_json

    def fail_on_explanations(path, obj):
        if path.name == "explanations.json":
            raise OSError("disk full (simulated)")
        return real(path, obj)

    monkeypatch.setattr(modeling, "write_json", fail_on_explanations)
    with pytest.raises(OSError, match="simulated"):
        run_training(paths, FAST)
    assert not paths.models.exists() or list(paths.models.iterdir()) == []


def test_run_manifest_records_inputs_and_model_hash(trained):
    paths, run_id = trained
    run_dir = paths.models / run_id
    manifest = read_json(run_dir / "manifest.json")
    fp = load_verified_inputs(paths).fingerprint
    assert manifest["data"]["split_content_sha256"] == fp.split_content_sha256
    assert manifest["data"]["customer_master_sha256"] == fp.customer_master_sha256
    assert manifest["model_sha256"] == manifest["artifacts"]["model.joblib"]
    assert manifest["tags"]["test_split_read"] == "no"
    assert set(manifest) >= {"params", "metrics", "tags", "artifacts", "code", "environment"}
    assert not (run_dir / "test_access.json").exists()


def test_test_split_is_read_once_unless_forced_and_every_read_is_logged(trained):
    paths, run_id = trained
    first = run_final_evaluation(paths, run_id, n_boot=200)
    assert first["output_dir"].name == "evaluation-01"
    with pytest.raises(TestSetAlreadyUsedError):
        run_final_evaluation(paths, run_id, n_boot=200)
    second = run_final_evaluation(paths, run_id, force=True, n_boot=200)
    assert second["output_dir"].name == "evaluation-02"
    log = read_json(paths.models / run_id / "test_access.json")
    assert [r["force"] for r in log["reads"]] == [False, True]
    # The threshold applied to test is the one fixed on validation.
    sel = read_json(paths.models / run_id / "selection.json")
    assert first["metrics"]["threshold"] == sel["threshold"]["chosen"]


def test_a_crashed_evaluation_still_counts_as_a_test_read(trained, monkeypatch):
    paths, run_id = trained
    import churn_pipeline.evaluation as ev
    monkeypatch.setattr(ev, "joblib", type("J", (), {"load": staticmethod(
        lambda p: (_ for _ in ()).throw(RuntimeError("crash after reading test")))}))
    with pytest.raises(RuntimeError, match="crash"):
        run_final_evaluation(paths, run_id, n_boot=200)
    monkeypatch.undo()
    assert len(read_json(paths.models / run_id / "test_access.json")["reads"]) == 1
    with pytest.raises(TestSetAlreadyUsedError):
        run_final_evaluation(paths, run_id, n_boot=200)
    assert not list((paths.models / run_id).glob("evaluation-*"))


def test_cli_evaluate_twice_exits_with_refusal(trained, capsys):
    paths, run_id = trained
    assert main(["--data-dir", str(paths.data_dir), "evaluate", "--run", run_id]) == 0
    assert main(["--data-dir", str(paths.data_dir), "evaluate", "--run", run_id]) == 2
    assert "refused" in capsys.readouterr().err


def test_compare_same_run_twice_reports_no_difference(trained):
    paths, a = trained
    b = clone_run(paths, a, "challenger")
    out = compare_runs(paths, a, b, n_boot=200)
    assert out["prediction_agreement"] == 1.0
    assert out["pr_auc_diff_challenger_minus_champion"]["diff"] == 0
    assert out["decision"] == "keep champion"
    assert (paths.models / "comparisons" / f"{a}__vs__{b}.json").exists()
