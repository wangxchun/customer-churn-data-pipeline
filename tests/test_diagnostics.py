"""Phase 3 diagnostics: the experiments demonstrate what they claim to demonstrate."""
import numpy as np
import pytest

from churn_pipeline.diagnostics import (
    SENSITIVE_OUTPUT_COLUMNS,
    convergence_experiment,
    drift_experiment,
    early_stopping_experiment,
    epoch_batch_experiment,
    without_sensitive,
)
from churn_pipeline.modeling import prepare_modeling_data
from sklearn.linear_model import LogisticRegression
from phase3_data import FAST, make_signal_master
from test_modeling import merged


@pytest.fixture(scope="module")
def data():
    return prepare_modeling_data(merged(make_signal_master()))


def test_too_few_iterations_trigger_a_recorded_convergence_warning(data):
    out = convergence_experiment(data, FAST, max_iters=(1, 1000), mlp_learning_rates=(1e-3,))
    rows = {(r["inputs"], r["max_iter"]): r for r in out["logistic_regression"]}
    assert rows[("scaled", 1)]["convergence_warning"] is True
    assert rows[("scaled", 1000)]["convergence_warning"] is False
    assert rows[("scaled", 1000)]["n_iter"] < 1000


def test_early_stopping_stops_before_max_iter(data):
    rows = {r["early_stopping"]: r for r in early_stopping_experiment(data, FAST, max_iter=300)}
    assert rows[False]["boosting_rounds_used"] == 300
    assert rows[True]["boosting_rounds_used"] < 300


def test_steps_are_epochs_times_batches_per_epoch(data):
    n = len(data.X_train)
    rows = epoch_batch_experiment(data, FAST, epochs=3, batch_sizes=(32, None))
    small, full = rows
    assert small["steps_per_epoch"] == int(np.ceil(n / 32)) and small["epochs"] == 3
    assert small["total_steps"] == 3 * int(np.ceil(n / 32))
    assert full["steps_per_epoch"] == 1 and full["samples_seen"] == 3 * n


def test_without_sensitive_removes_exactly_the_sensitive_model_inputs(data):
    pipe = without_sensitive(LogisticRegression(max_iter=500)).fit(data.X_train, data.y_train)
    Z = pipe[:-1].transform(data.X_train)
    assert not set(SENSITIVE_OUTPUT_COLUMNS) & set(Z.columns)
    assert Z.shape[1] == 40 - len(SENSITIVE_OUTPUT_COLUMNS)


def test_drift_scenarios_are_detected_by_psi(data):
    out = drift_experiment(data, FAST)
    rows = {r["scenario"].split(":")[0].split(" (")[0]: r for r in out["scenarios"]}
    assert rows["price increase +15%"]["top_feature_psi"]["monthly_charges"] > 0.1
    assert rows["unseen category"]["top_feature_psi"]["payment_method"] > 0.25
