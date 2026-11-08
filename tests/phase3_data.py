"""Shared Phase 3 test data: a small synthetic master *with* churn signal.

``make_master`` (Phase 2 tests) draws churn independently of the features, so
every model would be at chance. Here churn depends on contract, tenure and
internet service, so selection and thresholds behave like on real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from churn_pipeline.config import Paths
from churn_pipeline.features import run_features
from churn_pipeline.modeling import TrainConfig
from test_features import make_master

FAST = TrainConfig(n_iter=2, cv_folds=3, n_boot=200, n_jobs=1, rf_n_estimators=30,
                   permutation_repeats=2, latency_repeats=1, min_subgroup_n=50,
                   min_subgroup_class=5)


def make_signal_master(n: int = 600, seed: int = 0) -> pd.DataFrame:
    df = make_master(n, seed)
    rng = np.random.default_rng(seed + 1)
    logit = (-1.2 + 1.8 * (df["contract"] == "Month-to-month").astype(float)
             - 0.03 * df["tenure_months"].astype(float)
             + 0.9 * (df["internet_service"] == "Fiber optic").astype(float))
    p = 1 / (1 + np.exp(-logit.to_numpy()))
    df["churn"] = pd.array(np.where(rng.random(n) < p, "Yes", "No"), dtype="string")
    return df


def write_phase2(tmp_path, master: pd.DataFrame | None = None) -> Paths:
    paths = Paths(tmp_path / "data")
    paths.curated.mkdir(parents=True)
    (make_signal_master() if master is None else master).to_parquet(
        paths.curated / "customer_master.parquet", index=False)
    run_features(paths)
    return paths
