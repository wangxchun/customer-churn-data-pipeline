"""W3: monitoring records, alerts and the Markdown report.

Every test here is a synthetic-fixture test. The master, split, run, baseline,
release, key and traffic all come from synthetic fixtures under pytest tmp
directories. Nothing reads ``data/``, D7 outputs or the real champion. No real
batch is predicted; that is stage C, which needs separate approval.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_pipeline import cli
from churn_pipeline import modeling
from churn_pipeline import monitoring as mon
from churn_pipeline import predict as pr
from churn_pipeline import release as rel
from churn_pipeline import simulate as sim
from churn_pipeline.evaluation import feature_psi, score_baseline, score_psi
from churn_pipeline.features import (
    ID_COLUMN,
    _to_model_frame,
    build_features,
)
from churn_pipeline.inference import REQUIRED_COLUMNS
from test_inference import fit_model, synthetic_master
from test_predict import add_baseline
from test_release import KEY, RUN_ID, make_env, write_key_file

RELEASE_ID = "rel-syn-mon"
N, SEED = 2000, 11
RAW_CATEGORY_VALUES = ("Month-to-month", "Fiber optic", "Electronic check", "Crypto wallet",
                       "Two year", "No internet service")


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def master():
    return synthetic_master()


@pytest.fixture(scope="module")
def model(master):
    return fit_model(master)


@pytest.fixture(scope="module")
def assignments(master):
    return build_features(master).assignments


def build_env(root: Path, model, master, assignments):
    paths = make_env(root, model, assignments)
    baseline = add_baseline(paths, model, master, assignments)
    paths.curated.mkdir(parents=True, exist_ok=True)
    master.to_parquet(paths.curated / "customer_master.parquet", index=False)  # for D7 source
    rel.create_release(paths, RUN_ID, KEY, release_id=RELEASE_ID)
    rel.promote(paths, RELEASE_ID, KEY)
    return paths, baseline


@pytest.fixture(scope="module")
def shared(tmp_path_factory, model, master, assignments):
    """One release shared by read-only tests; every batch uses a unique batch_id."""
    root = tmp_path_factory.mktemp("mon")
    paths, baseline = build_env(root, model, master, assignments)
    source = sim.load_traffic_source(paths, suffix="SYNTR", expected_count=len(master))
    return {"paths": paths, "baseline": baseline, "source": source, "root": root}


@pytest.fixture
def fresh(tmp_path, model, master, assignments):
    paths, baseline = build_env(tmp_path, model, master, assignments)
    return paths


def predict_scenario(ctx, scenario, *, n=N, seed=SEED, batch_id=None, **kw):
    df = sim.generate_traffic(ctx["source"], scenario, n, seed)
    batch_id = batch_id or f"b-{scenario}-{n}-{seed}"
    inp = ctx["root"] / f"{batch_id}.parquet"
    df.to_parquet(inp, index=False)
    m = pr.run_predict(ctx["paths"], inp, "production", KEY, batch_id=batch_id, **kw)
    return df, m, record_for(ctx["paths"], batch_id)


def record_for(paths, batch_id):
    (rec,) = [r for r in mon.read_records(paths, RELEASE_ID) if r["batch_id"] == batch_id]
    return rec


def alert_names(rec):
    return {a["metric"] + (f":{a['feature']}" if "feature" in a else "") for a in rec["alerts"]}


# ---------------------------------------------------------------------------
# 1. Scenarios and alerts (P8 thresholds)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario,expected", [
    ("none", set()),
    ("contract_mix", {"feature_psi:contract", "score_psi"}),
    ("unseen_payment", {"reject_rate", "unseen_share"}),
    ("nulls", {"reject_rate"}),
])
def test_scenarios_raise_the_expected_alerts(shared, scenario, expected):
    _, _, rec = predict_scenario(shared, scenario)
    assert alert_names(rec) == expected
    assert rec["retrain_review_recommended"] is bool(expected)
    assert rec["insufficient_sample"] is False


# Monthly charges by internet-service tier, fixed up front from the domain (as in the
# IBM Telco data, the price depends on the service tier), NOT tuned to a PSI result.
# make_master's uniform(18, 120) charges have no such structure, so a +15% shift
# barely moves mass between decile bins (PSI ~0.16). The Phase 3 real-data drift
# experiment measured 0.915 for the same +15% shift (PHASE3_GUIDE, section N.3).
CHARGE_BANDS = {"No": (19.0, 26.0), "DSL": (40.0, 70.0), "Fiber optic": (70.0, 110.0)}


def tiered_charges_master(seed: int = 7) -> pd.DataFrame:
    m = synthetic_master()
    rng = np.random.default_rng(seed)
    lo = m["internet_service"].map(lambda s: CHARGE_BANDS[s][0]).to_numpy(float)
    hi = m["internet_service"].map(lambda s: CHARGE_BANDS[s][1]).to_numpy(float)
    monthly = np.round(rng.uniform(lo, hi), 2)
    new_customer = m["total_charges"].isna()            # tenure 0: blank total, kept
    m["monthly_charges"] = pd.array(monthly, dtype="Float64")
    m["total_charges"] = pd.array(
        np.round(m["tenure_months"].astype(float).to_numpy() * monthly, 2), dtype="Float64")
    m.loc[new_customer, "total_charges"] = pd.NA
    return m


@pytest.fixture(scope="module")
def tiered(tmp_path_factory):
    master = tiered_charges_master()
    model = fit_model(master)
    assignments = build_features(master).assignments
    root = tmp_path_factory.mktemp("mon-tiered")
    paths, baseline = build_env(root, model, master, assignments)
    source = sim.load_traffic_source(paths, suffix="SYNTR", expected_count=len(master))
    return {"paths": paths, "baseline": baseline, "source": source, "root": root}


def test_price_up_raises_a_feature_psi_alert_on_tiered_charges(tiered):
    """+15% monthly and total charges is a large shift once charges have realistic,
    tier-concentrated structure. The P8 threshold (PSI > 0.25, strict) is unchanged.
    Score-PSI alerts are deliberately NOT asserted: D7 resamples TRAIN rows while the
    score baseline is VALIDATION scores, and on this fixture even 'none' has a
    score PSI near 0.25 with no input shift (see the W3 score-PSI limitation)."""
    _, _, none = predict_scenario(tiered, "none")
    _, _, up = predict_scenario(tiered, "price_up")
    assert up["thresholds"]["psi"] == 0.25
    assert up["feature_psi"]["monthly_charges"] > 0.25
    assert "feature_psi:monthly_charges" in alert_names(up)
    assert up["retrain_review_recommended"] is True
    # The same fixture without the shift: no feature drift at all.
    assert max(none["feature_psi"].values()) < 0.05
    assert not [a for a in none["alerts"] if a["metric"] == "feature_psi"]


# ---------------------------------------------------------------------------
# 2. Small samples
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["none", "contract_mix"])
def test_small_batches_are_insufficient_samples_without_drift_alerts(shared, scenario):
    _, m, rec = predict_scenario(shared, scenario, n=50)
    assert rec["insufficient_sample"] is True and m["n_scored"] < rec["thresholds"]["min_batch_n"]
    assert rec["feature_psi"] is None and rec["score_psi"] is None
    assert not [a for a in rec["alerts"] if a["type"] == "drift"]


def test_data_quality_alerts_still_fire_on_small_batches(shared):
    _, _, rec = predict_scenario(shared, "unseen_payment", n=100, seed=3)
    assert rec["insufficient_sample"] is True
    assert {a["type"] for a in rec["alerts"]} == {"data_quality"}


# ---------------------------------------------------------------------------
# 3. Feature PSI uses the TRAIN baseline; score PSI the VALIDATION baseline
# ---------------------------------------------------------------------------
def test_feature_and_score_psi_use_their_own_baselines(shared, master, assignments, model):
    df, _, rec = predict_scenario(shared, "contract_mix", batch_id="b-baselines")
    preds = pd.read_parquet(pr.inference_dir(shared["paths"]) / "b-baselines" / pr.PREDICTIONS_FILE)
    x = _to_model_frame(df)  # all rows valid in this scenario
    expected_feat = {c: v["psi"] for c, v in feature_psi(shared["baseline"]["features"], x).items()}
    assert rec["feature_psi"] == pytest.approx(expected_feat)
    p = preds["churn_probability"].to_numpy()
    assert rec["score_psi"] == pytest.approx(score_psi(shared["baseline"]["score"], p))
    # A TRAIN-score baseline would give a different number: validation is what is used.
    data = master.assign(**{ID_COLUMN: master[ID_COLUMN].astype("string")}).merge(
        assignments, on=ID_COLUMN)
    p_train = model.predict_proba(_to_model_frame(data[data["split"] == "train"]))[:, 1]
    assert abs(score_psi(score_baseline(p_train), p) - rec["score_psi"]) > 1e-6
    assert rec["baseline_sources"] == {"features": "train rows (synthetic fixture)",
                                       "score": "validation scores (synthetic fixture)"}


# ---------------------------------------------------------------------------
# 4–5. Exact metric definitions (unit level, handcrafted frames)
# ---------------------------------------------------------------------------
def handcrafted(shared):
    df = sim.generate_traffic(shared["source"], "none", 4, seed=0)
    df = df.astype({"monthly_charges": object, "payment_method": object, "gender": object})
    df.loc[0, "monthly_charges"] = None
    df.loc[1, "monthly_charges"] = ""             # CSV-style blank counts as missing
    df.loc[2, "gender"] = "   "                    # whitespace-only counts as missing
    preds = pd.DataFrame({"churn_probability": [0.9, 0.1], "churn_flag": [True, False]})
    rejects = pd.DataFrame({"customer_id": ["a", "a", "b", "c"],
                            "reason_code": ["not_allowed", "null_not_allowed",
                                            "unseen_category", "not_allowed"],
                            "column": ["payment_method", "monthly_charges",
                                       "contract", "senior_citizen"],
                            "message": ["x"] * 4}, index=[0, 0, 1, 3])
    manifest = {"batch_id": "b-unit", "release_id": RELEASE_ID, "run_id": RUN_ID,
                "model_sha256": "m", "n_input": 4, "n_scored": 2, "n_rejected_rows": 3,
                "reject_counts_by_reason": {"not_allowed": 2, "null_not_allowed": 1,
                                            "unseen_category": 1},
                "reject_counts_unit": "reject records"}
    scored = _to_model_frame(sim.generate_traffic(shared["source"], "none", 2, seed=1))
    return dict(manifest=manifest, input_frame=df, predictions=preds, rejects=rejects,
                scored_features=scored, baseline=shared["baseline"])


def test_metric_definitions_are_exact(shared):
    rec = mon.batch_record(**handcrafted(shared))
    assert rec["null_rate"]["monthly_charges"] == 0.5 and rec["null_rate"]["gender"] == 0.25
    assert rec["null_rate"]["contract"] == 0.0
    assert set(rec["null_rate"]) == set(REQUIRED_COLUMNS)
    # unseen share: rows (not records) with unseen/not_allowed on CATEGORICAL columns only;
    # row 3's not_allowed is on senior_citizen (not categorical) -> excluded.
    assert rec["unseen_rows_by_column"] == {"contract": 1, "payment_method": 1}
    assert rec["unseen_share"] == 2 / 4
    # reject rate uses rows (3/4), never the record total (4).
    assert rec["reject_rate"] == 3 / 4
    assert sum(rec["reject_counts_by_reason"].values()) == 4
    assert rec["flag_rate"] == 0.5
    assert rec["flag_rate_diff"] == pytest.approx(0.5 - rec["baseline_flag_rate"])
    assert rec["insufficient_sample"] is True and rec["feature_psi"] is None


def test_empty_batch_has_undefined_rates_not_zeros(shared):
    _, _, rec = predict_scenario(shared, "none", n=1, batch_id="b-one")  # n>=1 by D7 cap
    assert rec["n_input"] == 1
    kw = handcrafted(shared)
    kw["input_frame"] = kw["input_frame"].iloc[0:0]
    kw["predictions"] = kw["predictions"].iloc[0:0]
    kw["rejects"] = kw["rejects"].iloc[0:0]
    kw["scored_features"] = kw["scored_features"].iloc[0:0]
    kw["manifest"] = {**kw["manifest"], "n_input": 0, "n_scored": 0, "n_rejected_rows": 0,
                      "reject_counts_by_reason": {}}
    rec = mon.batch_record(**kw)
    assert rec["reject_rate"] is None and rec["unseen_share"] is None
    assert rec["flag_rate"] is None and rec["alerts"] == []


# ---------------------------------------------------------------------------
# 6–7. Privacy; batches never merged by customer ID
# ---------------------------------------------------------------------------
def test_records_and_report_hold_no_ids_and_no_raw_values(shared):
    df, _, _ = predict_scenario(shared, "unseen_payment", batch_id="b-privacy")
    text = mon.records_path(shared["paths"], RELEASE_ID).read_text()
    report = mon.render_report(RELEASE_ID, mon.read_records(shared["paths"], RELEASE_ID))
    for cid in df[ID_COLUMN]:
        assert cid not in text and cid not in report
    for raw in RAW_CATEGORY_VALUES:
        assert raw not in text and raw not in report
    # "customer_id" may appear only as a COLUMN NAME (its null rate), never as a field
    # that carries identifiers.
    rec = record_for(shared["paths"], "b-privacy")
    assert "customer_id" not in rec
    assert json.dumps({k: v for k, v in rec.items() if k != "null_rate"}).count("customer_id") == 0
    assert isinstance(rec["null_rate"]["customer_id"], float)


def test_repeated_ids_across_batches_stay_separate(shared):
    a, _, ra = predict_scenario(shared, "none", n=300, seed=21, batch_id="b-rep-1")
    b, _, rb = predict_scenario(shared, "nulls", n=300, seed=21, batch_id="b-rep-2")
    assert list(a[ID_COLUMN]) == list(b[ID_COLUMN])          # same synthetic IDs
    assert ra["n_input"] == rb["n_input"] == 300             # not merged into 600
    assert ra["reject_rate"] == 0.0 and rb["reject_rate"] > 0
    report = mon.render_report(RELEASE_ID, mon.read_records(shared["paths"], RELEASE_ID))
    assert "| b-rep-1 |" in report and "| b-rep-2 |" in report


# ---------------------------------------------------------------------------
# 8–9. Append-only JSONL; reports never append
# ---------------------------------------------------------------------------
def minimal_record(batch_id):
    return {"batch_id": batch_id, "release_id": RELEASE_ID, "alerts": []}


def test_jsonl_is_append_only_and_refuses_duplicate_batches(tmp_path):
    from churn_pipeline.config import Paths
    paths = Paths(tmp_path / "data")
    path = mon.append_record(paths, minimal_record("b1"))
    first = path.read_bytes()
    mon.append_record(paths, minimal_record("b2"))
    assert path.read_bytes().startswith(first)
    before = path.read_bytes()
    with pytest.raises(mon.MonitoringError, match="already has a monitoring record"):
        mon.append_record(paths, minimal_record("b1"))
    assert path.read_bytes() == before


def test_a_failed_append_leaves_no_partial_line(tmp_path, monkeypatch):
    from churn_pipeline.config import Paths
    paths = Paths(tmp_path / "data")
    path = mon.append_record(paths, minimal_record("b1"))
    before = path.read_bytes()
    real_write = mon.os.write

    def torn_write(fd, data):
        real_write(fd, data[: len(data) // 2])        # half a line reaches the disk...
        raise OSError("simulated disk failure")      # ...then the write fails
    monkeypatch.setattr(mon.os, "write", torn_write)
    with pytest.raises(OSError):
        mon.append_record(paths, minimal_record("b2"))
    assert path.read_bytes() == before                 # rolled back to the old length
    monkeypatch.undo()
    assert [r["batch_id"] for r in mon.read_records(paths, RELEASE_ID)] == ["b1"]


def test_a_torn_log_fails_closed(tmp_path):
    from churn_pipeline.config import Paths
    paths = Paths(tmp_path / "data")
    path = mon.append_record(paths, minimal_record("b1"))
    with path.open("ab") as f:
        f.write(b'{"batch_id": "b2"')                   # no newline: a torn record
    with pytest.raises(mon.MonitoringError, match="incomplete record"):
        mon.read_records(paths, RELEASE_ID)
    with pytest.raises(mon.MonitoringError):
        mon.append_record(paths, minimal_record("b3"))


def test_report_generation_never_appends(shared, capsys):
    predict_scenario(shared, "none", n=300, seed=5, batch_id="b-report")
    path = mon.records_path(shared["paths"], RELEASE_ID)
    before = path.read_bytes()
    for _ in range(2):
        assert cli.main(["--data-dir", str(shared["paths"].data_dir), "monitor-report",
                         "--release", RELEASE_ID]) == 0
    assert path.read_bytes() == before
    assert "| b-report |" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 10. Fail closed on missing or invalid inputs
# ---------------------------------------------------------------------------
def tamper_baseline(paths, how):
    run_dir = paths.models / RUN_ID
    m = json.loads((run_dir / "manifest.json").read_text())
    if how == "missing_file":
        (run_dir / "baseline.json").unlink()
    elif how == "hash_mismatch":
        with (run_dir / "baseline.json").open("a") as f:
            f.write(" ")
    elif how == "no_manifest_hash":
        del m["artifacts"]["baseline.json"]
        (run_dir / "manifest.json").write_text(json.dumps(m))
    else:  # structurally invalid, but correctly hashed
        b = json.loads((run_dir / "baseline.json").read_text())
        del b["performance"]
        (run_dir / "baseline.json").write_text(json.dumps(b))
        m["artifacts"]["baseline.json"] = hashlib.sha256(
            (run_dir / "baseline.json").read_bytes()).hexdigest()
        (run_dir / "manifest.json").write_text(json.dumps(m))


@pytest.mark.parametrize("how", ["missing_file", "hash_mismatch", "no_manifest_hash",
                                 "invalid_structure"])
def test_bad_baseline_fails_closed_before_reading_input_or_writing(fresh, tmp_path, how,
                                                                   monkeypatch):
    tamper_baseline(fresh, how)
    reads = []
    monkeypatch.setattr(pr, "read_input", lambda p: reads.append(p))
    with pytest.raises(mon.MonitoringError):
        pr.run_predict(fresh, tmp_path / "in.parquet", "production", KEY, batch_id="b-x")
    assert reads == []
    assert not pr.inference_dir(fresh).exists()
    assert not mon.records_path(fresh, RELEASE_ID).exists()


@pytest.mark.parametrize("drop", ["churn_probability", "churn_flag"])
def test_record_with_missing_prediction_columns_fails_closed(shared, drop):
    kw = handcrafted(shared)
    kw["predictions"] = kw["predictions"].drop(columns=[drop])
    with pytest.raises(mon.MonitoringError, match=drop):
        mon.batch_record(**kw)


def test_record_with_inconsistent_counts_fails_closed(shared):
    kw = handcrafted(shared)
    kw["manifest"] = {**kw["manifest"], "n_scored": 99}
    with pytest.raises(mon.MonitoringError, match="disagree"):
        mon.batch_record(**kw)


# ---------------------------------------------------------------------------
# 11–12. No performance claims; never retrain
# ---------------------------------------------------------------------------
def test_records_make_no_performance_claims(shared):
    _, _, rec = predict_scenario(shared, "contract_mix", batch_id="b-claims")
    assert rec["labels_available"] is False
    flat = json.dumps(rec).lower()
    for word in ("auc", "precision", "recall", "accuracy", "degrad", "performance_drop"):
        assert word not in flat
    report = mon.render_report(RELEASE_ID, [rec])
    assert mon.NO_LABELS_NOTE in report and mon.RETRAIN_NOTE in report
    assert "not automatically" in mon.RETRAIN_NOTE.lower() or "nothing is retrained" in \
        mon.RETRAIN_NOTE.lower()


def test_alerts_never_trigger_training(shared, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("monitoring must never retrain")
    for name in ("run_training", "train_models", "tune", "build_ensemble"):
        monkeypatch.setattr(modeling, name, forbidden)
    _, _, rec = predict_scenario(shared, "contract_mix", batch_id="b-noretrain")
    assert rec["retrain_review_recommended"] is True                 # alerts fired
    source = Path(mon.__file__).read_text()
    for token in ("modeling", "run_training", "train_models", ".fit("):
        assert token not in source


# ---------------------------------------------------------------------------
# 13. Predict + monitoring failure contract
# ---------------------------------------------------------------------------
def test_append_failure_keeps_the_committed_batch_and_says_so(fresh, tmp_path, monkeypatch,
                                                              capsys, master, assignments):
    source = sim.load_traffic_source(fresh, suffix="SYNTR", expected_count=len(master))
    inp = tmp_path / "in.parquet"
    sim.generate_traffic(source, "none", 300, 1).to_parquet(inp, index=False)

    def fail(*a, **k):
        raise OSError("simulated monitoring disk failure")
    monkeypatch.setattr(mon.os, "write", fail)
    with pytest.raises(mon.MonitoringAppendError) as exc:
        pr.run_predict(fresh, inp, "production", KEY, batch_id="b-partial")
    d = pr.inference_dir(fresh) / "b-partial"
    assert "WAS written" in str(exc.value) and exc.value.batch_dir == str(d)
    assert sorted(p.name for p in d.iterdir()) == sorted(
        [pr.PREDICTIONS_FILE, pr.REJECTS_FILE, pr.MANIFEST_FILE])   # batch kept, complete
    monkeypatch.undo()
    assert [r["batch_id"] for r in mon.read_records(fresh, RELEASE_ID)] == []


def test_successful_predict_appends_exactly_one_record(fresh, tmp_path, master):
    source = sim.load_traffic_source(fresh, suffix="SYNTR", expected_count=len(master))
    inp = tmp_path / "in.parquet"
    sim.generate_traffic(source, "none", 300, 2).to_parquet(inp, index=False)
    m = pr.run_predict(fresh, inp, "production", KEY, batch_id="b-one-rec")
    recs = mon.read_records(fresh, RELEASE_ID)
    assert [r["batch_id"] for r in recs] == ["b-one-rec"]
    assert recs[0]["n_input"] == m["n_input"] and recs[0]["timings_s"] == json.loads(
        (pr.inference_dir(fresh) / "b-one-rec" / pr.MANIFEST_FILE).read_text())["timings_s"]


# ---------------------------------------------------------------------------
# 14. CLI
# ---------------------------------------------------------------------------
@pytest.fixture
def cli_key(tmp_path, monkeypatch):
    f = write_key_file(tmp_path / "guard.key", KEY.hex())
    monkeypatch.setenv(rel.GUARD_KEY_ENV, str(f))


def test_cli_predict_exit_3_when_only_monitoring_fails(fresh, tmp_path, master, monkeypatch,
                                                       cli_key, capsys):
    source = sim.load_traffic_source(fresh, suffix="SYNTR", expected_count=len(master))
    inp = tmp_path / "in.parquet"
    sim.generate_traffic(source, "none", 300, 3).to_parquet(inp, index=False)
    monkeypatch.setattr(mon.os, "write", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert cli.main(["--data-dir", str(fresh.data_dir), "predict", "--input", str(inp)]) == 3
    err = capsys.readouterr().err
    assert err.startswith("partial:") and "WAS written" in err
    assert len(list(pr.inference_dir(fresh).iterdir())) == 1


def test_cli_monitor_report_errors_exit_2(fresh, capsys):
    argv = ["--data-dir", str(fresh.data_dir), "monitor-report", "--release"]
    assert cli.main(argv + [RELEASE_ID]) == 2                 # no records yet
    assert "no monitoring records" in capsys.readouterr().err
    path = mon.records_path(fresh, RELEASE_ID)
    path.parent.mkdir(parents=True)
    path.write_text("not json\n")
    assert cli.main(argv + [RELEASE_ID]) == 2                 # corrupt log
    assert cli.main(argv + ["../escape"]) == 2               # invalid release id


@pytest.mark.parametrize("argv", [["monitor-report", "--release", RELEASE_ID],  # no data dir
                                  ["monitor-report"]])                          # no release
def test_cli_monitor_report_required_arguments(argv, tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
