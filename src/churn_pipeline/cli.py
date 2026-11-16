"""Command-line entry point: ``churn-pipeline <command>``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_PATHS, Paths
from .download import download_ibm_csv
from .features import SplitConfig, SplitDriftError, format_drift_report, plan_feature_write, run_features
from .features import render_markdown as render_feature_report
from .pipeline import run
from .summary import render_markdown
from .synthetic import write_synthetic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="churn-pipeline", description=__doc__)
    # Default None = "not given". Every command except simulate-traffic then falls back to
    # DEFAULT_PATHS.data_dir exactly as before; simulate-traffic requires an explicit value.
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="data directory (default: the project's data/; "
                             "simulate-traffic requires it explicitly)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="download and checksum-verify the IBM CSV")

    gen = sub.add_parser("generate-synthetic", help="write synthetic profiles + event batches")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--batches", type=int, default=3, choices=[1, 2, 3],
                     help="how many event batches have 'arrived' (simulate increments)")

    r = sub.add_parser("run", help="run Phase 1: validate, store, ingest events, merge, summarise")
    r.add_argument("--events", type=Path, nargs="*",
                   help="ingest only these event files (default: all *.jsonl in landing dir)")

    feat = sub.add_parser(
        "build-features",
        help="Phase 2: split customer_master, fit preprocessing on train, write features",
    )
    feat.add_argument("--seed", type=int, default=42, help="split random seed")

    # Phase 3. Only `evaluate` reads the test split.
    tr = sub.add_parser("train", help="Phase 3: tune, select, and threshold on train/validation "
                                      "(never reads test)")
    tr.add_argument("--seed", type=int, default=42, help="CV/search/bootstrap seed")
    tr.add_argument("--n-iter", type=int, default=20, help="random-search trials per family")
    tr.add_argument("--fn-cost", type=float, default=5.0,
                    help="cost of a missed churner (illustrative assumption)")
    tr.add_argument("--fp-cost", type=float, default=1.0,
                    help="cost of an unnecessary retention offer (illustrative assumption)")
    tr.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples")

    ev = sub.add_parser("evaluate", help="Phase 3: one-time final evaluation on the test split")
    ev.add_argument("--run", required=True, help="run id under models/")
    ev.add_argument("--force", action="store_true",
                    help="re-read the test split although it was already used (logged)")

    dg = sub.add_parser("diagnose", help="Phase 3: learning diagnostics (train/validation only)")
    dg.add_argument("--run", help="use this run's champion for the drift experiment")
    dg.add_argument("--seed", type=int, default=42)
    dg.add_argument("--n-iter", type=int, default=20)

    cp = sub.add_parser("compare", help="Phase 3: offline champion/challenger on validation")
    cp.add_argument("--champion", required=True)
    cp.add_argument("--challenger", required=True)

    # Phase 4 W1: release registry. Key: $CHURN_HOLDOUT_GUARD_KEY_FILE (outside the repo).
    rl = sub.add_parser("release", help="W1: gate a run and create an immutable release "
                                        "with its holdout guard")
    rl.add_argument("--run", required=True, help="run id under models/")
    rl.add_argument("--waive-dirty-code", metavar="REASON",
                    help="record a D6 provenance waiver for a run trained on uncommitted code")
    rl.add_argument("--waiver-approved-on", metavar="YYYY-MM-DD",
                    help="PM approval date of the waiver (required with --waive-dirty-code)")
    ck = sub.add_parser("check-release", help="W1: re-run release gates (never re-reads the split)")
    ck.add_argument("--release", required=True)
    pm = sub.add_parser("promote", help="W1: gate, move the production pointer, smoke-test; "
                                        "restore the pointer if the smoke test fails")
    pm.add_argument("--release", required=True)
    sub.add_parser("rollback", help="W1: point production back at the previous release")

    # Phase 4 W2 / D7: synthetic traffic from train rows. No defaults that could trigger a
    # real run: suffix, expected count, output dir and batch name must all be given.
    st = sub.add_parser("simulate-traffic", help="W2/D7: resample train rows into synthetic "
                                                 "traffic with certified synthetic IDs")
    # Own dest: a subparser default would otherwise overwrite a parent-level --data-dir.
    st.add_argument("--data-dir", dest="sim_data_dir", type=Path, default=None, metavar="DIR",
                    help="REQUIRED for this command (here or before 'simulate-traffic'); "
                         "there is no default data directory")
    st.add_argument("--suffix", required=True, help="pre-approved synthetic ID suffix")
    st.add_argument("--expected-count", required=True, type=int,
                    help="expected number of real customer IDs (collision check)")
    st.add_argument("--scenario", required=True,
                    choices=["none", "price_up", "contract_mix", "unseen_payment", "nulls"])
    st.add_argument("--n", required=True, type=int, help="rows to generate (max 10,000)")
    st.add_argument("--seed", type=int, default=42)
    st.add_argument("--out-dir", required=True, type=Path)
    st.add_argument("--name", required=True, help="batch name; an existing batch is never "
                                                 "overwritten")

    args = parser.parse_args(argv)
    if args.command == "simulate-traffic":
        # Checked before any handler runs: no stat, no read, no output without a data dir.
        given = {d for d in (args.data_dir, args.sim_data_dir) if d is not None}
        if not given:
            st.error("--data-dir is required for simulate-traffic (no default data directory)")
        if len(given) > 1:
            st.error("conflicting --data-dir values; give it once")
        paths = Paths(given.pop())
    else:
        paths = Paths(args.data_dir if args.data_dir is not None else DEFAULT_PATHS.data_dir)

    if args.command == "download":
        p = download_ibm_csv(paths.ibm_csv)
        print(f"OK: {p} (checksum verified)")
    elif args.command == "generate-synthetic":
        for p in write_synthetic(paths, seed=args.seed, batches=args.batches):
            print(f"wrote {p}")
    elif args.command == "run":
        summary = run(paths, event_files=args.events)
        print(render_markdown(summary))
        print(f"Summary written to {paths.summary_md}")
    elif args.command == "build-features":
        plan = plan_feature_write(paths.features)  # read-only; taken before the build writes
        try:
            report = run_features(paths, SplitConfig(seed=args.seed))
        except SplitDriftError as e:
            print(format_drift_report(e.report), file=sys.stderr)
            return 2
        print(render_feature_report(report, report["input_sha256"]))
        print("Files written:")
        for w in plan:
            print(f"  {w['action']}: {w['path']}")
        print(f"Features written to {paths.features}")
    elif args.command == "train":
        from .modeling import TrainConfig, run_training
        out = run_training(paths, TrainConfig(seed=args.seed, n_iter=args.n_iter, fn_cost=args.fn_cost,
                                              fp_cost=args.fp_cost, n_boot=args.n_boot))
        print((out["run_dir"] / "report.md").read_text())
        print(f"Run {out['run_id']} written to {out['run_dir']} (test split not read)")
    elif args.command == "evaluate":
        from .evaluation import TestSetAlreadyUsedError, run_final_evaluation
        try:
            out = run_final_evaluation(paths, args.run, force=args.force)
        except TestSetAlreadyUsedError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 2
        print((out["output_dir"] / "test_report.md").read_text())
        print(f"Test evaluation written to {out['output_dir']}")
    elif args.command == "diagnose":
        from .diagnostics import run_diagnostics
        from .modeling import TrainConfig
        out = run_diagnostics(paths, TrainConfig(seed=args.seed, n_iter=args.n_iter), run_id=args.run)
        print((out["dir"] / "diagnostics.md").read_text())
        print(f"Diagnostics written to {out['dir']}")
    elif args.command in ("release", "check-release", "promote", "rollback"):
        return _release_command(paths, args)
    elif args.command == "simulate-traffic":
        return _simulate_command(paths, args)
    elif args.command == "compare":
        import json
        from .evaluation import compare_runs
        print(json.dumps(compare_runs(paths, args.champion, args.challenger), indent=2, default=str))
    return 0


def _release_command(paths: Paths, args) -> int:
    from . import release as rel
    from .inference import ModelIntegrityError
    try:
        key = rel.load_guard_key()
        if args.command == "release":
            m = rel.create_release(paths, args.run, key, waiver_reason=args.waive_dirty_code,
                                   waiver_approved_on=args.waiver_approved_on)
            print(f"Release {m['release_id']} created for run {m['run_id']}")
            for gate, result in m["gates"].items():
                print(f"  {gate}: {result['status']}")
        elif args.command == "check-release":
            for gate, result in rel.check_release(paths, args.release, key).items():
                print(f"  {gate}: {result['status']}")
            print(f"Release {args.release}: all gates passed")
        elif args.command == "promote":
            p = rel.promote(paths, args.release, key)
            print(f"Production -> {p['release_id']} (previous: {p['previous_release_id']})")
        else:
            p = rel.rollback(paths, key)
            print(f"Production -> {p['release_id']} (rolled back)")
    except (rel.ReleaseError, ModelIntegrityError, FileExistsError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


def _simulate_command(paths: Paths, args) -> int:
    from . import simulate as sim
    from .traffic import SuffixCertificationError
    try:
        sim.ensure_batch_absent(args.out_dir, args.name)  # before any data is read
        source = sim.load_traffic_source(paths, suffix=args.suffix,
                                         expected_count=args.expected_count)
        df = sim.generate_traffic(source, args.scenario, args.n, args.seed)
        path = sim.write_traffic(df, args.out_dir, args.name, scenario=args.scenario,
                                 seed=args.seed, certificate=source.certificate)
    except (SuffixCertificationError, sim.TrafficSourceError, FileExistsError,
            FileNotFoundError, ValueError, OSError) as e:
        print(f"refused: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(f"Synthetic traffic written to {path} ({len(df)} rows, scenario {args.scenario}, "
          f"IDs NNNN-{source.certificate.suffix})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
