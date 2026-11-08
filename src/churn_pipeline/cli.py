"""Command-line entry point: ``churn-pipeline <command>``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_PATHS, Paths
from .download import download_ibm_csv
from .features import SplitConfig, run_features
from .features import render_markdown as render_feature_report
from .pipeline import run
from .summary import render_markdown
from .synthetic import write_synthetic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="churn-pipeline", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_PATHS.data_dir)
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

    args = parser.parse_args(argv)
    paths = Paths(args.data_dir)

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
        report = run_features(paths, SplitConfig(seed=args.seed))
        print(render_feature_report(report, report["input_sha256"]))
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
    elif args.command == "compare":
        import json
        from .evaluation import compare_runs
        print(json.dumps(compare_runs(paths, args.champion, args.challenger), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
