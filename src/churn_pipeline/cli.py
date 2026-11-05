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
    return 0


if __name__ == "__main__":
    sys.exit(main())
