from __future__ import annotations

import argparse

from _bootstrap import PROJECT_ROOT
from zipangu.data.pipeline import build_context, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the resumable local ZIPANGU dataset curation pipeline.")
    parser.add_argument(
        "--stage",
        choices=("discover", "inventory", "audit", "canonicalize", "filter", "quality", "dedup", "contamination", "rank", "reports", "pilot", "all"),
        default="all",
    )
    parser.add_argument("--dataset-root")
    parser.add_argument(
        "--dataset-id",
        dest="dataset_ids",
        action="append",
        help="Limit audit/dedup/contamination to one dataset; repeat for a cohort.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    context = build_context(
        PROJECT_ROOT,
        dataset_root=args.dataset_root,
        dataset_ids=args.dataset_ids,
    )
    return run_pipeline(
        context,
        stage=args.stage,
        resume=args.resume,
        max_rows=args.max_rows,
        progress_every=max(1, args.progress_every),
    )


if __name__ == "__main__":
    raise SystemExit(main())
