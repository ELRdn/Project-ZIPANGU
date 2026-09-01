from __future__ import annotations

import argparse

from _bootstrap import PROJECT_ROOT
from zipangu.data.pipeline import build_context, stage_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local dataset rows with canonicalization and hard filters.")
    parser.add_argument("--dataset-root")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    context = build_context(PROJECT_ROOT, dataset_root=args.dataset_root)
    result = stage_audit(
        context,
        resume=args.resume,
        max_rows=args.max_rows,
        progress_every=max(1, args.progress_every),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
