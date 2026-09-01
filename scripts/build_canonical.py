from __future__ import annotations

import argparse

from _bootstrap import PROJECT_ROOT
from zipangu.data.pipeline import build_context, stage_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the content-free canonical audit index.")
    parser.add_argument("--dataset-root")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()
    context = build_context(PROJECT_ROOT, dataset_root=args.dataset_root)
    print(stage_audit(context, max_rows=args.max_rows, progress_every=max(1, args.progress_every)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
