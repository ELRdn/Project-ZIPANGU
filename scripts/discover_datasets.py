from __future__ import annotations

import argparse

from _bootstrap import PROJECT_ROOT
from zipangu.data.pipeline import build_context, stage_discover


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover and inventory local ZIPANGU datasets.")
    parser.add_argument("--dataset-root")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    context = build_context(PROJECT_ROOT, dataset_root=args.dataset_root)
    result = stage_discover(context, resume=args.resume)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
