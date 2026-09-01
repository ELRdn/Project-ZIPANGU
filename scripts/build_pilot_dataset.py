from __future__ import annotations

import argparse
import json

from _bootstrap import PROJECT_ROOT
from zipangu.data.pipeline import build_context
from zipangu.data.pilot import build_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an approved-only ZIPANGU pilot dataset.")
    parser.add_argument("--dataset-root")
    parser.add_argument("--recipe", default="configs/datasets/c-i-v1-jp-heavy-curated.yaml")
    parser.add_argument("--target-tokens", type=int, default=1_000_000)
    parser.add_argument("--output-dir")
    parser.add_argument("--format", choices=("parquet", "jsonl.gz"), default="parquet")
    args = parser.parse_args()
    if args.target_tokens <= 0:
        parser.error("--target-tokens must be positive")
    context = build_context(PROJECT_ROOT, dataset_root=args.dataset_root)
    locations = {location.policy.dataset_id: location for location in context.locations}
    result = build_pilot(
        PROJECT_ROOT,
        args.recipe,
        locations,
        target_tokens=args.target_tokens,
        output_dir=args.output_dir,
        output_format=args.format,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return {"PASS": 0, "ERROR": 1, "BLOCKED": 2}.get(str(result.get("status")), 1)


if __name__ == "__main__":
    raise SystemExit(main())
