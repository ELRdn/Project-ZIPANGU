from __future__ import annotations

import argparse
import json

from _bootstrap import PROJECT_ROOT
from zipangu.data.pipeline import build_context, stage_contamination


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local-only contamination gate.")
    parser.add_argument("--dataset-root")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    context = build_context(PROJECT_ROOT, dataset_root=args.dataset_root)
    result = stage_contamination(context, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("status") != "clear" else 0


if __name__ == "__main__":
    raise SystemExit(main())
