"""CLI for ZIPANGU's local overnight finalization stages."""

from __future__ import annotations

import argparse

from _bootstrap import PROJECT_ROOT
from zipangu.finalization.core import DEFAULT_PATHS
from zipangu.models import DEFAULT_MODEL_ID
from zipangu.finalization.orchestrator import FinalizationRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare ZIPANGU eval fingerprints, token statistics, candidates, and baseline handoff."
    )
    parser.add_argument(
        "--stage",
        choices=(
            "preflight",
            "eval",
            "contamination",
            "tokenize",
            "nemotron",
            "fable",
            "quality",
            "license",
            "candidate",
            "format",
            "baseline",
            "reports",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_PATHS["dataset_root"]))
    parser.add_argument("--eval-root", default=str(DEFAULT_PATHS["eval_root"]))
    parser.add_argument("--cache-root", default=str(DEFAULT_PATHS["cache_root"]))
    parser.add_argument("--model-root", default=str(DEFAULT_PATHS["model_root"]))
    parser.add_argument("--temp-root", default=str(DEFAULT_PATHS["temp_root"]))
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="canonical ZIPANGU model id")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "pretrain" / "finalization-k-i.yaml"))
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    runner = FinalizationRunner(
        repo_root=PROJECT_ROOT,
        dataset_root=args.dataset_root,
        eval_root=args.eval_root,
        cache_root=args.cache_root,
        model_root=args.model_root,
        temp_root=args.temp_root,
        config_path=args.config,
        model_id=args.model,
    )
    return runner.run(args.stage, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
