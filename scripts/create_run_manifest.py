from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import yaml


def command_output(command: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "python": sys.version,
        "platform": platform.platform(),
        "train_config": yaml.safe_load(
            Path(args.train_config).read_text(encoding="utf-8")
        ),
        "dataset_config": yaml.safe_load(
            Path(args.dataset_config).read_text(encoding="utf-8")
        ),
        "environment": {
            "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
            "nvidia_smi": command_output(["nvidia-smi"]),
        },
        "status": "created",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
