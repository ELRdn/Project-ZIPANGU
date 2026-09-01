from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_validator(mode: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-B", "scripts/validate_registry.py", "--mode", mode],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_structure_mode_passes_current_config_shape() -> None:
    result = run_validator("structure")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STRUCTURE: PASS" in result.stdout
    assert "NOTICE [training_source_not_approved]" in result.stdout


def test_all_mode_blocks_unapproved_current_sources() -> None:
    result = run_validator("all")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "STRUCTURE: PASS" in result.stdout
    assert "RUNNABLE: BLOCKED" in result.stdout
    assert "target_policy_pending" in result.stdout
