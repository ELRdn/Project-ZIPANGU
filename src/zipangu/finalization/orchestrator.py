"""Public finalization runner API and exit-code policy."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .baseline import run_baseline
from .core import (
    DEFAULT_PATHS,
    FinalizationBlocked,
    RuntimeStore,
    config_hash,
    input_fingerprint,
    load_yaml,
)
from ..models import DEFAULT_MODEL_ID, ModelSpec, load_model_spec
from .formatting import stage_format
from .reporting import stage_reports
from .runner import (
    STAGE_ORDER,
    _read_json,
    _stage_contamination,
    _stage_eval,
    _stage_preflight,
    _stage_tokenize,
)
from .stages import (
    _stage_candidate,
    _stage_fable,
    _stage_license,
    _stage_nemotron,
    _stage_quality,
)


class FinalizationRunner:
    """Run independent local stages with fail-closed status reporting."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        dataset_root: str | Path = DEFAULT_PATHS["dataset_root"],
        eval_root: str | Path = DEFAULT_PATHS["eval_root"],
        cache_root: str | Path = DEFAULT_PATHS["cache_root"],
        model_root: str | Path = DEFAULT_PATHS["model_root"],
        temp_root: str | Path = DEFAULT_PATHS["temp_root"],
        config_path: str | Path | None = None,
        model_id: str | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.dataset_root = Path(dataset_root)
        self.eval_root = Path(eval_root)
        self.cache_root = Path(cache_root)
        self.model_root = Path(model_root)
        self.temp_root = Path(temp_root)
        self.config_path = Path(config_path) if config_path else self.repo_root / "configs" / "pretrain" / "finalization.yaml"
        self.config = load_yaml(self.config_path)
        selected_model_id = model_id or self.config.get("model_id") or DEFAULT_MODEL_ID
        self.model_spec: ModelSpec = load_model_spec(self.repo_root, str(selected_model_id))
        self.model_id = self.model_spec.id
        self.base_model_id = self.model_spec.base_model_id
        self.tokenizer_revision = self.model_spec.revision
        self.shared_candidate_root = self.repo_root / "data" / "processed" / "generation-i" / "pilot-1m-candidate"
        self.training_format_root = self.repo_root / "data" / "processed" / "models" / self.model_spec.artifact_slug
        self.runtime = RuntimeStore(self.temp_root / "finalization" / f"{self.model_spec.artifact_slug}-runtime.json")

    def _locations(self):
        from .runner import _locations

        return _locations(self.repo_root, self.dataset_root)

    @staticmethod
    def _read_json(path: Path, default: Any = None) -> Any:
        return _read_json(path, default)

    def _fingerprint(self, stage: str) -> str:
        inputs: list[Path] = [
            self.config_path,
            self.repo_root / "configs" / "datasets" / "registry.yaml",
            self.repo_root / "configs" / "models",
        ]
        if stage in {"preflight", "tokenize", "nemotron", "fable", "quality", "candidate", "contamination", "license"}:
            inputs.append(self.dataset_root)
        if stage in {"eval", "contamination", "baseline", "reports"}:
            inputs.append(self.eval_root)
        if stage in {"tokenize", "baseline"}:
            inputs.append(self.cache_root)
            inputs.append(self.model_root)
        return input_fingerprint(inputs, root=self.repo_root)

    def _stage_callable(self, stage: str) -> Callable[["FinalizationRunner"], dict[str, Any]]:
        return {
            "preflight": _stage_preflight,
            "eval": _stage_eval,
            "contamination": _stage_contamination,
            "tokenize": _stage_tokenize,
            "nemotron": _stage_nemotron,
            "fable": _stage_fable,
            "quality": _stage_quality,
            "license": _stage_license,
            "candidate": _stage_candidate,
            "format": stage_format,
            "baseline": self._baseline_stage,
            "reports": stage_reports,
        }[stage]

    def _baseline_stage(self, runner: "FinalizationRunner") -> dict[str, Any]:
        return run_baseline(
            runner.repo_root,
            eval_root=runner.eval_root,
            cache_root=runner.cache_root,
            model_root=runner.model_root,
            model_id=runner.model_id,
            configured_revision=runner.config.get("baseline", {}).get("revision") or runner.model_spec.revision,
            resume=runner.resume,
            max_hours=float(runner.config.get("baseline", {}).get("max_hours", 12)),
        )

    def _run_one(self, stage: str, *, resume: bool) -> str:
        fingerprint = self._fingerprint(stage)
        cfg_hash = config_hash(self.config)
        if not self.runtime.start(stage, fingerprint=fingerprint, cfg_hash=cfg_hash, resume=resume):
            status = str(self.runtime.stage(stage).get("status", "PASS"))
            print(f"{stage}: {status} (resume-skip)")
            return status
        try:
            result = self._stage_callable(stage)(self)
            status = str(result.get("status", "PASS"))
            if status not in {"PASS", "BLOCKED", "PARTIAL", "ERROR"}:
                status = "ERROR"
                result.setdefault("errors", []).append("invalid_stage_status")
        except KeyboardInterrupt:
            result = {"status": "PARTIAL", "blocked_reasons": ["interrupted_after_checkpoint"], "artifacts": []}
            status = "PARTIAL"
        except FinalizationBlocked as exc:
            result = {"status": "BLOCKED", "blocked_reasons": [str(exc)], "artifacts": []}
            status = "BLOCKED"
        except (OSError, PermissionError) as exc:
            result = {"status": "BLOCKED", "blocked_reasons": [f"filesystem:{type(exc).__name__}"], "artifacts": []}
            status = "BLOCKED"
        except Exception as exc:  # pragma: no cover - exercised by integration failures
            result = {"status": "ERROR", "errors": [f"{type(exc).__name__}: {exc}"], "artifacts": []}
            status = "ERROR"
        self.runtime.finish(
            stage,
            status,
            artifacts=result.get("artifacts", []),
            errors=result.get("errors", []),
            blocked_reasons=result.get("blocked_reasons", []),
            details=result.get("details"),
        )
        print(f"{stage}: {status}")
        return status

    def run(self, stage: str = "all", *, resume: bool = False) -> int:
        if stage != "all" and stage not in STAGE_ORDER:
            raise ValueError(f"unknown finalization stage: {stage}")
        self.resume = bool(resume)
        stages = STAGE_ORDER if stage == "all" else (stage,)
        statuses = {name: self._run_one(name, resume=resume) for name in stages}
        if any(value == "ERROR" for value in statuses.values()):
            code = 1
        elif any(value in {"BLOCKED", "PARTIAL"} for value in statuses.values()):
            code = 2
        else:
            code = 0
        print(f"MODEL_ID: {self.model_id}")
        print(f"BASE_MODEL_ID: {self.base_model_id}")
        print("TRAINING_ALLOWED: false")
        print(f"FINALIZATION_EXIT_CODE: {code}")
        return code


def run_finalization(**kwargs: Any) -> int:
    stage = str(kwargs.pop("stage", "all"))
    resume = bool(kwargs.pop("resume", False))
    return FinalizationRunner(**kwargs).run(stage, resume=resume)
