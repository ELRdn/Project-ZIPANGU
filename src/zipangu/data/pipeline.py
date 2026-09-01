"""Orchestration for the local, resumable ZIPANGU curation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .audit import audit_location, load_audit_summary, write_audit_summary
from .dedup import run_dedup_sqlite
from .discovery import (
    DatasetLocation,
    discover_sources,
    inventory_location,
    load_source_policies,
    policies_as_dict,
)
from .reports import (
    metadata_only_inventory,
    write_cpt_report,
    write_dataset_reports,
    write_final_recommendation,
    write_html_dashboard,
    write_inventory_reports,
)


@dataclass(frozen=True)
class PipelineContext:
    repo_root: Path
    policy_path: Path
    policies: tuple[Any, ...]
    locations: tuple[DatasetLocation, ...]
    audit_locations: tuple[DatasetLocation, ...]
    selected_dataset_ids: tuple[str, ...] | None
    inventories: Mapping[str, Mapping[str, Any]]
    quality_config: Mapping[str, Any]
    dedup_config: Mapping[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _artifact_dir(root: Path) -> Path:
    path = root / "reports" / "dataset_audit" / "_artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _runtime_dir(root: Path) -> Path:
    path = root / "data" / "manifests" / "_runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stage_marker(root: Path, stage: str) -> Path:
    path = _runtime_dir(root) / "stages" / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _is_completed(
    root: Path,
    stage: str,
    resume: bool,
    expected: Mapping[str, Any] | None = None,
) -> bool:
    if not resume:
        return False
    marker = _stage_marker(root, stage)
    if not marker.is_file():
        return False
    if expected is None:
        return True
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, Mapping) and all(payload.get(key) == value for key, value in expected.items())


def _mark_completed(root: Path, stage: str, details: Mapping[str, Any] | None = None) -> None:
    payload = {"stage": stage, "completed": True, **(dict(details or {}))}
    _stage_marker(root, stage).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_context(
    repo_root: str | Path,
    *,
    dataset_root: str | Path | None = None,
    dataset_ids: list[str] | tuple[str, ...] | None = None,
) -> PipelineContext:
    root = Path(repo_root).resolve()
    policy_path, policies, policy_payload = load_source_policies(root / "configs" / "datasets" / "source_policies.yaml")
    resolved_root, locations = discover_sources(
        policies,
        root,
        explicit_root=dataset_root,
        policy_payload=policy_payload,
    )
    if resolved_root:
        print(f"DATASET_ROOT: {resolved_root}")
    else:
        print("DATASET_ROOT: MISSING")
    all_locations = tuple(locations)
    selected_dataset_ids = None
    audit_locations = all_locations
    if dataset_ids is not None:
        normalized_ids = tuple(dict.fromkeys(str(dataset_id) for dataset_id in dataset_ids))
        known_ids = {location.policy.dataset_id for location in all_locations}
        unknown_ids = sorted(set(normalized_ids) - known_ids)
        if unknown_ids:
            raise ValueError(f"unknown dataset IDs: {', '.join(unknown_ids)}")
        selected_dataset_ids = tuple(sorted(normalized_ids))
        selected_set = set(selected_dataset_ids)
        audit_locations = tuple(location for location in all_locations if location.policy.dataset_id in selected_set)
        print(f"DATASET_SCOPE: {', '.join(selected_dataset_ids) or 'none'}")
    inventories: dict[str, Mapping[str, Any]] = {}
    quality_path = root / "configs" / "datasets" / "quality.yaml"
    dedup_path = root / "configs" / "datasets" / "dedup.yaml"
    quality_config = _load_yaml(quality_path) if quality_path.is_file() else {}
    dedup_config = _load_yaml(dedup_path) if dedup_path.is_file() else {}
    return PipelineContext(
        repo_root=root,
        policy_path=policy_path,
        policies=tuple(policies),
        locations=all_locations,
        audit_locations=audit_locations,
        selected_dataset_ids=selected_dataset_ids,
        inventories=inventories,
        quality_config=quality_config,
        dedup_config=dedup_config,
    )


def stage_discover(context: PipelineContext, *, resume: bool = False) -> dict[str, Any]:
    if _is_completed(context.repo_root, "discover", resume):
        payload = json.loads(_stage_marker(context.repo_root, "discover").read_text(encoding="utf-8"))
        print("RESUME: discover skipped (completed marker present)")
        return payload
    inventories: dict[str, dict[str, Any]] = {}
    for location in context.locations:
        print(f"INVENTORY dataset={location.policy.dataset_id}")
        inventories[location.policy.dataset_id] = inventory_location(location)
    audit_summaries = load_audit_summary(context.repo_root)
    write_inventory_reports(context.repo_root, context.locations, inventories, audit_summaries=audit_summaries)
    artifact = _artifact_dir(context.repo_root) / "inventory.json"
    safe_inventories = {dataset_id: metadata_only_inventory(item) for dataset_id, item in inventories.items()}
    artifact.write_text(json.dumps(safe_inventories, ensure_ascii=False, indent=2), encoding="utf-8")
    found = sum(1 for item in inventories.values() if item.get("found"))
    result = {"stage": "discover", "datasets": len(inventories), "found": found, "missing": len(inventories) - found}
    _mark_completed(context.repo_root, "discover", result)
    return result


def _load_inventory(context: PipelineContext) -> dict[str, Mapping[str, Any]]:
    path = _artifact_dir(context.repo_root) / "inventory.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return value
    return {location.policy.dataset_id: inventory_location(location) for location in context.locations}


def stage_audit(context: PipelineContext, *, resume: bool = False, max_rows: int | None = None, progress_every: int = 5000) -> dict[str, Any]:
    expected = {"max_rows": max_rows, "dataset_ids": list(context.selected_dataset_ids) if context.selected_dataset_ids is not None else None}
    if _is_completed(context.repo_root, "audit", resume, expected=expected):
        payload = json.loads(_stage_marker(context.repo_root, "audit").read_text(encoding="utf-8"))
        print("RESUME: audit skipped (completed marker present)")
        return payload
    inventory = _load_inventory(context)
    summaries = []
    for location in context.audit_locations:
        print(f"AUDIT dataset={location.policy.dataset_id} mode={location.policy.audit_mode}")
        summary = audit_location(
            location,
            repo_root=context.repo_root,
            inventory=inventory.get(location.policy.dataset_id),
            quality_config=context.quality_config,
            max_rows=max_rows,
            progress_every=progress_every,
        )
        summaries.append(summary)
    path = write_audit_summary(context.repo_root, summaries, merge=context.selected_dataset_ids is not None)
    merged_summaries = load_audit_summary(context.repo_root)
    write_inventory_reports(
        context.repo_root,
        context.locations,
        inventory,
        audit_summaries=merged_summaries,
    )
    cpt_summary = next((item for item in summaries if item.get("dataset_id") == "awesome_japanese_corpus"), None)
    if cpt_summary:
        write_cpt_report(context.repo_root, cpt_summary)
    result = {
        "stage": "audit",
        "datasets": len(summaries),
        "summary_path": str(path),
        "max_rows": max_rows,
        "dataset_ids": list(context.selected_dataset_ids) if context.selected_dataset_ids is not None else None,
    }
    _mark_completed(context.repo_root, "audit", result)
    return result


def stage_dedup(context: PipelineContext, *, resume: bool = False) -> dict[str, Any]:
    expected = {"dataset_ids": list(context.selected_dataset_ids) if context.selected_dataset_ids is not None else None}
    if _is_completed(context.repo_root, "dedup", resume, expected=expected):
        payload = json.loads(_stage_marker(context.repo_root, "dedup").read_text(encoding="utf-8"))
        print("RESUME: dedup skipped (completed marker present)")
        return payload
    near = context.dedup_config.get("near", {})
    summary = run_dedup_sqlite(
        context.repo_root / "data" / "manifests" / "_runtime" / "audit",
        context.repo_root / "data" / "manifests" / "_runtime" / "dedup",
        hamming_threshold=int(near.get("hamming_threshold", 3)),
        bucket_bits=int(near.get("bucket_bits", 16)),
        near_max_records=int(near.get("max_records", 250000)),
        dataset_ids=context.selected_dataset_ids,
    )
    result = {"stage": "dedup", **summary}
    _mark_completed(context.repo_root, "dedup", result)
    return result


def stage_contamination(context: PipelineContext, *, resume: bool = False) -> dict[str, Any]:
    expected = {"dataset_ids": list(context.selected_dataset_ids) if context.selected_dataset_ids is not None else None}
    if _is_completed(context.repo_root, "contamination", resume, expected=expected):
        payload = json.loads(_stage_marker(context.repo_root, "contamination").read_text(encoding="utf-8"))
        print("RESUME: contamination skipped (completed marker present)")
        return payload
    # Evaluation datasets are intentionally not present in the local raw root.
    # Keep this explicit rather than calling the network or claiming clean.
    summaries = load_audit_summary(context.repo_root)
    if context.selected_dataset_ids is not None:
        selected = set(context.selected_dataset_ids)
        summaries = {dataset_id: item for dataset_id, item in summaries.items() if dataset_id in selected}
    candidate_count = sum(int(item.get("eligible_rows") or 0) for item in summaries.values())
    summary = {
        "stage": "contamination",
        "status": "not_checked_missing_eval_source",
        "evaluation_records": 0,
        "candidate_records": candidate_count,
        "exact_matches": 0,
        "substring_matches": 0,
        "ngram_matches": 0,
        "manual_review": "required",
        "network_used": False,
        "dataset_ids": list(context.selected_dataset_ids) if context.selected_dataset_ids is not None else None,
    }
    path = _artifact_dir(context.repo_root) / "contamination_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _mark_completed(context.repo_root, "contamination", summary)
    return summary


def _load_recipes(context: PipelineContext) -> dict[str, dict[str, Any]]:
    recipes: dict[str, dict[str, Any]] = {}
    for path in sorted((context.repo_root / "configs" / "datasets").glob("c-i-*.yaml")):
        value = _load_yaml(path)
        recipes[path.stem] = value
    return recipes


def stage_reports(
    context: PipelineContext,
    *,
    resume: bool = False,
    dedup_summary: Mapping[str, Any] | None = None,
    contamination_summary: Mapping[str, Any] | None = None,
    pilot_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if _is_completed(context.repo_root, "reports", resume):
        payload = json.loads(_stage_marker(context.repo_root, "reports").read_text(encoding="utf-8"))
        print("RESUME: reports skipped (completed marker present)")
        return payload
    if dedup_summary is None:
        dedup_path = context.repo_root / "data" / "manifests" / "_runtime" / "dedup" / "dedup_summary.json"
        if dedup_path.is_file():
            loaded = json.loads(dedup_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                dedup_summary = loaded
    if contamination_summary is None:
        contamination_path = _artifact_dir(context.repo_root) / "contamination_summary.json"
        if contamination_path.is_file():
            loaded = json.loads(contamination_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                contamination_summary = loaded
    if pilot_status is None:
        pilot_path = _artifact_dir(context.repo_root) / "pilot_status.json"
        if pilot_path.is_file():
            loaded = json.loads(pilot_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                pilot_status = loaded
    summaries = load_audit_summary(context.repo_root)
    policy_dicts = policies_as_dict(context.policies)
    write_dataset_reports(context.repo_root, policy_dicts, summaries, dedup_summary)
    recipes = _load_recipes(context)
    pilot_status = pilot_status or {
        "status": "BLOCKED",
        "reason": "registry candidates remain quarantine; no approved manifest-backed source",
    }
    final = write_final_recommendation(
        context.repo_root,
        policy_dicts,
        summaries,
        recipes,
        dedup_summary=dedup_summary,
        contamination_summary=contamination_summary,
        pilot_status=pilot_status,
    )
    dashboard = write_html_dashboard(context.repo_root, summaries, recipes, dedup_summary=dedup_summary)
    result = {"stage": "reports", "final_report": str(final), "html_report": str(dashboard)}
    _mark_completed(context.repo_root, "reports", result)
    return result



def stage_pilot(context: PipelineContext, *, resume: bool = False) -> dict[str, Any]:
    """Build the default 1M pilot only when every approval gate passes."""

    if _is_completed(context.repo_root, "pilot", resume):
        payload = json.loads(_stage_marker(context.repo_root, "pilot").read_text(encoding="utf-8"))
        print("RESUME: pilot skipped (completed marker present)")
        return payload

    from .pilot import build_pilot

    recipe_path = context.repo_root / "configs" / "datasets" / "c-i-v1-jp-heavy-curated.yaml"
    if not recipe_path.is_file():
        recipe_path = context.repo_root / "configs" / "datasets" / "c-i-v0-jp-heavy.yaml"
    output_dir = context.repo_root / "data" / "processed" / "c-i" / "pilot-1m"
    locations = {location.policy.dataset_id: location for location in context.locations}
    try:
        pilot = build_pilot(
            context.repo_root,
            recipe_path,
            locations,
            target_tokens=1_000_000,
            output_dir=output_dir,
            output_format="parquet",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        pilot = {
            "status": "ERROR",
            "recipe": str(recipe_path),
            "target_tokens": 1_000_000,
            "records_selected": 0,
            "issues": [str(exc)],
            "auto_approved": False,
        }
    result = {"stage": "pilot", **pilot}
    artifact = _artifact_dir(context.repo_root) / "pilot_status.json"
    artifact.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _mark_completed(context.repo_root, "pilot", result)
    print(f"PILOT: {result.get('status', 'ERROR')} ({len(result.get('issues', []))} issues)")
    return result


def run_pipeline(
    context: PipelineContext,
    *,
    stage: str = "all",
    resume: bool = False,
    max_rows: int | None = None,
    progress_every: int = 5000,
) -> int:
    if stage in {"discover", "inventory"}:
        stage_discover(context, resume=resume)
        return 0
    if stage in {"audit", "canonicalize", "filter", "quality"}:
        stage_audit(context, resume=resume, max_rows=max_rows, progress_every=progress_every)
        return 0
    if stage == "dedup":
        stage_dedup(context, resume=resume)
        return 0
    if stage == "contamination":
        result = stage_contamination(context, resume=resume)
        return 0 if result.get("status") == "clear" else 1 if result.get("status") == "error" else 2
    if stage in {"rank", "reports"}:
        stage_reports(context, resume=resume)
        return 0
    if stage == "pilot":
        result = stage_pilot(context, resume=resume)
        return {"PASS": 0, "ERROR": 1, "BLOCKED": 2}.get(str(result.get("status")), 1)
    if stage != "all":
        raise ValueError(f"unsupported pipeline stage: {stage}")

    stage_discover(context, resume=resume)
    stage_audit(context, resume=resume, max_rows=max_rows, progress_every=progress_every)
    dedup_summary = stage_dedup(context, resume=resume)
    contamination_summary = stage_contamination(context, resume=resume)
    pilot_status = stage_pilot(context, resume=resume)
    stage_reports(
        context,
        resume=resume,
        dedup_summary=dedup_summary,
        contamination_summary=contamination_summary,
        pilot_status=pilot_status,
    )
    if str(pilot_status.get("status")) == "ERROR":
        return 1
    print("PILOT: BLOCKED (no registry entry was auto-approved)")
    return 2
