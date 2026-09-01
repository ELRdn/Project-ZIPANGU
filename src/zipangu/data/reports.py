"""Tracked Markdown/CSV/HTML reports built from lightweight audit artifacts."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .discovery import DatasetLocation, location_to_manifest
from .sampling import recipe_mass_plan


def _reports_dir(repo_root: Path) -> Path:
    path = repo_root / "reports" / "dataset_audit"
    path.mkdir(parents=True, exist_ok=True)
    return path


_METADATA_ONLY_OMIT_KEYS = frozenset({"samples"})


def metadata_only_inventory(value: Any) -> Any:
    """Return an inventory projection that cannot carry sampled row payloads."""

    if isinstance(value, Mapping):
        return {
            key: metadata_only_inventory(item)
            for key, item in value.items()
            if key not in _METADATA_ONLY_OMIT_KEYS
        }
    if isinstance(value, list):
        return [metadata_only_inventory(item) for item in value]
    if isinstance(value, tuple):
        return [metadata_only_inventory(item) for item in value]
    return value


def write_inventory_reports(
    repo_root: str | Path,
    locations: Iterable[DatasetLocation],
    inventories: Mapping[str, Mapping[str, Any]],
    audit_summaries: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Path, Path, Path]:
    root = Path(repo_root)
    reports = _reports_dir(root)
    inventory_path = reports / "00_inventory.md"
    csv_path = reports / "source_inventory.csv"
    raw_manifest_path = root / "data" / "manifests" / "raw_sources.json"
    raw_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    audit_summaries = audit_summaries or {}
    lines = [
        "# ZIPANGU Dataset Inventory",
        "",
        "This is a metadata-only inventory. Raw dataset files remain outside the repository and are read-only.",
        "",
    ]
    fields = [
        "dataset_id",
        "repo",
        "local_path",
        "found",
        "physical_file_count",
        "logical_file_count",
        "total_bytes",
        "logical_total_bytes",
        "row_count",
        "row_count_known",
        "unknown_row_count_files",
        "file_formats",
        "configs",
        "splits",
        "revision",
        "license_metadata",
        "language_metadata",
        "source_category",
        "source_metadata",
        "schema_summary",
        "teacher_source_fields",
        "teacher_counts",
        "length_median_chars",
        "length_p95_chars",
        "length_observed_rows",
        "token_median_estimate",
        "token_p95_estimate",
        "audit_scope",
        "adapter_warnings",
    ]
    rows: list[dict[str, Any]] = []
    for location in locations:
        item = metadata_only_inventory(dict(inventories.get(location.policy.dataset_id, {})))
        manifest[location.policy.dataset_id] = location_to_manifest(location, item)
        audit = dict(audit_summaries.get(location.policy.dataset_id, {}))
        char_stats = audit.get("char_stats") or item.get("length_profile", {})
        token_stats = audit.get("token_stats") or {}
        row = {field: item.get(field) for field in fields}
        row["file_formats"] = ",".join(str(value) for value in item.get("file_formats", []))
        row["configs"] = ",".join(str(value) for value in item.get("configs", []))
        row["splits"] = ",".join(str(value) for value in item.get("splits", []))
        row["license_metadata"] = json.dumps(item.get("card_metadata", {}).get("license", "unknown"), ensure_ascii=False)
        row["language_metadata"] = json.dumps(item.get("language_metadata", item.get("card_metadata", {}).get("language", [])), ensure_ascii=False)
        row["source_category"] = location.policy.category
        row["source_metadata"] = json.dumps(item.get("source_metadata", {}), ensure_ascii=False, sort_keys=True)
        row["schema_summary"] = json.dumps(item.get("schema", {}), ensure_ascii=False, sort_keys=True)
        row["teacher_source_fields"] = json.dumps(item.get("teacher_source_fields", []), ensure_ascii=False)
        row["teacher_counts"] = json.dumps(audit.get("teacher_counts", {}), ensure_ascii=False, sort_keys=True)
        row["length_median_chars"] = char_stats.get("median")
        row["length_p95_chars"] = char_stats.get("p95")
        row["length_observed_rows"] = char_stats.get("observed_values", char_stats.get("observed_rows"))
        row["token_median_estimate"] = token_stats.get("median_estimate")
        row["token_p95_estimate"] = token_stats.get("p95_estimate")
        row["audit_scope"] = audit.get("audit_scope", "not_run")
        row["adapter_warnings"] = ";".join(str(value) for value in item.get("adapter_warnings", []))
        rows.append(row)
        lines.extend(
            [
                f"## {location.policy.dataset_id}",
                "",
                f"- repository: `{location.policy.repo}`",
                f"- local path: `{item.get('local_path') or 'MISSING'}`",
                f"- found: `{bool(item.get('found'))}`",
                f"- files (physical / logical): `{item.get('physical_file_count', item.get('file_count', 0))}` / `{item.get('logical_file_count', item.get('file_count', 0))}`",
                f"- bytes (physical / logical): `{item.get('total_bytes', 0)}` / `{item.get('logical_total_bytes', item.get('total_bytes', 0))}`",
                f"- rows: `{item.get('row_count', 0)}` (known=`{bool(item.get('row_count_known', False))}`, unknown files=`{item.get('unknown_row_count_files', 0)}`)",
                f"- formats: `{', '.join(str(value) for value in item.get('file_formats', [])) or 'unknown'}`",
                f"- configs: `{', '.join(str(value) for value in item.get('configs', [])) or 'unknown'}`",
                f"- splits: `{', '.join(str(value) for value in item.get('splits', [])) or 'unknown'}`",
                f"- revision: `{item.get('revision', 'unknown')}`",
                f"- license metadata: `{json.dumps(item.get('card_metadata', {}).get('license', 'unknown'), ensure_ascii=False)}`",
                f"- language metadata: `{json.dumps(item.get('language_metadata', item.get('card_metadata', {}).get('language', [])), ensure_ascii=False)}`",
                f"- source category: `{location.policy.category}`",
                f"- teacher/source fields: `{', '.join(str(value) for value in item.get('teacher_source_fields', [])) or 'none detected'}`",
                f"- length profile: `{json.dumps(item.get('length_profile', {}), ensure_ascii=False)}`",
                f"- audit scope: `{audit_summaries.get(location.policy.dataset_id, {}).get('audit_scope', 'not_run')}`",
                f"- adapter warnings: `{'; '.join(str(value) for value in item.get('adapter_warnings', [])) or 'none'}`",
                "",
                "### Schema",
                "",
                "```json",
                json.dumps(item.get("schema", {}), ensure_ascii=False, indent=2),
                "```",
                "",
                "### Sample policy",
                "",
                "Raw row samples are intentionally omitted from tracked inventory reports.",
                "",
            ]
        )
    inventory_path.write_text("\n".join(lines), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    raw_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return inventory_path, csv_path, raw_manifest_path


def _pct(value: int | float, total: int | float) -> str:
    return f"{(100.0 * value / total):.2f}%" if total else "0.00%"


def _bar_rows(mapping: Mapping[str, Any], total: int) -> str:
    return ", ".join(f"{key}={value} ({_pct(value, total)})" for key, value in sorted(mapping.items())) or "none"


def write_dataset_reports(
    repo_root: str | Path,
    policies: Iterable[Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    dedup_summary: Mapping[str, Any] | None = None,
) -> list[Path]:
    reports = _reports_dir(Path(repo_root)) / "datasets"
    reports.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    dedup_summary = dedup_summary or {}
    for policy in policies:
        dataset_id = str(policy.get("id") or policy.get("dataset_id"))
        summary = summaries.get(dataset_id, {})
        raw = int(summary.get("raw_rows") or 0)
        valid = int(summary.get("valid_rows") or 0)
        quality = summary.get("quality", {})
        lines = [
            f"# Dataset audit — {dataset_id}",
            "",
            f"- repository: `{policy.get('repo')}`",
            f"- provisional tier: **{policy.get('recommended_tier', 'QUARANTINE')}**",
            f"- recommended usage: {policy.get('recommended_usage', 'requires_review')}",
            f"- recommended maximum weight: `{policy.get('recommended_max_weight', 0)}`",
            f"- audit scope: `{summary.get('audit_scope', 'not_run')}`",
            "",
            "## Counts",
            "",
            f"- raw rows: `{raw}`",
            f"- rows scanned: `{summary.get('rows_scanned', 0)}`",
            f"- valid/eligible after hard filter: `{valid}` ({_pct(valid, summary.get('rows_scanned', 0))})",
            f"- rejected: `{summary.get('rejected_rows', 0)}`",
            f"- excluded eval/test/validation rows: `{summary.get('excluded_eval_rows', 0)}`",
            f"- estimated eligible token mass: `{summary.get('eligible_token_count', 0)}` (tokenizer_pending=true)",
            "",
            "## Language / task / teacher",
            "",
            f"- language: {_bar_rows(summary.get('language_counts', {}), summary.get('rows_scanned', 0))}",
            f"- categories: {_bar_rows(summary.get('category_counts', {}), summary.get('rows_scanned', 0))}",
            f"- teachers: {_bar_rows(summary.get('teacher_counts', {}), summary.get('rows_scanned', 0))}",
            "",
            "## Quality",
            "",
            "The score is deterministic structural triage only; it does not verify factual correctness.",
            "",
            f"- mean / median / p10 / p90: `{quality.get('mean', 'n/a')}` / `{quality.get('median', 'n/a')}` / `{quality.get('p10', 'n/a')}` / `{quality.get('p90', 'n/a')}`",
            f"- quality method: `{quality.get('method', 'deterministic_structural_heuristic_v1')}`",
            "",
            "## Rejections and risks",
            "",
            f"- reasons: {_bar_rows(summary.get('rejection_reasons', {}), summary.get('rejected_rows', 0))}",
            f"- license expected: `{policy.get('license_expected', 'unknown')}`",
            f"- license metadata: `{json.dumps(summary.get('provenance', {}).get('license_metadata', 'unknown'), ensure_ascii=False)}`",
            f"- source revision: `{summary.get('provenance', {}).get('source_revision', 'unknown')}`",
            f"- contamination: `{summary.get('provenance', {}).get('contamination_status', 'not_checked_missing_eval_source')}`",
            f"- vendor_specific: `{bool(policy.get('vendor_specific'))}`",
            "",
            "## Recommendation",
            "",
            "This is a review priority, not an approval. Registry status is not changed by the pipeline.",
            "",
        ]
        path = reports / f"{dataset_id}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
    return paths


def write_final_recommendation(
    repo_root: str | Path,
    policies: Iterable[Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    recipes: Mapping[str, Mapping[str, Any]],
    *,
    dedup_summary: Mapping[str, Any] | None = None,
    contamination_summary: Mapping[str, Any] | None = None,
    pilot_status: Mapping[str, Any] | None = None,
) -> Path:
    reports = _reports_dir(Path(repo_root))
    policy_list = list(policies)
    lines = [
        "# ZIPANGU Dataset Curation — Final Recommendation",
        "",
        "## Executive summary",
        "",
        "Use the Japanese instruction, extraction and math sources as the first review priorities; keep frontier traces as a small support bucket. Do not approve or train yet: the registry remains quarantine, the local evaluation sources are unavailable for contamination clearance, and the base-model LoRA target policy is still pending.",
        "",
        "## Dataset tier list",
        "",
    ]
    for policy in policy_list:
        dataset_id = str(policy.get("id") or policy.get("dataset_id"))
        summary = summaries.get(dataset_id, {})
        lines.append(
            f"- **{policy.get('recommended_tier', 'QUARANTINE')}** `{dataset_id}` — scanned `{summary.get('rows_scanned', 0)}`; eligible `{summary.get('eligible_rows', 0)}`; usage: {policy.get('recommended_usage', 'requires_review')}"
        )
    first_wave_ids = (
        "math_japanese_8k",
        "ace_reason_math_japanese",
        "magpie_sft_v1",
        "extraction_wiki_ja",
        "nemotron_sft_multilingual_v2",
        "fable_5_premium",
    )
    lines.extend(
        [
            "",
            "## First-wave full audit",
            "",
            "The following six sources were fully scanned. Counts are audit results, not approval decisions; `eligible` means the deterministic hard filter passed.",
            "",
            "| Dataset | Raw rows | Scanned | Eligible | Rejected | Median chars | P95 chars | Quality mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset_id in first_wave_ids:
        summary = summaries.get(dataset_id, {})
        char_stats = summary.get("char_stats", {})
        quality = summary.get("quality", {})
        lines.append(
            "| `{}` | {:,} | {:,} | {:,} | {:,} | {} | {} | {} |".format(
                dataset_id,
                int(summary.get("raw_rows") or 0),
                int(summary.get("rows_scanned") or 0),
                int(summary.get("eligible_rows") or 0),
                int(summary.get("rejected_rows") or 0),
                char_stats.get("median", "n/a"),
                char_stats.get("p95", "n/a"),
                quality.get("mean", "n/a"),
            )
        )
    lines.extend(
        [
            "",
            "## Recommended JP-heavy mix",
            "",
            "Configured token-mass targets (estimated tokens; tokenizer_pending=true):",
            "",
        ]
    )
    for name, recipe in recipes.items():
        if "jp-heavy" not in name:
            continue
        plan = recipe_mass_plan(recipe, summaries)
        lines.extend([f"### {name}", "", "```json", json.dumps(plan, ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## Recommended balanced mix", ""])
    for name, recipe in recipes.items():
        if "balanced" not in name:
            continue
        plan = recipe_mass_plan(recipe, summaries)
        lines.extend([f"### {name}", "", "```json", json.dumps(plan, ensure_ascii=False, indent=2), "```", ""])
    lines.extend(
        [
            "## Data rejected",
            "",
            "Rejections are recorded as eligibility=false plus exclusion_reasons in the runtime audit index; raw rows are never deleted.",
            "",
            "## Duplicate analysis",
            "",
            f"- exact: `{json.dumps(dedup_summary or {}, ensure_ascii=False)}`",
            "- near duplicate screening uses SimHash/LSH buckets and is capped/configured for scalable review; it is not an O(N²) comparison.",
            "",
            "## Quality and language analysis",
            "",
            "Per-source distributions are in `reports/dataset_audit/datasets/`. The score is structural and factuality_verified remains false.",
            "",
            "## License/provenance risks",
            "",
            "Unknown or mixed license/provenance is marked for review. No provenance placeholder or registry approval was created.",
            "",
            "## Contamination status",
            "",
            f"`{json.dumps(contamination_summary or {'status': 'not_checked_missing_eval_source'}, ensure_ascii=False)}`",
            "",
            "## Pilot readiness",
            "",
            f"`{json.dumps(pilot_status or {'status': 'BLOCKED', 'reason': 'no approved sources'}, ensure_ascii=False)}`",
            "",
            "## Next steps",
            "",
            "1. Complete source revision, row identity, license and contamination review for each intended candidate.",
            "2. Create and review tracked provenance sidecars; only then change selected registry entries to approved.",
            "3. Decide explicit LoRA target_modules and update a new train config; do not overwrite old configs.",
            "4. Re-run the same pipeline and L1 validator, then generate a small pilot.",
            "5. Run untouched-base baseline evaluation before any improvement claim.",
            "",
        ]
    )
    path = reports / "FINAL_RECOMMENDATION.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_cpt_report(repo_root: str | Path, summary: Mapping[str, Any]) -> Path:
    reports = _reports_dir(Path(repo_root))
    path = reports / "cpt_candidate_report.md"
    path.write_text(
        "\n".join(
            [
                "# Future CPT Candidate Report",
                "",
                "`awesome_japanese_corpus` is not an SFT source in this phase. The pipeline uses a bounded streaming sample and Parquet metadata; it does not run a full expensive near-dedup or quality model over the entire corpus.",
                "",
                "```json",
                json.dumps(summary, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_html_dashboard(
    repo_root: str | Path,
    summaries: Mapping[str, Mapping[str, Any]],
    recipes: Mapping[str, Mapping[str, Any]],
    *,
    dedup_summary: Mapping[str, Any] | None = None,
) -> Path:
    reports = _reports_dir(Path(repo_root))
    rows = []
    for dataset_id, summary in sorted(summaries.items()):
        scanned = int(summary.get("rows_scanned") or 0)
        eligible = int(summary.get("eligible_rows") or 0)
        rows.append(
            "<tr>"
            f"<td>{html.escape(dataset_id)}</td>"
            f"<td>{scanned:,}</td>"
            f"<td>{eligible:,}</td>"
            f"<td>{html.escape(_pct(eligible, scanned))}</td>"
            f"<td>{html.escape(str(summary.get('quality', {}).get('mean', 'n/a')))}</td>"
            f"<td>{html.escape(str(summary.get('provenance', {}).get('contamination_status', 'unknown')))}</td>"
            "</tr>"
        )
    recipe_rows = []
    for name, recipe in sorted(recipes.items()):
        mass = recipe.get("sampling_mass", {})
        recipe_rows.append(f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(json.dumps(mass, ensure_ascii=False))}</code></td></tr>")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ZIPANGU dataset audit</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
body {{ margin: 2rem auto; max-width: 1200px; padding: 0 1rem; background: #10141c; color: #e8edf5; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0 2rem; background: #171d28; }}
th,td {{ padding: .65rem; border: 1px solid #313b4c; text-align: left; vertical-align: top; }}
th {{ background: #222c3b; }} code {{ white-space: pre-wrap; color: #a9e6c0; }}
.notice {{ border-left: 4px solid #e1a73a; padding: .8rem 1rem; background: #292318; }}
</style></head><body>
<h1>ZIPANGU dataset audit</h1>
<p class="notice">Read-only audit. Scores are heuristic; contamination is not clear until local eval sources and manual review are available. No registry status was changed.</p>
<h2>Dataset summary</h2>
<table><thead><tr><th>Dataset</th><th>Scanned rows</th><th>Eligible</th><th>Accepted %</th><th>Quality mean</th><th>Contamination</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Recipe sampling mass</h2>
<table><thead><tr><th>Recipe</th><th>Configured token mass</th></tr></thead><tbody>{''.join(recipe_rows)}</tbody></table>
<h2>Dedup</h2><pre>{html.escape(json.dumps(dedup_summary or {}, ensure_ascii=False, indent=2))}</pre>
</body></html>"""
    path = reports / "index.html"
    path.write_text(page, encoding="utf-8")
    return path
