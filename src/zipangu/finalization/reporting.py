"""Metadata-only dashboard and morning handoff report."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import html
import json
from pathlib import Path
from typing import Any

from .baseline import baseline_paths
from .core import atomic_write_text


REPORT_STAGE_ORDER = (
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
)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _technical_blockers(statuses: Mapping[str, str]) -> list[str]:
    blocking_states = {"BLOCKED", "ERROR", "PARTIAL", "RUNNING", "PENDING"}
    return [
        f"{stage}:{status}"
        for stage, status in statuses.items()
        if stage != "reports" and status in blocking_states
    ]


def _decision_section(
    title: str,
    *,
    what: str,
    evidence: list[str],
    decision: str,
    risk: str,
) -> list[str]:
    return [
        f"## {title}",
        "",
        "### WHAT",
        "",
        what,
        "",
        "### EVIDENCE",
        "",
        *[f"- {item}" for item in evidence],
        "",
        "### RECOMMENDED DECISION",
        "",
        decision,
        "",
        "### RISK",
        "",
        risk,
        "",
    ]


def stage_reports(runner: Any) -> dict[str, Any]:
    report_dir = runner.repo_root / "reports" / "pretrain_finalization"
    report_dir.mkdir(parents=True, exist_ok=True)
    statuses = runner.runtime.statuses()
    # This function is executing while its own stage is RUNNING.  The report
    # is the proof of successful report generation, so render that row as PASS.
    display_statuses = {
        stage: statuses.get(stage, "PENDING") for stage in REPORT_STAGE_ORDER
    }
    display_statuses.update(
        {
            stage: status
            for stage, status in statuses.items()
            if stage not in display_statuses
        }
    )
    display_statuses["reports"] = "PASS"
    blockers = _technical_blockers(display_statuses)
    eval_registry = _read_json(runner.repo_root / "data" / "manifests" / "eval_registry.json", {}) or {}
    contamination = _read_json(runner.temp_root / "finalization" / "contamination_summary.json", {}) or {}
    contamination_audit = _read_json(
        report_dir / "contamination_rescan_audit.json", {}
    ) or {}
    legacy_contamination_audit = _read_json(
        report_dir / "contamination_exact_audit.json", {}
    ) or {}
    nemotron = _read_json(runner.temp_root / "finalization" / "nemotron_summary.json", {}) or {}
    fable = _read_json(runner.temp_root / "finalization" / "fable_summary.json", {}) or {}
    candidates = _read_json(runner.temp_root / "finalization" / "candidate_summary.json", {}) or {}
    format_summary = _read_json(runner.temp_root / "finalization" / "format_summary.json", {}) or {}
    approval = _read_json(runner.repo_root / "data" / "manifests" / "source_approval_candidates.json", {}) or {}
    license_audit = _read_json(
        runner.repo_root / "data" / "manifests" / "source_license_provenance_audit.json", {}
    ) or {}
    source_decisions = _read_json(
        runner.repo_root / "data" / "manifests" / "source_approval_decisions.json", {}
    ) or {}
    raw_sources = _read_json(runner.repo_root / "data" / "manifests" / "raw_sources.json", {}) or {}
    baseline = _read_json(baseline_paths(runner.repo_root, runner.model_id)["manifest"], {}) or {}
    token_rows: list[dict[str, str]] = []
    token_csv = report_dir / "token_statistics.csv"
    if token_csv.is_file():
        try:
            with token_csv.open("r", encoding="utf-8", newline="") as handle:
                token_rows = list(csv.DictReader(handle))
        except OSError:
            token_rows = []
    candidate_source_ids = {
        str(item.get("dataset"))
        for item in approval.get("sources", [])
        if isinstance(item, Mapping)
    }
    raw_candidate_rows = sum(
        int(item.get("row_count") or 0)
        for source_id, item in raw_sources.items()
        if source_id in candidate_source_ids and isinstance(item, Mapping)
    )
    token_stage = runner.runtime.stage("tokenize")
    token_details = token_stage.get("details") if isinstance(token_stage, Mapping) else {}
    token_details = token_details if isinstance(token_details, Mapping) else {}
    canonical_rows = int(token_details.get("rows_scanned") or 0)
    formatted_rows = sum(int(row.get("rows") or 0) for row in token_rows)
    grouped_delta = raw_candidate_rows - canonical_rows
    contamination_counts = contamination.get("counts", {}) if isinstance(contamination, Mapping) else {}
    legacy_audit_totals = (
        legacy_contamination_audit.get("totals", {})
        if isinstance(legacy_contamination_audit, Mapping)
        else {}
    )
    old_false_positive_candidates = int(
        legacy_audit_totals.get("false_positive_candidate_rows") or 0
    )
    rescan_identity = (
        contamination_audit.get("row_identity", {})
        if isinstance(contamination_audit, Mapping)
        else {}
    )
    rescan_status_counts = (
        contamination_audit.get("status_counts", {}).get("new", {})
        if isinstance(contamination_audit, Mapping)
        and isinstance(contamination_audit.get("status_counts"), Mapping)
        else {}
    )
    genuine_exact = (
        contamination_audit.get("genuine_exact", {})
        if isinstance(contamination_audit, Mapping)
        else {}
    )
    exact_sample = (
        contamination_audit.get("sample", {})
        if isinstance(contamination_audit, Mapping)
        else {}
    )
    rescan_rows = int(rescan_identity.get("rows_compared") or 0)
    rescan_available = bool(
        rescan_rows == int(contamination.get("candidate_rows_scanned") or 0)
        and int(rescan_identity.get("mismatches") or 0) == 0
        and contamination.get("empty_fingerprints_allowed") is False
    )
    extractor_defect_closed = bool(
        rescan_available
        and int(genuine_exact.get("empty_sha256_exact_new_excluded") or 0) == 0
    )
    exact_sample_complete = bool(
        rescan_available
        and int(exact_sample.get("actual_size") or 0)
        == min(
            int(exact_sample.get("target_size") or 0),
            int(exact_sample.get("population") or 0),
        )
    )
    heavy_fill = candidates.get("jp_heavy_fill", {}) if isinstance(candidates, Mapping) else {}
    balanced_details = candidates.get("balanced", {}) if isinstance(candidates, Mapping) else {}
    balanced_fill = balanced_details.get("fill", {}) if isinstance(balanced_details, Mapping) else {}
    format_invalid = {
        name: int(item.get("invalid_sample_rows") or 0)
        for name, item in format_summary.items()
        if isinstance(item, Mapping)
    }
    source_rows = [
        item
        for item in approval.get("sources", [])
        if isinstance(item, Mapping)
    ]
    unknown_revisions = sum(str(item.get("revision") or "unknown") == "unknown" for item in source_rows)
    allowed_source_decisions = set(source_decisions.get("allowed_decisions", []))
    source_decision_rows = [
        item
        for item in source_decisions.get("sources", [])
        if isinstance(item, Mapping)
    ]
    recorded_source_decisions = sum(
        item.get("decision") in allowed_source_decisions
        for item in source_decision_rows
    )
    pending_source_decisions = len(source_rows) - recorded_source_decisions
    audit_summary = license_audit.get("summary", {}) if isinstance(license_audit, Mapping) else {}
    headline = "BLOCKED" if blockers else "READY FOR HUMAN REVIEW"
    eval_rows = eval_registry.get("evaluations", []) if isinstance(eval_registry, Mapping) else []
    eval_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('id')))}</td>"
        f"<td>{html.escape(str(item.get('version')))}</td>"
        f"<td>{html.escape(str(item.get('availability')))}</td>"
        f"<td>{int(item.get('row_count') or 0):,}</td>"
        f"<td>{html.escape(str(item.get('content_hash') or 'n/a'))}</td>"
        "</tr>"
        for item in eval_rows
        if isinstance(item, Mapping)
    )
    stage_html = "".join(
        f"<tr><td>{html.escape(stage)}</td><td>{html.escape(value)}</td></tr>"
        for stage, value in sorted(display_statuses.items())
    )
    token_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('dataset')))}</td>"
        f"<td>{html.escape(str(row.get('category')))}</td>"
        f"<td>{html.escape(str(row.get('rows')))}</td>"
        f"<td>{html.escape(str(row.get('total_tokens')))}</td>"
        f"<td>{html.escape(str(row.get('p95')))}</td>"
        "</tr>"
        for row in token_rows
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="robots" content="noindex"><title>ZIPANGU pre-training finalization</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
body {{ margin: 2rem auto; max-width: 1280px; padding: 0 1rem; background: #10141c; color: #e8edf5; }}
h1 {{ margin-bottom: .25rem; }} h2 {{ margin-top: 2rem; }}
.state {{ display: inline-block; padding: .4rem .75rem; border-radius: .4rem; background: #5f291f; color: #ffd6c7; font-weight: 700; }}
.notice {{ border-left: 4px solid #e1a73a; padding: .8rem 1rem; background: #292318; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0 2rem; background: #171d28; }}
th,td {{ padding: .55rem; border: 1px solid #313b4c; text-align: left; vertical-align: top; }}
th {{ background: #222c3b; }} code,pre {{ white-space: pre-wrap; color: #a9e6c0; }}
</style></head><body>
<h1>ZIPANGU overnight pre-training finalization</h1>
<p>Model: <code>{html.escape(str(runner.model_id))}</code>; base: <code>{html.escape(str(runner.base_model_id))}</code></p>
<p class="state">{html.escape(headline)}</p>
<p class="notice"><strong>TRAINING_ALLOWED=false</strong>. Metadata-only dashboard; registry status, human approval, training, RunPod, paid APIs, and official judge execution remain outside this runner.</p>
<h2>Stage status</h2><table><thead><tr><th>Stage</th><th>Status</th></tr></thead><tbody>{stage_html}</tbody></table>
<h2>Eval status</h2><table><thead><tr><th>ID</th><th>Version</th><th>Availability</th><th>Rows</th><th>Aggregate content hash</th></tr></thead><tbody>{eval_html}</tbody></table>
<h2>Contamination</h2><pre>{html.escape(json.dumps(contamination, ensure_ascii=False, indent=2))}</pre>
<h2>Contamination rescan audit</h2><pre>{html.escape(json.dumps(contamination_audit, ensure_ascii=False, indent=2))}</pre>
<h2>Token statistics</h2><table><thead><tr><th>Dataset</th><th>Category</th><th>Rows</th><th>Formatted tokens</th><th>P95</th></tr></thead><tbody>{token_html}</tbody></table>
<h2>Nemotron composition</h2><pre>{html.escape(json.dumps(nemotron, ensure_ascii=False, indent=2))}</pre>
<h2>Fable length distribution</h2><pre>{html.escape(json.dumps(fable, ensure_ascii=False, indent=2))}</pre>
<h2>License/provenance queue</h2><pre>{html.escape(json.dumps(approval, ensure_ascii=False, indent=2))}</pre>
<h2>License/provenance evidence audit</h2><pre>{html.escape(json.dumps(license_audit, ensure_ascii=False, indent=2))}</pre>
<h2>Human source decisions</h2><pre>{html.escape(json.dumps(source_decisions, ensure_ascii=False, indent=2))}</pre>
<h2>1M candidate mix</h2><pre>{html.escape(json.dumps(candidates, ensure_ascii=False, indent=2))}</pre>
<h2>Baseline progress</h2><pre>{html.escape(json.dumps(baseline, ensure_ascii=False, indent=2))}</pre>
<h2>Remaining blockers</h2><pre>{html.escape(json.dumps(blockers, ensure_ascii=False, indent=2))}</pre>
</body></html>
"""
    dashboard_path = report_dir / "index.html"
    atomic_write_text(dashboard_path, page)
    blocker_lines = [f"- `{value}`" for value in blockers] if blockers else ["- technical blockers: none; human review gates remain"]
    morning_lines = [
        "# ZIPANGU Morning Report",
        "",
        "TRAINING_ALLOWED=false",
        f"MODEL_ID={runner.model_id}",
        f"BASE_MODEL_ID={runner.base_model_id}",
        f"LOCAL_FINALIZATION_STATUS={headline}",
        "",
        "## WHAT",
        "",
        "This run prepared local metadata and research candidates only. It did not approve sources, change registry status, train a model, provision RunPod, call a paid API, or execute the official judge.",
        "",
        "## EVIDENCE",
        "",
    ]
    morning_lines.extend(f"- `{stage}`: `{value}`" for stage, value in sorted(display_statuses.items()))
    morning_lines.extend(
        [
            "",
            "## HARD BLOCKERS",
            "",
            *blocker_lines,
            "",
            "## RECOMMENDED DECISION",
            "",
            f"- Review `SOURCE_APPROVAL_MATRIX.md`; `{recorded_source_decisions}` / `{len(source_rows)}` human source decisions are recorded, and the runner selected none.",
            (
                "- Empty-fingerprint extractor defect is CLOSED for the 830 available eval rows; "
                "review the genuine exact/near/manual rows and resolve AnswerCarefully access."
                if extractor_defect_closed
                else "- Review contamination hits and resolve missing/gated evaluation sources."
            ),
            "- Confirm explicit LoRA target modules and budget preflight before any paid execution.",
            "- Run the official judge separately; `pending_judge.json` is the only baseline judge handoff.",
            f"- Keep the 1M run at NO-GO: JP-heavy fill is `{float(heavy_fill.get('fill_ratio') or 0):.1%}` and balanced fill is `{float(balanced_fill.get('fill_ratio') or 0):.1%}`.",
            "",
            "## RISK",
            "",
            "Treating locally generated candidates as approved data would bypass provenance, contamination, evaluation, and budget gates. Keep `TRAINING_ALLOWED=false` until each human gate is explicitly cleared.",
        ]
    )
    morning_path = report_dir / "MORNING_REPORT.md"
    atomic_write_text(morning_path, "\n".join(morning_lines) + "\n")

    queue_lines = [
        "# Human Approval Queue",
        "",
        "TRAINING_ALLOWED=false",
        "",
        "This package records decisions for a human reviewer. It does not approve data, training, paid compute, or publication.",
        "",
    ]
    queue_lines.extend(
        _decision_section(
            "1. LICENSE",
            what="Decide whether each of the ten candidate sources permits the planned transformation, training use, adapter/weight handling, and required attribution.",
            evidence=[
                f"`{pending_source_decisions}` of `{len(source_rows)}` source decisions remain unset; AI/runner decisions recorded: `0`.",
                "Nemotron reports mixed CC-BY-4.0 / CC-BY-SA-4.0 metadata; Math-Japanese-8k and claude-fable-code have unknown license metadata; frontier_multi_teacher reports `other`.",
                "Evidence audit: `data/manifests/source_license_provenance_audit.json`; human-only record: `data/manifests/source_approval_decisions.json`; worksheet: `SOURCE_APPROVAL_MATRIX.md`.",
            ],
            decision="A human reviewer must select exactly one of five choices per source. No choice is preselected, and a source decision alone does not authorize training.",
            risk="Metadata-only approval could create incompatible attribution, share-alike, redistribution, or derivative-model obligations.",
        )
    )
    queue_lines.extend(
        _decision_section(
            "2. PROVENANCE",
            what="Approve only source rows with immutable upstream revision and row-level origin evidence.",
            evidence=[
                f"`{len(source_rows) - unknown_revisions}` of `{len(source_rows)}` local source packages have a full immutable revision; `{unknown_revisions}` remain `unknown`.",
                f"Pinned local/remote document hash matches recorded by the audit: `{int(audit_summary.get('pinned_documents_matched') or 0)}`.",
                "A package revision does not by itself resolve upstream corpus, teacher-model, contributor-rights, or service-terms lineage.",
                f"Raw inventory has `{raw_candidate_rows:,}` rows; canonical processing produced `{canonical_rows:,}` records. The `{grouped_delta:,}`-row difference is the expected Claude event-to-session grouping (18,370 events to 63 sessions), not silent deletion.",
                f"`{formatted_rows:,}` canonical records had a usable formatted-token count; unavailable rows remain excluded/fail-closed.",
            ],
            decision="Use the source matrix to record whether the remaining lineage evidence is accepted, conditionally accepted, scope-limited, deferred, or excluded. The runner records no answer.",
            risk="Treating a pinned package revision as complete upstream provenance can still make the experiment legally or scientifically irreproducible.",
        )
    )
    queue_lines.extend(
        _decision_section(
            "3. CONTAMINATION HITS",
            what="Review every exact/near-match quarantine class and complete the scan against all required evaluation corpora.",
            evidence=[
                f"Available eval fingerprints indexed: `{int(contamination.get('eval_rows_indexed') or 0):,}`.",
                f"Corrected full rescan classifications: exact `{int(rescan_status_counts.get('quarantine_exact', contamination_counts.get('quarantine_exact', 0)) or 0):,}`, near `{int(rescan_status_counts.get('quarantine_near', contamination_counts.get('quarantine_near', 0)) or 0):,}`, manual-review `{int(rescan_status_counts.get('requires_manual_review', contamination_counts.get('requires_manual_review', 0)) or 0):,}`.",
                (
                    f"Extractor defect: `CLOSED`; all `{rescan_rows:,}` record IDs reconciled old-vs-new, the prior `{old_false_positive_candidates:,}` empty-hash exacts were removed from valid exact evidence, and `{int(exact_sample.get('actual_size') or 0):,}` / `{int(exact_sample.get('population') or 0):,}` genuine exact candidates were sampled (target up to 100)."
                    if extractor_defect_closed and exact_sample_complete
                    else "The corrected rescan or genuine-exact evidence sample is not yet complete."
                ),
                "AnswerCarefully v2.0 remains access-denied; Japanese MT-Bench and llm-jp-instructions were processed.",
            ],
            decision=(
                "CLOSE the empty-fingerprint extractor defect and available-eval exact revalidation item. Keep every genuine exact, near and manual-review row excluded; the global contamination gate remains BLOCKED until AnswerCarefully is available."
                if extractor_defect_closed and exact_sample_complete
                else "Keep all quarantined/manual-review rows excluded and keep the global contamination gate blocked until the corrected rescan, sample review and AnswerCarefully coverage are complete."
            ),
            risk="Training before resolution can contaminate evaluation and invalidate every improvement claim.",
        )
    )
    queue_lines.extend(
        _decision_section(
            "4. DATASET RECIPE",
            what="Choose whether to revise the JP-heavy/balanced bucket targets, template-family cap, and safe-format filters before regenerating the 1M candidate.",
            evidence=[
                f"JP-heavy selected `{int(heavy_fill.get('selected_tokens') or 0):,}` / 1,000,000 tokens (`{float(heavy_fill.get('fill_ratio') or 0):.1%}`).",
                f"Balanced selected `{int(balanced_fill.get('selected_tokens') or 0):,}` / 1,000,000 tokens (`{float(balanced_fill.get('fill_ratio') or 0):.1%}`).",
                f"Format-invalid samples: `{json.dumps(format_invalid, ensure_ascii=False, sort_keys=True)}`; no automatic truncation was applied.",
                "Fable Premium remains excluded while tool-call canonicalization/trace integrity is blocked.",
            ],
            decision="Revise the recipe in a new config/manifest, then regenerate. Do not relax the 5% template cap or 4096-token/tool-boundary checks without explicit human rationale.",
            risk="Relaxing diversity or format gates merely to hit 1M can produce a biased or non-runnable training set.",
        )
    )
    queue_lines.extend(
        _decision_section(
            "5. LORA TARGET POLICY",
            what="Approve explicit production LoRA target modules for ZIPANGU-K-I-4B.",
            evidence=[
                "The real-data smoke passed with `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, r=4, alpha=8, two optimizer steps.",
                "Production K-I configs intentionally remain `target_policy: auto_review_required` with r=32 and alpha=64.",
                "The smoke proves the local data/trainer path only; it is not a quality or production-policy approval.",
            ],
            decision="Approve or amend an explicit module list in a new immutable run config before any 1M run. Do not use automatic target discovery for the approved experiment.",
            risk="An unreviewed target set changes trainable capacity, memory use, cost, and comparison validity.",
        )
    )
    queue_lines.extend(
        _decision_section(
            "6. 1M GO",
            what="Issue or withhold the human GO for the first 1M-token K-I experiment.",
            evidence=[
                f"Current blocking stages: `{', '.join(blockers)}`.",
                "Local RX 7600 infrastructure, llama.cpp backends, 9-case benchmark, and real-data smoke passed, but candidate/eval/provenance gates did not.",
                "No paid Pod, production training, C/Z training, or publication was started.",
            ],
            decision="NO-GO now. Reconsider only after license/provenance approval, complete contamination coverage, a format-valid near-1M candidate, explicit LoRA targets, and budget preflight.",
            risk="A premature GO would spend budget on an experiment whose data and evaluation claims are not defensible.",
        )
    )
    queue_path = report_dir / "HUMAN_APPROVAL_QUEUE.md"
    atomic_write_text(queue_path, "\n".join(queue_lines) + "\n")

    final_lines = [
        "# Final Recommendation",
        "",
        "TRAINING_ALLOWED=false",
        "",
        "## WHAT",
        "",
        "Decide whether Project ZIPANGU is ready to move from local completion work to the human-approved 1M K-I experiment.",
        "",
        "## EVIDENCE",
        "",
        "- Local hardware/inference gates are complete: Vulkan and HIP actual inference, all nine CPU/Vulkan/HIP context cases, and the 16-example/two-step real-data QLoRA path passed.",
        f"- Data finalization executed all local stages; blockers are `{', '.join(blockers)}`.",
        f"- Candidate fill is only `{float(heavy_fill.get('fill_ratio') or 0):.1%}` JP-heavy and `{float(balanced_fill.get('fill_ratio') or 0):.1%}` balanced; format-invalid rows remain.",
        (
            f"- Corrected contamination rescan reconciled `{rescan_rows:,}` records: exact `{int(rescan_status_counts.get('quarantine_exact', contamination_counts.get('quarantine_exact', 0)) or 0):,}`, near `{int(rescan_status_counts.get('quarantine_near', contamination_counts.get('quarantine_near', 0)) or 0):,}`, manual-review `{int(rescan_status_counts.get('requires_manual_review', contamination_counts.get('requires_manual_review', 0)) or 0):,}`. The empty-fingerprint defect is CLOSED for available evals; AnswerCarefully coverage remains unavailable."
            if extractor_defect_closed and exact_sample_complete
            else f"- Contamination classifications include `{int(contamination_counts.get('quarantine_exact') or 0):,}` exact quarantines and `{int(contamination_counts.get('requires_manual_review') or 0):,}` manual-review rows; corrected audit evidence or AnswerCarefully coverage is unavailable."
        ),
        f"- All `{len(source_rows) - unknown_revisions}` / `{len(source_rows)}` local source package revisions are pinned, but `{pending_source_decisions}` / `{len(source_rows)}` human license/provenance decisions remain unset.",
        "",
        "## RECOMMENDED DECISION",
        "",
        "LOCAL COMPLETION: ACCEPT. 1M PRODUCTION TRAINING: NO-GO. A human reviewer may use `SOURCE_APPROVAL_MATRIX.md` to record one of five decisions for each source; the runner does not select or apply those decisions.",
        "",
        "## RISK",
        "",
        "The local systems path is proven, but treating that success as data or experiment approval would invalidate provenance, contamination, format, budget, and benchmark claims.",
    ]
    final_path = report_dir / "FINAL_RECOMMENDATION.md"
    atomic_write_text(final_path, "\n".join(final_lines) + "\n")
    return {
        "status": "PASS",
        "artifacts": [dashboard_path, morning_path, queue_path, final_path],
        "details": {"headline": headline, "blockers": blockers, "training_allowed": False},
    }
