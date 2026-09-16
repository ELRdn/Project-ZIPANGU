# Source Approval Matrix

TRAINING_ALLOWED=false

AUDIT_DATE=2026-09-05

HUMAN_DECISIONS_RECORDED=0/10

This is a human decision worksheet. The AI and finalization runner have selected no source decision. A checked box here does not update the dataset registry, authorize training or paid compute, or set `TRAINING_ALLOWED=true`; applying a completed review requires a separate reviewed manifest/config change and explicit human GO.

## Five allowed decisions

1. `APPROVE_AS_IS` — accept the recorded source scope without extra source-specific conditions.
2. `APPROVE_WITH_OBLIGATIONS` — accept only with recorded attribution, share-alike, notice, redistribution, or other obligations.
3. `APPROVE_SCOPED_SUBSET` — accept only the configs/files/rows/licenses/teachers/use conditions written into `approved_scope`.
4. `DEFER_PENDING_EVIDENCE` — leave unapproved until named evidence gaps are resolved.
5. `EXCLUDE` — exclude the source from the training-data pool.

Select exactly one decision per source and record reviewer, date, rationale, accepted evidence, obligations, and scope in `data/manifests/source_approval_decisions.json`. All boxes and all decision fields are intentionally blank.

## Matrix

| Source | APPROVE AS IS | WITH OBLIGATIONS | SCOPED SUBSET | DEFER | EXCLUDE | Reviewer / date |
| --- | --- | --- | --- | --- | --- | --- |
| `extraction_wiki_ja` | [ ] | [ ] | [ ] | [ ] | [ ] | _unset_ |
| `magpie_sft_v1` | [ ] | [ ] | [ ] | [ ] | [ ] | _unset_ |
| `nemotron_sft_multilingual_v2` | [ ] | [ ] | [ ] | [ ] | [ ] | _unset_ |
| `math_japanese_8k` | [ ] | [ ] | [ ] | [ ] | [ ] | _unset_ |
| `ace_reason_math_japanese` | [ ] | [ ] | [ ] | [ ] | [ ] | _unset_ |
| `gpt_5_6_traces` | [ ] | [ ] | [ ] | [ ] | [ ] | _unset_ |
| `fable_5_5_distillation` | [ ] | [ ] | [ ] | [ ] | [ ] | _unset_ |
| `claude_fable_code` | [ ] | [ ] | [ ] | [ ] | [ ] | _unset_ |
| `frontier_multi_teacher` | [ ] | [ ] | [ ] | [ ] | [ ] | _unset_ |
| `fable_5_premium` | [ ] | [ ] | [ ] | [ ] | [ ] | _unset_ |

## Evidence posture at handoff

These are observations, not recommendations or approvals.

| Source | Package SHA | Declared license evidence | Remaining evidence questions |
| --- | --- | --- | --- |
| `extraction_wiki_ja` | `b385b781defb9bf4266175cd8d7e7c53af0290b8` | card: Apache-2.0 | llm-jp-corpus-v3/Wikipedia revision and attribution chain; teacher revision/terms |
| `magpie_sft_v1` | `4f949e18cfa01d99abe6f92e5629157d69737e6d` | card: Apache-2.0 | CALM3/Qwen teacher revisions, generation terms and row identity |
| `nemotron_sft_multilingual_v2` | `971a252224b75414b1b67c55dbe0446d8b6606a0` | 241,889 CC-BY-4.0 + 128,192 CC-BY-SA-4.0 rows | obligation/derived-model treatment for the selected row mix |
| `math_japanese_8k` | `e86bb58896c228e91acec2cbbb768d37f14ecdb3` | no card license / no license file | grant, Kimi/Gemma revisions and terms, generation and row lineage |
| `ace_reason_math_japanese` | `037b823f6ab85aee4a7de51ca35476c6d1079f0a` | card + `LISENCE`: CC-BY-4.0 | upstream AceReason revision/rows; translation model revision/terms |
| `gpt_5_6_traces` | `07e1745da4cb8220c4811e200bfeb97768290e21` | card: CC-BY-4.0 | upstream revisions, publisher-asserted trace origin, provider-output terms |
| `fable_5_5_distillation` | `15ba38ac01b610cc986b728a2d55fcb8db9c2096` | card: CC-BY-4.0 for curation/merge; upstream licenses retained | partial snapshot; mixed upstream rights; personal-Codex trace restriction; provider terms |
| `claude_fable_code` | `c19fb6831700da833b22d1c9cdac47fe8603685c` | no card license / no license file | contributor rights, redistribution grant and Anthropic terms |
| `frontier_multi_teacher` | `7a3473446840bcc397928cd8183d4b3ba3ca13a7` | card: other; row-level mixed/custom/NC/unknown | 12,045 explicit NC rows; 15,122 unknown rows; stale provenance count/name; service terms |
| `fable_5_premium` | `684cb1f849fe4a1c96f55351e1d7366f9888bb28` | card tag: MIT; no LICENSE file | empty source table; upstream revisions/rights absent; physical-format double count |

Full evidence, pinned links, document hashes, inventory and contamination counts are in `SOURCE_LICENSE_PROVENANCE_AUDIT.md` and `data/manifests/source_license_provenance_audit.json`.

## Human completion record

For each source, fill all applicable fields in the human-owned JSON record:

- `decision`: exactly one allowed value;
- `approved_scope`: required for `APPROVE_SCOPED_SUBSET` and useful for every conditional decision;
- `obligations`: attribution/share-alike/notice/redistribution/provider-term controls;
- `accepted_evidence`: pinned URLs, local document hashes, counsel notes, or publisher confirmations relied on;
- `decided_by`, `decided_at`, `rationale`: reviewer identity, ISO timestamp, and reasoning.

Any missing required field keeps that source unapproved. Any source with exact/near/manual-review contamination rows still has those rows excluded regardless of the license/provenance decision.
