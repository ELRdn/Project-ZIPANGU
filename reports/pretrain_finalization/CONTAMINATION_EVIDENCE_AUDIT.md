# Contamination Exact-Match Evidence Audit

This audit is metadata-only. It does not copy candidate or eval text into the repository.

> Historical pre-fix snapshot: this report identified the empty-projection collision in the old scan. The repair and full 4,643,875-record rescan are complete. Current evidence and closure boundaries are in `CONTAMINATION_RESCAN_AUDIT.md`; do not use the counts below as current contamination classifications.

## Result

- exact quarantines audited: `1,393,704`
- exact content-hash clusters: `1`
- high-confidence false-positive candidates: `1,393,704` (`100.000000%`)
- plausible non-empty true positives: `0`
- unresolved exact rows: `0`
- decomposition reconciles to exact total: `True`

## Root finding

Every pre-fix `quarantine_exact` row belonged to the empty normalized-content cluster `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. The same hash appeared in empty eval projections, so the old status proved an extractor projection collision, not benchmark-content overlap.

Japanese MT-Bench rows use structures such as `turns`; the current generic eval extractor does not project those fields. Empty fingerprints were then admitted to the exact-content index. Candidate rows whose canonical projection was also empty matched all empty eval fingerprints.

## By source

| Source | Scanned | Exact | Exact % | FP candidates | Sample |
| --- | ---: | ---: | ---: | ---: | ---: |
| ace_reason_math_japanese | 10,000 | 0 | 0.000000 | 0 | 0 |
| claude_fable_code | 63 | 0 | 0.000000 | 0 | 0 |
| extraction_wiki_ja | 152,719 | 0 | 0.000000 | 0 | 0 |
| fable_5_5_distillation | 1,981,131 | 0 | 0.000000 | 0 | 0 |
| fable_5_premium | 6,365 | 0 | 0.000000 | 0 | 0 |
| frontier_multi_teacher | 1,976,658 | 1,393,704 | 70.508100 | 1,393,704 | 100 |
| gpt_5_6_traces | 6,288 | 0 | 0.000000 | 0 | 0 |
| magpie_sft_v1 | 132,476 | 0 | 0.000000 | 0 | 0 |
| math_japanese_8k | 8,094 | 0 | 0.000000 | 0 | 0 |
| nemotron_sft_multilingual_v2 | 370,081 | 0 | 0.000000 | 0 | 0 |

## By eval corpus

| Eval | Fingerprints | Empty | Clusters | Primary candidate rows |
| --- | ---: | ---: | ---: | ---: |
| answer_carefully | 0 | 0 | 0 | 0 |
| code_reasoning_sanity | 0 | 0 | 0 | 0 |
| english_general_regression | 0 | 0 | 0 | 0 |
| japanese_math_holdout | 0 | 0 | 0 | 0 |
| japanese_mt_bench | 430 | 430 | 1 | 1,393,704 |
| japanese_writing_holdout | 0 | 0 | 0 | 0 |
| llm_jp_instruction_eval | 400 | 0 | 0 | 0 |
| zipangu_private_holdout | 0 | 0 | 0 | 0 |

## Frontier source path families

| Source family | Exact | Exact share % | Files | Sample |
| --- | ---: | ---: | ---: | ---: |
| metadata/context_buckets | 922,866 | 66.216786 | 18 | 62 |
| data/canonical | 104,187 | 7.475547 | 13 | 8 |
| data/glm47_native | 104,187 | 7.475547 | 44 | 8 |
| data/token_stats | 104,187 | 7.475547 | 13 | 8 |
| metadata/parent_ids | 100,084 | 7.181152 | 5 | 8 |
| data/prompt_completion_text | 52,028 | 3.733074 | 7 | 4 |
| data/rl_tool_prompts | 5,909 | 0.423978 | 3 | 1 |
| data/smoke | 256 | 0.018368 | 3 | 1 |

### Source-side projection findings

- `metadata/context_buckets`: metadata-only fields (id, split, token counts); should not enter text contamination scan (`922,866` rows).
- `data/canonical`: trainable text is stored in messages_json, which the current canonical extractor ignores (`104,187` rows).
- `data/glm47_native`: tokenized input_ids/labels derivative; no raw text field for the generic extractor (`104,187` rows).
- `data/token_stats`: token-statistics derivative; should not enter text contamination scan (`104,187` rows).
- `metadata/parent_ids`: metadata-only id/split rows; should not enter text contamination scan (`100,084` rows).
- `data/prompt_completion_text`: text is stored in prompt_text/completion_text, which the current extractor ignores (`52,028` rows).
- `data/rl_tool_prompts`: prompt is stored in prompt_messages_json, which the current extractor ignores (`5,909` rows).
- `data/smoke`: trainable text is stored in messages_json, which the current canonical extractor ignores (`256` rows).

## By exact cluster

| Content hash | Candidates | Sources | Eval rows | Classification |
| --- | ---: | ---: | ---: | ---: |
| e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | 1,393,704 | 1 | 430 | high_confidence_false_positive_projection_collision |

## 100-row stratified evidence sample

`contamination_stratified_evidence_sample.csv` contains `100` rows allocated across source-family and content-hash-cluster strata, with at least one row per positive stratum and proportional remainder. Selection is deterministic using seed `3407` and SHA-256 hash priority.

The sample contains identifiers and fingerprints only. It includes no raw candidate text and no eval-only text.

## Decision

Treat the pre-fix 1,393,704 exact count as invalid evidence of contamination. Those rows were not automatically marked clean: the extractor was repaired, empty fingerprints were rejected, and a fresh full scan was completed before the replacement classifications were accepted.

At the time of this snapshot, the corrected full-corpus scan had not been executed. It has since completed; the available-eval extractor and exact-revalidation item is closed, while the global contamination gate remains blocked because AnswerCarefully is still unavailable.

## Artifacts

- `contamination_exact_by_source.csv`
- `contamination_exact_by_source_family.csv`
- `contamination_exact_by_source_file.csv`
- `contamination_exact_by_eval.csv`
- `contamination_exact_by_cluster.csv`
- `contamination_stratified_evidence_sample.csv`
- `contamination_exact_audit.json`
