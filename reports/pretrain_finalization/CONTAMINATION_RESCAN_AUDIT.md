# Corrected Contamination Rescan Audit

This report is metadata-only. It contains record identifiers and hashes, but no candidate text or eval-only text.

## Decision

- available-eval extractor defect: `CLOSED`
- available-eval exact revalidation: `CLOSED`
- global contamination gate: `BLOCKED_REQUIRED_EVAL_MISSING`
- missing required corpus: `llm-jp/AnswerCarefully` v2.0
- `TRAINING_ALLOWED=false`

Closing the available-eval item does not mark the remaining hits clean. All `103` genuine exact candidates and `5` manual-review rows remain excluded from candidate selection. The global gate remains fail-closed until AnswerCarefully can be indexed and the affected candidate scope is rescanned.

## Extractor correction

- Registered explicit schema adapters for all `10` candidate datasets and all `8` eval datasets.
- Projected Japanese MT-Bench from `turns` and `choices[].turns`, and llm-jp instructions from `text` plus `output`.
- Rejected empty normalized eval fingerprints at index build and lookup time.
- Excluded empty candidate projections instead of hashing them as valid content.
- Limited `frontier_multi_teacher` scanning to its authoritative canonical train/validation/test Parquet views.
- Classified Frontier `metadata/*` as `excluded_metadata` and all non-authoritative materialized views as `excluded_derivative`.
- Calibrated near-match review to require meaningful length and similarity agreement. The previous shared-13-gram plus low SimHash threshold admitted random-baseline similarities into manual review.
- Recorded the policy as immutable config `configs/pretrain/finalization-k-i-contamination-v2.yaml`.

The corrected eval index contains `830` non-empty fingerprints: Japanese MT-Bench `430` via `mt_bench_turns_and_choices_v1` and llm-jp instructions `400` via `llm_jp_text_output_v1`. Empty fingerprints are forbidden.

## Full canonical rescan

| Classification | Rows |
| --- | ---: |
| canonical records reconciled | 4,643,875 |
| fingerprinted authoritative records | 2,725,154 |
| excluded metadata | 1,022,950 |
| excluded derivative | 895,771 |
| quarantine exact | 103 |
| quarantine near | 0 |
| requires manual review | 5 |
| not checked because a required eval is missing | 2,725,046 |

The old and corrected Parquet files contain the same `4,643,875` `record_id` values in source order with `0` mismatches. The corrected artifact has no exact hit using the empty SHA-256 value.

## Old-vs-new waterfall

| Old status | Corrected status | Rows |
| --- | --- | ---: |
| not_checked_missing_eval_source | not_checked_missing_eval_source | 2,479,865 |
| quarantine_exact | excluded_metadata | 1,022,950 |
| not_checked_missing_eval_source | excluded_derivative | 545,919 |
| quarantine_exact | excluded_derivative | 312,817 |
| requires_manual_review | not_checked_missing_eval_source | 187,301 |
| quarantine_exact | not_checked_missing_eval_source | 57,880 |
| requires_manual_review | excluded_derivative | 37,035 |
| quarantine_exact | quarantine_exact | 57 |
| not_checked_missing_eval_source | quarantine_exact | 44 |
| requires_manual_review | requires_manual_review | 5 |
| requires_manual_review | quarantine_exact | 2 |

Old totals were exact `1,393,704`, manual-review `224,343`, and not-checked `3,025,828`. Corrected totals are exact `103`, near `0`, manual-review `5`, not-checked `2,725,046`, excluded metadata `1,022,950`, and excluded derivative `895,771`.

## Genuine exact population and sample

All `103` structurally valid exact candidates are non-empty `long_substring_ge_80` matches against Japanese MT-Bench. They remain quarantine candidates rather than adjudicated benchmark contamination.

| Source dataset | Exact population | Sample |
| --- | ---: | ---: |
| `frontier_multi_teacher` | 57 | 57 |
| `fable_5_5_distillation` | 37 | 37 |
| `nemotron_sft_multilingual_v2` | 6 | 3 |
| `magpie_sft_v1` | 3 | 3 |
| **Total** | **103** | **100** |

The sample is deterministic with seed `3407`, stratified by source dataset, eval corpus, exact reason, and content-hash cluster. It contains `100` distinct content hashes across `100` rows and no raw text fields. The remaining three rows are in the same Nemotron/eval/reason family and stay quarantined with the full population.

The five manual-review rows are from `fable_5_5_distillation` and carry `benchmark_or_source_metadata_signal`; they have no asserted eval match and remain excluded pending human review.

## Integrity evidence

- old scan SHA-256: `4ec7f0b97ac5186d59653503c328cae4958b1a95d0e4064bcf515dfb112f1b9a`
- corrected scan SHA-256: `ccd18839cc565c08d58ca510e3532c2a91da4254ecc0404cf1fdfcbd8f44ebbb`
- waterfall SHA-256: `e414c1bce1a4914a3191b1fe78e80abcaf19fce0182927ee6c0e4d18854e445c`
- sample SHA-256: `fac952a0096981bd9f9b4abffc51ad1aa09a03afd318a76a474a6d3001bf9214`

Artifacts:

- `data/manifests/contamination_results.parquet`
- `reports/pretrain_finalization/contamination_rescan_audit.json`
- `reports/pretrain_finalization/contamination_rescan_waterfall.csv`
- `reports/pretrain_finalization/contamination_rescan_genuine_exact_sample.csv`
- `reports/pretrain_finalization/contamination_old_scan_snapshot.json`
- `reports/pretrain_finalization/contamination_near_threshold_calibration.json`

## Close rationale for item 1

The available-eval extractor defect and exact revalidation item can be closed because the empty-fingerprint path now fails closed, the schema projections are explicit, metadata and derivatives are excluded, all canonical record IDs reconcile across old and new outputs, the full corrected status waterfall is reproducible, and the valid exact population has a deterministic 100-row metadata-only evidence sample.

This closure is deliberately narrower than the global contamination gate. AnswerCarefully access is still required before any contamination PASS or training approval.
