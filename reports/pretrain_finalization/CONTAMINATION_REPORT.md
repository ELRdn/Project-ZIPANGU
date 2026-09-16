# Contamination Report

The report is metadata-only. Candidate text is never written to the repository.

- global status: `not_checked_missing_eval_source`
- evaluation rows indexed: `830`
- candidate rows scanned: `4,643,875`
- candidate rows fingerprinted: `2,725,154`
- metadata/derivative/empty rows excluded: `1,918,721`
- required eval missing: `True`
- classifications: `{"excluded_derivative": 895771, "excluded_metadata": 1022950, "not_checked_missing_eval_source": 2725046, "quarantine_exact": 103, "requires_manual_review": 5}`
- comparison strategy: inverted indexes and LSH; O(N²) pairwise comparison is not used.

## Corrected-result interpretation

- exact: `103`
- near: `0`
- manual review: `5`
- empty fingerprints admitted: `False`
- old-vs-new `record_id` reconciliation: `4,643,875 / 4,643,875`, mismatches `0`
- prior invalid empty-hash exact count: `1,393,704`
- corrected genuine-exact evidence sample: `100 / 103`, seed `3407`, raw text absent

All `103` exact candidates are non-empty `long_substring_ge_80` matches against Japanese MT-Bench. They remain quarantined; this report does not adjudicate them as confirmed benchmark leakage. The five manual-review rows also remain excluded.

Available-eval extractor defect and exact revalidation: `CLOSED`. Global contamination gate: `BLOCKED_REQUIRED_EVAL_MISSING` because AnswerCarefully v2.0 remains unavailable. See `CONTAMINATION_RESCAN_AUDIT.md` and `contamination_rescan_waterfall.csv` for the complete old-vs-new evidence.
