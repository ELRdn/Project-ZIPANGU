# Future CPT Candidate Report

`awesome_japanese_corpus` is not an SFT source in this phase. The pipeline uses a bounded streaming sample and Parquet metadata; it does not run a full expensive near-dedup or quality model over the entire corpus.

```json
{
  "dataset_id": "awesome_japanese_corpus",
  "repo": "nakasyou/awesome-japanese-corpus",
  "local_path": "E:\\zipangu-datesets\\nakasyou__awesome-japanese-corpus",
  "audit_scope": "sampled",
  "raw_rows": 0,
  "raw_rows_known": false,
  "rows_scanned": 0,
  "canonical_rows": 0,
  "valid_rows": 0,
  "eligible_rows": 0,
  "rejected_rows": 0,
  "eligible_token_count": 0,
  "excluded_eval_rows": 0,
  "reasoning_ready_rows": 0,
  "tool_ready_rows": 0,
  "language_counts": {},
  "category_counts": {},
  "teacher_counts": {},
  "split_counts": {},
  "rejection_reasons": {},
  "quality": {
    "mean": null,
    "median": null,
    "p10": null,
    "p90": null,
    "sampled_values": 0,
    "observed_values": 0
  },
  "provenance": {
    "source_revision": "unknown",
    "license_metadata": [
      "odc-by",
      "cc-by-4.0"
    ],
    "contamination_status": "not_checked_missing_eval_source"
  },
  "warnings": [
    "Parquet input requires pyarrow; install the audit extra without downloading dataset files",
    "full_row_audit_skipped_by_future_cpt_policy"
  ],
  "char_stats": {
    "median": null,
    "p95": null,
    "observed_values": 0
  },
  "token_stats": {
    "median_estimate": null,
    "p95_estimate": null,
    "tokenizer_pending": true,
    "observed_values": 0
  }
}
```
