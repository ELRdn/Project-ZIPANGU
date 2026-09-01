# ZIPANGU Dataset Curation — Final Recommendation

## Executive summary

Use the Japanese instruction, extraction and math sources as the first review priorities; keep frontier traces as a small support bucket. Do not approve or train yet: the registry remains quarantine, the local evaluation sources are unavailable for contamination clearance, and the base-model LoRA target policy is still pending.

## Dataset tier list

- **CORE** `extraction_wiki_ja` — scanned `152719`; eligible `151922`; usage: Japanese instruction following and structured extraction
- **CORE** `magpie_sft_v1` — scanned `132476`; eligible `131337`; usage: Japanese general assistant
- **CONDITIONAL** `nemotron_sft_multilingual_v2` — scanned `370081`; eligible `368950`; usage: Japanese STEM
- **CORE** `math_japanese_8k` — scanned `8094`; eligible `8094`; usage: Japanese worked math reasoning
- **CONDITIONAL** `ace_reason_math_japanese` — scanned `10000`; eligible `9989`; usage: Verified-final math seed; use reasoning_sft_ready rows only
- **FUTURE_CPT** `awesome_japanese_corpus` — scanned `0`; eligible `0`; usage: Lightweight statistics and future CPT review only
- **SUPPORT** `gpt_5_6_traces` — scanned `0`; eligible `0`; usage: Frontier coding traces; separate Sol/Luna and persona-injected rows
- **CONDITIONAL** `fable_5_5_distillation` — scanned `0`; eligible `0`; usage: Provenance-filtered frontier reasoning subset; never use all rows
- **SUPPORT** `claude_fable_code` — scanned `10`; eligible `3`; usage: Small agent/tool/code retention bucket after vendor and secret review
- **CONDITIONAL** `frontier_multi_teacher` — scanned `0`; eligible `0`; usage: Metadata-preserving multilingual frontier retention subset
- **CONDITIONAL** `fable_5_premium` — scanned `6365`; eligible `5469`; usage: Cross-deduplicated frontier supplement

## First-wave full audit

The following six sources were fully scanned. Counts are audit results, not approval decisions; `eligible` means the deterministic hard filter passed.

| Dataset | Raw rows | Scanned | Eligible | Rejected | Median chars | P95 chars | Quality mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `math_japanese_8k` | 8,094 | 8,094 | 8,094 | 0 | 574.5 | 907.25 | 93.8623 |
| `ace_reason_math_japanese` | 10,000 | 10,000 | 9,989 | 11 | 105.0 | 257.0 | 78.9148 |
| `magpie_sft_v1` | 132,476 | 132,476 | 131,337 | 1,139 | 630.0 | 1023.25 | 90.6974 |
| `extraction_wiki_ja` | 152,719 | 152,719 | 151,922 | 797 | 691.0 | 1846.25 | 93.566 |
| `nemotron_sft_multilingual_v2` | 370,081 | 370,081 | 368,950 | 1,131 | 2517.0 | 7986.75 | 87.6127 |
| `fable_5_premium` | 6,365 | 6,365 | 5,469 | 896 | 70641.0 | 121963.0 | 72.3091 |

## Recommended JP-heavy mix

Configured token-mass targets (estimated tokens; tokenizer_pending=true):

### c-i-v0-jp-heavy

```json
{
  "tokenizer_pending": true,
  "buckets": {
    "japanese": {
      "target_mass": 0.75,
      "available_token_mass": 3012809.0,
      "candidates": [
        {
          "dataset_id": "math_japanese_8k",
          "available_tokens": 2394896.0,
          "target_mass": 0.59617852
        },
        {
          "dataset_id": "ace_reason_math_japanese",
          "available_tokens": 617913.0,
          "target_mass": 0.15382148
        }
      ]
    },
    "frontier_reasoning": {
      "target_mass": 0.25,
      "available_token_mass": 181550241.0,
      "candidates": [
        {
          "dataset_id": "fable_5_premium",
          "available_tokens": 181513066.0,
          "target_mass": 0.24994881
        },
        {
          "dataset_id": "claude_fable_code",
          "available_tokens": 37175.0,
          "target_mass": 5.119e-05
        }
      ]
    }
  },
  "total_mass": 1.0
}
```

### c-i-v1-jp-heavy-curated

```json
{
  "tokenizer_pending": true,
  "buckets": {
    "japanese_instruction": {
      "target_mass": 0.35,
      "available_token_mass": 109346546.0,
      "candidates": [
        {
          "dataset_id": "extraction_wiki_ja",
          "available_tokens": 67172216.0,
          "target_mass": 0.21500703
        },
        {
          "dataset_id": "magpie_sft_v1",
          "available_tokens": 42174330.0,
          "target_mass": 0.13499297
        }
      ]
    },
    "japanese_math": {
      "target_mass": 0.2,
      "available_token_mass": 3012809.0,
      "candidates": [
        {
          "dataset_id": "math_japanese_8k",
          "available_tokens": 2394896.0,
          "target_mass": 0.15898094
        },
        {
          "dataset_id": "ace_reason_math_japanese",
          "available_tokens": 617913.0,
          "target_mass": 0.04101906
        }
      ]
    },
    "japanese_stem": {
      "target_mass": 0.1,
      "available_token_mass": 538098578.0,
      "candidates": [
        {
          "dataset_id": "nemotron_sft_multilingual_v2",
          "available_tokens": 538098578.0,
          "target_mass": 0.1
        }
      ]
    },
    "frontier_reasoning": {
      "target_mass": 0.35,
      "available_token_mass": 181550241.0,
      "candidates": [
        {
          "dataset_id": "claude_fable_code",
          "available_tokens": 37175.0,
          "target_mass": 7.167e-05
        },
        {
          "dataset_id": "fable_5_premium",
          "available_tokens": 181513066.0,
          "target_mass": 0.34992833
        }
      ]
    }
  },
  "total_mass": 1.0
}
```

## Recommended balanced mix

### c-i-v0-balanced

```json
{
  "tokenizer_pending": true,
  "buckets": {
    "japanese": {
      "target_mass": 0.5,
      "available_token_mass": 3012809.0,
      "candidates": [
        {
          "dataset_id": "math_japanese_8k",
          "available_tokens": 2394896.0,
          "target_mass": 0.39745234
        },
        {
          "dataset_id": "ace_reason_math_japanese",
          "available_tokens": 617913.0,
          "target_mass": 0.10254766
        }
      ]
    },
    "frontier_reasoning": {
      "target_mass": 0.5,
      "available_token_mass": 181550241.0,
      "candidates": [
        {
          "dataset_id": "fable_5_premium",
          "available_tokens": 181513066.0,
          "target_mass": 0.49989762
        },
        {
          "dataset_id": "claude_fable_code",
          "available_tokens": 37175.0,
          "target_mass": 0.00010238
        }
      ]
    }
  },
  "total_mass": 1.0
}
```

### c-i-v1-balanced-curated

```json
{
  "tokenizer_pending": true,
  "buckets": {
    "japanese_instruction": {
      "target_mass": 0.25,
      "available_token_mass": 109346546.0,
      "candidates": [
        {
          "dataset_id": "extraction_wiki_ja",
          "available_tokens": 67172216.0,
          "target_mass": 0.15357645
        },
        {
          "dataset_id": "magpie_sft_v1",
          "available_tokens": 42174330.0,
          "target_mass": 0.09642355
        }
      ]
    },
    "japanese_math": {
      "target_mass": 0.15,
      "available_token_mass": 3012809.0,
      "candidates": [
        {
          "dataset_id": "math_japanese_8k",
          "available_tokens": 2394896.0,
          "target_mass": 0.1192357
        },
        {
          "dataset_id": "ace_reason_math_japanese",
          "available_tokens": 617913.0,
          "target_mass": 0.0307643
        }
      ]
    },
    "japanese_stem": {
      "target_mass": 0.1,
      "available_token_mass": 538098578.0,
      "candidates": [
        {
          "dataset_id": "nemotron_sft_multilingual_v2",
          "available_tokens": 538098578.0,
          "target_mass": 0.1
        }
      ]
    },
    "frontier_reasoning": {
      "target_mass": 0.5,
      "available_token_mass": 181550241.0,
      "candidates": [
        {
          "dataset_id": "claude_fable_code",
          "available_tokens": 37175.0,
          "target_mass": 0.00010238
        },
        {
          "dataset_id": "fable_5_premium",
          "available_tokens": 181513066.0,
          "target_mass": 0.49989762
        }
      ]
    }
  },
  "total_mass": 1.0
}
```

## Data rejected

Rejections are recorded as eligibility=false plus exclusion_reasons in the runtime audit index; raw rows are never deleted.

## Duplicate analysis

- exact: `{"dataset_ids": ["ace_reason_math_japanese", "extraction_wiki_ja", "fable_5_premium", "magpie_sft_v1", "math_japanese_8k", "nemotron_sft_multilingual_v2"], "eligible_records": 675761, "exact_unique_records": 673311, "exact_duplicate_records": 2450, "near_duplicate_records": 11809, "near": {"considered_eligible": 673311, "inspected": 250000, "max_records": 250000, "capped": true, "algorithm": "simhash_lsh", "hamming_threshold": 3, "bucket_bits": 16}, "artifacts": ["D:\\VibeCoding\\project-zipangu\\data\\manifests\\_runtime\\dedup\\exact_winners.jsonl.gz", "D:\\VibeCoding\\project-zipangu\\data\\manifests\\_runtime\\dedup\\near_duplicates.jsonl.gz", "D:\\VibeCoding\\project-zipangu\\data\\manifests\\_runtime\\dedup\\exact_index.sqlite3"]}`
- near duplicate screening uses SimHash/LSH buckets and is capped/configured for scalable review; it is not an O(N²) comparison.

## Quality and language analysis

Per-source distributions are in `reports/dataset_audit/datasets/`. The score is structural and factuality_verified remains false.

## License/provenance risks

Unknown or mixed license/provenance is marked for review. No provenance placeholder or registry approval was created.

## Contamination status

`{"stage": "contamination", "status": "not_checked_missing_eval_source", "evaluation_records": 0, "candidate_records": 675761, "exact_matches": 0, "substring_matches": 0, "ngram_matches": 0, "manual_review": "required", "network_used": false, "dataset_ids": ["ace_reason_math_japanese", "extraction_wiki_ja", "fable_5_premium", "magpie_sft_v1", "math_japanese_8k", "nemotron_sft_multilingual_v2"]}`

## Pilot readiness

`{"stage": "pilot", "status": "BLOCKED", "recipe": "D:\\VibeCoding\\project-zipangu\\configs\\datasets\\c-i-v1-jp-heavy-curated.yaml", "target_tokens": 1000000, "records_selected": 0, "issues": ["BLOCKED [training_source_not_approved] c-i-v1-jp-heavy-curated.buckets.japanese_instruction.candidates: training source is not approved: extraction_wiki_ja", "BLOCKED [training_source_not_approved] c-i-v1-jp-heavy-curated.buckets.japanese_instruction.candidates: training source is not approved: magpie_sft_v1", "BLOCKED [training_source_not_approved] c-i-v1-jp-heavy-curated.buckets.japanese_math.candidates: training source is not approved: math_japanese_8k", "BLOCKED [training_source_not_approved] c-i-v1-jp-heavy-curated.buckets.japanese_math.candidates: training source is not approved: ace_reason_math_japanese", "BLOCKED [training_source_not_approved] c-i-v1-jp-heavy-curated.buckets.japanese_stem.candidates: training source is not approved: nemotron_sft_multilingual_v2", "BLOCKED [training_source_not_approved] c-i-v1-jp-heavy-curated.buckets.frontier_reasoning.candidates: training source is not approved: frontier_multi_teacher", "BLOCKED [training_source_not_approved] c-i-v1-jp-heavy-curated.buckets.frontier_reasoning.candidates: training source is not approved: fable_5_5_distillation", "BLOCKED [training_source_not_approved] c-i-v1-jp-heavy-curated.buckets.frontier_reasoning.candidates: training source is not approved: gpt_5_6_traces", "BLOCKED [training_source_not_approved] c-i-v1-jp-heavy-curated.buckets.frontier_reasoning.candidates: training source is not approved: claude_fable_code", "BLOCKED [training_source_not_approved] c-i-v1-jp-heavy-curated.buckets.frontier_reasoning.candidates: training source is not approved: fable_5_premium"], "auto_approved": false}`

## Next steps

1. Complete source revision, row identity, license and contamination review for each intended candidate.
2. Create and review tracked provenance sidecars; only then change selected registry entries to approved.
3. Decide explicit LoRA target_modules and update a new train config; do not overwrite old configs.
4. Re-run the same pipeline and L1 validator, then generate a small pilot.
5. Run untouched-base baseline evaluation before any improvement claim.
