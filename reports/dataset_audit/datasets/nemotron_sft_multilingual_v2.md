# Dataset audit — nemotron_sft_multilingual_v2

- repository: `nvidia/Nemotron-SFT-Multilingual-v2`
- provisional tier: **CONDITIONAL**
- recommended usage: Japanese STEM
- recommended maximum weight: `0.1`
- audit scope: `full`

## Counts

- raw rows: `370081`
- rows scanned: `370081`
- valid/eligible after hard filter: `368950` (99.69%)
- rejected: `1131`
- excluded eval/test/validation rows: `0`
- estimated eligible token mass: `538098578` (tokenizer_pending=true)

## Language / task / teacher

- language: code-heavy=201723 (54.51%), english=32450 (8.77%), japanese=15822 (4.28%), mixed=58739 (15.87%), unknown=61347 (16.58%)
- categories: code=138948 (37.55%), math=122949 (33.22%), stem=108184 (29.23%)
- teachers: unknown=370081 (100.00%)

## Quality

The score is deterministic structural triage only; it does not verify factual correctness.

- mean / median / p10 / p90: `87.6127` / `89.611` / `74.042` / `91.90899999999999`
- quality method: `deterministic_structural_heuristic_v1`

## Rejections and risks

- reasons: empty_assistant=1120 (99.03%), private_path_exposure=11 (0.97%), secret_like_string=1 (0.09%), stack_trace_only=1 (0.09%)
- license expected: `mixed_cc_by_and_cc_by_sa`
- license metadata: `["cc-by-4.0", "cc-by-sa-4.0"]`
- source revision: `unknown`
- contamination: `not_checked_missing_eval_source`
- vendor_specific: `False`

## Recommendation

This is a review priority, not an approval. Registry status is not changed by the pipeline.
