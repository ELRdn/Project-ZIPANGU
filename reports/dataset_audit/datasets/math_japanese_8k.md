# Dataset audit — math_japanese_8k

- repository: `awakara/Math-Japanese-8k`
- provisional tier: **CORE**
- recommended usage: Japanese worked math reasoning
- recommended maximum weight: `0.2`
- audit scope: `full`

## Counts

- raw rows: `8094`
- rows scanned: `8094`
- valid/eligible after hard filter: `8094` (100.00%)
- rejected: `0`
- excluded eval/test/validation rows: `0`
- estimated eligible token mass: `2394896` (tokenizer_pending=true)

## Language / task / teacher

- language: japanese=8079 (99.81%), mixed=15 (0.19%)
- categories: math_japanese_8k=8094 (100.00%)
- teachers: unknown=8094 (100.00%)

## Quality

The score is deterministic structural triage only; it does not verify factual correctness.

- mean / median / p10 / p90: `93.8623` / `93.652` / `92.59` / `95.417`
- quality method: `deterministic_structural_heuristic_v1`

## Rejections and risks

- reasons: none
- license expected: `unknown`
- license metadata: `"unknown"`
- source revision: `unknown`
- contamination: `not_checked_missing_eval_source`
- vendor_specific: `False`

## Recommendation

This is a review priority, not an approval. Registry status is not changed by the pipeline.
