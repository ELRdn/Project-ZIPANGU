# Dataset audit — fable_5_premium

- repository: `saidutta69/fable-5-premium`
- provisional tier: **CONDITIONAL**
- recommended usage: Cross-deduplicated frontier supplement
- recommended maximum weight: `0.1`
- audit scope: `full`

## Counts

- raw rows: `6365`
- rows scanned: `6365`
- valid/eligible after hard filter: `5469` (85.92%)
- rejected: `896`
- excluded eval/test/validation rows: `637`
- estimated eligible token mass: `181513066` (tokenizer_pending=true)

## Language / task / teacher

- language: code-heavy=939 (14.75%), english=5426 (85.25%)
- categories: fable_5_premium=6365 (100.00%)
- teachers: claude-fable-5=591 (9.29%), claude-opus-4-6=15 (0.24%), claude-opus-4-7=28 (0.44%), claude-opus-4-8=26 (0.41%), unknown=5705 (89.63%)

## Quality

The score is deterministic structural triage only; it does not verify factual correctness.

- mean / median / p10 / p90: `72.3091` / `71.264` / `71.022` / `83.33449999999999`
- quality method: `deterministic_structural_heuristic_v1`

## Rejections and risks

- reasons: empty_assistant=248 (27.68%), eval_split_not_train=637 (71.09%), html_navigation_garbage=3 (0.33%), private_path_exposure=265 (29.58%)
- license expected: `inspect_source`
- license metadata: `"mit"`
- source revision: `unknown`
- contamination: `not_checked_missing_eval_source`
- vendor_specific: `True`

## Recommendation

This is a review priority, not an approval. Registry status is not changed by the pipeline.
