# Dataset audit — claude_fable_code

- repository: `armand0e/claude-fable-5-claude-code`
- provisional tier: **SUPPORT**
- recommended usage: Small agent/tool/code retention bucket after vendor and secret review
- recommended maximum weight: `0.1`
- audit scope: `full`

## Counts

- raw rows: `18370`
- rows scanned: `10`
- valid/eligible after hard filter: `3` (30.00%)
- rejected: `7`
- excluded eval/test/validation rows: `0`
- estimated eligible token mass: `37175` (tokenizer_pending=true)

## Language / task / teacher

- language: code-heavy=5 (50.00%), english=5 (50.00%)
- categories: claude_fable_code=10 (100.00%)
- teachers: unknown=10 (100.00%)

## Quality

The score is deterministic structural triage only; it does not verify factual correctness.

- mean / median / p10 / p90: `82.9333` / `83.78450000000001` / `75.80090000000001` / `88.3051`
- quality method: `deterministic_structural_heuristic_v1`

## Rejections and risks

- reasons: private_path_exposure=7 (100.00%)
- license expected: `unknown`
- license metadata: `"unknown"`
- source revision: `unknown`
- contamination: `not_checked_missing_eval_source`
- vendor_specific: `True`

## Recommendation

This is a review priority, not an approval. Registry status is not changed by the pipeline.
