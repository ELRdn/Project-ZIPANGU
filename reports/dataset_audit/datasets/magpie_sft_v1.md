# Dataset audit — magpie_sft_v1

- repository: `llm-jp/magpie-sft-v1.0`
- provisional tier: **CORE**
- recommended usage: Japanese general assistant
- recommended maximum weight: `0.3`
- audit scope: `full`

## Counts

- raw rows: `132476`
- rows scanned: `132476`
- valid/eligible after hard filter: `131337` (99.14%)
- rejected: `1139`
- excluded eval/test/validation rows: `0`
- estimated eligible token mass: `42174330` (tokenizer_pending=true)

## Language / task / teacher

- language: code-heavy=152 (0.11%), english=345 (0.26%), japanese=123222 (93.01%), mixed=8747 (6.60%), unknown=10 (0.01%)
- categories: magpie_sft_v1=132476 (100.00%)
- teachers: Qwen/Qwen2.5-32B-Instruct=132476 (100.00%)

## Quality

The score is deterministic structural triage only; it does not verify factual correctness.

- mean / median / p10 / p90: `90.6974` / `91.2` / `89.0` / `92.0`
- quality method: `deterministic_structural_heuristic_v1`

## Rejections and risks

- reasons: empty_user=3 (0.26%), html_navigation_garbage=6 (0.53%), meaningless_repetition=2 (0.18%), private_path_exposure=9 (0.79%), refusal_only=1103 (96.84%), secret_like_string=12 (1.05%), stack_trace_only=3 (0.26%), unicode_corruption=1 (0.09%)
- license expected: `unknown`
- license metadata: `"apache-2.0"`
- source revision: `unknown`
- contamination: `not_checked_missing_eval_source`
- vendor_specific: `False`

## Recommendation

This is a review priority, not an approval. Registry status is not changed by the pipeline.
