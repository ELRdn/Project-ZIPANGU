# Dataset audit — extraction_wiki_ja

- repository: `llm-jp/extraction-wiki-ja`
- provisional tier: **CORE**
- recommended usage: Japanese instruction following and structured extraction
- recommended maximum weight: `0.15`
- audit scope: `full`

## Counts

- raw rows: `152719`
- rows scanned: `152719`
- valid/eligible after hard filter: `151922` (99.48%)
- rejected: `797`
- excluded eval/test/validation rows: `0`
- estimated eligible token mass: `67172216` (tokenizer_pending=true)

## Language / task / teacher

- language: code-heavy=2 (0.00%), english=9 (0.01%), japanese=144124 (94.37%), mixed=8581 (5.62%), unknown=3 (0.00%)
- categories: extraction_wiki_ja=152719 (100.00%)
- teachers: unknown=152719 (100.00%)

## Quality

The score is deterministic structural triage only; it does not verify factual correctness.

- mean / median / p10 / p90: `93.566` / `94.65350000000001` / `90.0` / `97.0`
- quality method: `deterministic_structural_heuristic_v1`

## Rejections and risks

- reasons: empty_assistant=679 (85.19%), html_navigation_garbage=1 (0.13%), meaningless_repetition=2 (0.25%), private_path_exposure=2 (0.25%), refusal_only=99 (12.42%), stack_trace_only=11 (1.38%), unicode_corruption=3 (0.38%)
- license expected: `apache-2.0`
- license metadata: `"apache-2.0"`
- source revision: `unknown`
- contamination: `not_checked_missing_eval_source`
- vendor_specific: `False`

## Recommendation

This is a review priority, not an approval. Registry status is not changed by the pipeline.
