# Dataset audit — ace_reason_math_japanese

- repository: `SousiOmine/AceReason-Math-Japanese`
- provisional tier: **CONDITIONAL**
- recommended usage: Verified-final math seed; use reasoning_sft_ready rows only
- recommended maximum weight: `0.1`
- audit scope: `full`

## Counts

- raw rows: `10000`
- rows scanned: `10000`
- valid/eligible after hard filter: `9989` (99.89%)
- rejected: `11`
- excluded eval/test/validation rows: `0`
- estimated eligible token mass: `617913` (tokenizer_pending=true)

## Language / task / teacher

- language: code-heavy=325 (3.25%), english=17 (0.17%), japanese=6562 (65.62%), mixed=3049 (30.49%), unknown=47 (0.47%)
- categories: ace_reason_math_japanese=10000 (100.00%)
- teachers: unknown=10000 (100.00%)

## Quality

The score is deterministic structural triage only; it does not verify factual correctness.

- mean / median / p10 / p90: `78.9148` / `79.0` / `79.0` / `79.0`
- quality method: `deterministic_structural_heuristic_v1`

## Rejections and risks

- reasons: empty_assistant=11 (100.00%)
- license expected: `unknown`
- license metadata: `"cc-by-4.0"`
- source revision: `unknown`
- contamination: `not_checked_missing_eval_source`
- vendor_specific: `False`

## Recommendation

This is a review priority, not an approval. Registry status is not changed by the pipeline.
