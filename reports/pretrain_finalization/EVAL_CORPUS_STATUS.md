# Evaluation Corpus Status

Raw evaluation data is stored only under the external eval root. This report contains metadata and availability; it does not contain evaluation prompts.

| ID | Version | Split | Rows | Availability | Contamination ready | Required |
| --- | --- | --- | ---: | --- | --- | --- |
| `japanese_mt_bench` | `v2.0.0` | `embedded` | 430 | `available` | `True` | `True` |
| `llm_jp_instruction_eval` | `v1.0` | `test` | 400 | `available` | `True` | `True` |
| `answer_carefully` | `v2.0` | `test` | 0 | `blocked_access_denied` | `False` | `True` |
| `zipangu_private_holdout` | `local` | `holdout` | 0 | `unavailable_not_provided` | `False` | `False` |
| `english_general_regression` | `local` | `holdout` | 0 | `unavailable_not_provided` | `False` | `False` |
| `japanese_math_holdout` | `local` | `holdout` | 0 | `unavailable_not_provided` | `False` | `False` |
| `japanese_writing_holdout` | `local` | `holdout` | 0 | `unavailable_not_provided` | `False` | `False` |
| `code_reasoning_sanity` | `local` | `holdout` | 0 | `unavailable_not_provided` | `False` | `False` |

- required sources missing or blocked: `answer_carefully`
- AnswerCarefully access is never bypassed; a gated or denied source remains blocked.
- training_forbidden is true for every entry.
