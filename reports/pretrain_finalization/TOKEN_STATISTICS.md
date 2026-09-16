# Token Statistics

Statistics use the actual Qwen tokenizer at the resolved full revision. Raw rows and generated content are not included in this report.
Quantiles are exact for groups up to 100,000 rows and use a deterministic bounded reservoir for larger groups.

- model: `ZIPANGU-K-I-4B`
- base model/tokenizer: `empero-ai/Qwen3.8-4B-Distill`
- resolved tokenizer revision: `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e`
- assistant masks unavailable are recorded as unavailable; they are not estimated.

| Dataset | Category | Rows | Formatted tokens | Median | Mean | P95 | Max | Supervised rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ace_reason_math_japanese` | `japanese_math` | 10,000 | 891,646 | 80.0 | 89.165 | 164.0 | 771 | 0 |
| `claude_fable_code` | `coding_agent` | 43 | 1,032,283 | 16962 | 24006.581 | 63187.9 | 142642 | 0 |
| `extraction_wiki_ja` | `japanese_instruction_extraction` | 152,719 | 85,133,153 | 449.0 | 557.45 | 1108.0 | 7578 | 0 |
| `fable_5_5_distillation` | `frontier_reasoning_merged` | 1,978,248 | 10,495,856,438 | 4844.0 | 5305.632 | 17029.0 | 1367165 | 0 |
| `fable_5_premium` | `frontier_reasoning` | 6,365 | 117,270,622 | 20987 | 18424.293 | 32115.0 | 48949 | 0 |
| `frontier_multi_teacher` | `frontier_reasoning` | 582,954 | 508,986,050 | 426.0 | 873.115 | 2254.0 | 64337 | 0 |
| `gpt_5_6_traces` | `frontier_reasoning` | 6,288 | 24,017,823 | 3138.0 | 3819.628 | 10262.0 | 49790 | 0 |
| `magpie_sft_v1` | `japanese_general_instruction` | 132,473 | 42,736,969 | 322.0 | 322.609 | 509.0 | 4077 | 0 |
| `math_japanese_8k` | `japanese_math` | 8,094 | 3,303,751 | 397.0 | 408.173 | 636.35 | 3228 | 0 |
| `nemotron_sft_multilingual_v2` | `japanese_stem_code_math` | 370,081 | 468,178,923 | 1329.0 | 1265.071 | 3079.05 | 52508 | 0 |
