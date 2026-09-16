# Evaluation Protocol

## Model-generic comparison

The harness accepts a canonical `model_id` and resolves its base/revision from `configs/models/registry.yaml`. It must evaluate the untouched base and the ZIPANGU artifact with the same protocol:

```text
model_id → base_model + pinned revision + model-specific tokenizer revision
```

K, C and Z therefore have separate baseline/candidate pairs. Do not reuse Qwen token counts for Gemma. `configs/eval/k-i.yaml`, `configs/eval/c-i.yaml` and `configs/eval/z-i.yaml` describe the class-specific comparison policy.

## Primary Japanese buckets

1. Japanese MT-Bench-compatible conversational/instruction quality
2. LLM-jp AnswerCarefully
3. llm-jp instruction-following evaluation

Exact harness revision, judge version, prompt source and split are frozen in each eval manifest. Evaluation-only data is never copied into training data or used for teacher generation.

Contamination fingerprints use named, corpus-specific projections. Japanese MT-Bench projects both question `turns` and reference-answer `choices[].turns`; llm-jp instructions project `text` plus `output`. An empty normalized projection aborts that corpus fingerprint instead of emitting the SHA-256 of empty text. Fingerprint artifacts remain metadata-only and record the schema-adapter ID used for each row.

## Class-specific success criteria

### K-I

The primary comparison is untouched `empero-ai/Qwen3.8-4B-Distill` versus `ZIPANGU-K-I-4B`. Required: clear improvement on multiple Japanese metrics, no material general-reasoning collapse, and contamination/provenance gates pass. LLM-jp-4-8B Thinking is optional context/bonus, not the K gate.

### C-I

The primary opponent is `llm-jp/llm-jp-4-8b-thinking`. A release-candidate claim requires at least 2 of 3 primary Japanese wins, the general-regression gate, contamination PASS, untouched-base comparison, repeatable evaluation, raw outputs and provenance/license review.

### Z-I

Z evaluation is future-only during this migration. Its canonical base is clean Gemma 4 26B-A4B-it. Any QAT comparison must report BF16 score, Q4 score, BF16-to-Q4 degradation, Japanese metrics and general metrics separately.

## Secondary regression checks

At minimum:

- English/general reasoning;
- Japanese math holdout;
- Japanese writing holdout;
- code/reasoning sanity set;
- refusal/safety behavior where the approved evaluation permits it.

At least one broad/English regression check accompanies every primary Japanese result.

## Equivalence and repeats

Record temperature, top_p, top_k, max_new_tokens, reasoning mode/effort, chat template, system prompt, judge model/version, judge prompt/version, repeats and seeds. Do not call a tiny one-run difference decisive. Preserve individual scores plus mean/spread for stochastic evaluation.

Regression tolerance is frozen after baseline variance measurement and before the serious run. A baseline must be untouched; tuning or post-hoc threshold changes invalidate the comparison.

## Artifacts

Preserve, per model and per run:

```text
raw generations
parsed answers
judge outputs
aggregate metrics
model/base/tokenizer revisions
dataset and provenance manifests
environment/GPU manifest
git commit
eval config
```

Raw generations are not public by default. A leaderboard screenshot without these artifacts is not evidence of a result.
