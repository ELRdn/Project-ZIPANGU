# Evaluation Protocol

## Primary Japanese buckets

1. Japanese MT-Bench-compatible conversational/instruction quality
2. LLM-jp AnswerCarefully
3. llm-jp instruction-following evaluation

Exact harness revision, judge version and prompt are frozen in each eval manifest.

## Secondary regression checks

At minimum:

- English/general reasoning
- Japanese math holdout
- Japanese writing holdout
- code/reasoning sanity set

## Equivalence

Record temperature, top_p, top_k, max_new_tokens, reasoning mode/effort, chat template, system prompt, judge model/version, judge prompt/version, repeats and seeds.

## Repeats

Do not call a tiny one-run difference decisive. Preserve individual scores plus mean/spread when stochastic.

## Victory

- win >=2/3 primary Japanese buckets vs LLM-jp-4-8B Thinking;
- pass a predeclared general-capability regression tolerance;
- pass contamination audit.

Regression tolerance is frozen after baseline variance measurement and **before** the serious run.

## Artifacts

Preserve per-prompt generations, parsing, judge outputs, aggregate metrics, environment manifest, git commit and eval config.
