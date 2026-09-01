# ZIPANGU C-I Research Plan

## Question

Can a strong 9B open-weight reasoning model, adapted with carefully selected Japanese post-training data, outperform a Japanese-focused 8B model on multiple uncontaminated Japanese evaluations at low GPU cost?

## Hypothesis

The upstream base already contains substantial knowledge/reasoning/math/code capability. C-I therefore begins with **SFT/reasoning distillation**, not scratch pretraining, tokenizer replacement or CPT.

## Base / reference

- Base: `empero-ai/Qwen3.8-9B-Distill`
- Reference opponent: `llm-jp/llm-jp-4-8b-thinking`

## Phases

### P0 Baseline
Freeze untouched base/reference reports.

### P1 Smoke
Tiny end-to-end pipeline run. No performance claim.

### P2 1M
First useful signal. Stop if Japanese metrics regress, general ability collapses, outputs become unhealthy or contamination fails.

### P3 10M
Run only after P2 human GO. Compare JP-heavy vs balanced if budget permits.

### P4 50M
Only the better recipe progresses.

### P5 Release candidate
Requires 2/3 primary Japanese wins, regression gate pass, contamination pass, repeatable eval, provenance and license review.

## Non-goals for C-I

- scratch training
- tokenizer replacement
- billion-token CPT
- multimodal specialization
- claiming Japanese SOTA from one benchmark
