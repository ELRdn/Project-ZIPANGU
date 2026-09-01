# AGENTS.md — Project ZIPANGU

These rules apply to Codex and any other research/coding agent operating in this repository.

## Mission

Test whether low-cost Japanese post-training of `empero-ai/Qwen3.8-9B-Distill` can outperform `llm-jp/llm-jp-4-8b-thinking` on multiple clean Japanese evaluations without materially damaging general capability.

## Absolute rules

### 1. Never contaminate evaluation data

Never train on, translate, paraphrase, distill from, retrieve, or synthesize from anything marked `eval_only`.

Locked evaluation resources include AnswerCarefully, Japanese MT-Bench evaluation prompts, llm-jp instruction evaluation prompts, ZIPANGU private holdout, and any future source marked `role: eval_only`.

If a training source looks benchmark-derived, quarantine it for human review.

### 2. Never silently change experiments

Changes to data mix, formatting, LoRA targets, LR, sequence length, batch size, optimizer, seed, token budget, filtering or sampling require a new config/manifest. Never overwrite an old run config.

### 3. Budget is a hard boundary

Pilot cap: **¥3,000**.

Before paid RunPod creation:

1. inspect live offer/price;
2. confirm JPY/USD rate;
3. estimate runtime;
4. run budget preflight;
5. show projected cost;
6. obtain human GO.

Never provision if projected cost exceeds remaining budget.

### 4. RunPod is semi-automatic

After a human has approved a specific run, an agent may bootstrap, train, evaluate, persist artifacts and stop that Pod.

Explicit human GO is always required for:

- creation of a new paid Pod;
- cost-increasing GPU/tier change;
- extension of approved GPU hours;
- next token-scale tier;
- destructive deletion of Pods, volumes, datasets or checkpoints.

Preferred compute: **RTX 4090 Community Cloud**.

### 5. Stop paid compute

On success or unrecoverable failure: persist artifacts → record status → stop Pod → verify stopped → report GPU time/cost.

### 6. Preserve provenance

Every accepted training source needs repo, config/subset, revision, split, row identity, license, language/category, transform chain, content hash and contamination status.

Unknown provenance = not allowed in serious run.

### 7. Baseline before tuning

No improvement claim without untouched-base evaluation using the same protocol.

### 8. Do not optimize only one benchmark

At least one broad/English regression check accompanies primary Japanese metrics.

### 9. Preserve raw evaluation artifacts

Keep generations, parsed answers, judge outputs, configs and aggregates. A leaderboard screenshot alone is not evidence.

### 10. Never auto-publish weights

Code/docs may be public. Dataset exports, adapters and weights require human approval after license/provenance review.

## Workflow

```text
read configs
→ validate registry
→ resolve run config
→ budget preflight
→ HUMAN GO
→ provision approved Pod
→ smoke
→ train
→ eval
→ persist manifest/results
→ stop Pod
→ compare baseline
→ HUMAN GO for next scale
```

When uncertain, choose the cheaper and more conservative action. Never solve uncertainty by spending more GPU money.
