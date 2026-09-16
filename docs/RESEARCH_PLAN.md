# Project ZIPANGU Research Plan — K/C/Z Generation I

## Research question

Can careful, low-cost Japanese adaptation of strong open-weight models compete with Japanese-focused scratch-trained foundation models without materially damaging general capability?

The experiment is a staged scale-up, not three unrelated model runs:

```text
K (Kei / 軽量級) → C (Chū / 中量級) → Z (Zyū / 重量級)
```

Generation I (`I`) identifies the first ZIPANGU training-recipe generation. The model class is independent from the recipe generation and from the upstream model version.

## Canonical Generation-I family

| Class | ZIPANGU id | Canonical base | Research role | Training status |
|---|---|---|---|---|
| K | `ZIPANGU-K-I-4B` | `empero-ai/Qwen3.8-4B-Distill` | recipe validation, local AMD research, cheap scaling | local completion gate passed; 1M data/training NO-GO |
| C | `ZIPANGU-C-I-9B` | `empero-ai/Qwen3.8-9B-Distill` | primary LLM-jp-4-8B Thinking challenge | configured; follows K |
| Z | `ZIPANGU-Z-I-26B-A4B` | `google/gemma-4-26B-A4B-it` | MoE scale-up, quantization and QAT research | future only |

Canonical means a clean upstream base plus the ZIPANGU recipe. Community Heretic, Abliterated and refusal-removal checkpoints are separate non-canonical variants and never replace a canonical base.

## Roadmap

### Phase K — cheap validation first

1. Freeze the untouched `Qwen3.8-4B-Distill` baseline.
2. Validate the shared Generation-I dataset recipe, format and QLoRA path. **Local processing completed; QLoRA path PASS. The corrected 4,643,875-record contamination rescan closed the available-eval extractor/exact item, while AnswerCarefully keeps the global contamination gate BLOCKED; candidate fill/format and human data gates remain BLOCKED.**
3. Run the RX 7600 local research track: device selection, Vulkan, ROCm/HIP, PyTorch and Unsloth diagnostics. **Completed for the current isolated environment.**
4. Complete the pinned llama.cpp gate: official Empero GGUF, Vulkan build, HIP build for `gfx1102`, Japanese/English inference, llama-bench and the 9-case CPU/Vulkan/HIP matrix. **Completed; all required cases PASS.**
5. If the environment is healthy, run only a tiny local K-I QLoRA smoke. **Completed as both a two-step synthetic infrastructure smoke and a 16-example research-only real-data-path smoke; neither is ZIPANGU training.**
6. With human GO, run 1M tokens and evaluate. **Current decision: NO-GO until the six-item human approval queue is closed and a format-valid near-1M candidate is regenerated.**
7. With a new human GO only after a useful 1M signal, run 10M and evaluate.
8. Treat 50M as optional and proceed only with the better recipe.

K success does not require beating LLM-jp. Required evidence is clear improvement over the untouched K base on multiple Japanese metrics, no material general-reasoning collapse, and clean contamination/provenance. LLM-jp proximity or a win is a desired bonus.

### Phase C — primary Japanese challenge

1. Freeze the untouched `Qwen3.8-9B-Distill` baseline.
2. Transfer the K-winning recipe with sources, token-mass targets, quality thresholds, deduplication, contamination policy, chat template, LoRA targets, learning rate, sequence length and assistant-loss policy frozen wherever possible.
3. Change model size as the principal factor; do not change several recipe variables at once.
4. Compare against `llm-jp/llm-jp-4-8b-thinking` on the three primary Japanese evaluations.

C release-candidate success requires at least 2 of 3 primary Japanese wins, a predeclared general-regression gate, contamination PASS, untouched-base comparison, preserved raw outputs and complete provenance/license review.

### Phase Z — future MoE/QAT research

Z is a design target only in this migration. Do not train it now. The canonical base is the clean Gemma 4 26B-A4B-it checkpoint, not its community uncensored variant. Gemma tokenization is always model-specific and must be recomputed rather than reused from Qwen.

The future QAT A/B/C design is:

```text
Route A: Gemma BF16 → ZIPANGU training → merge → Q4
Route B: Google QAT-unquantized checkpoint → ZIPANGU training → merge → Q4
Route C: Gemma → ZIPANGU training → QAT-aware finalization → Q4
```

Compare BF16 score, Q4 score, BF16-to-Q4 degradation, Japanese metrics and general metrics. Do not assume normal SFT preserves QAT behavior.

## Shared Generation-I recipe

The dataset philosophy is shared across K/C/Z, while materialized tokenized training formats are model-specific:

```text
shared candidate data: data/processed/generation-i/
model-specific format: data/processed/models/<model-id>/
```

Use `configs/datasets/i-jp-heavy.yaml` for 75% Japanese / 25% frontier-retention mass and `configs/datasets/i-balanced.yaml` for 50% / 50%. Ratios are token/sampling mass, not row counts. Every source remains read-only until provenance, license, contamination and explicit approval gates pass.

The Qwen family is not assumed to have an interchangeable tokenizer revision. K and C each pin their own base revision; Z uses the Gemma tokenizer and its own revision. Actual token mass is calculated per model even when semantic category targets are shared.

## Evaluation and baseline policy

Every class has a generic `model_id` comparison path:

```text
class base vanilla → ZIPANGU adapted model
```

The untouched base must be evaluated with the same prompt, chat template, generation settings, judge revision, repeats and seeds as the candidate. C additionally has the LLM-jp opponent gate; K treats it as optional context; Z comparison is future-only.

Primary Japanese buckets are Japanese MT-Bench-compatible quality, AnswerCarefully and llm-jp instruction following. Secondary checks include English/general reasoning, Japanese math, Japanese writing and code/reasoning sanity. Preserve generations, parsed answers, judge outputs, configs and aggregates.

## Local RX 7600 research track

Target hardware:

```text
GPU: AMD Radeon RX 7600, 8GB VRAM
CPU: Ryzen 9 7900X
RAM: 96GB
```

The RX 7600 role is `Local Research / CI GPU`, not `production training GPU`. The current isolated environment (`.envs/rocm-windows-py313`) verified PyTorch `2.11.0+rocm7.14.0`, HIP `7.14.60850`, one RX 7600 device, Vulkan runtime, Unsloth import, 4bit K-I model load, the pinned llama.cpp Vulkan/HIP completion gate, and both synthetic and research-only real-data-path QLoRA smokes. Allowed local checks remain hardware detection, Vulkan and ROCm/HIP diagnostics, PyTorch ROCm inspection, Unsloth diagnosis, GGUF inference, quantization and adapter sanity checks. The smokes are infrastructure evidence only and do not authorize a ZIPANGU training run.

The device manifest must retain selected GPU, available devices, PCI ID, gfx architecture, VRAM, backend, driver, ROCm/Vulkan versions and llama.cpp commit. If the Ryzen iGPU is enumerated, record `device_role: iGPU` and `exclude_from_primary_benchmark: true`; never label its result as RX 7600.

Compare CPU, Vulkan and ROCm/HIP with the same model, GGUF, prompt and context. Use one warmup and three measurements, recording prompt-processing tok/s, generation tok/s, time to first token, VRAM, RAM, utilization, stability, errors and backend versions. Do not force settings above 8GB VRAM; OOM retry loops are prohibited.

ROCm 10 is an isolated experimental track. Use a separate venv, WSL distro, build directory or container where possible. Never replace the stable torch/ROCm environment in place.

Current local evidence is split by scope: `reports/hardware/hip_smoke.json` records an actual 512×512 HIP matrix operation; `reports/hardware/results.json` records the RX 7600 runtime and completion-gate summary; `reports/hardware/unsloth_qlora_smoke.json` records the two-step synthetic adapter training; `reports/hardware/unsloth_real_data_smoke.json` records the 16-example research-only real data path; and `reports/hardware/llamacpp_completion_gate.json` plus the CSV/raw logs record the pinned same-GGUF 9-case matrix. ROCm 10 official feasibility is recorded separately in `reports/hardware/ROCM10_FEASIBILITY.md`; the working ROCm 7.14 lane was not replaced.

## Safety and cost gates

`TRAINING_ALLOWED=false` remains active after this migration. No K 1M run, C/Z training, RunPod creation, Heretic processing, model upload, Hugging Face publication, GitHub push or destructive cleanup starts automatically.

The order of compute preference is:

```text
RX 7600 local → RTX 4090 Community Cloud → H100 → future Z-scale compute
```

K falls back to RTX 4090 Community only if local execution is impossible or impractical. C defaults to RTX 4090 Community. Z compute is undecided. The pilot hard cap remains ¥3,000; live price, FX, runtime and budget preflight are required before any paid Pod. Every scale tier requires a separate human GO.

## Non-goals for this migration

- starting 1M/10M/50M training;
- starting C or Z training;
- auto-approving quarantined data;
- replacing or deleting existing C-I configs;
- using Heretic/Abliterated weights as canonical bases;
- treating a local smoke or one benchmark as a release claim;
- damaging the stable Python, ROCm, Unsloth or ComfyUI environment to obtain PASS.

## Completion evidence

The migration report is `reports/migration/KCZ_MIGRATION_REPORT.md`. Hardware evidence belongs in `reports/hardware/`. The completed local finalization handoff is `reports/pretrain_finalization/FINAL_RECOMMENDATION.md`, with the six explicit human gates in `HUMAN_APPROVAL_QUEUE.md`. The source-specific review uses the evidence-only `SOURCE_LICENSE_PROVENANCE_AUDIT.md` and the five-choice, human-owned `SOURCE_APPROVAL_MATRIX.md`; all ten package revisions are pinned, but all ten human source decisions remain unset and no AI/runner decision is permitted. The current run processed 4,643,875 canonical records; the 18,307-row difference from the raw 4,662,182-row inventory is the expected grouping of 18,370 Claude event rows into 63 sessions. The corrected contamination scan found exact `103`, near `0`, manual-review `5`, excluded metadata `1,022,950`, and excluded derivative `895,771`; the old/new record IDs reconcile with zero mismatches, and `CONTAMINATION_RESCAN_AUDIT.md` records the 100-row stratified exact sample. The available-eval extractor/exact item is `CLOSED`, but missing AnswerCarefully coverage keeps the global contamination gate `BLOCKED`. Every unresolved gate is recorded as `BLOCKED`; no unknown host or release gate is promoted to PASS.
