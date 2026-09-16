# Project ZIPANGU

**Low-cost, reproducible Japanese post-training research for strong open-weight language models.**

Project ZIPANGU asks:

> Can careful low-cost Japanese adaptation of strong open-weight models compete with Japanese-focused scratch-trained foundation models?

The research strategy is a controlled scale-up:

```text
K (Kei / 軽量級) → C (Chū / 中量級) → Z (Zyū / 重量級)
```

Generation I is the first ZIPANGU recipe generation. Model class and recipe generation are intentionally separate:

```text
Generation I
              ┌─ K-I-4B
Recipe I ─────┼─ C-I-9B
              └─ Z-I-26B-A4B
```

## Model family

| Class | Canonical model | Upstream base | Role | Status |
|---|---|---|---|---|
| K — lightweight | `ZIPANGU-K-I-4B` | `empero-ai/Qwen3.8-4B-Distill` | recipe, local AMD and cheap-pilot validation | local completion gate passed; 1M data/training NO-GO pending human review |
| C — middleweight | `ZIPANGU-C-I-9B` | `empero-ai/Qwen3.8-9B-Distill` | primary LLM-jp-4-8B Thinking challenge | configured; follows K |
| Z — heavyweight | `ZIPANGU-Z-I-26B-A4B` | `google/gemma-4-26B-A4B-it` | MoE scale-up and future QAT research | future only |

K is not required to defeat LLM-jp directly. K succeeds when the vanilla-to-adapted Japanese improvement is repeatable, general reasoning does not materially collapse, and contamination/provenance gates remain clean. C inherits the established research question and is the first primary comparison against `llm-jp/llm-jp-4-8b-thinking`. Z is designed now but is not an immediate training target.

Community Heretic/Abliterated checkpoints are separate, non-canonical artifacts. Official ZIPANGU focuses on Japanese adaptation, dataset curation, benchmarking and reproducibility.

## Research roadmap

```text
Phase K: ZIPANGU-K-I-4B
  → recipe validation
  → RX 7600 local research
  → llama.cpp Vulkan/HIP completion gate PASS
  → isolated HIP/Unsloth QLoRA + real-data path smoke PASS
  → corrected 4,643,875-record contamination rescan completed
  → available-eval extractor/exact revalidation CLOSED; global contamination gate BLOCKED on AnswerCarefully
  → recipe/format/provenance gates BLOCKED
  → human approval → 1M → evaluation → 10M → optional 50M

Phase C: ZIPANGU-C-I-9B
  → transfer the winning K recipe
  → LLM-jp-4-8B Thinking challenge

Phase Z: ZIPANGU-Z-I-26B-A4B
  → scale the K/C recipe to Gemma MoE
  → QAT research → future 32B-class comparison
```

Every scale-up is a separate experiment and requires human GO. `TRAINING_ALLOWED=false` remains in force for this migration.

The local completion run processed 4,643,875 canonical records and generated both research candidate recipes. The corrected contamination rescan found exact `103`, near `0`, manual-review `5`, excluded metadata `1,022,950`, and excluded derivative `895,771`; all old/new record IDs reconcile, and a metadata-only stratified sample covers `100 / 103` genuine exact candidates. The available-eval extractor/exact item is closed, but AnswerCarefully coverage keeps the global contamination gate blocked. All ten local source-package revisions are now pinned and 14 named local evidence documents match their pinned remote versions, but package identity is not source approval: all `10 / 10` human license/provenance decisions remain unset. Neither candidate is runnable: JP-heavy reached 208,924/1,000,000 tokens and balanced reached 316,249/1,000,000 tokens; format, license/provenance and LoRA-target approval gates also remain open. See `reports/pretrain_finalization/CONTAMINATION_RESCAN_AUDIT.md`, `SOURCE_LICENSE_PROVENANCE_AUDIT.md`, `SOURCE_APPROVAL_MATRIX.md`, `FINAL_RECOMMENDATION.md` and `HUMAN_APPROVAL_QUEUE.md`. Local infrastructure completion is not a production-training approval.

## K-I immediate target

The default metadata finalization and local pilot target is `ZIPANGU-K-I-4B`. Use the model-scoped configs under `configs/train/k-i/` and the shared Generation-I recipes under `configs/datasets/i-*.yaml`. Existing C-I configs remain available for the later transfer stage.

The K-I success criteria are:

1. Clear improvement on multiple Japanese primary metrics against the untouched K base.
2. No material collapse on general reasoning or English/general regression checks.
3. No training/evaluation contamination and complete source provenance.
4. An LLM-jp-4-8B comparison is a desired bonus, not the K gate.

## Dataset strategy

Generation-I recipes are class-independent. Ratios mean token/sampling mass, not raw row counts:

- **JP-heavy**: 75% Japanese / 25% frontier-retention support.
- **Balanced**: 50% Japanese / 50% frontier-retention support.

Shared candidate data lives under `data/processed/generation-i/`; model-tokenized or materialized training formats must be kept model-specific. Each model uses its own pinned tokenizer revision, including a fresh Gemma tokenization for Z.

Anything marked `eval_only` is permanently excluded from SFT, CPT, translation, paraphrase, synthetic augmentation, teacher generation and training-time retrieval. Quarantined or provenance-uncertain sources remain blocked and are never auto-approved.

See `docs/DATASET_POLICY.md` and `docs/RESEARCH_PLAN.md`.

## Evaluation isolation

Primary Japanese evaluations:

- Japanese MT-Bench-compatible conversational/instruction quality
- `llm-jp/AnswerCarefully`
- llm-jp instruction-following evaluation

Secondary checks include English/general reasoning, Japanese math and writing holdouts, and a code/reasoning sanity set. Preserve raw generations, parsed answers, judge outputs, configs and manifests. C-I victory remains at least 2 of 3 primary Japanese wins plus the general-regression and contamination gates.

## RX 7600 local research

The AMD Radeon RX 7600 8GB is a **Local Research / CI GPU**, not the production training GPU. The isolated local lane now has a verified PyTorch HIP tensor smoke, Vulkan runtime probe, a pinned llama.cpp Vulkan/HIP completion gate, the 9-case CPU/Vulkan/HIP same-GGUF matrix, and both synthetic and research-only real-data-path QLoRA smokes. A smoke or local benchmark is infrastructure evidence, not a training or release result.

ROCm 10 is experimental and isolated from the stable environment. No global pip uninstall, in-place torch replacement, system ROCm overwrite or working-venv destruction is allowed. The Ryzen 9 7900X iGPU must be recorded and excluded from the primary RX 7600 benchmark when detected.

```bash
python scripts/hardware_probe.py
python scripts/benchmark_local_inference.py
python scripts/run_hip_smoke.py
python scripts/run_unsloth_qlora_smoke.py --output-dir .cache/smoke-runs/manual-k-i
python scripts/run_unsloth_real_data_smoke.py --dataset-id math_japanese_8k --examples 16 --steps 2
```

The synthetic command is limited to two invented records and two steps. The real-data command is limited to 16–64 train-candidate examples and 2–10 steps, never runs evaluation, never changes registry approval, and remains research-only while provenance and contamination gates are open. The pinned llama.cpp executables, official GGUF hash, raw logs and matrix are recorded under `reports/hardware/`.

## Training and RunPod policy

- Local RX 7600 first for K research; RTX 4090 Community Cloud is the fallback.
- C defaults to RTX 4090 Community Cloud.
- Z compute is decided later.
- Pilot hard cap: **¥3,000**.
- Resolve live price, FX, runtime and budget before any paid Pod.
- Human GO is required for every 1M/10M/50M scale, paid Pod, GPU/tier increase, extension, C/Z training, Heretic processing, upload/publication and destructive cleanup.
- Persist artifacts, stop the Pod and verify it stopped after success or unrecoverable failure.

See `docs/RUNPOD_OPERATIONS.md`.

## Quick start

```bash
python -m venv .venv
# activate .venv
pip install -e ".[train,dev]"

python scripts/validate_registry.py --mode structure
python scripts/validate_registry.py --mode all
python scripts/pretrain_finalization.py --model ZIPANGU-K-I-4B --stage preflight
python scripts/preflight.py --hourly-usd 0.50 --fx-jpy-per-usd 160 --planned-hours 8
```

The price/FX values above are examples only. Use live values. The validator returns `0` for a structural pass, `1` for malformed configuration and `2` when the configuration is valid but intentionally not runnable because safety or approval gates remain open.

Approved training sources must reference a tracked provenance manifest under `configs/datasets/manifests/`. Raw dataset files remain local and ignored.

## Repository layout

```text
project-zipangu/
├─ README.md
├─ AGENTS.md
├─ PROJECT_DECISIONS.json
├─ pyproject.toml
├─ docs/
├─ configs/
│  ├─ models/        # K/C/Z canonical registry and variants
│  ├─ datasets/      # shared Generation-I and legacy C-I recipes
│  ├─ train/         # model-scoped training configs
│  └─ eval/          # class-generic evaluation configs
├─ src/zipangu/
├─ scripts/
├─ reports/
│  ├─ migration/
│  └─ hardware/
├─ runs/              # artifacts ignored except .gitkeep
└─ data/              # local data ignored except manifests/.gitkeep
```

## Reproducibility rule

Every publishable result must map to:

```text
model id + exact base/revision + model tokenizer/revision
+ dataset recipe + source manifests/revisions/transforms
+ train config + seed + git commit + dependency/GPU manifest
+ eval config + raw outputs
```

If that chain is incomplete, the result is exploratory. Model family names and quantization names are separate metadata fields; for example `ZIPANGU-K-I-4B-Q4_K_M` is a quantized artifact, not a new model family.

## License

Project code license is intentionally **TBD** until dataset/model redistribution constraints are audited. A public repository does not imply permission to redistribute upstream datasets, adapters or derived weights.
