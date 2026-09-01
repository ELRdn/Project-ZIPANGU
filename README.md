# Project ZIPANGU

**Low-cost, reproducible Japanese post-training research for strong open-weight language models.**

Project ZIPANGU asks:

> How far can careful Japanese post-training, dataset curation, and clean evaluation push an already-strong open-weight model without scratch pretraining?

The first model is **ZIPANGU-C-I-9B**, based on `empero-ai/Qwen3.8-9B-Distill`.

## First target

Beat **LLM-jp-4-8B Thinking** on a clean Japanese evaluation suite while preserving broad reasoning ability.

A first-stage win requires:

1. Win at least **2 of 3** primary Japanese evaluations.
2. No material general/English capability regression.
3. Evaluation data remains strictly isolated from training.
4. Exact data/config/code/environment/eval provenance is preserved.

A contaminated benchmark win does **not** count.

## Naming

`ZIPANGU-{Class}-{Generation}-{Size}`

- `K` = **Kei / 軽** — lightweight
- `C` = **Chū / 中** — middleweight
- `Z` = **Zyū / 重** — heavyweight

Generation uses *iroha* order:

`I → RO → HA → NI → HO → HE → TO → ...`

The generation means the **ZIPANGU training-recipe generation**, not the upstream base-model version.

First model: **ZIPANGU-C-I-9B**.

## Research stages

```text
Baseline → Smoke → 1M → 10M → 50M → Release candidate
```

Every scale-up is a separate experiment and requires human GO.

### Stage 0 — Baseline

Evaluate untouched:

- `empero-ai/Qwen3.8-9B-Distill`
- `llm-jp/llm-jp-4-8b-thinking`

No tuning before the baseline report is committed.

### Stage 1 — Smoke

Verify tokenizer/chat formatting, loss, checkpoint save/reload, inference, evaluation, provenance and budget guard.

### Stage 2+ — Pilots

Scale only when the previous stage shows healthy behavior and a useful signal.

## Dataset strategy

Two explicit variants:

- **JP-heavy**: initial target 75% Japanese / 25% frontier reasoning.
- **Balanced ablation**: 50% Japanese / 50% frontier reasoning.

Some English reasoning traces stay in English; a selected subset may receive high-quality Japanese rewrites with pair provenance.

See `docs/DATASET_POLICY.md`.

## Evaluation isolation

Evaluation-only by default:

- `llm-jp/AnswerCarefully`
- Japanese MT-Bench-compatible evaluation
- llm-jp instruction-following evaluation
- ZIPANGU private holdout

`llm-jp/HakushoBench` is tracked as eval-only but is not a primary C-I text benchmark.

## Training stack

Reference implementation:

- Transformers
- TRL
- PEFT
- Accelerate

Optimization backend:

- Unsloth only after the reference path is validated.

`Qwen3.8-9B-Distill` requires recent Qwen3.5-compatible Transformers and optimized Gated DeltaNet kernels for practical performance, so CUDA/PyTorch/kernel versions are recorded per run rather than guessed globally.

## RunPod policy

- Preferred GPU: **RTX 4090**
- Preferred tier: **Community Cloud**
- Pilot hard cap: **¥3,000**
- Live price check immediately before provisioning
- Secure Cloud / GPU upgrade requires human GO
- New scale tier requires human GO
- Completed or failed runs must stop the Pod after artifacts are persisted
- Destructive deletion requires human GO

RunPod MCP is used for semi-automatic infrastructure management. See `docs/RUNPOD_OPERATIONS.md`.

## Quick start

```bash
python -m venv .venv
# activate .venv
pip install -e ".[train,dev]"

python scripts/validate_registry.py --mode structure
python scripts/validate_registry.py --mode all
python scripts/preflight.py --hourly-usd 0.50 --fx-jpy-per-usd 160 --planned-hours 8
```

The price/FX values above are examples only. Use live values.

The registry validator returns `0` for a passing check, `1` for a malformed
configuration, and `2` when the configuration is valid but not runnable yet
(for example, because sources remain quarantined or LoRA targets are pending
review). `--mode structure` intentionally allows quarantined candidates so
dataset preparation can proceed without approving them.

Approved training sources must reference a tracked provenance manifest under
`configs/datasets/manifests/`. Raw dataset files remain local and ignored.

## Repository layout

```text
project-zipangu/
├─ README.md
├─ AGENTS.md
├─ PROJECT_DECISIONS.json
├─ pyproject.toml
├─ requirements.lock.txt
├─ docs/
├─ configs/
├─ src/zipangu/
├─ scripts/
├─ tests/
├─ runs/            # artifacts ignored except .gitkeep
└─ data/            # local data ignored except .gitkeep
```

## Reproducibility rule

Every publishable result must map to:

```text
model + dataset manifest + source revisions + transforms + train config
+ seed + git commit + dependency/CUDA/GPU environment + eval config
+ raw outputs
```

If that chain is incomplete, the result is exploratory.

## License

Project code license is intentionally **TBD** until dataset/model redistribution constraints are audited. A public repository does not imply permission to redistribute upstream datasets or derived weights.
