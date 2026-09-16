# ZIPANGU model naming

## Canonical identifier

The canonical form is:

```text
ZIPANGU-{Class}-{Generation}-{Size}
```

| Component | Values | Meaning |
|---|---|---|
| Class | `K`, `C`, `Z` | Kei / 軽, Chū / 中, Zyū / 重 model scale |
| Generation | `I`, `RO`, `HA`, `NI`, ... | ZIPANGU recipe generation in iroha order |
| Size | `4B`, `9B`, `26B-A4B`, ... | Model-family size descriptor |

Generation identifies a ZIPANGU training recipe, not the upstream checkpoint version. A model-size change is a class change; a substantial recipe change advances the generation. A bugfix does not by itself advance the generation.

## Generation-I canonical family

```text
ZIPANGU-K-I-4B       = empero-ai/Qwen3.8-4B-Distill
ZIPANGU-C-I-9B       = empero-ai/Qwen3.8-9B-Distill
ZIPANGU-Z-I-26B-A4B  = google/gemma-4-26B-A4B-it
```

K is the immediate research target. C is the primary later LLM-jp challenge. Z is a future MoE/QAT track and is not an immediate training target.

## Canonical versus variants

Canonical artifacts use the clean upstream base plus the ZIPANGU recipe. Heretic, Abliterated and refusal-removal checkpoints are recorded in `configs/models/variants.yaml` only as non-canonical community variants.

The required order for an optional variant is:

```text
clean base → ZIPANGU Japanese post-training → merge
→ canonical evaluation → optional Heretic/Abliteration → repeat evaluation
```

Variant IDs are separate artifacts and must never overwrite or be substituted for their `canonical_id`.

## Quantized artifacts

Quantization is metadata, not a new model family:

```text
ZIPANGU-K-I-4B-Q4_K_M
ZIPANGU-C-I-9B-Q4_K_M
ZIPANGU-Z-I-26B-A4B-Q4_0
```

The base model ID, resolved revision, tokenizer/revision, recipe, quantization format and artifact hash are stored separately in the release manifest. A Q4 artifact cannot be used to infer the BF16 model's evaluation result.
