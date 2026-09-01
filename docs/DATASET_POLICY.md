# Dataset Policy

## Isolation rule

Anything `eval_only` can never be used for SFT, CPT, preference training, translation, paraphrase, synthetic augmentation, teacher generation or training-time retrieval.

## Initial candidates

Training candidates are quarantined by default until schema/license/provenance review:

- `awakara/Math-Japanese-8k`
- `SousiOmine/AceReason-Math-Japanese`
- `nakasyou/awesome-japanese-corpus` (future CPT, not default C-I SFT)
- `Manusagents/GPT-5.6-Sol-Luna-Terra-Traces`
- `RESMP-DEV/Fable-GPT-5.5-Distillation-Traces`
- `armand0e/claude-fable-5-claude-code`
- `r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation`
- `saidutta69/fable-5-premium`

Evaluation-only:

- `llm-jp/AnswerCarefully`
- Japanese MT-Bench-compatible evaluation
- llm-jp instruction-following evaluation
- ZIPANGU private holdout
- `llm-jp/HakushoBench` (separate multimodal evaluation track)

`Anthropic/enabling-independent-research` is reference-only unless a later audit establishes a valid training use.

## Recipes

### JP-heavy
Initial sampling mass: 75% Japanese / 25% frontier reasoning. Changes within the 70–80/20–30 envelope require a new config.

### Balanced ablation
50% Japanese / 50% frontier reasoning.

Ratios mean **sampling/token mass**, not raw row counts.

## English traces

Do not translate everything. Keep an English subset, create Japanese rewrites for a selected subset, and preserve original↔rewrite provenance.

## Contamination gate

At minimum before serious run:

- normalized exact hashes
- substring checks
- n-gram similarity
- MinHash/LSH near-duplicate screening
- benchmark-name/source filtering
- manual review of top-similarity hits

## Provenance fields

`source_repo, source_config, source_revision, source_split, source_row_id, license, language, category, transform_chain, content_hash, contamination_status`

Approved registry entries reference a tracked sidecar manifest at
`configs/datasets/manifests/<dataset_id>.json` and store its canonical SHA-256
in `provenance.manifest_sha256`. The manifest contains source metadata and
`records` with `source_row_id` and `content_hash`; it contains no raw dataset
text. A source is runnable only when the manifest hash matches and
`contamination_status` is `clear`.

## License rule

Merged datasets require source-level auditing. Unknown or incompatible sources are excluded from public release training until resolved.
