# Dataset Policy

## Isolation rule

Anything marked `eval_only` can never be used for SFT, CPT, preference training, translation, paraphrase, synthetic augmentation, teacher generation or training-time retrieval. This includes AnswerCarefully, Japanese MT-Bench evaluation prompts, llm-jp instruction evaluation prompts, the ZIPANGU private holdout and any future source marked `role: eval_only`.

## Generation-I recipe separation

Generation-I semantic recipes are shared across K/C/Z:

- `configs/datasets/i-jp-heavy.yaml`: 75% Japanese / 25% frontier-retention support;
- `configs/datasets/i-balanced.yaml`: 50% Japanese / 50% frontier-retention support.

Ratios are token/sampling mass, not raw row counts. Shared candidate data is kept under `data/processed/generation-i/`. Tokenized or materialized training output is model-specific under `data/processed/models/<model-id>/`, because tokenizer, chat template and actual token mass are not assumed identical across K/C/Z.

The legacy `c-i-*.yaml` recipes remain for compatibility and historical C-I work; they are not silently rewritten into K recipes.

## Initial candidates

Training candidates are quarantined by default until schema, license and provenance review:

- `awakara/Math-Japanese-8k`;
- `SousiOmine/AceReason-Math-Japanese`;
- `nakasyou/awesome-japanese-corpus` (future CPT, not default Generation-I SFT);
- `Manusagents/GPT-5.6-Sol-Luna-Terra-Traces`;
- `RESMP-DEV/Fable-GPT-5.5-Distillation-Traces`;
- `armand0e/claude-fable-5-claude-code`;
- `r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation`;
- `saidutta69/fable-5-premium`.

Evaluation-only sources are locked:

- `llm-jp/AnswerCarefully`;
- Japanese MT-Bench-compatible evaluation;
- llm-jp instruction-following evaluation;
- ZIPANGU private holdout;
- `llm-jp/HakushoBench` (separate multimodal evaluation track).

`Anthropic/enabling-independent-research` is reference-only unless a later audit establishes a valid training use. Unknown, mixed, contradictory or vendor-specific provenance remains `REVIEW_REQUIRED`; it is never auto-approved.

## English traces

Do not translate everything. Keep an English subset, create Japanese rewrites only for a selected subset, and preserve original↔rewrite provenance. Teacher or trace content does not override dataset license or contamination status.

## Contamination gate

Before any serious run, require:

- an explicit dataset-specific schema adapter for every candidate and eval corpus;
- exclusion of metadata tables, token statistics, smoke subsets and other materialized derivatives before text fingerprinting;
- rejection of empty normalized eval fingerprints and fail-closed exclusion of empty candidate projections;
- normalized exact hashes;
- substring checks;
- n-gram similarity;
- length-calibrated MinHash/LSH and SimHash near-duplicate screening;
- benchmark-name/source filtering;
- manual review of top-similarity hits.

For repositories that expose several physical views of the same logical rows, only the declared authoritative canonical view enters fingerprint comparison. Excluded metadata/derivative rows remain visible as metadata-only audit dispositions and can never become training candidates. A random SimHash baseline is not near-match evidence: SimHash review requires both a high similarity threshold and comparable text length, while MinHash requires its own overlap and length-ratio thresholds.

Missing evaluation access is a blocked or partial result, never `clear`. Trust only atomic completed contamination results; a `.partial` result is not evidence of cleanliness. A fresh scan never implicitly resumes stale chunks; resume must be explicit and must reconcile source order and record IDs.

## Provenance fields

Every accepted source needs:

```text
source_repo
source_config
source_revision
source_split
source_row_id
license
language
category
transform_chain
content_hash
contamination_status
```

Approved registry entries reference a tracked sidecar under `configs/datasets/manifests/<dataset_id>.json` and store its canonical SHA-256 in `provenance.manifest_sha256`. The sidecar contains source metadata and record identities, not raw dataset text. A source is runnable only when the hash matches and `contamination_status` is `clear`.

## Human source approval

License/provenance evidence is recorded in `data/manifests/source_license_provenance_audit.json`; the human worksheet is `reports/pretrain_finalization/SOURCE_APPROVAL_MATRIX.md`, and structured decisions belong only in `data/manifests/source_approval_decisions.json`. The five mutually exclusive values are `APPROVE_AS_IS`, `APPROVE_WITH_OBLIGATIONS`, `APPROVE_SCOPED_SUBSET`, `DEFER_PENDING_EVIDENCE`, and `EXCLUDE`.

An AI or pipeline may collect evidence, validate revisions, flag gaps, and keep a source blocked. It may not select or apply a human decision. A completed source decision still does not update registry state or authorize training; those require a separate reviewed manifest/config change and explicit human GO.

## License and raw-data rule

Merged datasets require source-level auditing. Unknown or incompatible sources are excluded from public release training until resolved. Raw external dataset files remain read-only; the local curation pipeline writes metadata and derived candidates only when the relevant gate allows it.
