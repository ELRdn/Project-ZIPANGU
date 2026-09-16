# Human Approval Queue

TRAINING_ALLOWED=false

This package records decisions for a human reviewer. It does not approve data, training, paid compute, or publication.

## 1. LICENSE

### WHAT

Decide whether each of the ten candidate sources permits the planned transformation, training use, adapter/weight handling, and required attribution.

### EVIDENCE

- `10` of `10` source decisions remain unset; AI/runner decisions recorded: `0`.
- Nemotron reports mixed CC-BY-4.0 / CC-BY-SA-4.0 metadata; Math-Japanese-8k and claude-fable-code have unknown license metadata; frontier_multi_teacher reports `other`.
- Evidence audit: `data/manifests/source_license_provenance_audit.json`; human-only record: `data/manifests/source_approval_decisions.json`; worksheet: `SOURCE_APPROVAL_MATRIX.md`.

### RECOMMENDED DECISION

A human reviewer must select exactly one of five choices per source. No choice is preselected, and a source decision alone does not authorize training.

### RISK

Metadata-only approval could create incompatible attribution, share-alike, redistribution, or derivative-model obligations.

## 2. PROVENANCE

### WHAT

Approve only source rows with immutable upstream revision and row-level origin evidence.

### EVIDENCE

- `10` of `10` local source packages have a full immutable revision; `0` remain `unknown`.
- Pinned local/remote document hash matches recorded by the audit: `14`.
- A package revision does not by itself resolve upstream corpus, teacher-model, contributor-rights, or service-terms lineage.
- Raw inventory has `4,662,182` rows; canonical processing produced `4,643,875` records. The `18,307`-row difference is the expected Claude event-to-session grouping (18,370 events to 63 sessions), not silent deletion.
- `3,247,265` canonical records had a usable formatted-token count; unavailable rows remain excluded/fail-closed.

### RECOMMENDED DECISION

Use the source matrix to record whether the remaining lineage evidence is accepted, conditionally accepted, scope-limited, deferred, or excluded. The runner records no answer.

### RISK

Treating a pinned package revision as complete upstream provenance can still make the experiment legally or scientifically irreproducible.

## 3. CONTAMINATION HITS

### WHAT

Review every exact/near-match quarantine class and complete the scan against all required evaluation corpora.

### EVIDENCE

- Available eval fingerprints indexed: `830`.
- Corrected full rescan classifications: exact `103`, near `0`, manual-review `5`.
- Extractor defect: `CLOSED`; all `4,643,875` record IDs reconciled old-vs-new, the prior `1,393,704` empty-hash exacts were removed from valid exact evidence, and `100` / `103` genuine exact candidates were sampled (target up to 100).
- AnswerCarefully v2.0 remains access-denied; Japanese MT-Bench and llm-jp-instructions were processed.

### RECOMMENDED DECISION

CLOSE the empty-fingerprint extractor defect and available-eval exact revalidation item. Keep every genuine exact, near and manual-review row excluded; the global contamination gate remains BLOCKED until AnswerCarefully is available.

### RISK

Training before resolution can contaminate evaluation and invalidate every improvement claim.

## 4. DATASET RECIPE

### WHAT

Choose whether to revise the JP-heavy/balanced bucket targets, template-family cap, and safe-format filters before regenerating the 1M candidate.

### EVIDENCE

- JP-heavy selected `208,924` / 1,000,000 tokens (`20.9%`).
- Balanced selected `316,249` / 1,000,000 tokens (`31.6%`).
- Format-invalid samples: `{"pilot-1m-balanced": 22, "pilot-1m-jp-heavy": 13}`; no automatic truncation was applied.
- Fable Premium remains excluded while tool-call canonicalization/trace integrity is blocked.

### RECOMMENDED DECISION

Revise the recipe in a new config/manifest, then regenerate. Do not relax the 5% template cap or 4096-token/tool-boundary checks without explicit human rationale.

### RISK

Relaxing diversity or format gates merely to hit 1M can produce a biased or non-runnable training set.

## 5. LORA TARGET POLICY

### WHAT

Approve explicit production LoRA target modules for ZIPANGU-K-I-4B.

### EVIDENCE

- The real-data smoke passed with `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, r=4, alpha=8, two optimizer steps.
- Production K-I configs intentionally remain `target_policy: auto_review_required` with r=32 and alpha=64.
- The smoke proves the local data/trainer path only; it is not a quality or production-policy approval.

### RECOMMENDED DECISION

Approve or amend an explicit module list in a new immutable run config before any 1M run. Do not use automatic target discovery for the approved experiment.

### RISK

An unreviewed target set changes trainable capacity, memory use, cost, and comparison validity.

## 6. 1M GO

### WHAT

Issue or withhold the human GO for the first 1M-token K-I experiment.

### EVIDENCE

- Current blocking stages: `eval:BLOCKED, contamination:BLOCKED, fable:BLOCKED, candidate:BLOCKED, format:BLOCKED, baseline:BLOCKED`.
- Local RX 7600 infrastructure, llama.cpp backends, 9-case benchmark, and real-data smoke passed, but candidate/eval/provenance gates did not.
- No paid Pod, production training, C/Z training, or publication was started.

### RECOMMENDED DECISION

NO-GO now. Reconsider only after license/provenance approval, complete contamination coverage, a format-valid near-1M candidate, explicit LoRA targets, and budget preflight.

### RISK

A premature GO would spend budget on an experiment whose data and evaluation claims are not defensible.
