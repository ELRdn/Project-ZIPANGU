# Source License and Provenance Audit

TRAINING_ALLOWED=false

AUDIT_DATE=2026-09-05

This report records evidence for human review. It is not legal advice, does not approve any source, and does not infer training permission from a public repository or a dataset-card license tag.

## Audit result

- Local source packages found: `10 / 10`.
- Full package revisions resolved from local Hugging Face cache metadata: `10 / 10`.
- Local package revisions matching the publisher API revision observed on 2026-09-05: `10 / 10`.
- Named local documents matching their pinned remote version by SHA-256: `14 / 14`.
- Human decisions recorded: `0 / 10`.
- Global contamination coverage: `BLOCKED`; AnswerCarefully is still unavailable, so `not_checked_missing_eval_source` is not a clean verdict.

The previous `10 / 10 revision unknown` result was an extractor defect: current Hugging Face `.metadata` sidecars are line-oriented, while `read_local_revision` accepted JSON only. The parser now accepts an immutable 40–64 hex first-line commit while retaining JSON compatibility and returning `mixed:` when sidecars disagree.

Hugging Face documents that dataset-card YAML controls displayed metadata such as the license tag. That makes the tag useful publisher evidence, but not a substitute for checking upstream rights and terms: [Dataset Cards documentation](https://huggingface.co/docs/hub/datasets-cards).

## 1. extraction_wiki_ja

- Package: [llm-jp/extraction-wiki-ja pinned card](https://huggingface.co/datasets/llm-jp/extraction-wiki-ja/blob/b385b781defb9bf4266175cd8d7e7c53af0290b8/README.md), SHA `b385b781defb9bf4266175cd8d7e7c53af0290b8`.
- Local scope: `152,719` rows, 3 files, configs `v0.1`, `v0.2`, `v0.3`.
- License evidence: card declares Apache-2.0; official text: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- Provenance evidence: card says Japanese Wikipedia rows came from `llm-jp-corpus-v3`; Qwen2.5-32B generated instructions/responses and performed filtering.
- Open evidence: exact corpus revision/row mapping, Wikipedia/corpus attribution chain, teacher revision and generation terms are not pinned.
- Contamination: 152,719 `not_checked_missing_eval_source`; no current exact/near/manual hit.
- Local/remote document match: `README.md` SHA-256 `bf74259d2bfb7165cdbc952a119068bd980b98eaab5246db1247fec70523b883`.

## 2. magpie_sft_v1

- Package: [llm-jp/magpie-sft-v1.0 pinned card](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0/blob/4f949e18cfa01d99abe6f92e5629157d69737e6d/README.md), SHA `4f949e18cfa01d99abe6f92e5629157d69737e6d`.
- Local scope: `132,476` rows, 1 JSONL.
- License evidence: card declares Apache-2.0.
- Provenance evidence: card describes CALM3-22B-Chat instruction generation and Qwen2.5-32B-Instruct response generation.
- Open evidence: teacher revisions, generation-time terms and row-level generation identity are not pinned.
- Contamination: 132,473 `not_checked_missing_eval_source`; 3 exact rows remain excluded.
- Local/remote document match: `README.md` SHA-256 `3d500ff31df5473e4b98ef310262e2518d3bc6a4b86444998cf1bf0978670d27`.

## 3. nemotron_sft_multilingual_v2

- Package: [nvidia/Nemotron-SFT-Multilingual-v2 pinned card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v2/blob/971a252224b75414b1b67c55dbe0446d8b6606a0/README.md), SHA `971a252224b75414b1b67c55dbe0446d8b6606a0`.
- Local scope: `370,081` rows across 12 JSONLs.
- License evidence from a full local row scan: 241,889 `cc-by-4.0`; 128,192 `cc-by-sa-4.0`. Official texts: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode).
- Open evidence: the two obligation sets cannot be collapsed into one package-wide treatment; selected-row obligations and derived-model/redistribution handling need an explicit human interpretation.
- Contamination: 370,075 `not_checked_missing_eval_source`; 6 exact rows remain excluded.
- Local/remote document match: `README.md` SHA-256 `41e21b069e2f61d2897a6422e425ff2890a66981bc00aa18dc84fb65b26bdd11`.

## 4. math_japanese_8k

- Package: [awakara/Math-Japanese-8k pinned card](https://huggingface.co/datasets/awakara/Math-Japanese-8k/blob/e86bb58896c228e91acec2cbbb768d37f14ecdb3/README.md), SHA `e86bb58896c228e91acec2cbbb768d37f14ecdb3`.
- Local scope: `8,094` rows.
- License evidence: no card license and no package license file.
- Provenance evidence: card says Kimi K2.6/K2.7-code generated problems and Gemma4 31B generated answers.
- Open evidence: grant, teacher revisions, generation method, provider-output terms and row identities are absent.
- Contamination: 8,094 `not_checked_missing_eval_source`; no current exact/near/manual hit.
- Local/remote document match: `README.md` SHA-256 `49ac144b95c6d9623552465bc9a436e4fb65b8f7d43d7ee47b2d306b264a160b`.

## 5. ace_reason_math_japanese

- Package: [SousiOmine/AceReason-Math-Japanese pinned card](https://huggingface.co/datasets/SousiOmine/AceReason-Math-Japanese/blob/037b823f6ab85aee4a7de51ca35476c6d1079f0a/README.md), SHA `037b823f6ab85aee4a7de51ca35476c6d1079f0a`.
- Local scope: `10,000` rows.
- License evidence: card and `LISENCE` file declare CC-BY-4.0.
- Provenance evidence: card says the first 10,000 rows of `nvidia/AceReason-Math` were translated with Deepseek-V4-Flash-0731; answers were unchanged.
- Open evidence: upstream dataset revision/row mapping and translator revision/provider terms are not pinned.
- Contamination: 10,000 `not_checked_missing_eval_source`; no current exact/near/manual hit.
- Local/remote document matches: `README.md` SHA-256 `bfc1864fc1a24f701f19d298c53031f777ed363aaf4772e57ca9129b620288a9`; `LISENCE` SHA-256 `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411`.

## 6. gpt_5_6_traces

- Package: [Manusagents/GPT-5.6-Sol-Luna-Terra-Traces pinned card](https://huggingface.co/datasets/Manusagents/GPT-5.6-Sol-Luna-Terra-Traces/blob/07e1745da4cb8220c4811e200bfeb97768290e21/README.md), SHA `07e1745da4cb8220c4811e200bfeb97768290e21`.
- Local scope: `6,288` rows: 5,402 from `greghavens/gpt-5.6-sol-coding-and-debugging-traces`, 886 from `empero-ai/gpt-5.6-luna-sft-900x`.
- License evidence: card declares CC-BY-4.0.
- Provenance evidence: rows include upstream dataset/config/split/row index and `row_hash`; the card calls provenance source-asserted/content-verified rather than OpenAI-certified.
- Open evidence: upstream revisions are unpinned; trace origin remains publisher-asserted; current provider-output restrictions must be reviewed against the planned use. Primary terms: [OpenAI Terms of Use](https://openai.com/policies/terms-of-use/), [OpenAI Services Agreement](https://openai.com/en-GB/policies/services-agreement/).
- Contamination: 6,288 `not_checked_missing_eval_source`; no current exact/near/manual hit.
- Local/remote document match: `README.md` SHA-256 `7ea232b4a821d0c26014774bf65bd5f6ea717eb9228b0bfee7d3d2b16102305c`.

## 7. fable_5_5_distillation

- Package: [RESMP-DEV/Fable-GPT-5.5-Distillation-Traces pinned card](https://huggingface.co/datasets/RESMP-DEV/Fable-GPT-5.5-Distillation-Traces/blob/15ba38ac01b610cc986b728a2d55fcb8db9c2096/README.md), SHA `15ba38ac01b610cc986b728a2d55fcb8db9c2096`.
- Local scope: `1,981,131` rows across 20 Parquet files; this is partial against the card's `9,057,143` rows.
- License evidence: card declares CC-BY-4.0 for curation/merge and states original upstream licenses remain in force.
- Provenance evidence: local rows preserve a `source` field across many corpora and trace origins.
- Open evidence: the package tag is not a blanket replacement for upstream licenses; per-record license is incomplete; the local subset contains Codex/personal-Codex origins while the card marks personal Codex traces personal-use/not-for-redistribution; provider terms need source-specific review.
- Contamination: 1,981,089 `not_checked_missing_eval_source`; 37 exact and 5 manual-review rows remain excluded.
- Local/remote document match: `README.md` SHA-256 `7b61f075eb3b881071247f4c52be0e2185e22215e43fd63e81b0d7d7759e8747`.

## 8. claude_fable_code

- Package: [armand0e/claude-fable-5-claude-code pinned card](https://huggingface.co/datasets/armand0e/claude-fable-5-claude-code/blob/c19fb6831700da833b22d1c9cdac47fe8603685c/README.md), SHA `c19fb6831700da833b22d1c9cdac47fe8603685c`.
- Local scope: 18,370 raw events in 63 JSONLs, grouped into 63 canonical sessions.
- License evidence: no card license and no package license file.
- Provenance evidence: card describes anonymized raw traces from two teams and overlap with Glint Research/Fable-5-traces.
- Open evidence: contributor rights, redistribution/training grant and the applicable provider terms are not recorded. Primary terms: [Anthropic Commercial Terms](https://www.anthropic.com/legal/commercial-terms), [Anthropic Consumer Terms](https://www.anthropic.com/legal/consumer-terms).
- Contamination: 63 canonical sessions are `not_checked_missing_eval_source`; no current exact/near/manual hit.
- Local/remote document match: `README.md` SHA-256 `2a090d783e93372383a7d58c5c14da4f688ddebfebb9fcdcc60c4e99dd75a0b7`.

## 9. frontier_multi_teacher

- Package: [r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation pinned card](https://huggingface.co/datasets/r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation/blob/7a3473446840bcc397928cd8183d4b3ba3ca13a7/README.md), SHA `7a3473446840bcc397928cd8183d4b3ba3ca13a7`.
- Local scope: 1,976,658 physical scan records. The schema adapter marks 1,022,950 metadata and 895,771 derivative/materialized records out of scan scope, leaving 57,937 authoritative content rows across 8 files.
- License evidence: card says `other`. Row-level evidence includes 12,045 explicitly noncommercial rows and 15,122 `unknown-see-provenance` rows, plus MIT, Apache, CC-BY, CC-BY-SA, ODC-BY and custom/other classes.
- Provenance evidence: all 57,937 authoritative rows include source repository/revision fields.
- Open evidence: local/pinned `PROVENANCE.md` says `distillation-51389` and 51,389 rows while the card/canonical package says 57,937; package `LICENSE` limits use to controlled noncommercial research pending review; teacher-service terms require source-specific review. Alibaba's Model Studio terms are primary evidence for one named service path: [Alibaba Cloud Model Studio terms](https://www.alibabacloud.com/help/en/legal/latest/alibaba-cloud-international-website-product-terms-of-service-v-3-8-0).
- Contamination: 57,880 `not_checked_missing_eval_source`; 57 exact rows remain excluded; metadata/derivative records were excluded before fingerprinting.
- Local/remote document matches: `README.md` `69c9b2776d57c7450aca5ffee91a472631cc931749d32d6ab340fad20fc81cad`; `LICENSE` `3cec953e36c3e8b41b469d1ed6c25590d551c29bf58e09905496a2e24a1be4ec`; `PROVENANCE.md` `e049247f9687fa16e2bb7ca6ccec53f9d51772dde6b7e51d0ebb5c01587d0257`; `manifest.json` `9c9c6404c6e5123fcd3d552710d2d23a94eb294f16031b3aedae29dc40f70cf0`.

## 10. fable_5_premium

- Package: [saidutta69/fable-5-premium pinned card](https://huggingface.co/datasets/saidutta69/fable-5-premium/blob/684cb1f849fe4a1c96f55351e1d7366f9888bb28/README.md), SHA `684cb1f849fe4a1c96f55351e1d7366f9888bb28`.
- Local scope: `6,365` logical rows across JSONL and Parquet train/validation/test files. The card's `12,730` total counts both physical formats of the same logical rows.
- License evidence: card tag says MIT; no package `LICENSE` file is present.
- Provenance evidence: the current card's source table is empty.
- Open evidence: upstream source/revisions, teacher rights and row-level lineage are absent; the card tag has no accompanying license file.
- Contamination: 6,365 `not_checked_missing_eval_source`; no current exact/near/manual hit.
- Local/remote document match: `README.md` SHA-256 `3e951aa46a9931bcf5fa2eab0ca24bd5beb8cda9663ac429d49754ea4d817527`.

## Handoff boundary

The five permitted human outcomes are defined in `SOURCE_APPROVAL_MATRIX.md` and `data/manifests/source_approval_decisions.json`. Every `decision`, reviewer, date, scope and rationale field is unset. No evidence state in this report maps automatically to an outcome.
