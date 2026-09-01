# ZIPANGU Dataset Inventory

This is a metadata-only inventory. Raw dataset files remain outside the repository and are read-only.

## extraction_wiki_ja

- repository: `llm-jp/extraction-wiki-ja`
- local path: `E:\zipangu-datesets\llm-jp__extraction-wiki-ja`
- found: `True`
- files (physical / logical): `3` / `3`
- bytes (physical / logical): `107120056` / `107120056`
- rows: `152719` (known=`True`, unknown files=`0`)
- formats: `.parquet`
- configs: `v0.1, v0.2, v0.3`
- splits: `train`
- revision: `unknown`
- license metadata: `"apache-2.0"`
- language metadata: `["ja"]`
- source category: `japanese_instruction_extraction`
- teacher/source fields: `none detected`
- length profile: `{"unit": "raw_text_field_characters", "selection": "deterministic_logical_file_order_prefix", "sample_rows_requested": 10000, "observed_rows": 10000, "median": 563.0, "p95": 1285.0}`
- audit scope: `full`
- adapter warnings: `none`

### Schema

```json
{
  "conversations": [
    "list<element: struct<content: string, role: string>>"
  ],
  "id": [
    "string"
  ]
}
```

### Sample policy

Raw row samples are intentionally omitted from tracked inventory reports.

## magpie_sft_v1

- repository: `llm-jp/magpie-sft-v1.0`
- local path: `E:\zipangu-datesets\llm-jp__magpie-sft-v1.0`
- found: `True`
- files (physical / logical): `1` / `1`
- bytes (physical / logical): `284154581` / `284154581`
- rows: `132476` (known=`True`, unknown files=`0`)
- formats: `.jsonl`
- configs: `default`
- splits: `magpie-sft-v1.0`
- revision: `unknown`
- license metadata: `"apache-2.0"`
- language metadata: `["ja"]`
- source category: `japanese_general_instruction`
- teacher/source fields: `none detected`
- length profile: `{"unit": "raw_text_field_characters", "selection": "deterministic_logical_file_order_prefix", "sample_rows_requested": 10000, "observed_rows": 10000, "median": 622.0, "p95": 1015.0}`
- audit scope: `full`
- adapter warnings: `none`

### Schema

```json
{
  "conversations": [
    "list"
  ],
  "gen_asst_configs": [
    "dict"
  ],
  "gen_usr_configs": [
    "dict"
  ],
  "id": [
    "int"
  ]
}
```

### Sample policy

Raw row samples are intentionally omitted from tracked inventory reports.

## nemotron_sft_multilingual_v2

- repository: `nvidia/Nemotron-SFT-Multilingual-v2`
- local path: `E:\zipangu-datesets\nvidia__Nemotron-SFT-Multilingual-v2`
- found: `True`
- files (physical / logical): `12` / `12`
- bytes (physical / logical): `12460641142` / `12460641142`
- rows: `370081` (known=`True`, unknown files=`0`)
- formats: `.jsonl`
- configs: `default`
- splits: `code_hi, code_ja, code_ko, code_pt, math_hi, math_ja, math_ko, math_pt, stem_hi, stem_ja, stem_ko, stem_pt`
- revision: `unknown`
- license metadata: `["cc-by-4.0", "cc-by-sa-4.0"]`
- language metadata: `["en", "hi", "ja", "ko", "pt"]`
- source category: `japanese_stem_code_math`
- teacher/source fields: `none detected`
- length profile: `{"unit": "raw_text_field_characters", "selection": "deterministic_logical_file_order_prefix", "sample_rows_requested": 10000, "observed_rows": 10000, "median": 36010.0, "p95": 95542.05}`
- audit scope: `full`
- adapter warnings: `none`

### Schema

```json
{
  "license": [
    "str"
  ],
  "messages": [
    "list"
  ],
  "metadata": [
    "dict"
  ],
  "tools": [
    "list"
  ],
  "used_in": [
    "list"
  ],
  "uuid": [
    "str"
  ]
}
```

### Sample policy

Raw row samples are intentionally omitted from tracked inventory reports.

## math_japanese_8k

- repository: `awakara/Math-Japanese-8k`
- local path: `E:\zipangu-datesets\awakara__Math-Japanese-8k`
- found: `True`
- files (physical / logical): `1` / `1`
- bytes (physical / logical): `11866128` / `11866128`
- rows: `8094` (known=`True`, unknown files=`0`)
- formats: `.jsonl`
- configs: `default`
- splits: `train`
- revision: `unknown`
- license metadata: `"unknown"`
- language metadata: `["ja"]`
- source category: `japanese_math`
- teacher/source fields: `none detected`
- length profile: `{"unit": "raw_text_field_characters", "selection": "deterministic_logical_file_order_prefix", "sample_rows_requested": 10000, "observed_rows": 8094, "median": 577.5, "p95": 909.35}`
- audit scope: `full`
- adapter warnings: `none`

### Schema

```json
{
  "input": [
    "str"
  ],
  "instruction": [
    "str"
  ],
  "output": [
    "str"
  ]
}
```

### Sample policy

Raw row samples are intentionally omitted from tracked inventory reports.

## ace_reason_math_japanese

- repository: `SousiOmine/AceReason-Math-Japanese`
- local path: `E:\zipangu-datesets\SousiOmine__AceReason-Math-Japanese`
- found: `True`
- files (physical / logical): `1` / `1`
- bytes (physical / logical): `2707053` / `2707053`
- rows: `10000` (known=`True`, unknown files=`0`)
- formats: `.parquet`
- configs: `default`
- splits: `train`
- revision: `unknown`
- license metadata: `"cc-by-4.0"`
- language metadata: `["ja"]`
- source category: `japanese_math`
- teacher/source fields: `none detected`
- length profile: `{"unit": "raw_text_field_characters", "selection": "deterministic_logical_file_order_prefix", "sample_rows_requested": 10000, "observed_rows": 10000, "median": 106.0, "p95": 257.0}`
- audit scope: `full`
- adapter warnings: `none`

### Schema

```json
{
  "answer": [
    "string"
  ],
  "problem": [
    "string"
  ],
  "problem_en": [
    "string"
  ]
}
```

### Sample policy

Raw row samples are intentionally omitted from tracked inventory reports.

## awesome_japanese_corpus

- repository: `nakasyou/awesome-japanese-corpus`
- local path: `E:\zipangu-datesets\nakasyou__awesome-japanese-corpus`
- found: `True`
- files (physical / logical): `1228` / `1228`
- bytes (physical / logical): `97421096463` / `97421096463`
- rows: `169958848` (known=`True`, unknown files=`0`)
- formats: `.parquet`
- configs: `default`
- splits: `train`
- revision: `unknown`
- license metadata: `["odc-by", "cc-by-4.0"]`
- language metadata: `["ja"]`
- source category: `japanese_corpus`
- teacher/source fields: `none detected`
- length profile: `{"unit": "raw_text_field_characters", "selection": "deterministic_logical_file_order_prefix", "sample_rows_requested": 10000, "observed_rows": 10000, "median": 923.5, "p95": 5047.25}`
- audit scope: `sampled`
- adapter warnings: `none`

### Schema

```json
{
  "from": [
    "string"
  ],
  "from_license": [
    "string"
  ],
  "text": [
    "string"
  ]
}
```

### Sample policy

Raw row samples are intentionally omitted from tracked inventory reports.

## gpt_5_6_traces

- repository: `Manusagents/GPT-5.6-Sol-Luna-Terra-Traces`
- local path: `E:\zipangu-datesets\Manusagents__GPT-5.6-Sol-Luna-Terra-Traces`
- found: `True`
- files (physical / logical): `1` / `1`
- bytes (physical / logical): `36197525` / `36197525`
- rows: `6288` (known=`True`, unknown files=`0`)
- formats: `.parquet`
- configs: `default`
- splits: `train`
- revision: `unknown`
- license metadata: `"cc-by-4.0"`
- language metadata: `["en"]`
- source category: `frontier_reasoning`
- teacher/source fields: `first_source_config, first_source_dataset, first_source_row_index, first_source_split`
- length profile: `{"unit": "raw_text_field_characters", "selection": "deterministic_logical_file_order_prefix", "sample_rows_requested": 10000, "observed_rows": 6288, "median": 16967.0, "p95": 58782.95}`
- audit scope: `full`
- adapter warnings: `none`

### Schema

```json
{
  "first_source_config": [
    "string"
  ],
  "first_source_dataset": [
    "string"
  ],
  "first_source_row_index": [
    "int64"
  ],
  "first_source_split": [
    "string"
  ],
  "row_hash": [
    "string"
  ],
  "row_json": [
    "string"
  ],
  "seen_count": [
    "int64"
  ]
}
```

### Sample policy

Raw row samples are intentionally omitted from tracked inventory reports.

## fable_5_5_distillation

- repository: `RESMP-DEV/Fable-GPT-5.5-Distillation-Traces`
- local path: `E:\zipangu-datesets\RESMP-DEV__Fable-GPT-5.5-Distillation-Traces`
- found: `True`
- files (physical / logical): `20` / `20`
- bytes (physical / logical): `13232592182` / `13232592182`
- rows: `1981131` (known=`True`, unknown files=`0`)
- formats: `.parquet`
- configs: `default`
- splits: `eval, train`
- revision: `unknown`
- license metadata: `"cc-by-4.0"`
- language metadata: `["en", "ru", "tr"]`
- source category: `frontier_reasoning_merged`
- teacher/source fields: `model, source`
- length profile: `{"unit": "raw_text_field_characters", "selection": "deterministic_logical_file_order_prefix", "sample_rows_requested": 10000, "observed_rows": 10000, "median": 20448.5, "p95": 47862.9}`
- audit scope: `full`
- adapter warnings: `none`

### Schema

```json
{
  "messages": [
    "string"
  ],
  "metadata": [
    "string"
  ],
  "model": [
    "string"
  ],
  "n_turns": [
    "int64"
  ],
  "session_id": [
    "string"
  ],
  "source": [
    "string"
  ],
  "split": [
    "string"
  ]
}
```

### Sample policy

Raw row samples are intentionally omitted from tracked inventory reports.

## claude_fable_code

- repository: `armand0e/claude-fable-5-claude-code`
- local path: `E:\zipangu-datesets\armand0e__claude-fable-5-claude-code`
- found: `True`
- files (physical / logical): `63` / `63`
- bytes (physical / logical): `75104357` / `75104357`
- rows: `18370` (known=`True`, unknown files=`0`)
- formats: `.jsonl`
- configs: `default`
- splits: `004c0d63-c96f-431d-8cf8-f78ca9367eb5, 06ec42c3-2184-40c5-b0ee-98c3235b4c4c, 17643245-7004-426a-b3d0-40a5fb6fd397, 189f0549-c2fc-4fc7-b50d-37cd255eabf2, 1afc3cc3-792e-49a2-83c4-19ddb66ba008, 1d29fdca-8b1a-4087-b14f-7c4e69701fbe, 21576682-1c94-4e42-a9d2-0f9e0137ffa7, 2f6b46c2-a83e-43e7-9012-76c7b434abba, 311883ad-f031-4619-ab72-1aeb410a9123, 32bd3eac-d079-4d7f-8f49-b34180656abd, 32c96eea-ab6f-4268-9f48-60fb2c8838e7, 3691aa67-12fe-404f-b33e-9bc0bce0d4d1, 3b47d4c6-c3ee-4a3b-92b0-4b434ab3c033, 44b273b6-9dbc-4b68-9520-965d98377043, 471accf0-3df7-4e2e-a567-a094f8f8d585, 485078ae-64bc-4bf5-a385-2d65a1108dae, 4b456a37-0524-420a-a4a1-67db0efa4881, 4b853fa6-aed6-494b-9a13-2cda8c3531ec, 4c99dff8-2213-4365-a651-3d78e202a489, 539c47c6-0b8c-40ff-8947-bf3ebb30900a, 554c2eb9-9cf9-4078-a6a2-82456a53f189, 58344d60-96b2-4ff5-8987-08b3e4f6f515, 5c0d7bef-0751-49d2-b55e-72fd00abedd2, 5eeb7e10-99a4-4d7d-a505-3ef47dcab480, 602033e7-dc55-4b7f-909f-1c67135d8f4b, 660bf750-e54d-4619-93ac-7c1877343220, 66d893d9-ee7f-4ca7-bdb6-99e8dabb601d, 68396a0d-8585-4d00-9321-54b2b00c07f4, 6917f657-870f-48a6-84fa-a774d4699ccf, 6ab67974-013e-44ab-8ae8-3c64ef6cb770, 6ae0aeed-340e-4b4d-9d74-f905a74ca2fc, 7274969b-ab5c-461c-bc7c-8434eb9eb42e, 743a2f8c-6846-4e89-848c-9c4679492491, 7758cfa4-f4c8-4cb9-8ea3-30697ba1dfad, 7da8c055-9995-4731-a147-bf2a9e1251c2, 89becffe-10a0-4022-b92b-c0bd8ddc3206, 90a22dda-0de2-4658-89f8-f55f7facd6bc, 92726ffc-0be8-4bd6-8d60-9164883f67d5, 9fc7c249-f2cb-4fd6-8417-85caec1816df, a13efdaa-96ed-4ab0-a4c4-240692be0f10, ac6c27f7-e781-47f0-afb4-b089c7f12742, agent-a68c3d43360a5ed83, agent-a7fa04d5e85bd5c0a, agent-a8466eb290f99ea00, agent-a87ca9b22d0fee3f9, agent-a8d979113b5d895ee, agent-a8da0aea214c5e781, agent-a93f2cd3c99c3d4ed, agent-aa3612085487e6586, b0e959f7-df98-445f-808f-4d38b25e4e01, bcfbd960-e7f5-42ee-bb37-183b81ab12e3, bd2f5aa9-a725-436b-b4ef-c1e05496c7cb, c6d4788b-6bb6-4774-8294-5dc897346ca9, c7c6924c-09c0-4c89-9ce1-a2080fd18f8f, c8264cd7-40ec-4d52-b3e8-e07a0d8a710f, d9c29410-80a6-4e34-aab1-0c7e5bb7110f, e1b4c95f-4884-49f5-b266-3f8b69929958, e3d0c93f-5cf8-4067-9a49-657ba5c67c80, e8b69c49-ad36-49d0-8998-c50a1e83e457, e91a5346-6059-4f85-a564-37dcd53380db, f587cec0-5329-4595-86e5-b9513e82e1a1, f8b99555-306c-482a-923f-395b40aadff3, f956721a-0af7-4bdc-8678-3a493d8fcd39`
- revision: `unknown`
- license metadata: `"unknown"`
- language metadata: `[]`
- source category: `coding_agent`
- teacher/source fields: `promptSource, sourceToolAssistantUUID, sourceToolUseID`
- length profile: `{"unit": "raw_text_field_characters", "selection": "deterministic_logical_file_order_prefix", "sample_rows_requested": 10000, "observed_rows": 3631, "median": 190.0, "p95": 11258.0}`
- audit scope: `full`
- adapter warnings: `none`

### Schema

```json
{
  "agentId": [
    "str"
  ],
  "aiTitle": [
    "str"
  ],
  "apiErrorStatus": [
    "int"
  ],
  "attachment": [
    "dict"
  ],
  "attributionAgent": [
    "str"
  ],
  "attributionMcpServer": [
    "str"
  ],
  "attributionMcpTool": [
    "str"
  ],
  "attributionSkill": [
    "str"
  ],
  "bridgeSessionId": [
    "str"
  ],
  "content": [
    "str"
  ],
  "customTitle": [
    "str"
  ],
  "cwd": [
    "str"
  ],
  "durationMs": [
    "int"
  ],
  "entrypoint": [
    "str"
  ],
  "error": [
    "dict",
    "str"
  ],
  "gitBranch": [
    "str"
  ],
  "hasOutput": [
    "bool"
  ],
  "hookAdditionalContext": [
    "list"
  ],
  "hookCount": [
    "int"
  ],
  "hookErrors": [
    "list"
  ],
  "hookInfos": [
    "list"
  ],
  "imagePasteIds": [
    "list"
  ],
  "interruptedMessageId": [
    "str"
  ],
  "isApiErrorMessage": [
    "bool"
  ],
  "isMeta": [
    "bool"
  ],
  "isSidechain": [
    "bool"
  ],
  "isSnapshotUpdate": [
    "bool"
  ],
  "lastPrompt": [
    "str"
  ],
  "lastSequenceNum": [
    "int"
  ],
  "leafUuid": [
    "str"
  ],
  "level": [
    "str"
  ],
  "maxRetries": [
    "int"
  ],
  "message": [
    "dict"
  ],
  "messageCount": [
    "int"
  ],
  "messageId": [
    "str"
  ],
  "mode": [
    "str"
  ],
  "operation": [
    "str"
  ],
  "origin": [
    "dict"
  ],
  "parentUuid": [
    "NoneType",
    "str"
  ],
  "pendingWorkflowCount": [
    "int"
  ],
  "permissionMode": [
    "str"
  ],
  "preventedContinuation": [
    "bool"
  ],
  "promptId": [
    "str"
  ],
  "promptSource": [
    "str"
  ],
  "requestId": [
    "str"
  ],
  "retryAttempt": [
    "int"
  ],
  "retryInMs": [
    "float"
  ],
  "sessionId": [
    "str"
  ],
  "slug": [
    "str"
  ],
  "snapshot": [
    "dict"
  ],
  "sourceToolAssistantUUID": [
    "str"
  ],
  "sourceToolUseID": [
    "str"
  ],
  "stopReason": [
    "str"
  ],
  "subtype": [
    "str"
  ],
  "timestamp": [
    "str"
  ],
  "toolUseID": [
    "str"
  ],
  "toolUseResult": [
    "dict",
    "list",
    "str"
  ],
  "type": [
    "str"
  ],
  "userType": [
    "str"
  ],
  "uuid": [
    "str"
  ],
  "version": [
    "str"
  ]
}
```

### Sample policy

Raw row samples are intentionally omitted from tracked inventory reports.

## frontier_multi_teacher

- repository: `r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation`
- local path: `E:\zipangu-datesets\r0b0tlab__qwen3.8-max-glm5.2-kimi-k3-distillation`
- found: `True`
- files (physical / logical): `213` / `213`
- bytes (physical / logical): `827086944` / `827086944`
- rows: `1976658` (known=`True`, unknown files=`0`)
- formats: `.parquet`
- configs: `default`
- splits: `glm47-le-131072, glm47-le-16384, glm47-le-32768, glm47-le-4096, glm47-le-65536, glm47-le-8192, glm47-long-outlier, llama31_final-le-131072, llama31_final-le-16384, llama31_final-le-32768, llama31_final-le-4096, llama31_final-le-65536, llama31_final-le-8192, llama31_final-long-outlier, prompt_completion_text, qwen3-le-131072, qwen3-le-16384, qwen3-le-32768, qwen3-le-4096, qwen3-le-65536, qwen3-le-8192, qwen3-long-outlier, rl_tool_prompts, sft_reasoning, sft_tools, smoke, test, train, validation`
- revision: `unknown`
- license metadata: `"other"`
- language metadata: `["en", "zh", "es", "fr", "de", "ja"]`
- source category: `frontier_reasoning`
- teacher/source fields: `source, source_item_id, source_license, source_record_hash, source_repository, source_revision, source_split, teacher_model, teacher_provider`
- length profile: `{"unit": "raw_text_field_characters", "selection": "deterministic_logical_file_order_prefix", "sample_rows_requested": 10000, "observed_rows": 10000, "median": 2510.0, "p95": 11546.4}`
- audit scope: `full`
- adapter warnings: `none`

### Schema

```json
{
  "assistant_tokens": [
    "int32"
  ],
  "completion_text": [
    "string"
  ],
  "dedup_cluster_id": [
    "string"
  ],
  "disposition": [
    "string"
  ],
  "domain": [
    "string"
  ],
  "expected_terminal_oracle_json": [
    "string"
  ],
  "family": [
    "string"
  ],
  "family_oracle_passed": [
    "bool"
  ],
  "format_score_raw": [
    "double"
  ],
  "glm47_assistant_tokens": [
    "int32"
  ],
  "glm47_context_bucket": [
    "string"
  ],
  "glm47_final_payload_tokens": [
    "int32"
  ],
  "glm47_loss_ratio": [
    "double"
  ],
  "glm47_mask_method": [
    "string"
  ],
  "glm47_reasoning_payload_tokens": [
    "int32"
  ],
  "glm47_tool_call_payload_tokens": [
    "int32"
  ],
  "glm47_tool_observation_tokens": [
    "int32"
  ],
  "glm47_total_tokens": [
    "int32"
  ],
  "ground_truth_json": [
    "string"
  ],
  "id": [
    "string"
  ],
  "input_ids": [
    "list<element: int32>"
  ],
  "labels": [
    "list<element: int32>"
  ],
  "llama31_context_bucket": [
    "string"
  ],
  "llama31_final_assistant_tokens": [
    "int32"
  ],
  "llama31_final_total_tokens": [
    "int32"
  ],
  "llama31_mask_method": [
    "string"
  ],
  "mask_method": [
    "string"
  ],
  "max_validated_tokens": [
    "int32"
  ],
  "messages": [
    "list<element: struct<role: string, content: string, reasoning_content: string, tool_calls: list<element: struct<id: string, type: string, function: struct<name: string, arguments: string>>>, tool_call_id: string, name: string, trainable: bool>>"
  ],
  "messages_json": [
    "string"
  ],
  "n_assistant_turns": [
    "int32"
  ],
  "n_tool_calls": [
    "int32"
  ],
  "parent_id": [
    "string"
  ],
  "plain_completion_tokens": [
    "int32"
  ],
  "plain_context_bucket": [
    "string"
  ],
  "plain_prompt_tokens": [
    "int32"
  ],
  "prompt_cluster_id": [
    "string"
  ],
  "prompt_messages_json": [
    "string"
  ],
  "prompt_text": [
    "string"
  ],
  "quality_flags": [
    "list<element: string>"
  ],
  "qwen3_assistant_tokens": [
    "int32"
  ],
  "qwen3_context_bucket": [
    "string"
  ],
  "qwen3_final_payload_tokens": [
    "int32"
  ],
  "qwen3_loss_ratio": [
    "double"
  ],
  "qwen3_mask_method": [
    "string"
  ],
  "qwen3_reasoning_payload_tokens": [
    "int32"
  ],
  "qwen3_tool_call_payload_tokens": [
    "int32"
  ],
  "qwen3_tool_observation_tokens": [
    "int32"
  ],
  "qwen3_total_tokens": [
    "int32"
  ],
  "reference_agreement": [
    "bool"
  ],
  "renderer_revision": [
    "string"
  ],
  "reward_contract_json": [
    "string"
  ],
  "sampling_weight": [
    "double"
  ],
  "sampling_weight_glm47": [
    "double"
  ],
  "sampling_weight_llama31_final": [
    "double"
  ],
  "sampling_weight_qwen3": [
    "double"
  ],
  "schema_version": [
    "string"
  ],
  "source": [
    "string"
  ],
  "source_item_id": [
    "string"
  ],
  "source_license": [
    "string"
  ],
  "source_record_hash": [
    "string"
  ],
  "source_repository": [
    "string"
  ],
  "source_revision": [
    "string"
  ],
  "source_split": [
    "string"
  ],
  "split": [
    "string"
  ],
  "subdomain": [
    "string"
  ],
  "task_type": [
    "string"
  ],
  "teacher_model": [
    "string"
  ],
  "teacher_provider": [
    "string"
  ],
  "template_cluster_id": [
    "string"
  ],
  "tools": [
    "list<element: struct<type: string, function: struct<name: string, description: string, parameters_json: string>>>"
  ],
  "tools_json": [
    "string"
  ],
  "total_tokens": [
    "int32"
  ],
  "trace_kind": [
    "string"
  ],
  "verifier_passed": [
    "bool"
  ]
}
```

### Sample policy

Raw row samples are intentionally omitted from tracked inventory reports.

## fable_5_premium

- repository: `saidutta69/fable-5-premium`
- local path: `E:\zipangu-datesets\saidutta69__fable-5-premium`
- found: `True`
- files (physical / logical): `12` / `3`
- bytes (physical / logical): `2335985660` / `799836185`
- rows: `6365` (known=`True`, unknown files=`0`)
- formats: `.jsonl, .parquet`
- configs: `agent_traces`
- splits: `test, train, validation`
- revision: `unknown`
- license metadata: `"mit"`
- language metadata: `["en"]`
- source category: `frontier_reasoning`
- teacher/source fields: `model, source`
- length profile: `{"unit": "raw_text_field_characters", "selection": "deterministic_logical_file_order_prefix", "sample_rows_requested": 10000, "observed_rows": 6365, "median": 79814.0, "p95": 164981.4}`
- audit scope: `full`
- adapter warnings: `none`

### Schema

```json
{
  "messages": [
    "list"
  ],
  "model": [
    "str"
  ],
  "quality_scores": [
    "dict"
  ],
  "reasoning": [
    "str"
  ],
  "session_id": [
    "str"
  ],
  "source": [
    "str"
  ]
}
```

### Sample policy

Raw row samples are intentionally omitted from tracked inventory reports.
