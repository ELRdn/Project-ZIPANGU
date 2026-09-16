#!/usr/bin/env python3
"""Run a tiny, research-only QLoRA pass through the real ZIPANGU data path.

This script is intentionally separate from the synthetic hardware smoke.  It
reads only a local train-candidate source, canonicalizes rows through the
ZIPANGU pipeline, checks available evaluation fingerprints without copying
evaluation content into the training dataset, formats/tokenizes the resulting
chat examples, and runs a bounded LoRA update.  It never changes the registry
approval gate and never authorizes a real training run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

DEFAULT_DATASET_ROOT = Path(r"E:\zipangu-datesets")
DEFAULT_EVAL_ROOT = Path(r"E:\zipangu-eval")
DEFAULT_MODEL = ROOT / ".cache" / "models" / "ZIPANGU-K-I-4B-base"
DEFAULT_CACHE = ROOT / ".cache" / "huggingface-real-data-smoke"
DEFAULT_REPORT = ROOT / "reports" / "hardware" / "unsloth_real_data_smoke.json"
DEFAULT_OUTPUT = ROOT / ".cache" / "smoke-runs" / "k-i-real-data-qlora"
DEFAULT_REVISION = "c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e"
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def resolve_inside_repo(value: str | Path) -> Path:
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"path must stay inside the repository: {resolved}")
    return resolved


def normalize_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        result[str(key)] = value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
    return result


def configure_cache(cache_dir: Path) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "hub": cache_dir / "hub",
        "xet": cache_dir / "xet",
        "miopen": cache_dir / "miopen" / "kernel",
        "miopen_db": cache_dir / "miopen" / "db",
        "triton": cache_dir / "triton" / "cache",
        "torchinductor": cache_dir / "torchinductor",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HF_HOME": str(cache_dir),
            "HF_HUB_CACHE": str(paths["hub"]),
            "HF_XET_CACHE": str(paths["xet"]),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_XET": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MIOPEN_CUSTOM_CACHE_DIR": str(paths["miopen"]),
            "MIOPEN_USER_DB_PATH": str(paths["miopen_db"]),
            "MIOPEN_CACHE_DIR": str(paths["miopen"]),
            "TRITON_CACHE_DIR": str(paths["triton"]),
            "TRITON_HOME": str(cache_dir / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(paths["torchinductor"]),
        }
    )
    original_bnb = os.environ.pop("BNB_ROCM_VERSION", None)
    return {
        "cache_dir": str(cache_dir),
        "hf_home": str(cache_dir),
        "miopen_cache": str(paths["miopen"]),
        "triton_cache": str(paths["triton"]),
        "torchinductor_cache": str(paths["torchinductor"]),
        "bnb_rocm_version_original": str(original_bnb) if original_bnb is not None else "unset",
        "bnb_rocm_version_for_process": "unset",
    }


def load_eval_fingerprints(eval_root: Path) -> tuple[set[str], list[str]]:
    hashes: set[str] = set()
    files: list[str] = []
    fingerprint_dir = eval_root / "_fingerprints"
    for path in sorted(fingerprint_dir.glob("*.jsonl")):
        files.append(str(path))
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, Mapping) and item.get("content_hash"):
                    hashes.add(str(item["content_hash"]))
    return hashes, files


def available_answer_carefully_data(eval_root: Path) -> bool:
    root = eval_root / "sources" / "AnswerCarefully-v2.0-test"
    if not root.is_dir():
        return False
    data_suffixes = {".json", ".jsonl", ".gz", ".parquet", ".arrow", ".csv"}
    return any(path.is_file() and path.suffix.casefold() in data_suffixes for path in root.rglob("*"))


def select_canonical_records(
    *,
    dataset_root: Path,
    eval_root: Path,
    dataset_id: str,
    requested: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from zipangu.data.adapters import iter_source_rows
    from zipangu.data.audit import _runtime_audit_dir
    from zipangu.data.canonical import canonicalize_source_row
    from zipangu.data.dedup import iter_audit_records
    from zipangu.data.filtering import apply_filter, hard_filter
    from zipangu.data.pipeline import build_context
    from zipangu.data.quality import score_record

    context = build_context(ROOT, dataset_root=dataset_root, dataset_ids=[dataset_id])
    locations = {location.policy.dataset_id: location for location in context.locations}
    location = locations.get(dataset_id)
    if location is None:
        raise ValueError(f"dataset not found in registry: {dataset_id}")
    if not location.found:
        raise FileNotFoundError(f"dataset path not found: {location.path}")
    if location.policy.role != "train_candidate":
        raise ValueError(f"refusing non-train-candidate source role: {location.policy.role}")

    eval_hashes, eval_fingerprint_files = load_eval_fingerprints(eval_root)
    audit_candidates: list[dict[str, Any]] = []
    audit_seen = 0
    exact_fingerprint_matches = 0
    for audit in iter_audit_records(_runtime_audit_dir(ROOT)):
        if str(audit.get("source_dataset")) != dataset_id:
            continue
        audit_seen += 1
        split = str(audit.get("source_split", "")).casefold()
        contamination = str(audit.get("contamination_status", "")).casefold()
        if audit.get("eligibility") is not True:
            continue
        if split in {"eval", "test", "validation"}:
            continue
        if contamination in {"suspect", "contaminated", "match", "blocked"}:
            continue
        if str(audit.get("content_hash", "")) in eval_hashes:
            exact_fingerprint_matches += 1
            continue
        if audit.get("contamination_reasons"):
            continue
        audit_candidates.append(dict(audit))
        if len(audit_candidates) >= requested * 8:
            break

    wanted_ids = {str(item.get("source_row_id")) for item in audit_candidates}
    by_row_id = {str(item.get("source_row_id")): item for item in audit_candidates}
    selected: list[dict[str, Any]] = []
    seen_content_hashes: set[str] = set()
    scanned_source_rows = 0
    for source_row in iter_source_rows(location, batch_size=256):
        scanned_source_rows += 1
        if source_row.source_row_id not in wanted_ids:
            continue
        record = canonicalize_source_row(
            source_row,
            source_metadata={
                "revision": location.revision,
                "license": location.card_metadata.get("license", "unknown"),
                "vendor_specific": location.policy.vendor_specific,
            },
        )
        record = apply_filter(record, hard_filter(record, source_role=location.policy.role))
        record.update(score_record(record))
        audit_record = by_row_id[source_row.source_row_id]
        if str(record.get("record_id")) != str(audit_record.get("record_id")):
            continue
        if record.get("eligibility") is not True:
            continue
        content_hash = str(record.get("content_hash", ""))
        if not content_hash or content_hash in seen_content_hashes:
            continue
        if content_hash in eval_hashes:
            exact_fingerprint_matches += 1
            continue
        if str(record.get("source_split", "")).casefold() in {"eval", "test", "validation"}:
            continue
        selected.append(record)
        seen_content_hashes.add(content_hash)
        if len(selected) >= requested:
            break

    evidence = {
        "dataset_id": dataset_id,
        "dataset_path": str(location.path),
        "repo": location.policy.repo,
        "source_role": location.policy.role,
        "source_revision": location.revision,
        "license_metadata": location.card_metadata.get("license", "unknown"),
        "audit_records_seen": audit_seen,
        "audit_candidates_after_safety_filters": len(audit_candidates),
        "source_rows_scanned": scanned_source_rows,
        "selected_records": len(selected),
        "eval_fingerprint_files_read": eval_fingerprint_files,
        "available_eval_fingerprint_count": len(eval_hashes),
        "exact_fingerprint_matches_excluded": exact_fingerprint_matches,
        "answer_carefully_data_present": available_answer_carefully_data(eval_root),
        "contamination_status_in_audit": sorted(
            {str(item.get("contamination_status", "")) for item in audit_candidates}
        ),
        "research_only": True,
        "serious_training_allowed": False,
    }
    return selected, evidence


def format_and_tokenize(records: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int) -> tuple[Any, dict[str, Any]]:
    from datasets import Dataset

    def single_token_ids(encoded: Mapping[str, Any]) -> list[int]:
        ids = encoded["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(item) for item in ids]

    tokenized: list[dict[str, Any]] = []
    assistant_label_counts: list[int] = []
    truncated_count = 0
    for record in records:
        messages = [
            {"role": str(item.get("role")), "content": str(item.get("content", ""))}
            for item in (record.get("messages") or [])
            if isinstance(item, Mapping) and item.get("role") in {"system", "user", "assistant"}
        ]
        assistant_indices = [index for index, item in enumerate(messages) if item["role"] == "assistant"]
        if not messages or not assistant_indices:
            continue
        assistant_index = assistant_indices[-1]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prefix_text = tokenizer.apply_chat_template(
            messages[:assistant_index], tokenize=False, add_generation_prompt=True
        )
        full_ids = single_token_ids(tokenizer(
            text=full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        ))
        prefix_ids = single_token_ids(tokenizer(text=prefix_text, add_special_tokens=False))
        common = 0
        for left, right in zip(full_ids, prefix_ids):
            if left != right:
                break
            common += 1
        prefix_length = min(common, len(full_ids))
        labels = [-100] * prefix_length + list(full_ids[prefix_length:])
        if len(full_ids) < len(single_token_ids(tokenizer(text=full_text, add_special_tokens=False))):
            truncated_count += 1
        label_count = sum(1 for label in labels if label != -100)
        if label_count < 1:
            continue
        assistant_label_counts.append(label_count)
        tokenized.append(
            {
                "input_ids": list(full_ids),
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
                "record_id": str(record.get("record_id")),
            }
        )
    if not tokenized:
        raise ValueError("chat formatting/tokenization produced no assistant-supervised examples")
    return Dataset.from_list(tokenized), {
        "formatted_examples": len(tokenized),
        "assistant_supervised_examples": len(assistant_label_counts),
        "assistant_label_tokens_min": min(assistant_label_counts),
        "assistant_label_tokens_max": max(assistant_label_counts),
        "assistant_label_tokens_total": sum(assistant_label_counts),
        "max_length": max_length,
        "truncated_examples": truncated_count,
    }


class CausalCollator:
    """Right-pad pretokenized causal examples and retain assistant-only labels."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        width = max(len(item["input_ids"]) for item in features)
        input_ids: list[list[int]] = []
        attention: list[list[int]] = []
        labels: list[list[int]] = []
        for item in features:
            length = len(item["input_ids"])
            padding = width - length
            input_ids.append(list(item["input_ids"]) + [self.pad_token_id] * padding)
            attention.append(list(item["attention_mask"]) + [0] * padding)
            labels.append(list(item["labels"]) + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--dataset-id", default="math_japanese_8k")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=256)
    args = parser.parse_args(argv)
    if not 16 <= args.examples <= 64:
        parser.error("--examples must be between 16 and 64")
    if not 2 <= args.steps <= 10:
        parser.error("--steps must be between 2 and 10")
    if not 32 <= args.max_seq_length <= 512:
        parser.error("--max-seq-length must be between 32 and 512")
    args.model_path = resolve_inside_repo(args.model_path)
    args.cache_dir = resolve_inside_repo(args.cache_dir)
    args.output_dir = resolve_inside_repo(args.output_dir)
    args.report = resolve_inside_repo(args.report)
    args.dataset_root = Path(args.dataset_root).expanduser().resolve()
    args.eval_root = Path(args.eval_root).expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "HARD_BLOCKED",
        "phase": "k_i_real_data_path_smoke",
        "generated_at": utc_now(),
        "K_I_REAL_DATA_PATH_SMOKE": "HARD_BLOCKED",
        "TRAINING_ALLOWED": False,
        "scope": {
            "research_only": True,
            "evaluation_run": False,
            "paid_compute": False,
            "registry_approval_changed": False,
            "raw_dataset_mutated": False,
            "examples_requested": args.examples,
            "steps_requested": args.steps,
        },
        "model": {
            "path": str(args.model_path),
            "revision": args.revision,
            "load_in_4bit": True,
            "full_finetuning": False,
        },
        "training": {
            "max_steps": args.steps,
            "max_seq_length": args.max_seq_length,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-4,
            "optimizer": "adamw_torch",
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_target_modules": LORA_TARGET_MODULES,
            "assistant_only_loss": True,
        },
        "pipeline": {},
        "warnings": [],
    }
    try:
        if not args.model_path.is_dir() or not (args.model_path / "config.json").is_file():
            raise FileNotFoundError(f"local base model is missing: {args.model_path}")
        if not args.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset root is missing: {args.dataset_root}")
        if not args.eval_root.is_dir():
            raise FileNotFoundError(f"evaluation root is missing: {args.eval_root}")
        if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
            raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report["environment"] = configure_cache(args.cache_dir)

        records, source_evidence = select_canonical_records(
            dataset_root=args.dataset_root,
            eval_root=args.eval_root,
            dataset_id=args.dataset_id,
            requested=args.examples,
        )
        report["pipeline"]["canonical_dataset"] = {
            "status": "PASS" if len(records) >= args.examples else "HARD_BLOCKED",
            **source_evidence,
        }
        if len(records) < args.examples:
            raise RuntimeError(
                f"safe real-data selection yielded {len(records)} examples; {args.examples} required"
            )

        import torch
        from unsloth import FastLanguageModel

        report["environment"].update(
            {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "hip": getattr(torch.version, "hip", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() else None,
            }
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.model_path),
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
            full_finetuning=False,
            trust_remote_code=False,
            device_map="auto",
            use_exact_model_name=True,
            disable_log_stats=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dataset, tokenization_evidence = format_and_tokenize(records, tokenizer, args.max_seq_length)
        report["pipeline"]["chat_template_tokenize_collator"] = {
            "status": "PASS",
            **tokenization_evidence,
            "collator": "CausalCollator",
            "assistant_supervision": "labels=-100 before final assistant span",
        }
        model.config.use_cache = False
        model = FastLanguageModel.get_peft_model(
            model,
            r=4,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            max_seq_length=args.max_seq_length,
        )
        report["model"].update(
            {
                "class": type(model).__name__,
                "tokenizer_class": type(tokenizer).__name__,
                "first_parameter_device": str(next(model.parameters()).device),
                "first_parameter_dtype": str(next(model.parameters()).dtype),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
                "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
        )

        from transformers import Trainer, TrainingArguments

        train_args = TrainingArguments(
            output_dir=str(args.output_dir),
            max_steps=args.steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            logging_strategy="steps",
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            optim="adamw_torch",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            fp16=False,
            bf16=False,
            seed=3407,
            data_seed=3407,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            label_names=["labels"],
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=dataset,
            data_collator=CausalCollator(int(tokenizer.pad_token_id)),
        )
        train_result = trainer.train()
        trainer.save_model(str(args.output_dir))
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(args.output_dir))
        report["pipeline"]["forward_backward_optimizer"] = {
            "status": "PASS" if int(getattr(trainer.state, "global_step", 0)) >= args.steps else "HARD_BLOCKED",
            "global_step": int(getattr(trainer.state, "global_step", 0)),
            "metrics": normalize_metrics(getattr(train_result, "metrics", None)),
            "log_history_tail": [
                normalize_metrics(item) for item in getattr(trainer.state, "log_history", [])[-6:]
            ],
        }
        adapter_files = sorted(path.name for path in args.output_dir.glob("adapter*") if path.is_file())
        report["pipeline"]["adapter_save"] = {
            "status": "PASS" if (args.output_dir / "adapter_config.json").is_file() else "HARD_BLOCKED",
            "output_dir": str(args.output_dir),
            "files": adapter_files,
        }

        # Reload the saved adapter into a fresh local 4-bit base model and run
        # one short sample.  The sample is evidence of reload/inference only;
        # it is not an evaluation result and is not published.
        reloaded_base, reload_tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.model_path),
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
            full_finetuning=False,
            trust_remote_code=False,
            device_map="auto",
            use_exact_model_name=True,
            disable_log_stats=True,
        )
        from peft import PeftModel

        reloaded = PeftModel.from_pretrained(reloaded_base, str(args.output_dir), is_trainable=False)
        reloaded.eval()
        sample_messages = [
            {"role": str(item.get("role")), "content": str(item.get("content", ""))}
            for item in (records[0].get("messages") or [])
            if isinstance(item, Mapping) and item.get("role") in {"system", "user", "assistant"}
        ]
        sample_prefix = reload_tokenizer.apply_chat_template(
            sample_messages[:-1], tokenize=False, add_generation_prompt=True
        )
        inputs = reload_tokenizer(text=sample_prefix, return_tensors="pt")
        device = next(reloaded.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = reloaded.generate(**inputs, max_new_tokens=16, do_sample=False)
        generated = reload_tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        if not generated.strip():
            raise RuntimeError("reloaded adapter sample inference returned empty output")
        report["pipeline"]["adapter_reload_sample_inference"] = {
            "status": "PASS",
            "generated_text": generated[:2000],
            "max_new_tokens": 16,
        }
        try:
            torch.cuda.synchronize()
            report["environment"]["max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
            report["environment"]["max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))
        except Exception as exc:
            report["warnings"].append(f"GPU memory post-check unavailable: {type(exc).__name__}: {exc}")
        report["status"] = "PASS"
        report["K_I_REAL_DATA_PATH_SMOKE"] = "PASS"
    except Exception as exc:
        report["status"] = "HARD_BLOCKED"
        report["K_I_REAL_DATA_PATH_SMOKE"] = "HARD_BLOCKED"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc)[:3000],
            "traceback_tail": traceback.format_exc()[-7000:],
        }
    finally:
        report["duration_seconds"] = round(time.time() - started, 2)
        report["generated_at"] = utc_now()
        write_json(args.report, report)
    print(json.dumps({"status": report["status"], "output": str(args.report)}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
