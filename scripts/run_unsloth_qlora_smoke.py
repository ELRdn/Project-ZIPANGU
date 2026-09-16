#!/usr/bin/env python3
"""Run a bounded, synthetic-data Unsloth QLoRA smoke on the local RX 7600.

This is deliberately not a ZIPANGU experiment.  It never reads the project
dataset registry, evaluation data, or a remote model identifier.  The script
exists to prove that the isolated AMD/Windows stack can load a local base
checkpoint, attach a LoRA adapter, and execute a tiny training step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = ".cache/models/ZIPANGU-K-I-4B-base"
DEFAULT_CACHE = ".cache/huggingface"
DEFAULT_REPORT = "reports/hardware/unsloth_qlora_smoke.json"
DEFAULT_REVISION = "c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e"
MAX_SMOKE_STEPS = 2
MAX_SMOKE_SEQUENCE_LENGTH = 256
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_inside_repo(raw: str, root: Path, label: str) -> Path:
    candidate = Path(raw)
    resolved = candidate if candidate.is_absolute() else root / candidate
    resolved = resolved.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} must stay inside the repository: {resolved}")
    return resolved


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(root / DEFAULT_MODEL))
    parser.add_argument("--cache-dir", default=str(root / DEFAULT_CACHE))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", default=str(root / DEFAULT_REPORT))
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.max_steps <= MAX_SMOKE_STEPS:
        parser.error(f"--max-steps must be between 1 and {MAX_SMOKE_STEPS}")
    if not 32 <= args.max_seq_length <= MAX_SMOKE_SEQUENCE_LENGTH:
        parser.error(
            f"--max-seq-length must be between 32 and {MAX_SMOKE_SEQUENCE_LENGTH}"
        )
    args.model_path = resolve_inside_repo(args.model_path, root, "--model-path")
    args.cache_dir = resolve_inside_repo(args.cache_dir, root, "--cache-dir")
    args.output_dir = resolve_inside_repo(args.output_dir, root, "--output-dir")
    args.report = resolve_inside_repo(args.report, root, "--report")
    return args


def configure_process_environment(cache_dir: Path) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    hub_cache = cache_dir / "hub"
    xet_cache = cache_dir / "xet"
    miopen_cache = cache_dir / "miopen" / "kernel"
    miopen_db = cache_dir / "miopen" / "db"
    triton_cache = cache_dir / "triton" / "cache"
    torchinductor_cache = cache_dir / "torchinductor"
    miopen_cache.mkdir(parents=True, exist_ok=True)
    miopen_db.mkdir(parents=True, exist_ok=True)
    triton_cache.mkdir(parents=True, exist_ok=True)
    torchinductor_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_XET_CACHE"] = str(xet_cache)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # The default Windows MIOpen cache may be ACL-protected.  Keep kernel and
    # tuning artifacts in the same disposable project cache as the model.
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = str(miopen_cache)
    os.environ["MIOPEN_USER_DB_PATH"] = str(miopen_db)
    os.environ["MIOPEN_CACHE_DIR"] = str(miopen_cache)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TRITON_HOME"] = str(cache_dir / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(torchinductor_cache)
    original_bnb_rocm_version = os.environ.pop("BNB_ROCM_VERSION", None)
    return {
        "hf_home": str(cache_dir),
        "hf_hub_cache": str(hub_cache),
        "hf_xet_cache": str(xet_cache),
        "miopen_custom_cache_dir": str(miopen_cache),
        "miopen_user_db_path": str(miopen_db),
        "triton_cache_dir": str(triton_cache),
        "torchinductor_cache_dir": str(torchinductor_cache),
        "bnb_rocm_version_original": original_bnb_rocm_version,
        "bnb_rocm_version_for_process": None,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def synthetic_dataset() -> Any:
    from datasets import Dataset

    # These are invented smoke records, not copied from any ZIPANGU source or
    # evaluation set.  Keep the dataset in memory so provenance cannot be
    # confused with a runnable training manifest.
    return Dataset.from_list(
        [
            {
                "text": (
                    "ユーザー: 日本の首都を一語で答えてください。\n"
                    "アシスタント: 東京"
                )
            },
            {
                "text": (
                    "ユーザー: 2足す3の答えだけを書いてください。\n"
                    "アシスタント: 5"
                )
            },
        ]
    )


def gpu_snapshot(torch_module: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "cuda_available": bool(torch_module.cuda.is_available()),
        "device_count": int(torch_module.cuda.device_count()),
    }
    if snapshot["device_count"]:
        snapshot["device"] = str(torch_module.cuda.current_device())
        snapshot["device_name"] = torch_module.cuda.get_device_name(0)
        snapshot["hip_version"] = getattr(torch_module.version, "hip", None)
        try:
            free_bytes, total_bytes = torch_module.cuda.mem_get_info(0)
            snapshot["free_bytes"] = int(free_bytes)
            snapshot["total_bytes"] = int(total_bytes)
        except Exception as exc:  # diagnostic only; training can still proceed
            snapshot["memory_probe_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def normalize_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    normalized: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def main() -> int:
    args = parse_args()
    started = time.time()
    environment: dict[str, Any] = {}
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "BLOCKED",
        "phase": "local_qlora_smoke",
        "generated_at": utc_now(),
        "scope": {
            "training_type": "synthetic_two_record_smoke_only",
            "evaluation_run": False,
            "project_dataset_registry_read": False,
            "remote_model_access": False,
            "max_steps_hard_cap": MAX_SMOKE_STEPS,
        },
        "model": {
            "path": str(args.model_path.relative_to(repo_root())),
            "revision": args.revision,
            "load_in_4bit": True,
            "full_finetuning": False,
        },
        "training": {
            "max_steps": args.max_steps,
            "max_seq_length": args.max_seq_length,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": "adamw_torch",
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_target_modules": LORA_TARGET_MODULES,
        },
        "warnings": [],
    }

    try:
        if not args.model_path.is_dir():
            raise FileNotFoundError(f"local model directory not found: {args.model_path}")
        if not (args.model_path / "config.json").is_file():
            raise FileNotFoundError(f"local model config not found: {args.model_path / 'config.json'}")
        if args.output_dir.exists():
            if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
                raise FileExistsError(
                    "refusing to overwrite a non-empty smoke output directory: "
                    f"{args.output_dir}"
                )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        environment = configure_process_environment(args.cache_dir)
        report["environment"] = environment

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            import torch
            import unsloth
            from unsloth import FastLanguageModel

            report["environment"].update(
                {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "hip": getattr(torch.version, "hip", None),
                    "unsloth": getattr(unsloth, "__version__", "unknown"),
                    "gpu_before": gpu_snapshot(torch),
                }
            )
            model, processor = FastLanguageModel.from_pretrained(
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
                    "processor_class": type(processor).__name__,
                    "first_parameter_device": str(next(model.parameters()).device),
                    "first_parameter_dtype": str(next(model.parameters()).dtype),
                    "trainable_parameters": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                }
            )

            from trl import SFTConfig, SFTTrainer

            train_args = SFTConfig(
                output_dir=str(args.output_dir),
                max_steps=args.max_steps,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=2e-4,
                logging_strategy="steps",
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                dataset_text_field="text",
                max_length=args.max_seq_length,
                packing=False,
                optim="adamw_torch",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                fp16=False,
                bf16=False,
                seed=3407,
                data_seed=3407,
                dataloader_num_workers=0,
                remove_unused_columns=False,
            )
            trainer = SFTTrainer(
                model=model,
                args=train_args,
                train_dataset=synthetic_dataset(),
                processing_class=processor,
            )
            train_result = trainer.train()
            trainer.save_model(str(args.output_dir))
            if hasattr(processor, "save_pretrained"):
                processor.save_pretrained(str(args.output_dir))

            report["training"].update(
                {
                    "global_step": int(getattr(trainer.state, "global_step", 0)),
                    "metrics": normalize_metrics(getattr(train_result, "metrics", None)),
                    "log_history_tail": [
                        normalize_metrics(item)
                        for item in getattr(trainer.state, "log_history", [])[-4:]
                    ],
                    "output_dir": str(args.output_dir.relative_to(repo_root())),
                }
            )
            try:
                torch.cuda.synchronize()
                report["environment"]["gpu_after"] = gpu_snapshot(torch)
                report["environment"]["max_memory_allocated_bytes"] = int(
                    torch.cuda.max_memory_allocated(0)
                )
                report["environment"]["max_memory_reserved_bytes"] = int(
                    torch.cuda.max_memory_reserved(0)
                )
            except Exception as exc:
                report["environment"]["gpu_after_error"] = f"{type(exc).__name__}: {exc}"

            report["warnings"] = [
                f"{item.category.__name__}: {item.message}" for item in caught_warnings
            ]
            report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "BLOCKED"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
            "traceback_tail": traceback.format_exc()[-5000:],
        }
    finally:
        report["duration_seconds"] = round(time.time() - started, 2)
        report["generated_at"] = utc_now()
        atomic_write_json(args.report, report)

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"REPORT={args.report}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
