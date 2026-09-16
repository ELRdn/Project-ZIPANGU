"""Safe vanilla-baseline preflight and resumable generation scaffolding."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import time
from typing import Any

from .core import atomic_write_json
from ..models import DEFAULT_MODEL_ID, load_model_spec
from .eval import iter_eval_rows
from .tokenization import TokenizationUnavailable, load_qwen_tokenizer


GENERATION_SETTINGS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "do_sample": True,
    "max_new_tokens": 16_384,
    "round_seeds": [3407, 3408, 3409],
}


def baseline_paths(
    repo_root: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
) -> dict[str, Path]:
    slug = model_id.casefold().replace("zipangu-", "", 1).replace("-", "_")
    root = Path(repo_root) / "baselines" / slug
    return {
        "root": root,
        "manifest": root / "manifest.json",
        "throughput": root / "throughput.json",
        "pending": root / "pending_judge.json",
    }


def _write_pending(path: Path, *, status: str, reason: str | None = None, pending: int | None = None) -> Path:
    payload = {
        "schema_version": 1,
        "status": status,
        "official_judge_required": True,
        "judge_executed": False,
        "judge": "llm-jp-judge v2.0.0",
        "judge_execution_policy": "not executed by finalization runner",
        "pending_prompts": pending,
    }
    if reason:
        payload["reason"] = reason
    return atomic_write_json(path, payload)


def _eval_prompts(eval_root: str | Path, *, limit: int = 2_000) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    root = Path(eval_root)
    for source in sorted(root.glob("sources/*")):
        if not source.is_dir():
            continue
        for relative, row_index, row in iter_eval_rows(source, split_hint="test"):
            prompt = row.get("prompt") or row.get("instruction") or row.get("question")
            if not prompt and isinstance(row.get("messages"), list):
                prompt = "\n".join(
                    str(item.get("content", ""))
                    for item in row["messages"]
                    if isinstance(item, Mapping) and str(item.get("role", "")).casefold() == "user"
                )
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            prompts.append({"source_row_id": f"{relative}#{row_index}", "prompt": prompt})
            if len(prompts) >= limit:
                return prompts
    return prompts


def _safe_baseline_preflight(model_root: Path, cache_root: Path) -> dict[str, Any]:
    model_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    free = __import__("shutil").disk_usage(model_root).free
    # The exact repository size is model/revision dependent.  Keep a
    # conservative reserve so a failed download cannot fill the drive.
    reserve = 25 * 1024**3
    return {
        "model_root": str(model_root),
        "cache_root": str(cache_root),
        "free_bytes": int(free),
        "reserve_bytes": reserve,
        "allowed": free > reserve,
    }


def _required_eval_sources_missing(repo_root: str | Path) -> list[str]:
    registry_path = Path(repo_root) / "data" / "manifests" / "eval_registry.json"
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["eval_registry"]
    evaluations = payload.get("evaluations", []) if isinstance(payload, Mapping) else []
    return [
        str(item.get("id") or "unknown")
        for item in evaluations
        if isinstance(item, Mapping)
        and bool(item.get("required", False))
        and item.get("availability") != "available"
    ]


def run_baseline(
    repo_root: str | Path,
    *,
    eval_root: str | Path,
    cache_root: str | Path,
    model_root: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    configured_revision: str | None = None,
    resume: bool = False,
    max_hours: float = 12.0,
) -> dict[str, Any]:
    spec = load_model_spec(repo_root, model_id)
    base_model_id = spec.base_model_id
    paths = baseline_paths(repo_root, spec.id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    pending_path = paths["pending"]
    preflight = _safe_baseline_preflight(Path(model_root), Path(cache_root))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_id": spec.id,
        "model": base_model_id,
        "tokenizer": base_model_id,
        "revision": None,
        "generation": dict(GENERATION_SETTINGS),
        "baseline_type": "untouched_vanilla",
        "official_judge_required": True,
        "judge_executed": False,
        "training_allowed": False,
        "preflight": preflight,
    }
    if not preflight["allowed"]:
        manifest["status"] = "BLOCKED"
        manifest["blocked_reason"] = "disk_reserve_insufficient"
        atomic_write_json(paths["manifest"], manifest)
        _write_pending(pending_path, status="BLOCKED", reason="disk_reserve_insufficient")
        return {"status": "BLOCKED", "artifacts": [paths["manifest"], pending_path], "reason": "disk_reserve_insufficient"}

    required_missing = _required_eval_sources_missing(repo_root)
    if required_missing:
        reason = f"required_eval_sources_missing:{','.join(required_missing)}"
        manifest["status"] = "BLOCKED"
        manifest["blocked_reason"] = reason
        manifest["required_eval_sources_missing"] = required_missing
        atomic_write_json(paths["manifest"], manifest)
        _write_pending(pending_path, status="BLOCKED", reason=reason, pending=0)
        return {
            "status": "BLOCKED",
            "artifacts": [paths["manifest"], pending_path],
            "blocked_reasons": [reason],
            "details": {"required_eval_sources_missing": required_missing},
        }

    try:
        tokenizer, revision = load_qwen_tokenizer(
            cache_root=cache_root,
            model_root=model_root,
            model_id=base_model_id,
            configured_revision=configured_revision,
        )
    except TokenizationUnavailable as exc:
        manifest["status"] = "BLOCKED"
        manifest["blocked_reason"] = str(exc)
        atomic_write_json(paths["manifest"], manifest)
        _write_pending(pending_path, status="BLOCKED", reason=str(exc))
        return {"status": "BLOCKED", "artifacts": [paths["manifest"], pending_path], "reason": str(exc)}
    manifest["revision"] = revision
    manifest["tokenizer_loaded"] = True

    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:  # pragma: no cover - environment dependent
        reason = "torch_and_transformers_required_for_baseline"
        manifest["status"] = "BLOCKED"
        manifest["blocked_reason"] = reason
        atomic_write_json(paths["manifest"], manifest)
        _write_pending(pending_path, status="BLOCKED", reason=reason)
        return {"status": "BLOCKED", "artifacts": [paths["manifest"], pending_path], "reason": reason}

    try:
        config = AutoConfig.from_pretrained(
            base_model_id,
            revision=revision,
            cache_dir=str(Path(cache_root) / "hub"),
        )
        model_type = str(getattr(config, "model_type", "")).casefold()
        if model_type not in {"qwen3_5", "qwen3.5", "qwen3_5_moe"}:
            raise RuntimeError(f"unsupported_qwen_architecture:{model_type or 'unknown'}")
        if not torch.cuda.is_available():
            raise RuntimeError("no_cuda_device_for_safe_baseline")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            revision=revision,
            cache_dir=str(Path(model_root) / "hub"),
            device_map="auto",
            torch_dtype="auto",
        )
        model.eval()
    except Exception as exc:  # pragma: no cover - model/runtime dependent
        reason = str(exc)
        manifest["status"] = "BLOCKED"
        manifest["blocked_reason"] = reason
        atomic_write_json(paths["manifest"], manifest)
        _write_pending(pending_path, status="BLOCKED", reason=reason)
        return {"status": "BLOCKED", "artifacts": [paths["manifest"], pending_path], "reason": reason}

    prompts = _eval_prompts(eval_root)
    if not prompts:
        reason = "no_available_eval_prompts"
        manifest["status"] = "BLOCKED"
        manifest["blocked_reason"] = reason
        atomic_write_json(paths["manifest"], manifest)
        _write_pending(pending_path, status="BLOCKED", reason=reason, pending=0)
        return {"status": "BLOCKED", "artifacts": [paths["manifest"], pending_path], "reason": reason}

    started = time.monotonic()
    throughput: dict[str, Any] = {"probe": {}, "rounds": [], "max_hours": max_hours}
    pending = len(prompts) * len(GENERATION_SETTINGS["round_seeds"])
    status = "PASS"
    for round_number, seed in enumerate(GENERATION_SETTINGS["round_seeds"], start=1):
        output_dir = paths["root"] / "generation" / f"round-{round_number}"
        output_path = output_dir / "generations.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        if resume and output_path.is_file():
            pending -= len(prompts)
            continue
        if time.monotonic() - started > max_hours * 3600:
            status = "PARTIAL"
            break
        torch.manual_seed(int(seed))
        samples: list[dict[str, Any]] = []
        round_started = time.monotonic()
        for index, item in enumerate(prompts):
            if time.monotonic() - started > max_hours * 3600:
                status = "PARTIAL"
                break
            encoded = tokenizer(item["prompt"], return_tensors="pt")
            device = getattr(model, "device", None)
            if device is not None:
                encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    temperature=GENERATION_SETTINGS["temperature"],
                    top_p=GENERATION_SETTINGS["top_p"],
                    top_k=GENERATION_SETTINGS["top_k"],
                    do_sample=True,
                    max_new_tokens=GENERATION_SETTINGS["max_new_tokens"],
                )
            prompt_length = int(encoded["input_ids"].shape[-1])
            generated_ids = generated[0][prompt_length:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            samples.append(
                {
                    "source_row_id": item["source_row_id"],
                    "round": round_number,
                    "seed": seed,
                    "generation": text,
                }
            )
            pending -= 1
            if index == 1:
                elapsed = max(1e-9, time.monotonic() - round_started)
                throughput["probe"] = {
                    "prompts": 2,
                    "tokens_per_second": round(sum(len(generated[0]) / elapsed for _ in [0]), 3),
                    "elapsed_seconds": round(elapsed, 3),
                }
        with output_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        elapsed = max(1e-9, time.monotonic() - round_started)
        throughput["rounds"].append({"round": round_number, "seed": seed, "rows": len(samples), "elapsed_seconds": round(elapsed, 3)})
        if status == "PARTIAL":
            break
    throughput["pending_prompts"] = max(0, pending)
    atomic_write_json(paths["throughput"], throughput)
    manifest["status"] = status
    manifest["completed_rounds"] = len(throughput["rounds"])
    manifest["pending_prompts"] = max(0, pending)
    atomic_write_json(paths["manifest"], manifest)
    _write_pending(pending_path, status=status, pending=max(0, pending))
    return {"status": status, "artifacts": [paths["manifest"], paths["throughput"], pending_path]}
