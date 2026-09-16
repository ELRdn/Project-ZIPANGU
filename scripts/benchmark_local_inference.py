#!/usr/bin/env python3
"""Create or explicitly run a safe llama.cpp local backend comparison matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_status(path: Path) -> str:
    if not path.is_file():
        return "NOT_TESTED"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "PARTIAL"
    return str(payload.get("status", "NOT_TESTED"))


def read_nested_status(path: Path, *keys: str) -> str:
    if not path.is_file():
        return "NOT_TESTED"
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "PARTIAL"
    for key in keys:
        if not isinstance(payload, dict):
            return "PARTIAL"
        payload = payload.get(key)
    if not isinstance(payload, dict):
        return "PARTIAL"
    return str(payload.get("status", "NOT_TESTED"))


def build_matrix(
    model: str | None,
    executable: str | None,
    backends: Sequence[str],
    contexts: Sequence[int],
    warmup: int,
    measurements: int,
) -> list[dict[str, Any]]:
    return [
        {
            "backend": backend,
            "context_length": context,
            "model": model,
            "executable": executable,
            "warmup": warmup,
            "measurements": measurements,
            "status": "NOT_TESTED",
        }
        for backend in backends
        for context in contexts
    ]


def run_case(
    executable: str,
    template: str,
    model: str,
    backend: str,
    context: int,
    measurements: int,
) -> dict[str, Any]:
    rendered = template.format(
        model=model,
        backend=backend,
        context=context,
        measurements=measurements,
    )
    command = [executable, *shlex.split(rendered, posix=False)]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "BLOCKED", "command": command, "reason": "timeout"}
    except OSError as exc:
        return {
            "status": "BLOCKED",
            "command": command,
            "reason": type(exc).__name__,
        }
    return {
        "status": "PASS" if completed.returncode == 0 else "PARTIAL",
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stdout": completed.stdout[-3000:],
        "stderr": completed.stderr[-2000:],
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    model = str(args.model) if args.model else None
    executable = str(args.executable) if args.executable else None
    contexts = [int(value) for value in args.contexts.split(",") if value.strip()]
    backends = [value.strip() for value in args.backends.split(",") if value.strip()]
    matrix = build_matrix(
        model,
        executable,
        backends,
        contexts,
        args.warmup,
        args.measurements,
    )
    plan: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": now(),
        "status": "NOT_TESTED",
        "policy": {
            "same_model": True,
            "same_prompt": True,
            "same_context_matrix": True,
            "warmup": args.warmup,
            "measurements": args.measurements,
            "exclude_igpu": True,
            "no_implicit_download": True,
        },
        "matrix": matrix,
        "runtime_smoke_evidence": {
            "vulkan_runtime": {
                "status": read_nested_status(
                    Path("reports/hardware/results.json"), "runtime", "vulkaninfo"
                ),
                "source": "reports/hardware/results.json:runtime.vulkaninfo",
            },
            "hip_tensor": {
                "status": read_status(Path("reports/hardware/hip_smoke.json")),
                "source": "reports/hardware/hip_smoke.json",
            },
            "unsloth_qlora": {
                "status": read_status(Path("reports/hardware/unsloth_qlora_smoke.json")),
                "source": "reports/hardware/unsloth_qlora_smoke.json",
            },
        },
    }
    if not model or not executable:
        plan["reason"] = "explicit_model_and_executable_required"
        return plan
    model_path = Path(model)
    if not model_path.exists():
        plan["status"] = "BLOCKED"
        plan["reason"] = "model_path_not_found"
        return plan
    if not args.run:
        plan["reason"] = "manifest_only_use_run_for_explicit_execution"
        return plan
    if not args.args_template:
        plan["status"] = "BLOCKED"
        plan["reason"] = "args_template_required_for_execution"
        return plan
    for item in matrix:
        result = run_case(
            executable,
            args.args_template,
            model,
            str(item["backend"]),
            int(item["context_length"]),
            args.measurements,
        )
        item.update(result)
    statuses = {str(item.get("status")) for item in matrix}
    plan["status"] = "PASS" if statuses == {"PASS"} else "PARTIAL"
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Existing local GGUF path; never downloaded")
    parser.add_argument("--executable", help="Existing llama-bench/runner executable")
    parser.add_argument("--backends", default="cpu,vulkan,rocm")
    parser.add_argument("--contexts", default="2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measurements", type=int, default=3)
    parser.add_argument("--run", action="store_true", help="Run only with --args-template")
    parser.add_argument(
        "--args-template",
        help="Shell-free argument template, e.g. '-m {model} -c {context}'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/hardware/benchmark_manifest.json"),
    )
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.measurements < 1:
        parser.error("warmup must be >= 0 and measurements must be >= 1")
    plan = build_plan(args)
    write_json(args.output, plan)
    print(json.dumps({"status": plan["status"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if plan["status"] in {"PASS", "PARTIAL", "NOT_TESTED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
