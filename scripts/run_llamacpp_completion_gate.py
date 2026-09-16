#!/usr/bin/env python3
"""Run the local llama.cpp completion gate with reproducible evidence.

The script never downloads a model.  It runs the explicitly supplied
executables against one already-hashed GGUF, keeps raw stdout/stderr for every
attempt, and marks an individual matrix case HARD_BLOCKED when the attempt
fails.  This is a local compatibility/performance gate, not a training run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from statistics import median
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    "japanese": "日本の四季を、それぞれの特徴が分かるように短く説明してください。",
    "english": "Explain the four seasons of Japan briefly, mentioning one characteristic of each season.",
}
TIMING_RE = re.compile(
    r"\[\s*Prompt:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s\s*\|\s*"
    r"Generation:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s\s*\]",
    re.IGNORECASE,
)
PROMPT_TIME_RE = re.compile(
    r"prompt\s+eval\s+time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
    re.IGNORECASE,
)
CRASH_MARKERS = (
    "access violation",
    "segmentation fault",
    "stack overflow",
    "fatal error",
    "abort() has been called",
    "exception code",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def tail_text(path: Path, limit: int = 12_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<read failed: {type(exc).__name__}: {exc}>"
    return text[-limit:]


def runtime_environment(backend: str, rocm_path: Path, hip_visible_devices: str | None) -> dict[str, str]:
    environment = dict(os.environ)
    if backend == "hip":
        environment["ROCM_PATH"] = str(rocm_path)
        environment["HIP_PATH"] = str(rocm_path)
        environment["HIP_DEVICE_LIB_PATH"] = str(rocm_path / "amdgcn" / "bitcode")
        environment["HSA_OVERRIDE_GFX_VERSION"] = environment.get(
            "HSA_OVERRIDE_GFX_VERSION", "11.0.2"
        )
        if hip_visible_devices:
            environment["HIP_VISIBLE_DEVICES"] = hip_visible_devices
        rocm_bin = str(rocm_path / "bin")
        path_key = "Path" if "Path" in environment else "PATH"
        old_path = environment.get(path_key, "")
        if rocm_bin.casefold() not in {item.casefold() for item in old_path.split(";") if item}:
            environment[path_key] = f"{rocm_bin};{old_path}"
    return environment


def run_process(
    command: Sequence[str],
    *,
    backend: str,
    raw_dir: Path,
    label: str,
    timeout_seconds: int,
    rocm_path: Path,
    hip_visible_devices: str | None,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = raw_dir / f"{label}.stdout.txt"
    stderr_path = raw_dir / f"{label}.stderr.txt"
    started = time.perf_counter()
    peak_rss: int | None = None
    timed_out = False
    returncode: int | None = None
    error: str | None = None
    try:
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr:
            process = subprocess.Popen(
                list(command),
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                cwd=str(REPO_ROOT),
                env=runtime_environment(backend, rocm_path, hip_visible_devices),
                shell=False,
            )
            try:
                import psutil

                process_info = psutil.Process(process.pid)
            except Exception:
                process_info = None
            deadline = time.perf_counter() + timeout_seconds
            while process.poll() is None:
                if process_info is not None:
                    try:
                        peak_rss = max(peak_rss or 0, int(process_info.memory_info().rss))
                    except (OSError, psutil.Error):
                        pass
                if time.perf_counter() >= deadline:
                    timed_out = True
                    process.kill()
                    break
                time.sleep(0.1)
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=10)
            if process_info is not None:
                try:
                    peak_rss = max(peak_rss or 0, int(process_info.memory_info().rss))
                except (OSError, psutil.Error):
                    pass
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = round(time.perf_counter() - started, 3)
    stdout = tail_text(stdout_path)
    stderr = tail_text(stderr_path)
    combined = f"{stdout}\n{stderr}".casefold()
    crash_marker = next((marker for marker in CRASH_MARKERS if marker in combined), None)
    timing_match = TIMING_RE.search(stdout + "\n" + stderr)
    prompt_speed = float(timing_match.group(1)) if timing_match else None
    generation_speed = float(timing_match.group(2)) if timing_match else None
    prompt_time_match = PROMPT_TIME_RE.search(stdout + "\n" + stderr)
    prompt_time_seconds = float(prompt_time_match.group(1)) / 1000 if prompt_time_match else None
    passed = (
        error is None
        and not timed_out
        and returncode == 0
        and bool(stdout.strip())
        and crash_marker is None
    )
    result: dict[str, Any] = {
        "status": "PASS" if passed else "HARD_BLOCKED",
        "command": [str(item) for item in command],
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
        "stdout_tail": stdout,
        "stderr_tail": stderr,
        "prompt_tokens_per_second": prompt_speed,
        "generation_tokens_per_second": generation_speed,
        "ttft_seconds": prompt_time_seconds,
    }
    if error:
        result["error"] = error
    if timed_out:
        result["reason"] = f"timeout_after_{timeout_seconds}s"
    elif crash_marker:
        result["reason"] = f"crash_marker:{crash_marker}"
    elif not stdout.strip() and passed is False:
        result["reason"] = "empty_stdout_or_no_generation"
    elif returncode not in (None, 0):
        result["reason"] = "nonzero_returncode"
    return result


def model_info(model: Path) -> dict[str, Any]:
    if not model.is_file():
        return {"path": str(model), "exists": False}
    return {
        "path": str(model),
        "exists": True,
        "size_bytes": model.stat().st_size,
        "sha256": sha256_file(model),
    }


def llama_args(
    model: Path,
    *,
    backend: str,
    device: str,
    context: int,
    prompt: str,
    threads: int,
    batch_size: int,
    n_predict: int,
    seed: int,
) -> list[str]:
    args = [
        "--model",
        str(model),
        "--offline",
        "--no-mmproj",
        "--ctx-size",
        str(context),
        "--batch-size",
        str(batch_size),
        "--ubatch-size",
        str(batch_size),
        "--threads",
        str(threads),
        "--threads-batch",
        str(threads),
        "--predict",
        str(n_predict),
        "--seed",
        str(seed),
        "--temperature",
        "0.0",
        "--top-k",
        "1",
        "--top-p",
        "1.0",
        "--fit",
        "off",
        "--split-mode",
        "none",
        "--flash-attn",
        "off",
        "--no-warmup",
        "--single-turn",
        "--no-display-prompt",
        "--simple-io",
        "--reasoning",
        "off",
        "--show-timings",
        "--log-verbosity",
        "1",
        "--prompt",
        prompt,
    ]
    if backend == "cpu":
        args.extend(["--device", "none", "--gpu-layers", "0"])
    else:
        args.extend(["--device", device, "--gpu-layers", "all"])
    return args


def executable_info(executable: Path, backend: str, rocm_path: Path, hip_visible_devices: str | None) -> dict[str, Any]:
    result = run_process(
        [str(executable), "--version"],
        backend=backend,
        raw_dir=executable.parent / "_completion_gate_probe",
        label=f"{backend}-version",
        timeout_seconds=30,
        rocm_path=rocm_path,
        hip_visible_devices=hip_visible_devices,
    )
    return {key: value for key, value in result.items() if key not in {"stdout_tail", "stderr_tail"}}


def device_probe(
    executable: Path,
    *,
    backend: str,
    requested_device: str,
    raw_dir: Path,
    rocm_path: Path,
    hip_visible_devices: str | None,
) -> dict[str, Any]:
    """Verify the requested accelerator is exposed by the matching build."""

    result = run_process(
        [str(executable), "--list-devices"],
        backend=backend,
        raw_dir=raw_dir,
        label=f"devices-{backend}",
        timeout_seconds=60,
        rocm_path=rocm_path,
        hip_visible_devices=hip_visible_devices,
    )
    combined = f"{result.get('stdout_tail', '')}\n{result.get('stderr_tail', '')}"
    device_seen = requested_device.casefold() in combined.casefold()
    result.update(
        {
            "backend": backend,
            "requested_device": requested_device,
            "device_seen": device_seen,
        }
    )
    if result["status"] == "PASS" and not device_seen:
        result["status"] = "HARD_BLOCKED"
        result["reason"] = "requested_device_not_listed"
    return result


def run_smoke(
    *,
    model: Path,
    executables: dict[str, Path],
    devices: dict[str, str],
    raw_dir: Path,
    rocm_path: Path,
    hip_visible_devices: str | None,
    device_probes: dict[str, dict[str, Any]],
    threads: int,
    batch_size: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for backend in ("vulkan", "hip"):
        executable = executables[backend]
        for language, prompt in PROMPTS.items():
            label = f"smoke-{backend}-{language}"
            result = run_process(
                [
                    str(executable),
                    *llama_args(
                        model,
                        backend=backend,
                        device=devices[backend],
                        context=2048,
                        prompt=prompt,
                        threads=threads,
                        batch_size=batch_size,
                        n_predict=48,
                        seed=3407,
                    ),
                ],
                backend=backend,
                raw_dir=raw_dir,
                label=label,
                timeout_seconds=timeout_seconds,
                rocm_path=rocm_path,
                hip_visible_devices=hip_visible_devices,
            )
            combined = f"{result.get('stdout_tail', '')}\n{result.get('stderr_tail', '')}"
            result.update(
                {
                    "backend": backend,
                    "language": language,
                    "model": str(model),
                    "device": devices[backend],
                    "nonempty_generation": bool(str(result.get("stdout_tail", "")).strip()),
                    "finite_timing": all(
                        value is None or finite_number(value)
                        for value in (
                            result.get("prompt_tokens_per_second"),
                            result.get("generation_tokens_per_second"),
                            result.get("ttft_seconds"),
                        )
                    ),
                    "backend_marker_seen": backend.casefold() in combined.casefold()
                    or devices[backend].casefold() in combined.casefold(),
                    "backend_device_probe_pass": device_probes.get(backend, {}).get("device_seen") is True,
                }
            )
            if result["status"] == "PASS" and not result["backend_device_probe_pass"]:
                result["status"] = "HARD_BLOCKED"
                result["reason"] = "requested_device_probe_failed"
            results.append(result)
    status = "PASS" if results and all(item["status"] == "PASS" for item in results) else "HARD_BLOCKED"
    return {
        "schema_version": 1,
        "status": status,
        "prompts": PROMPTS,
        "configuration": {
            "context": 2048,
            "n_predict": 48,
            "threads": threads,
            "batch_size": batch_size,
            "seed": 3407,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "reasoning": "off",
        },
        "results": results,
    }


def run_bench_binary(
    *,
    model: Path,
    executable: Path,
    backend: str,
    device: str,
    raw_dir: Path,
    rocm_path: Path,
    hip_visible_devices: str | None,
    threads: int,
    batch_size: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    gpu_args = ["--device", device, "--n-gpu-layers", "-1"] if backend != "cpu" else [
        "--device",
        "none",
        "--n-gpu-layers",
        "0",
    ]
    command = [
        str(executable),
        "--offline",
        "--model",
        str(model),
        "--repetitions",
        "1",
        "--n-prompt",
        "32",
        "--n-gen",
        "16",
        "--batch-size",
        str(batch_size),
        "--ubatch-size",
        str(batch_size),
        "--threads",
        str(threads),
        "--split-mode",
        "none",
        "--no-warmup",
        "--output",
        "json",
        *gpu_args,
    ]
    return run_process(
        command,
        backend=backend,
        raw_dir=raw_dir,
        label=f"llama-bench-{backend}",
        timeout_seconds=timeout_seconds,
        rocm_path=rocm_path,
        hip_visible_devices=hip_visible_devices,
    )


def median_or_none(values: Sequence[Any]) -> float | None:
    numbers = [float(value) for value in values if finite_number(value)]
    return round(float(median(numbers)), 6) if numbers else None


def run_matrix(
    *,
    model: Path,
    executables: dict[str, Path],
    devices: dict[str, str],
    contexts: Sequence[int],
    measurements: int,
    raw_dir: Path,
    rocm_path: Path,
    hip_visible_devices: str | None,
    threads: int,
    batch_size: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    # CPU uses the Vulkan build with offload explicitly disabled.  This keeps
    # the GGUF, loader and source revision identical across the three lanes.
    lanes = {
        "cpu": executables["vulkan"],
        "vulkan": executables["vulkan"],
        "hip": executables["hip"],
    }
    for backend, executable in lanes.items():
        for context in contexts:
            prompt = (
                "次の問いに日本語と英語で短く答えてください。"
                "日本の首都はどこですか？ What is the capital of Japan?"
            )
            fixed = {
                "backend": backend,
                "context": int(context),
                "executable": str(executable),
                "device": "none" if backend == "cpu" else devices[backend],
                "warmup_count": 1,
                "measurement_count": measurements,
                "runs": [],
            }
            lane_device = "none" if backend == "cpu" else devices[backend]
            warmup = run_process(
                [
                    str(executable),
                    *llama_args(
                        model,
                        backend=backend,
                        device=lane_device,
                        context=int(context),
                        prompt=prompt,
                        threads=threads,
                        batch_size=batch_size,
                        n_predict=32,
                        seed=3407,
                    ),
                ],
                backend=backend,
                raw_dir=raw_dir,
                label=f"matrix-{backend}-ctx{context}-warmup",
                timeout_seconds=timeout_seconds,
                rocm_path=rocm_path,
                hip_visible_devices=hip_visible_devices,
            )
            fixed["warmup"] = warmup
            for repetition in range(1, measurements + 1):
                run = run_process(
                    [
                        str(executable),
                        *llama_args(
                            model,
                            backend=backend,
                            device=lane_device,
                            context=int(context),
                            prompt=prompt,
                            threads=threads,
                            batch_size=batch_size,
                            n_predict=32,
                            seed=3407,
                        ),
                    ],
                    backend=backend,
                    raw_dir=raw_dir,
                    label=f"matrix-{backend}-ctx{context}-run{repetition}",
                    timeout_seconds=timeout_seconds,
                    rocm_path=rocm_path,
                    hip_visible_devices=hip_visible_devices,
                )
                run["repetition"] = repetition
                fixed["runs"].append(run)
            successful = [run for run in fixed["runs"] if run.get("status") == "PASS"]
            all_attempts_passed = (
                warmup.get("status") == "PASS"
                and len(successful) == measurements
            )
            fixed.update(
                {
                    "status": "PASS" if all_attempts_passed else "HARD_BLOCKED",
                    "metrics": {
                        "prompt_tokens_per_second_median": median_or_none(
                            [run.get("prompt_tokens_per_second") for run in successful]
                        ),
                        "generation_tokens_per_second_median": median_or_none(
                            [run.get("generation_tokens_per_second") for run in successful]
                        ),
                        "ttft_seconds_median": median_or_none(
                            [run.get("ttft_seconds") for run in successful]
                        ),
                        "wall_seconds_median": median_or_none(
                            [run.get("elapsed_seconds") for run in successful]
                        ),
                        "peak_ram_bytes_max": max(
                            (int(run["peak_rss_bytes"]) for run in successful if run.get("peak_rss_bytes")),
                            default=None,
                        ),
                        "peak_vram_bytes": None,
                        "gpu_utilization_percent": None,
                    },
                    "telemetry": {
                        "peak_vram_bytes": "unavailable: llama.cpp Windows CLI does not expose a stable peak allocation counter",
                        "gpu_utilization_percent": "unavailable: no ROCm SMI/telemetry provider was available without altering the working driver stack",
                    },
                }
            )
            if not all_attempts_passed:
                fixed["reason"] = "warmup_or_measurement_failed"
            cases.append(fixed)
    status = "PASS" if cases and all(case["status"] == "PASS" for case in cases) else "HARD_BLOCKED"
    return {
        "schema_version": 1,
        "status": status,
        "matrix_shape": "CPU/Vulkan/HIP x context 2048/4096/8192",
        "configuration": {
            "contexts": [int(item) for item in contexts],
            "warmup_count": 1,
            "measurements": measurements,
            "prompt": "次の問いに日本語と英語で短く答えてください。日本の首都はどこですか？ What is the capital of Japan?",
            "n_predict": 32,
            "threads": threads,
            "batch_size": batch_size,
            "seed": 3407,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "split_mode": "none",
            "cpu_offload": "none / n_gpu_layers=0",
            "gpu_offload": "selected device / n_gpu_layers=all",
        },
        "cases": cases,
    }


def write_matrix_csv(path: Path, matrix: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend",
        "context",
        "status",
        "device",
        "prompt_tokens_per_second_median",
        "generation_tokens_per_second_median",
        "ttft_seconds_median",
        "wall_seconds_median",
        "peak_ram_bytes_max",
        "peak_vram_bytes",
        "gpu_utilization_percent",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in matrix.get("cases", []):
            row = {field: case.get(field) for field in fields if field not in case}
            row.update(case.get("metrics", {}))
            row.update({field: case.get(field) for field in fields if field in case})
            writer.writerow({field: row.get(field) for field in fields})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing official GGUF; never downloaded")
    parser.add_argument("--vulkan-cli", required=True)
    parser.add_argument("--hip-cli", required=True)
    parser.add_argument("--vulkan-bench", required=True)
    parser.add_argument("--hip-bench", required=True)
    parser.add_argument("--vulkan-device", default="Vulkan0")
    parser.add_argument("--hip-device", default="HIP0")
    parser.add_argument("--rocm-path", default=r"C:\Program Files\AMD\ROCm\7.1")
    parser.add_argument("--hip-visible-devices", default=os.environ.get("HIP_VISIBLE_DEVICES", "1"))
    parser.add_argument("--contexts", default="2048,4096,8192")
    parser.add_argument("--measurements", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        default="reports/hardware",
        help="Directory for JSON/CSV evidence",
    )
    parser.add_argument(
        "--raw-dir",
        default="reports/hardware/llamacpp_raw",
        help="Directory for raw stdout/stderr evidence",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.measurements < 1:
        raise SystemExit("--measurements must be >= 1")
    contexts = [int(value.strip()) for value in args.contexts.split(",") if value.strip()]
    if contexts != [2048, 4096, 8192]:
        raise SystemExit("completion gate requires exactly contexts 2048,4096,8192")
    if args.threads < 1 or args.batch_size < 1 or args.timeout_seconds < 1:
        raise SystemExit("threads, batch-size and timeout-seconds must be positive")

    model = resolve_path(args.model)
    output_dir = resolve_path(args.output_dir)
    raw_dir = resolve_path(args.raw_dir)
    rocm_path = resolve_path(args.rocm_path)
    executables = {
        "vulkan": resolve_path(args.vulkan_cli),
        "hip": resolve_path(args.hip_cli),
    }
    benchmarks = {
        "vulkan": resolve_path(args.vulkan_bench),
        "hip": resolve_path(args.hip_bench),
    }
    devices = {"vulkan": args.vulkan_device, "hip": args.hip_device}

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "HARD_BLOCKED",
        "generated_at": utc_now(),
        "scope": {
            "local_only": True,
            "model_downloaded_by_script": False,
            "training": False,
            "paid_compute": False,
        },
        "model": model_info(model),
        "source": {
            "vulkan_executable": str(executables["vulkan"]),
            "hip_executable": str(executables["hip"]),
            "vulkan_bench": str(benchmarks["vulkan"]),
            "hip_bench": str(benchmarks["hip"]),
            "rocm_path": str(rocm_path),
            "hip_visible_devices": args.hip_visible_devices,
        },
    }
    if not model.is_file():
        report["reason"] = "model_path_not_found"
    else:
        report["executable_probes"] = {
            backend: executable_info(executables[backend], backend, rocm_path, args.hip_visible_devices)
            for backend in ("vulkan", "hip")
        }
        report["device_probes"] = {
            backend: device_probe(
                executables[backend],
                backend=backend,
                requested_device=devices[backend],
                raw_dir=raw_dir,
                rocm_path=rocm_path,
                hip_visible_devices=args.hip_visible_devices,
            )
            for backend in ("vulkan", "hip")
        }
        report["llama_bench"] = {
            backend: run_bench_binary(
                model=model,
                executable=benchmarks[backend],
                backend=backend,
                device=devices[backend],
                raw_dir=raw_dir,
                rocm_path=rocm_path,
                hip_visible_devices=args.hip_visible_devices,
                threads=args.threads,
                batch_size=args.batch_size,
                timeout_seconds=args.timeout_seconds,
            )
            for backend in ("vulkan", "hip")
        }
        report["inference_smoke"] = run_smoke(
            model=model,
            executables=executables,
            devices=devices,
            raw_dir=raw_dir,
            rocm_path=rocm_path,
            hip_visible_devices=args.hip_visible_devices,
            device_probes=report["device_probes"],
            threads=args.threads,
            batch_size=args.batch_size,
            timeout_seconds=args.timeout_seconds,
        )
        report["matrix"] = run_matrix(
            model=model,
            executables=executables,
            devices=devices,
            contexts=contexts,
            measurements=args.measurements,
            raw_dir=raw_dir,
            rocm_path=rocm_path,
            hip_visible_devices=args.hip_visible_devices,
            threads=args.threads,
            batch_size=args.batch_size,
            timeout_seconds=args.timeout_seconds,
        )
        bench_pass = all(item.get("status") == "PASS" for item in report["llama_bench"].values())
        device_probe_pass = all(item.get("status") == "PASS" for item in report["device_probes"].values())
        smoke_pass = report["inference_smoke"].get("status") == "PASS"
        matrix_pass = report["matrix"].get("status") == "PASS"
        report["status"] = "PASS" if bench_pass and device_probe_pass and smoke_pass and matrix_pass else "HARD_BLOCKED"

    write_json(output_dir / "llamacpp_inference_smoke.json", report.get("inference_smoke", report))
    write_json(output_dir / "llamacpp_benchmark_results.json", report.get("matrix", report))
    write_matrix_csv(output_dir / "llamacpp_benchmark_results.csv", report.get("matrix", {}))
    write_json(output_dir / "llamacpp_completion_gate.json", report)
    print(json.dumps({"status": report["status"], "output": str(output_dir / "llamacpp_completion_gate.json")}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
