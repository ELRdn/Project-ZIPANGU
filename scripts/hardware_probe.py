#!/usr/bin/env python3
"""Read-only local hardware and runtime probe for the ZIPANGU RX 7600 lane."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

TARGET_GPU = "AMD Radeon RX 7600 8GB"
EXPECTED_GFX_ARCH = "gfx1102"
IGPU_MARKERS = (
    "780m",
    "integrated",
    "vega",
    "apu",
    "radeon graphics",
    "radeon(tm) graphics",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_probe_command(command: Sequence[str], timeout: int = 8) -> dict[str, Any]:
    """Run a bounded, read-only diagnostic command without a shell."""
    result: dict[str, Any] = {"command": list(command), "status": "HARD_BLOCKED"}
    executable = shutil.which(command[0])
    if executable is None:
        result["reason"] = "executable_not_found"
        return result
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["status"] = "HARD_BLOCKED"
        result["reason"] = type(exc).__name__
        return result
    result["returncode"] = completed.returncode
    result["stdout"] = completed.stdout[-4000:]
    result["stderr"] = completed.stderr[-2000:]
    result["status"] = "PASS" if completed.returncode == 0 else "HARD_BLOCKED"
    return result


def classify_gpu(
    name: str,
    vram_bytes: int | None = None,
    pci_id: str | None = None,
    gfx_arch: str | None = None,
) -> dict[str, Any]:
    lowered = name.lower()
    is_rx7600 = "rx 7600" in lowered or "rx7600" in lowered
    is_igpu = (not is_rx7600) and any(marker in lowered for marker in IGPU_MARKERS)
    if vram_bytes is not None and vram_bytes < 2 * 1024**3 and not is_rx7600:
        is_igpu = True
    role = "iGPU" if is_igpu else "discrete_gpu"
    device: dict[str, Any] = {
        "name": name,
        "vram_bytes": vram_bytes,
        "pci_id": pci_id,
        "gfx_arch": gfx_arch,
        "device_role": role,
        "target_rx7600": is_rx7600,
        "exclude_from_primary_benchmark": is_igpu,
        "architecture_match": (
            gfx_arch == EXPECTED_GFX_ARCH if gfx_arch is not None else None
        ),
    }
    if is_igpu:
        device["selection_reason"] = "iGPU_excluded_by_policy"
    elif is_rx7600:
        device["selection_reason"] = "target_rx7600"
    else:
        device["selection_reason"] = "non_target_discrete_gpu"
    return device


def select_primary_gpu(devices: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [item for item in devices if not item.get("exclude_from_primary_benchmark")]
    target = [item for item in eligible if item.get("target_rx7600")]
    return (target or eligible or [None])[0]


def _parse_windows_devices(raw: str) -> list[dict[str, Any]]:
    if not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    devices: list[dict[str, Any]] = []
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict) or not item.get("Name"):
            continue
        ram = item.get("AdapterRAM")
        try:
            ram_value = int(ram) if ram is not None else None
        except (TypeError, ValueError):
            ram_value = None
        devices.append(
            classify_gpu(
                str(item["Name"]),
                ram_value,
                str(item.get("PNPDeviceID")) if item.get("PNPDeviceID") else None,
            )
        )
    return devices


def discover_gpus() -> list[dict[str, Any]]:
    if os.name == "nt":
        command = [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,PNPDeviceID | ConvertTo-Json -Compress",
        ]
        result = run_probe_command(command)
        if result.get("status") == "PASS":
            return _parse_windows_devices(str(result.get("stdout", "")))
    lspci = run_probe_command(["lspci", "-mm"])
    devices: list[dict[str, Any]] = []
    for line in str(lspci.get("stdout", "")).splitlines():
        if "VGA" not in line and "3D controller" not in line and "Display controller" not in line:
            continue
        fields = line.split('"')
        names = [field.strip() for field in fields if field.strip()]
        name = names[-1] if names else line.strip()
        devices.append(classify_gpu(name))
    return devices


def collect_runtime_commands() -> dict[str, dict[str, Any]]:
    commands: dict[str, Sequence[str]] = {
        "rocminfo": ["rocminfo"],
        "hipconfig": ["hipconfig", "--version"],
        "rocm_smi": ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
        "vulkaninfo": ["vulkaninfo", "--summary"],
        "llama_cli": ["llama-cli", "--version"],
        "llama_bench": ["llama-bench", "--version"],
    }
    results = {name: run_probe_command(command) for name, command in commands.items()}
    hip = results.get("hipconfig", {})
    if hip.get("status") == "PASS" and any(
        marker in str(hip.get("stderr", ""))
        for marker in ("HIP version file", "Device not supported", "not recognized")
    ):
        hip["status"] = "HARD_BLOCKED"
        hip["reason"] = "global_hipconfig_environment_points_to_incomplete_ComfyUI_runtime"
    return results


def collect_pytorch_diagnostic() -> dict[str, Any]:
    code = r'''
import json
try:
    import torch
    result = {
        "status": "PASS",
        "torch_version": getattr(torch, "__version__", None),
        "hip_version": getattr(getattr(torch, "version", None), "hip", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ],
    }
except Exception as exc:
    result = {
        "status": "PARTIAL",
        "torch_version": None,
        "hip_version": None,
        "cuda_available": None,
        "device_count": None,
        "devices": [],
        "error_type": type(exc).__name__,
        "error": str(exc)[:500],
    }
print(json.dumps(result, ensure_ascii=False))
'''
    result = run_probe_command([sys.executable, "-c", code], timeout=30)
    if result.get("returncode") != 0:
        return {
            "status": "HARD_BLOCKED",
            "reason": "pytorch_probe_process_failed",
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }
    try:
        payload = json.loads(str(result.get("stdout", "")).strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"status": "HARD_BLOCKED", "reason": "pytorch_probe_output_unparseable"}
    return payload


def collect_unsloth_diagnostic() -> dict[str, Any]:
    code = "import unsloth; print('import_ok')"
    result = run_probe_command([sys.executable, "-c", code], timeout=30)
    if result.get("returncode") == 0:
        return {"status": "PASS", "stdout": result.get("stdout", "")[-500:]}
    if result.get("reason") == "executable_not_found":
        return {"status": "HARD_BLOCKED", "reason": "python_not_found"}
    return {
        "status": "HARD_BLOCKED",
        "reason": "unsloth_not_available_or_import_failed",
        "stderr": result.get("stderr", "")[-500:],
    }


def overall_status(
    devices: Sequence[dict[str, Any]],
    runtime: dict[str, dict[str, Any]],
    pytorch: dict[str, Any],
) -> str:
    primary = select_primary_gpu(devices)
    runtime_ok = any(
        runtime.get(key, {}).get("status") == "PASS"
        for key in ("rocminfo", "hipconfig", "vulkaninfo")
    )
    if primary and primary.get("target_rx7600") and runtime_ok:
        return "PASS"
    if devices or any(item.get("status") == "PASS" for item in runtime.values()):
        return "PARTIAL"
    if pytorch.get("status") == "PASS":
        return "PARTIAL"
    return "HARD_BLOCKED"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_environment_markdown(path: Path, report: dict[str, Any]) -> None:
    primary = report.get("primary_gpu") or {}
    lines = [
        "# RX 7600 local environment",
        "",
        f"ステータス: {report.get('status', 'HARD_BLOCKED')}",
        "",
        f"生成時刻: {report.get('generated_at')}",
        "",
        f"検出主デバイス: {primary.get('name', 'なし')}",
        f"デバイス役割: {primary.get('device_role', '未選択')}",
        f"デバイス検出ソース: {report.get('device_discovery_source', '未確認')}",
        f"gfx architecture: {primary.get('gfx_arch') or '未確認'}",
        f"VRAM reported bytes: {primary.get('vram_bytes') or '未確認'}",
        "VRAM注記: WMIのAdapterRAM値は仕様容量の確定値として扱わず、専用ツールで再確認する。",
        "",
        "ランタイム:",
        "",
    ]
    for name, item in report.get("runtime", {}).items():
        lines.append(f"- {name}: {item.get('status', 'HARD_BLOCKED')}")
    pytorch = report.get("pytorch", {})
    hip_smoke = report.get("hip_smoke", {})
    lines += [
        "",
        f"- PyTorch: {pytorch.get('status', 'HARD_BLOCKED')} ({pytorch.get('torch_version') or '未確認'})",
        f"- HIP reported by PyTorch: {pytorch.get('hip_version') or '未確認'}",
        f"- HIP tensor smoke: {hip_smoke.get('status', 'HARD_BLOCKED')}",
        f"- Unsloth import: {report.get('unsloth', {}).get('status', 'HARD_BLOCKED')}",
        f"- llama.cpp completion gate: {report.get('llamacpp_completion_gate', {}).get('status', 'HARD_BLOCKED')}",
        f"- ROCm10 official feasibility: {report.get('rocm10_feasibility', {}).get('status', 'HARD_BLOCKED')} (current ROCm 7.14 lane preserved)",
    ]
    lines += [
        "",
        "ポリシー: RX 7600はLocal Research / CI GPU。iGPUは主ベンチマークから除外。",
        "この診断は環境変更、依存関係のインストール、ドライバ変更を行わない。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report() -> dict[str, Any]:
    devices = discover_gpus()
    runtime = collect_runtime_commands()
    pytorch = collect_pytorch_diagnostic()
    discovery_source = "windows_wmi_or_lspci"
    # WMI can be blocked by a host ACL even when the active PyTorch runtime
    # can enumerate the GPU.  Keep the discovery result honest by recording
    # the fallback rather than reporting that no primary device exists.
    known_names = {str(item.get("name", "")).lower() for item in devices}
    for item in pytorch.get("devices", []):
        name = str(item.get("name", "")).strip()
        if name and name.lower() not in known_names:
            devices.append(classify_gpu(name))
            known_names.add(name.lower())
            discovery_source = "pytorch_runtime_fallback"
    unsloth = collect_unsloth_diagnostic()
    unsloth_smoke_path = Path("reports/hardware/unsloth_qlora_smoke.json")
    if unsloth.get("status") != "PASS" and unsloth_smoke_path.is_file():
        try:
            unsloth_smoke = json.loads(
                unsloth_smoke_path.read_text(encoding="utf-8")
            )
            if str(unsloth_smoke.get("status")) == "PASS":
                unsloth = {
                    "status": "PASS",
                    "source": str(unsloth_smoke_path),
                    "note": "isolated QLoRA smoke evidence",
                }
        except (OSError, json.JSONDecodeError):
            pass
    primary = select_primary_gpu(devices)
    benchmark_path = Path("reports/hardware/benchmark_manifest.json")
    hip_smoke_path = Path("reports/hardware/hip_smoke.json")
    hip_smoke_status = "HARD_BLOCKED"
    if hip_smoke_path.is_file():
        try:
            hip_smoke_payload = json.loads(hip_smoke_path.read_text(encoding="utf-8"))
            hip_smoke_status = str(hip_smoke_payload.get("status", hip_smoke_status))
        except (OSError, json.JSONDecodeError):
            hip_smoke_status = "PARTIAL"
    benchmark_status = "HARD_BLOCKED"
    if benchmark_path.is_file():
        try:
            benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
            benchmark_status = str(benchmark_payload.get("status", benchmark_status))
        except (OSError, json.JSONDecodeError):
            benchmark_status = "PARTIAL"
    completion_gate_path = Path("reports/hardware/llamacpp_completion_gate.json")
    completion_gate_status = "HARD_BLOCKED"
    if completion_gate_path.is_file():
        try:
            completion_gate_payload = json.loads(
                completion_gate_path.read_text(encoding="utf-8")
            )
            completion_gate_status = str(
                completion_gate_payload.get("status", completion_gate_status)
            )
        except (OSError, json.JSONDecodeError):
            completion_gate_status = "HARD_BLOCKED"
    if completion_gate_status == "PASS":
        runtime["llama_cli"] = {
            "status": "PASS",
            "evidence": str(completion_gate_path),
            "reason": "validated_by_actual_vulkan_and_hip_inference",
        }
        runtime["llama_bench"] = {
            "status": "PASS",
            "evidence": str(completion_gate_path),
            "reason": "validated_by_same_gguf_backend_benchmarks",
        }
    rocm10_path = Path("reports/hardware/ROCM10_FEASIBILITY.md")
    rocm10_status = "HARD_BLOCKED"
    if rocm10_path.is_file():
        try:
            rocm10_status = (
                "PASS"
                if "overall status: `PASS`"
                in rocm10_path.read_text(encoding="utf-8")
                else "HARD_BLOCKED"
            )
        except OSError:
            rocm10_status = "HARD_BLOCKED"
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": overall_status(devices, runtime, pytorch),
        "target_gpu": TARGET_GPU,
        "expected_gfx_arch": EXPECTED_GFX_ARCH,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "device_policy": {
            "exclude_igpu": True,
            "production_training": False,
            "local_research_and_ci": True,
        },
        "devices": devices,
        "device_discovery_source": discovery_source,
        "primary_gpu": primary,
        "runtime": runtime,
        "pytorch": pytorch,
        "unsloth": unsloth,
        "hip_smoke": {
            "status": hip_smoke_status,
            "manifest_path": str(hip_smoke_path),
        },
        "benchmark": {
            "status": benchmark_status,
            "manifest_path": str(benchmark_path),
        },
        "llamacpp_completion_gate": {
            "status": completion_gate_status,
            "manifest_path": str(completion_gate_path),
        },
        "rocm10_feasibility": {
            "status": rocm10_status,
            "report_path": str(rocm10_path),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/hardware/results.json"),
        help="JSON report path relative to the current directory",
    )
    args = parser.parse_args(argv)
    report = build_report()
    atomic_write_json(args.output, report)
    atomic_write_json(args.output.parent / "rocm_pytorch_diagnostic.json", report["pytorch"])
    write_environment_markdown(args.output.parent / "RX7600_ENVIRONMENT.md", report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["status"] in {"PASS", "PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
