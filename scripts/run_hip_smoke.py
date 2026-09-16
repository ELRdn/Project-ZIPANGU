#!/usr/bin/env python3
"""Execute a bounded PyTorch HIP tensor smoke on the local GPU."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/hardware/hip_smoke.json"),
    )
    args = parser.parse_args()
    if not 32 <= args.size <= 1024:
        parser.error("--size must be between 32 and 1024")

    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "BLOCKED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "operation": "torch_cuda_matmul_on_hip",
        "size": args.size,
        "device_policy": "active discrete GPU only; no iGPU selection",
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["hip"] = getattr(torch.version, "hip", None)
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("PyTorch HIP device is not available")
        device = torch.device("cuda:0")
        report["device"] = torch.cuda.get_device_name(0)
        torch.manual_seed(3407)
        left = torch.randn((args.size, args.size), device=device, dtype=torch.float32)
        right = torch.randn((args.size, args.size), device=device, dtype=torch.float32)
        result = left @ right
        torch.cuda.synchronize()
        report.update(
            {
                "shape": [args.size, args.size],
                "dtype": str(result.dtype),
                "finite": bool(torch.isfinite(result).all().item()),
                "checksum": float(result.sum().item()),
                "status": "PASS",
            }
        )
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    report["duration_seconds"] = round(time.perf_counter() - started, 3)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"REPORT={args.output}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
