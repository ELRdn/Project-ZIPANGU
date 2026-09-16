# Vulkan vs ROCm comparison

ステータス: `PASS`

同一の公式 Empero GGUF、同一の llama.cpp pinned source、同一の prompt/seed/thread/batch/生成長で、warmup 1回＋測定3回を実行した。CPU は同じ Vulkan build で offload を無効化し、loader と GGUF を揃えている。

| backend | context | prompt tok/s median | generation tok/s median | wall s median | status |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU | 2048 | 79.30 | 13.00 | 4.72 | PASS |
| CPU | 4096 | 82.60 | 13.30 | 4.72 | PASS |
| CPU | 8192 | 79.80 | 13.00 | 4.92 | PASS |
| Vulkan / RX 7600 | 2048 | 25.60 | 35.60 | 6.22 | PASS |
| Vulkan / RX 7600 | 4096 | 25.50 | 35.60 | 6.23 | PASS |
| Vulkan / RX 7600 | 8192 | 25.80 | 36.20 | 6.22 | PASS |
| HIP / RX 7600 | 2048 | 76.50 | 4.30 | 8.13 | PASS |
| HIP / RX 7600 | 4096 | 77.00 | 4.30 | 8.13 | PASS |
| HIP / RX 7600 | 8192 | 78.30 | 4.30 | 8.13 | PASS |

## Gate evidence

- device probes: Vulkan `Vulkan0`, HIP `ROCm0`; both resolved to RX 7600
- llama-bench: Vulkan `PASS`, HIP `PASS`
- functional inference: Japanese + English × Vulkan/HIP, all `PASS`
- matrix: all 9 cases `PASS`
- prompt timing: not emitted by this CLI build; all generation/prompt throughput values are finite
- peak VRAM and utilization: `unavailable` without changing the working Windows driver/telemetry stack
- raw logs: `reports/hardware/llamacpp_raw/`
- JSON/CSV: `reports/hardware/llamacpp_completion_gate.json`, `reports/hardware/llamacpp_benchmark_results.json`, `reports/hardware/llamacpp_benchmark_results.csv`

これは速度の優劣を一般化する結果ではなく、今回固定した GGUF・commit・設定での RX 7600 local compatibility/benchmark evidence である。
