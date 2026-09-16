# llama.cpp ROCm / HIP track

ステータス: `PASS`

- pinned source: `ggml-org/llama.cpp@159b741427337a2e9a58b08121001545d66b5825`
- build: `GGML_HIP=ON`, target `gfx1102`, isolated ROCm 7.1 compiler lane
- executable: `PASS` (`llama-cli.exe` / `llama-bench.exe`)
- device probe: `PASS` (`ROCm0: AMD Radeon RX 7600`)
- official Empero GGUF load: `PASS`
- Japanese and English functional inference: `PASS`
- `llama-bench` execution: `PASS`
- controlled matrix lane: `PASS` for context `2048`, `4096`, and `8192`
- PyTorch/Unsloth ROCm 7.14 lane: `PASS`

Windows HIPCC の compile-definition quoting に対しては、ソースを改変せず isolated build-only forced-include shim を使用した。実行結果の commit 表示は `e3fa55c` で、完全な upstream revision は manifest に記録している。

実測証跡は [llamacpp_completion_gate.json](./llamacpp_completion_gate.json)、
[llamacpp_benchmark_results.csv](./llamacpp_benchmark_results.csv)、
および `reports/hardware/llamacpp_raw/` に保存した。`rocminfo`/`rocm-smi` のような別 telemetry CLI は環境に存在しないため、VRAM peak・GPU utilization は `unavailable` として記録した。

ROCm 10 は [ROCM10_FEASIBILITY.md](./ROCM10_FEASIBILITY.md) の公式 feasibility 結果を参照する。現在の ROCm 7.14/driver は置換していない。
