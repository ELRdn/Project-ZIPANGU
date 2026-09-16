# llama.cpp Vulkan track

ステータス: `PASS`

- pinned source: `ggml-org/llama.cpp@159b741427337a2e9a58b08121001545d66b5825`
- build: `GGML_VULKAN=ON`, modern LunarG `glslc`, isolated build directory
- executable: `PASS` (`llama-cli.exe` / `llama-bench.exe`)
- device probe: `PASS` (`Vulkan0: AMD Radeon RX 7600`)
- official Empero GGUF load: `PASS`
- Japanese and English functional inference: `PASS`
- `llama-bench` execution: `PASS`
- controlled matrix lane: `PASS` for context `2048`, `4096`, and `8192`

実測証跡は [llamacpp_completion_gate.json](./llamacpp_completion_gate.json)、
[llamacpp_benchmark_results.csv](./llamacpp_benchmark_results.csv)、
および `reports/hardware/llamacpp_raw/` に保存した。使用モデルの repo、revision、サイズ、SHA-256 は
[LLAMACPP_MODEL_MANIFEST.json](./LLAMACPP_MODEL_MANIFEST.json) に固定している。

GPU VRAM peak と GPU utilization は、Windows 上で安定した ROCm telemetry provider を変更なしに利用できなかったため `unavailable` と記録した。これは推測値で補完していない。
