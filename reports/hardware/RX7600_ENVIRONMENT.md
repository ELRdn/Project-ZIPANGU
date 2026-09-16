# RX 7600 local environment

ステータス: PASS

生成時刻: 2026-09-04T01:17:54.432917+00:00

検出主デバイス: AMD Radeon RX 7600
デバイス役割: discrete_gpu
デバイス検出ソース: pytorch_runtime_fallback
gfx architecture: 未確認
VRAM reported bytes: 未確認
VRAM注記: WMIのAdapterRAM値は仕様容量の確定値として扱わず、専用ツールで再確認する。

ランタイム:

- rocminfo: HARD_BLOCKED
- hipconfig: HARD_BLOCKED
- rocm_smi: HARD_BLOCKED
- vulkaninfo: PASS
- llama_cli: PASS
- llama_bench: PASS

- PyTorch: PASS (2.11.0+rocm7.14.0)
- HIP reported by PyTorch: 7.14.60850
- HIP tensor smoke: PASS
- Unsloth import: PASS
- llama.cpp completion gate: PASS
- ROCm10 official feasibility: PASS (current ROCm 7.14 lane preserved)

ポリシー: RX 7600はLocal Research / CI GPU。iGPUは主ベンチマークから除外。
この診断は環境変更、依存関係のインストール、ドライバ変更を行わない。
