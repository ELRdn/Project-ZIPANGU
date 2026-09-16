# Unsloth AMD status

ステータス: `PASS`（隔離venvで実モデルQLoRA smoke完走）

- isolated venv: `.envs/rocm-windows-py313`（Python 3.13.5）
- GPU: `AMD Radeon RX 7600`、PyTorch device count `1`
- PyTorch: `2.11.0+rocm7.14.0`
- HIP reported by PyTorch: `7.14.60850`
- Unsloth: `2026.9.2`
- bitsandbytes: `0.50.2`
- Transformers / TRL: `5.5.0` / `0.24.0`
- import: `PASS`
- 4bit model load: `PASS`（固定revision `c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e`）
- LoRA attach: `PASS`（trainable `5,308,416` parameters）
- QLoRA smoke: `PASS`（合成2件、sequence length 64、2 steps、optimizer `adamw_torch`、評価なし）
- train loss: `3.297 → 2.989`、adapter output: `.cache/smoke-runs/k-i-qlora-20260902-230942`
- raw report: `reports/hardware/unsloth_qlora_smoke.json`

既知の注意点:

- Triton/Unslothの一部fast pathと`torch.compile`はWindows/AMD上でfallback警告が出るが、今回の2 stepは完走した。
- bitsandbytesは`libbitsandbytes_rocm715.dll`選択警告を出した。性能評価や本番runの前に、ROCm/HIP対応組合せを固定して再確認する。
- MIOpen/Triton/TorchInductorのキャッシュはプロジェクト内`.cache/huggingface`へ隔離した。
- ROCm10 experimental environment: `FEASIBILITY_PASS / LOCAL_RUNTIME_HARD_BLOCKED`（公式RX 7600/gfx1102 supportは確認済み。WSL列挙は`E_ACCESSDENIED`で、安定系ROCm 7.14を壊す導入は行っていない）
- production training: `BLOCKED`（`TRAINING_ALLOWED=false`、provenance・評価・人間GOが未完了）
