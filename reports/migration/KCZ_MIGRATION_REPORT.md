# Project ZIPANGU K/C/Z migration report

更新日: 2026-09-05

## KCZ MIGRATION RESULT

ステータス: `LOCAL_COMPLETE / EXPERIMENT_BLOCKED`（設計移行とローカルK-I completion gateは完了、実験runは未承認）

- `ZIPANGU-K-I-4B` を直近の標準ターゲットとして登録。
- `ZIPANGU-C-I-9B` は既存の後続チャレンジとして保持。
- `ZIPANGU-Z-I-26B-A4B` は将来専用として登録し、実行可能化していない。
- Generation-I の `i-jp-heavy` / `i-balanced` をモデル非依存レシピとして追加。
- canonical model と Heretic / Abliterated community variant の名前空間を分離。
- 既存のC-I成果物・設定は削除していない。

## RX7600 RESULT

`PASS`（RX 7600を主デバイスとして検出）。PyTorch `2.11.0+rocm7.14.0` / HIP `7.14.60850`でdevice count 1、512×512行列積（finite=true）を実行した。Vulkan runtime（`vulkaninfo --summary`）も`PASS`。RX 7600 8GBはLocal Research / CI GPUとして扱い、AMD Radeon(TM) GraphicsのiGPUは主ベンチマークから除外した。ROCm 10は公式 feasibility `PASS`、既存の安定環境は変更していない。

## VULKAN VS ROCM

Vulkan/HIP の llama.cpp は同一 pinned source `159b741427337a2e9a58b08121001545d66b5825`、同一公式 Empero GGUFで実測した。`llama-cli`/`llama-bench`、日英 functional inference、CPU/Vulkan/ROCm × context 2048/4096/8192 の9ケース（warmup 1回・測定3回）はすべて`PASS`。中央値と生ログは`reports/hardware/llamacpp_completion_gate.json`および`reports/hardware/llamacpp_raw/`に保存した。

## UNSLOTH STATUS

`PASS`。`.envs/rocm-windows-py313`にPyTorch `2.11.0+rocm7.14.0`、HIP `7.14.60850`、Unsloth `2026.9.2`、bitsandbytes `0.50.2`、Transformers `5.5.0`、TRL `0.24.0`を隔離導入した。固定revisionのK-I 4Bを4bit読込し、LoRA（trainable `5,308,416` params）を接続して、合成2件・2 stepsと、`math_japanese_8k`から実際に選別した16例・2 stepsの研究専用 path smokeをRX 7600で完走した。実データ側はchat template、assistant-only supervision、forward/backward、optimizer、adapter save/reload、sample inferenceまで`PASS`。Triton fast path / `torch.compile` fallbackとbitsandbytesのROCm DLL選択警告は残るため、本番性能の合格判定とは分ける。

## K-I LOCAL TRAINING STATUS

`SMOKE_PASS / PRODUCTION_BLOCKED`。ローカル隔離環境で安全な極小smokeは実行済みだが、ZIPANGUの実験runは`TRAINING_ALLOWED=false`、provenance・評価・ベースライン・人間GOを維持する。K 1M、C/Z学習、RunPod、Heretic merge、アップロード、公開、pushは未実行。

## DATA FINALIZATION STATUS

`LOCAL_EXECUTION_COMPLETE / DATA_GATES_BLOCKED`。4,643,875 canonical recordsをトークナイズし、利用可能なJapanese MT-Bench / llm-jp-instructions 830件を使うcontamination処理、Nemotron分類、品質比較、license/provenance台帳、candidate生成、format検証、baseline gate、最終report生成まで実行した。raw inventory 4,662,182行との差18,307行は、Claude 18,370イベントを63セッションへgroup化した結果で欠落ではない。

contamination extractorは全10 candidate / 全8 evalの明示schema adapterへ移行し、空fingerprintを禁止、Frontierのmetadata `1,022,950`件とnon-authoritative derivative `895,771`件をscan対象外にした。全4,643,875件の修正scanは旧結果とrecord IDが全件一致し、mismatchは0件。修正後はexact `103`、near `0`、manual-review `5`、missing-required-evalによるnot-checked `2,725,046`。真正exact候補のmetadata-only stratified sampleは`100 / 103`件で、raw textを含まない。旧`1,393,704` empty-hash exactを生んだextractor defectと利用可能eval範囲のexact再検証は`CLOSED`、ただしAnswerCarefully未取得のためglobal contamination gateは`BLOCKED_REQUIRED_EVAL_MISSING`を維持する。

JP-heavy candidateは208,924/1,000,000 tokens、balanced candidateは316,249/1,000,000 tokensで、両方ともunderfilled。format検証もそれぞれ13/144件、22/168件の不整合を検出したため、candidate/formatは`BLOCKED`。AnswerCarefully欠落、全10 sourceのhuman license/provenance review、assistant-mask/LoRA target policyも未承認のため、1M判断は`NO-GO`。詳細は`reports/pretrain_finalization/FINAL_RECOMMENDATION.md`と`HUMAN_APPROVAL_QUEUE.md`に固定した。

## FILES CHANGED

主要な更新先は README、研究計画、モデル命名、評価プロトコル、データセットポリシー、RunPod運用、モデルレジストリ、K/C/Z設定、Generation-I共有レシピ、ローカルハードウェア診断・ベンチマーク計画。contamination修正は`schema_adapters.py`、`finalization-k-i-contamination-v2.yaml`、`audit_contamination_rescan.py`、`CONTAMINATION_RESCAN_AUDIT.md`とmetadata-only CSV/JSON evidenceへ固定した。今回追加・更新した実行系は`scripts/run_hip_smoke.py`、`scripts/run_unsloth_qlora_smoke.py`、`scripts/run_unsloth_real_data_smoke.py`、`scripts/run_llamacpp_completion_gate.py`、`scripts/hardware_probe.py`、`scripts/benchmark_local_inference.py`と`reports/hardware/*`。

## BLOCKERS

- AnswerCarefully本体はアクセス拒否のため`HARD_BLOCKED_EXTERNAL_ACCESS`。利用可能な830件のextractor/exact再検証は`CLOSED`だが、欠落ソースの穴埋めはしておらずglobal contamination PASSではない。
- exact 103件とmanual-review 5件の隔離維持、および各候補ソースの provenance / license clearanceとregistry quarantine解除。
- Triton fast path / torch.compile fallback、bitsandbytes ROCm DLL選択の互換性・性能評価は本番判断から分離。
- K-I各スケールの明示的な人間GO。
- C-I移行前にK-Iレシピとベースラインを凍結すること。

## NEXT EXACT STEP

まず`reports/pretrain_finalization/`の人間レビュー package と `reports/hardware/llamacpp_completion_gate.json` をレビューする。ZIPANGUのK-I本番run・モデル評価・有料GPU作成は、provenance/評価/予算確認と別途人間GOの後にだけ進める。
