# configs/train priority

Execution config priority is: `configs/train/k-i/` first, `configs/train/c-i/` after K success, `configs/train/z-i/` future-only. Root-level legacy configs are C-I compatible and must not be used for new K runs.

1. K (current default, immediate target): `configs/train/k-i/` for `ZIPANGU-K-I-4B` (`empero-ai/Qwen3.8-4B-Distill`).
2. C (configured follow-up): `configs/train/c-i/` for `ZIPANGU-C-I-9B` (`empero-ai/Qwen3.8-9B-Distill`); use only after K validation and explicit human GO.
3. Z (future-only placeholder): `configs/train/z-i/` for `ZIPANGU-Z-I-26B-A4B` (`google/gemma-4-26B-A4B-it`); no run authorized.
4. Legacy: root-level `configs/train/*.yaml` (for example `smoke.yaml`, `pilot-1m.yaml`) remain C-I compatible and are not the K default.

Safety boundary: every config keeps `safety.training_allowed: false`. Paid RunPod creation, GPU/tier escalation, and next token-scale tier all require explicit human GO. This matches `PROJECT_DECISIONS.json` and `configs/models/registry.yaml`.
