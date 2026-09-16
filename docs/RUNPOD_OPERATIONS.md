# RunPod Operations

## Class-aware defaults

```text
K: RX 7600 local first; RTX 4090 Community fallback
C: RTX 4090 Community
Z: decide separately later
Pilot hard cap: ¥3,000
```

Local RX 7600 work is research/CI validation, not production training. A local failure may conclude `LOCAL_TRAINING_NOT_PRACTICAL`; it is not a reason to force unsafe settings or silently spend more compute.

Always resolve live Community pricing immediately before provisioning. Confirm live/manual USD→JPY FX and planned runtime as well.

## MCP

Preferred hosted endpoint:

```text
https://mcp.getrunpod.io/
```

Local Codex fallback:

```bash
codex mcp add runpod --env RUNPOD_API_KEY=YOUR_KEY -- npx -y @runpod/mcp-server@latest
```

Docs MCP:

```bash
codex mcp add runpod-docs --url https://docs.runpod.io/mcp
```

Never commit API keys or ask for secrets in a report.

## Approval boundary

The agent may inspect availability, prices and resources without GO. Human GO is required for:

- a new paid Pod;
- any GPU-hour extension;
- Community → Secure or a more expensive GPU/tier;
- the next K token tier;
- any C or Z training;
- destructive deletion of Pods, volumes, datasets or checkpoints.

After GO for one specific run, the approved config may be bootstrapped, trained, evaluated, persisted and stopped. No adjacent scale-up is implied by that approval.

## Budget preflight

```text
live hourly USD × approved hours × live/manual USD→JPY <= remaining JPY budget
```

Run:

```bash
python scripts/preflight.py --hourly-usd <LIVE> --fx-jpy-per-usd <LIVE> --planned-hours <HOURS>
```

Do not provision when projected cost exceeds the remaining ¥3,000 cap. If no acceptable 4090 Community offer exists, show alternatives and wait for a new human decision.

## Shutdown and evidence

```text
persist logs/manifest/checkpoint if safe
→ record status and actual/estimated cost
→ stop Pod
→ verify stopped
```

For each run, preserve model ID/base revision, dataset/provenance manifest, train/eval config, environment/GPU manifest, raw outputs and the stop verification. A successful push or upload is not authorization to publish adapters or weights; publication remains a separate human gate.

## Local safety

ROCm 10 is an isolated experimental track. Do not perform global pip uninstall, global torch replacement, system ROCm overwrite or working-venv destruction. Use separate venv/WSL/build/container state where possible. `TRAINING_ALLOWED=false` remains active until the relevant human approvals and provenance gates are explicitly satisfied.
