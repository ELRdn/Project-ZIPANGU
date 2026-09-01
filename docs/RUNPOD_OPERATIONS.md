# RunPod Operations

## Defaults

```text
GPU: RTX 4090
Cloud: Community Cloud
Pilot hard cap: ¥3,000
```

Always resolve live Community pricing immediately before provisioning.

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

Never commit API keys.

## Approval boundary

Agent may inspect availability/prices/resources without GO.

Human GO required for:

- new paid Pod;
- GPU-hour extension;
- Community → Secure;
- more expensive GPU;
- next scale tier;
- destructive deletion.

After GO for one specific run, Codex may provision that approved config, bootstrap, train, evaluate, sync artifacts and stop it.

## Budget preflight

```text
live hourly USD × approved hours × live/manual USD→JPY <= remaining JPY budget
```

Run:

```bash
python scripts/preflight.py --hourly-usd <LIVE> --fx-jpy-per-usd <LIVE> --planned-hours <HOURS>
```

## Failure/success shutdown

```text
persist logs/manifest/checkpoint if safe
→ stop Pod
→ verify stopped
→ record actual/estimated cost
```

If no acceptable 4090 Community offer exists, do not silently upgrade. Show alternatives and wait for human GO.
