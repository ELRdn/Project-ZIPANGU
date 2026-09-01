from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from zipangu.budget import check_budget


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hourly-usd", type=float, required=True)
    parser.add_argument("--fx-jpy-per-usd", type=float, required=True)
    parser.add_argument("--planned-hours", type=float, required=True)
    parser.add_argument("--budget-config", default="configs/budget.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.budget_config).read_text(encoding="utf-8"))
    result = check_budget(
        args.hourly_usd,
        args.fx_jpy_per_usd,
        args.planned_hours,
        float(config["pilot"]["hard_cap_jpy"]),
    )
    print(f"Projected GPU cost: ¥{result.projected_jpy:,.0f}")
    print(f"Pilot hard cap:      ¥{result.hard_cap_jpy:,.0f}")
    if not result.allowed:
        print("BLOCKED: projected cost exceeds hard cap.")
        return 2
    print("PASS: within budget. Pod creation still requires human GO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
