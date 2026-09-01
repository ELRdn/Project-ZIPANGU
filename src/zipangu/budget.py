from dataclasses import dataclass

@dataclass(frozen=True)
class BudgetResult:
    projected_jpy: float
    hard_cap_jpy: float
    allowed: bool

def check_budget(hourly_usd: float, fx_jpy_per_usd: float, planned_hours: float, hard_cap_jpy: float) -> BudgetResult:
    for name, value in {"hourly_usd":hourly_usd,"fx_jpy_per_usd":fx_jpy_per_usd,"planned_hours":planned_hours,"hard_cap_jpy":hard_cap_jpy}.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
    projected = hourly_usd * fx_jpy_per_usd * planned_hours
    return BudgetResult(projected, hard_cap_jpy, projected <= hard_cap_jpy)
