from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import PROJECT_ROOT
from zipangu.registry import (
    ValidationIssue,
    load_yaml,
    split_issues,
    validate_eval_config,
    validate_model_registry,
    validate_recipe_against_registry,
    validate_registry,
    validate_train_config,
)


def _resolve_path(raw_path: str, root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_config(path: Path) -> tuple[dict[str, Any] | None, list[ValidationIssue]]:
    try:
        return load_yaml(path), []
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return None, [
            ValidationIssue(
                code="config_unreadable",
                message=str(exc),
                path=str(path),
                level="error",
            )
        ]


def _paths_or_defaults(
    raw_paths: list[str] | None,
    *,
    root: Path,
    pattern: str | tuple[str, ...],
) -> list[Path]:
    if raw_paths:
        return [_resolve_path(raw_path, root) for raw_path in raw_paths]
    patterns = (pattern,) if isinstance(pattern, str) else pattern
    return sorted({path for item in patterns for path in root.glob(item)})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate ZIPANGU registry and experiment configs.")
    parser.add_argument(
        "--mode",
        choices=("structure", "runnable", "all"),
        default="all",
        help="structure checks only, runnable checks, or both (default: all)",
    )
    parser.add_argument("--registry", default="configs/datasets/registry.yaml")
    parser.add_argument("--recipe", dest="recipes", action="append")
    parser.add_argument("--train-config", dest="train_configs", action="append")
    parser.add_argument("--eval-config", dest="eval_configs", action="append")
    return parser


def _deduplicate(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[ValidationIssue] = []
    for issue in issues:
        key = (issue.level, issue.code, issue.path, issue.message)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def main() -> int:
    args = _parser().parse_args()
    root = PROJECT_ROOT
    issues: list[ValidationIssue] = []
    registry_path = _resolve_path(args.registry, root)
    registry, load_issues = _load_config(registry_path)
    issues.extend(load_issues)
    if registry is None:
        for issue in issues:
            print(issue.render())
        print("STRUCTURE: ERROR")
        return 1

    issues.extend(validate_registry(registry))
    for message in validate_model_registry(root):
        issues.append(
            ValidationIssue(
                code="model_registry_issue",
                message=message,
                path="configs/models/registry.yaml",
                level="error",
            )
        )

    recipe_paths = _paths_or_defaults(
        args.recipes,
        root=root,
        pattern=("configs/datasets/i-jp-*.yaml", "configs/datasets/i-balanced.yaml", "configs/datasets/c-i-*.yaml"),
    )
    train_paths = _paths_or_defaults(
        args.train_configs,
        root=root,
        pattern="configs/train/**/*.yaml",
    )
    eval_paths = _paths_or_defaults(
        args.eval_configs,
        root=root,
        pattern="configs/eval/*.yaml",
    )
    for label, paths in (
        ("recipe", recipe_paths),
        ("train config", train_paths),
        ("eval config", eval_paths),
    ):
        if not paths:
            issues.append(
                ValidationIssue(
                    code=f"{label.replace(' ', '_')}_missing",
                    message=f"no {label} files were found",
                    path=str(root),
                    level="error",
                )
            )

    for path in recipe_paths:
        config, config_issues = _load_config(path)
        issues.extend(
            ValidationIssue(
                code=issue.code,
                message=issue.message,
                path=_display_path(path, root),
                level=issue.level,
            )
            for issue in config_issues
        )
        if config is not None:
            issues.extend(
                validate_recipe_against_registry(
                    config,
                    registry,
                    repo_root=root,
                )
            )

    for path in train_paths:
        config, config_issues = _load_config(path)
        issues.extend(
            ValidationIssue(
                code=issue.code,
                message=issue.message,
                path=_display_path(path, root),
                level=issue.level,
            )
            for issue in config_issues
        )
        if config is not None:
            issues.extend(
                validate_train_config(
                    config,
                    registry,
                    repo_root=root,
                    config_path=_display_path(path, root),
                )
            )

    for path in eval_paths:
        config, config_issues = _load_config(path)
        issues.extend(
            ValidationIssue(
                code=issue.code,
                message=issue.message,
                path=_display_path(path, root),
                level=issue.level,
            )
            for issue in config_issues
        )
        if config is not None:
            issues.extend(
                validate_eval_config(
                    config,
                    registry,
                    config_path=_display_path(path, root),
                )
            )

    issues = _deduplicate(issues)
    errors, blocked = split_issues(issues)
    print(f"STRUCTURE: {'ERROR' if errors else 'PASS'}")
    for issue in errors:
        print(issue.render())

    if args.mode == "structure":
        for issue in blocked:
            print(f"NOTICE [{issue.code}] {issue.path}: {issue.message}")
        return 1 if errors else 0

    if errors:
        print("RUNNABLE: BLOCKED")
        for issue in blocked:
            print(issue.render())
        return 1
    if blocked:
        print("RUNNABLE: BLOCKED")
        for issue in blocked:
            print(issue.render())
        return 2
    print("RUNNABLE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
