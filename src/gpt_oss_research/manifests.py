from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import load_json, load_yaml

REQUIRED_HOLDOUTS = {
    "livecodebench",
    "apps_holdout",
    "taco_holdout",
    "codecontests_holdout",
    "codeforces_holdout",
    "ml_research_eval_holdout",
}


@dataclass(slots=True)
class ValidationResult:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_manifest_spec(path: str | Path, frozen: bool = False) -> dict[str, Any]:
    suffix = Path(path).suffix.lower()
    if frozen or suffix == ".json":
        return load_json(path)
    return load_yaml(path)


def validate_manifest_spec(spec: dict[str, Any], *, frozen: bool = False) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    required_top_level = [
        "manifest_name",
        "version",
        "task_type",
        "description",
        "base_train_target",
        "teacher_model",
        "benchmark_holdouts",
        "split_strategy",
        "buckets",
    ]
    for key in required_top_level:
        if key not in spec:
            errors.append(f"missing required top-level key: {key}")

    if errors:
        return ValidationResult(errors=errors, warnings=warnings)

    task_type = spec["task_type"]
    if task_type not in {"sft", "grpo"}:
        errors.append(f"task_type must be 'sft' or 'grpo', got {task_type!r}")

    buckets = spec["buckets"]
    if not isinstance(buckets, list) or not buckets:
        errors.append("buckets must be a non-empty list")
        return ValidationResult(errors=errors, warnings=warnings)

    share_total = 0.0
    bucket_names: set[str] = set()
    for index, bucket in enumerate(buckets):
        if not isinstance(bucket, dict):
            errors.append(f"bucket {index} must be an object")
            continue
        for key in ["name", "share", "category", "purpose", "sources"]:
            if key not in bucket:
                errors.append(f"bucket {index} missing key: {key}")
        if "name" in bucket:
            if bucket["name"] in bucket_names:
                errors.append(f"duplicate bucket name: {bucket['name']}")
            bucket_names.add(bucket["name"])
        share = bucket.get("share", 0)
        if not isinstance(share, (int, float)) or share <= 0:
            errors.append(f"bucket {bucket.get('name', index)} has invalid share: {share!r}")
        else:
            share_total += float(share)
        sources = bucket.get("sources", [])
        if not isinstance(sources, list) or not sources:
            errors.append(f"bucket {bucket.get('name', index)} must declare at least one source")
        if task_type == "grpo" and not bucket.get("verifiable", False):
            errors.append(f"GRPO bucket {bucket.get('name', index)} must be marked verifiable")
        if bucket.get("category") == "synthetic_reasoning" and float(share) > 0.05:
            errors.append("synthetic_reasoning share exceeds the 5% cap encoded from AGENTS.md")
        if bucket.get("category") == "synthetic_reasoning" and not bucket.get("capped", False):
            warnings.append(
                f"bucket {bucket.get('name', index)} is synthetic reasoning and should record a cap rationale"
            )

    if abs(share_total - 1.0) > 1e-6:
        errors.append(f"bucket shares must sum to 1.0, got {share_total:.6f}")

    holdouts = set(spec.get("benchmark_holdouts", []))
    missing_holdouts = sorted(REQUIRED_HOLDOUTS - holdouts)
    if missing_holdouts:
        errors.append(f"benchmark_holdouts missing required entries: {', '.join(missing_holdouts)}")

    split_strategy = spec.get("split_strategy", {})
    if not split_strategy.get("group_by"):
        errors.append("split_strategy.group_by must be set")
    if split_strategy.get("group_by") != "repository_or_task_source":
        warnings.append("split_strategy.group_by is expected to be 'repository_or_task_source'")
    if not split_strategy.get("prevent_eval_contamination", False):
        errors.append("split_strategy.prevent_eval_contamination must be true")

    if spec.get("base_train_target") != "openai/gpt-oss-20b":
        errors.append("base_train_target must remain openai/gpt-oss-20b for the initial project milestone")

    teacher_model = spec.get("teacher_model")
    if "120b" not in str(teacher_model):
        warnings.append("teacher_model does not appear to reference gpt-oss-120b")

    if frozen and "frozen_at_utc" not in spec:
        errors.append("frozen manifests must include frozen_at_utc")

    return ValidationResult(errors=errors, warnings=warnings)


def build_frozen_manifest(spec: dict[str, Any], *, source_path: str | Path | None = None) -> dict[str, Any]:
    validation = validate_manifest_spec(spec)
    if not validation.ok:
        joined = "\n".join(validation.errors)
        raise ValueError(f"manifest spec failed validation:\n{joined}")

    frozen_manifest = dict(spec)
    frozen_manifest["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    if source_path is not None:
        frozen_manifest["source_spec_path"] = str(source_path)

    normalized_buckets = []
    for bucket in spec["buckets"]:
        normalized_bucket = dict(bucket)
        normalized_bucket["share_pct"] = round(float(bucket["share"]) * 100, 2)
        normalized_bucket["source_count"] = len(bucket["sources"])
        normalized_buckets.append(normalized_bucket)
    frozen_manifest["buckets"] = normalized_buckets
    return frozen_manifest

