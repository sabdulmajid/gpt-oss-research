from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..io import load_yaml
from ..manifests import load_manifest_spec, validate_manifest_spec


@dataclass(slots=True)
class TrainingPlan:
    config_path: str
    experiment_name: str
    model_name_or_path: str
    teacher_model: str
    manifest_path: str
    target_parameters: list[str]
    warnings: list[str]
    details: dict[str, Any]


def load_training_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def validate_training_manifest(path: str | Path) -> None:
    manifest = load_manifest_spec(path, frozen=Path(path).suffix == ".json")
    validation = validate_manifest_spec(manifest, frozen=Path(path).suffix == ".json")
    if not validation.ok:
        raise ValueError("\n".join(validation.errors))


def build_target_parameters(expert_targeting: dict[str, Any]) -> list[str]:
    if not expert_targeting.get("enabled", False):
        return []
    layer_indices = expert_targeting.get("layer_indices", [])
    templates = expert_targeting.get("parameter_templates", [])
    targets = []
    for layer_index in layer_indices:
        for template in templates:
            targets.append(template.format(layer=layer_index))
    return targets


def materialized_dataset_summary(dataset_path: str | Path | None) -> dict[str, Any]:
    if not dataset_path:
        return {"present": False, "row_count": 0}
    path = Path(dataset_path)
    if not path.exists():
        return {"present": False, "row_count": 0}
    with path.open("r", encoding="utf-8") as handle:
        row_count = sum(1 for line in handle if line.strip())
    return {"present": True, "row_count": row_count, "path": str(path)}

