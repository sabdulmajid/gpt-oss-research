from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

TORCH_SIGNAL_TERMS = (
    "import torch",
    "from torch",
    "torch.nn",
    "torch.utils.data",
    "torch.distributed",
    "torch.cuda",
    "torch.autograd",
    "torch.compile",
    "torchvision",
    "torchaudio",
    "transformers",
    "accelerate",
    "datasets",
    "deepspeed",
    "lightning",
    "pytorch_lightning",
    "xformers",
    "flash_attn",
    "triton",
)

PATH_SIGNAL_TERMS = (
    "train.py",
    "trainer.py",
    "finetune.py",
    "dataset.py",
    "dataloader.py",
    "modeling_",
    "fsdp",
    "ddp",
    "amp",
    "examples/",
    "tutorials/",
    "benchmarks/",
)

REJECT_PATH_TERMS = (
    "site-packages/",
    "dist-packages/",
    "node_modules/",
    ".venv/",
    "__pycache__/",
    "vendor/",
    "third_party/",
    ".ipynb_checkpoints/",
)


def _repo_key(record: dict[str, Any]) -> str:
    return str(record.get("repo_name") or record.get("repo") or record.get("repository") or "unknown-repo")


def _path_key(record: dict[str, Any]) -> str:
    return str(record.get("path") or record.get("filepath") or record.get("file_path") or "")


def _content_key(record: dict[str, Any]) -> str:
    return str(record.get("content") or record.get("code") or "")


def score_record(record: dict[str, Any]) -> int:
    path = _path_key(record).lower()
    content = _content_key(record)
    repo_name = _repo_key(record).lower()
    score = 0

    for term in TORCH_SIGNAL_TERMS:
        if term in content:
            score += 1
    for term in PATH_SIGNAL_TERMS:
        if term in path or term in repo_name:
            score += 1
    return score


def reject_reason(record: dict[str, Any], *, require_python: bool = True, syntax_check: bool = True) -> str | None:
    path = _path_key(record).lower()
    language = str(record.get("language", "")).lower()
    content = _content_key(record)
    if require_python and language not in {"python", "py"}:
        return "non_python_language"
    if any(term in path for term in REJECT_PATH_TERMS):
        return "rejected_path_pattern"
    if path.endswith(".ipynb"):
        return "notebook_source"
    if "generated" in path or "minified" in path:
        return "generated_or_minified"
    if score_record(record) == 0:
        return "missing_ml_signals"
    if syntax_check:
        try:
            ast.parse(content)
        except SyntaxError:
            return "syntax_error"
    return None


def filter_records(
    records: list[dict[str, Any]],
    *,
    require_python: bool = True,
    syntax_check: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    seen_hashes: set[str] = set()

    for record in records:
        reason = reject_reason(record, require_python=require_python, syntax_check=syntax_check)
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue

        content_hash = hashlib.sha256(_content_key(record).encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            rejected["exact_duplicate"] = rejected.get("exact_duplicate", 0) + 1
            continue
        seen_hashes.add(content_hash)

        annotated = dict(record)
        annotated["signal_score"] = score_record(record)
        annotated["content_sha256"] = content_hash
        annotated["repository_identity"] = _repo_key(record)
        kept.append(annotated)

    summary = {
        "input_records": len(records),
        "kept_records": len(kept),
        "rejected_records": len(records) - len(kept),
        "reject_reasons": rejected,
    }
    return kept, summary


def stable_split(
    records: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.1,
    seed: str = "gpt-oss-research-v1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    assignments: dict[str, str] = {}

    for record in records:
        repo_key = str(record.get("repository_identity") or _repo_key(record))
        if repo_key not in assignments:
            digest = hashlib.sha256(f"{seed}:{repo_key}".encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) / 0xFFFFFFFF
            assignments[repo_key] = "validation" if bucket < validation_fraction else "train"
        if assignments[repo_key] == "validation":
            validation.append(record)
        else:
            train.append(record)
    return train, validation


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

