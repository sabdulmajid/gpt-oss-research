from __future__ import annotations

import ast
import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.errors import GatedRepoError

from .execution import normalize_output
from .filtering import filter_records
from .internal_eval import discover_tasks
from .io import load_json, load_yaml, write_json, write_jsonl
from .manifests import load_manifest_spec, validate_manifest_spec

SYSTEM_PROMPT = "You are a precise ML research coding assistant."
DEFAULT_HF_CACHE = "/pub7/neel2/hf-cache/hub"


@dataclass(slots=True)
class MaterializationConfig:
    manifest_path: str
    task_type: str
    total_examples: int
    output_dir: str
    max_files_per_source: int = 1
    pyarrow_batch_size: int = 64
    validation_fraction: float = 0.05
    hf_cache_dir: str = DEFAULT_HF_CACHE


def load_materialization_config(path: str | Path) -> MaterializationConfig:
    data = load_yaml(path)
    return MaterializationConfig(
        manifest_path=data["manifest_path"],
        task_type=data["task_type"],
        total_examples=int(data["total_examples"]),
        output_dir=data["output_dir"],
        max_files_per_source=int(data.get("max_files_per_source", 1)),
        pyarrow_batch_size=int(data.get("pyarrow_batch_size", 64)),
        validation_fraction=float(data.get("validation_fraction", 0.05)),
        hf_cache_dir=data.get("hf_cache_dir", DEFAULT_HF_CACHE),
    )


def _build_messages(user: str, assistant: str, *, thinking: str | None = None) -> list[dict[str, str]]:
    assistant_message: dict[str, str] = {"role": "assistant", "content": assistant}
    if thinking:
        assistant_message["thinking"] = thinking
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        assistant_message,
    ]


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _choose_python_solution(candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        stripped = candidate.strip()
        if not stripped:
            continue
        if _is_valid_python(stripped):
            return stripped
    return None


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _examples_to_cases(examples: Any) -> list[dict[str, str]]:
    if isinstance(examples, dict):
        inputs = list(examples.get("inputs", examples.get("input", [])))
        outputs = list(examples.get("outputs", examples.get("output", [])))
        return [{"input": inp, "output": out} for inp, out in zip(inputs, outputs)]
    if isinstance(examples, list):
        return [{"input": item["input"], "output": item["output"]} for item in examples if isinstance(item, dict)]
    return []


def _build_problem_prompt(title: str, description: str, input_format: str | None = None, output_format: str | None = None, starter_code: str | None = None) -> str:
    parts = []
    if title:
        parts.append(f"Problem: {title}")
    parts.append(description.strip())
    if input_format:
        parts.append(f"Input format:\n{input_format.strip()}")
    if output_format:
        parts.append(f"Output format:\n{output_format.strip()}")
    if starter_code:
        parts.append(f"Starter code:\n{starter_code.rstrip()}")
    parts.append("Return only Python code.")
    return "\n\n".join(part for part in parts if part)


def _wrap_code_file_prompt(repo_name: str, path: str) -> str:
    return (
        f"Write the Python source file `{path}` for repository `{repo_name}`. "
        "Return only code."
    )


def _adapt_apps_sft(row: dict[str, Any], source_name: str, bucket_name: str) -> dict[str, Any] | None:
    solutions = _safe_json_loads(row.get("solutions")) or []
    solution = _choose_python_solution(solutions)
    if not solution:
        return None
    prompt = _build_problem_prompt("", row["question"], starter_code=row.get("starter_code"))
    return {
        "id": f"{source_name}:{row.get('id')}",
        "source_name": source_name,
        "bucket_name": bucket_name,
        "repository_identity": f"{source_name}:{row.get('id')}",
        "messages": _build_messages(prompt, solution),
        "metadata": {"difficulty": row.get("difficulty"), "url": row.get("url")},
    }


def _adapt_taco_sft(row: dict[str, Any], source_name: str, bucket_name: str) -> dict[str, Any] | None:
    solutions = _safe_json_loads(row.get("solutions")) or []
    solution = _choose_python_solution(solutions)
    if not solution:
        return None
    prompt = _build_problem_prompt(row.get("name", ""), row["question"], starter_code=row.get("starter_code"))
    return {
        "id": f"{source_name}:{row.get('url') or row.get('name')}",
        "source_name": source_name,
        "bucket_name": bucket_name,
        "repository_identity": str(row.get("url") or row.get("name")),
        "messages": _build_messages(prompt, solution),
        "metadata": {"difficulty": row.get("difficulty"), "source": row.get("source")},
    }


def _adapt_codecontests_sft(row: dict[str, Any], source_name: str, bucket_name: str) -> dict[str, Any] | None:
    solutions = row.get("solutions", {})
    candidate_solutions = list((solutions or {}).get("solution", []))
    solution = _choose_python_solution(candidate_solutions)
    if not solution:
        return None
    prompt = _build_problem_prompt(row.get("name", ""), row["description"])
    return {
        "id": f"{source_name}:{row.get('name')}",
        "source_name": source_name,
        "bucket_name": bucket_name,
        "repository_identity": f"{source_name}:{row.get('name')}",
        "messages": _build_messages(prompt, solution),
        "metadata": {"difficulty": row.get("difficulty"), "cf_rating": row.get("cf_rating")},
    }


def _adapt_stackoverflow_sft(row: dict[str, Any], source_name: str, bucket_name: str) -> dict[str, Any]:
    user_parts = [row.get("instruction", "").strip(), row.get("input", "").strip()]
    prompt = "\n\n".join(part for part in user_parts if part)
    return {
        "id": f"{source_name}:{row.get('source')}",
        "source_name": source_name,
        "bucket_name": bucket_name,
        "repository_identity": str(row.get("source")),
        "messages": _build_messages(prompt, row["output"]),
        "metadata": {"url": row.get("source")},
    }


def _adapt_pytorch_discuss_sft(row: dict[str, Any], source_name: str, bucket_name: str) -> dict[str, Any]:
    prompt = "\n\n".join(part for part in [row.get("question", "").strip(), row.get("context", "").strip()] if part)
    return {
        "id": f"{source_name}:{row.get('source')}:{hash(row.get('question', ''))}",
        "source_name": source_name,
        "bucket_name": bucket_name,
        "repository_identity": str(row.get("source")),
        "messages": _build_messages(prompt, row["answer"]),
        "metadata": {"source": row.get("source")},
    }


def _adapt_github_code_sft(row: dict[str, Any], source_name: str, bucket_name: str, *, require_transformers_repo: bool = False) -> dict[str, Any] | None:
    filter_input = [
        {
            "repo_name": row.get("repo_name"),
            "path": row.get("path"),
            "language": "Python" if str(row.get("path", "")).endswith(".py") else "Unknown",
            "content": row.get("content"),
        }
    ]
    kept, _ = filter_records(filter_input)
    if not kept:
        return None
    repo_name = str(row.get("repo_name"))
    if require_transformers_repo and not any(term in repo_name.lower() for term in ("transformers", "huggingface", "accelerate", "trl")):
        return None
    content = str(row.get("content", ""))
    if not _is_valid_python(content):
        return None
    prompt = _wrap_code_file_prompt(repo_name, str(row.get("path")))
    return {
        "id": f"{source_name}:{repo_name}:{row.get('path')}",
        "source_name": source_name,
        "bucket_name": bucket_name,
        "repository_identity": repo_name,
        "messages": _build_messages(prompt, content),
        "metadata": {"path": row.get("path"), "license": row.get("license")},
    }


def _adapt_transformers_code_sft(row: dict[str, Any], source_name: str, bucket_name: str) -> dict[str, Any] | None:
    metadata = row.get("metadata") or {}
    code = str(row.get("text", ""))
    if not _is_valid_python(code):
        return None
    prompt = _wrap_code_file_prompt(str(metadata.get("repo_id")), str(metadata.get("file_path")))
    return {
        "id": f"{source_name}:{metadata.get('repo_id')}:{metadata.get('file_path')}",
        "source_name": source_name,
        "bucket_name": bucket_name,
        "repository_identity": str(metadata.get("repo_id")),
        "messages": _build_messages(prompt, code),
        "metadata": {"path": metadata.get("file_path"), "token_count": metadata.get("token_count")},
    }


def _adapt_open_code_reasoning_sft(row: dict[str, Any], source_name: str, bucket_name: str) -> dict[str, Any] | None:
    solution = str(row.get("solution") or "").strip()
    output = str(row.get("output") or "").strip()
    if not solution:
        solution = output
        output = None
    if not solution:
        return None
    return {
        "id": f"{source_name}:{row.get('id')}",
        "source_name": source_name,
        "bucket_name": bucket_name,
        "repository_identity": f"{source_name}:{row.get('id')}",
        "messages": _build_messages(str(row.get("input", "")).strip(), solution, thinking=output),
        "metadata": {"difficulty": row.get("difficulty"), "dataset": row.get("dataset")},
    }


def _adapt_apps_grpo(row: dict[str, Any], source_name: str, bucket_name: str) -> dict[str, Any] | None:
    test_blob = _safe_json_loads(row.get("input_output"))
    cases = _examples_to_cases(test_blob)
    if not cases:
        return None
    return {
        "task_id": f"{source_name}:{row.get('id')}",
        "source_name": source_name,
        "bucket_name": bucket_name,
        "prompt": _build_problem_prompt("", row["question"], starter_code=row.get("starter_code")),
        "checker_type": "python_io_tests",
        "tests": {"cases": cases[:8], "timeout_sec": 2.0},
        "metadata": {"difficulty": row.get("difficulty"), "url": row.get("url")},
    }


def _adapt_taco_grpo(row: dict[str, Any], source_name: str, bucket_name: str) -> dict[str, Any] | None:
    test_blob = _safe_json_loads(row.get("input_output"))
    cases = _examples_to_cases(test_blob)
    if not cases:
        return None
    return {
        "task_id": f"{source_name}:{row.get('url') or row.get('name')}",
        "source_name": source_name,
        "bucket_name": bucket_name,
        "prompt": _build_problem_prompt(row.get("name", ""), row["question"], starter_code=row.get("starter_code")),
        "checker_type": "python_io_tests",
        "tests": {"cases": cases[:8], "timeout_sec": 2.0},
        "metadata": {"difficulty": row.get("difficulty"), "source": row.get("source")},
    }


def _adapt_codecontests_grpo(row: dict[str, Any], source_name: str, bucket_name: str) -> dict[str, Any] | None:
    public_cases = _examples_to_cases(row.get("public_tests"))
    generated_cases = _examples_to_cases(row.get("generated_tests"))
    cases = public_cases + generated_cases[:6]
    if not cases:
        return None
    return {
        "task_id": f"{source_name}:{row.get('name')}",
        "source_name": source_name,
        "bucket_name": bucket_name,
        "prompt": _build_problem_prompt(row.get("name", ""), row["description"]),
        "checker_type": "python_io_tests",
        "tests": {"cases": cases, "timeout_sec": 2.0},
        "metadata": {"difficulty": row.get("difficulty"), "cf_rating": row.get("cf_rating")},
    }


def _adapt_codeforces_grpo(row: dict[str, Any], source_name: str, bucket_name: str) -> dict[str, Any] | None:
    cases = _examples_to_cases(row.get("official_tests"))
    if not cases:
        return None
    return {
        "task_id": f"{source_name}:{row.get('id')}",
        "source_name": source_name,
        "bucket_name": bucket_name,
        "prompt": _build_problem_prompt(
            row.get("title", ""),
            row.get("description", ""),
            input_format=row.get("input_format"),
            output_format=row.get("output_format"),
        ),
        "checker_type": "python_io_tests",
        "tests": {"cases": cases[:8], "timeout_sec": 2.0},
        "metadata": {"rating": row.get("rating"), "contest": row.get("contest_name")},
    }


def _load_internal_eval_examples(bucket_name: str) -> list[dict[str, Any]]:
    tasks = discover_tasks(Path.cwd() / "eval" / "ml_research_eval" / "tasks")
    rows = []
    for task in tasks:
        prompt = task.prompt_path.read_text(encoding="utf-8")
        rows.append(
            {
                "task_id": f"internal_eval:{task.task_id}",
                "source_name": "internal/ml-research-eval-train",
                "bucket_name": bucket_name,
                "prompt": prompt,
                "checker_type": "internal_eval_task",
                "tests": {"task_id": task.task_id},
                "metadata": {"topic": task.topic, "difficulty": task.difficulty},
            }
        )
    return rows


def _resolve_source_files(source_name: str, *, max_files: int, cache_dir: str) -> list[tuple[str, str]]:
    if source_name == "codeparrot/apps":
        return [(source_name, "train.jsonl")]
    if source_name == "BAAI/TACO":
        files = sorted(f for f in list_repo_files(source_name, repo_type="dataset") if fnmatch.fnmatch(f, "ALL/train-*.parquet"))
        return [(source_name, file_name) for file_name in files[:max_files]]
    if source_name == "deepmind/code_contests":
        files = sorted(f for f in list_repo_files(source_name, repo_type="dataset") if fnmatch.fnmatch(f, "data/train-*.parquet"))
        return [(source_name, file_name) for file_name in files[:max_files]]
    if source_name == "open-r1/codeforces":
        files = sorted(f for f in list_repo_files(source_name, repo_type="dataset") if fnmatch.fnmatch(f, "data/train-*.parquet"))
        return [(source_name, file_name) for file_name in files[:max_files]]
    if source_name == "shrinath-suresh/stack_overflow_pytorch":
        files = sorted(f for f in list_repo_files(source_name, repo_type="dataset") if f.endswith(".jsonl"))
        return [(source_name, files[0])]
    if source_name == "shrinath-suresh/pytorch-discuss-tutorial-1000":
        files = sorted(f for f in list_repo_files(source_name, repo_type="dataset") if f.endswith(".parquet"))
        return [(source_name, files[0])]
    if source_name == "BigTimeCoderSean/pytorch-issues":
        files = sorted(f for f in list_repo_files(source_name, repo_type="dataset") if f.endswith(".parquet"))
        return [(source_name, files[0])]
    if source_name == "suvadityamuk/huggingface-transformers-code-dataset":
        files = sorted(f for f in list_repo_files(source_name, repo_type="dataset") if f.endswith(".parquet"))
        return [(source_name, files[0])]
    if source_name == "nvidia/OpenCodeReasoning":
        files = sorted(f for f in list_repo_files(source_name, repo_type="dataset") if fnmatch.fnmatch(f, "split_0/train-*.parquet"))
        return [(source_name, file_name) for file_name in files[:max_files]]
    if source_name == "codeparrot/github-code":
        files = sorted(f for f in list_repo_files(source_name, repo_type="dataset") if fnmatch.fnmatch(f, "data/train-*.parquet"))
        return [(source_name, file_name) for file_name in files[:max_files]]
    if source_name == "filtered_transformers_repos":
        files = sorted(f for f in list_repo_files("codeparrot/github-code", repo_type="dataset") if fnmatch.fnmatch(f, "data/train-*.parquet"))
        return [("codeparrot/github-code", file_name) for file_name in files[:max_files]]
    if source_name == "bigcode/the-stack-v2-dedup":
        raise GatedRepoError("bigcode/the-stack-v2-dedup is gated in this environment")
    if source_name == "internal/ml-research-eval-train":
        return []
    raise KeyError(f"unsupported source: {source_name}")


def _iter_file_rows(repo_id: str, file_name: str, *, cache_dir: str, batch_size: int) -> Iterable[dict[str, Any]]:
    local_path = hf_hub_download(repo_id, file_name, repo_type="dataset", cache_dir=cache_dir)
    path = Path(local_path)
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if path.suffix == ".parquet":
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield row
        return
    raise ValueError(f"unsupported file format: {path}")


def _materialize_sft_source(source_name: str, bucket_name: str, target_count: int, config: MaterializationConfig, seen_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    handlers: dict[str, Callable[[dict[str, Any], str, str], dict[str, Any] | None]] = {
        "codeparrot/apps": _adapt_apps_sft,
        "BAAI/TACO": _adapt_taco_sft,
        "deepmind/code_contests": _adapt_codecontests_sft,
        "shrinath-suresh/stack_overflow_pytorch": _adapt_stackoverflow_sft,
        "shrinath-suresh/pytorch-discuss-tutorial-1000": _adapt_pytorch_discuss_sft,
        "suvadityamuk/huggingface-transformers-code-dataset": _adapt_transformers_code_sft,
        "nvidia/OpenCodeReasoning": _adapt_open_code_reasoning_sft,
    }

    examples: list[dict[str, Any]] = []
    skipped_reason = None
    attempted = 0

    if source_name in {"BigTimeCoderSean/pytorch-issues", "open-r1/codeforces"}:
        skipped_reason = "source does not provide reference solutions for direct SFT materialization"
        return examples, {"target_count": target_count, "materialized_count": 0, "skipped_reason": skipped_reason}
    if source_name == "bigcode/the-stack-v2-dedup":
        skipped_reason = "bigcode/the-stack-v2-dedup is gated in this environment"
        return examples, {"target_count": target_count, "materialized_count": 0, "skipped_reason": skipped_reason}

    if source_name in {"codeparrot/github-code", "filtered_transformers_repos"}:
        require_transformers = source_name == "filtered_transformers_repos"
        for repo_id, file_name in _resolve_source_files(source_name, max_files=config.max_files_per_source, cache_dir=config.hf_cache_dir):
            for row in _iter_file_rows(repo_id, file_name, cache_dir=config.hf_cache_dir, batch_size=config.pyarrow_batch_size):
                attempted += 1
                example = _adapt_github_code_sft(row, source_name, bucket_name, require_transformers_repo=require_transformers)
                if not example or example["id"] in seen_ids:
                    continue
                seen_ids.add(example["id"])
                examples.append(example)
                if len(examples) >= target_count:
                    return examples, {"target_count": target_count, "materialized_count": len(examples), "rows_scanned": attempted}
        return examples, {"target_count": target_count, "materialized_count": len(examples), "rows_scanned": attempted}

    handler = handlers.get(source_name)
    if handler is None:
        return examples, {"target_count": target_count, "materialized_count": 0, "skipped_reason": "no handler"}

    for repo_id, file_name in _resolve_source_files(source_name, max_files=config.max_files_per_source, cache_dir=config.hf_cache_dir):
        for row in _iter_file_rows(repo_id, file_name, cache_dir=config.hf_cache_dir, batch_size=config.pyarrow_batch_size):
            attempted += 1
            example = handler(row, source_name, bucket_name)
            if not example or example["id"] in seen_ids:
                continue
            seen_ids.add(example["id"])
            examples.append(example)
            if len(examples) >= target_count:
                return examples, {"target_count": target_count, "materialized_count": len(examples), "rows_scanned": attempted}
    return examples, {"target_count": target_count, "materialized_count": len(examples), "rows_scanned": attempted}


def _materialize_grpo_source(source_name: str, bucket_name: str, target_count: int, config: MaterializationConfig, seen_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    handlers: dict[str, Callable[[dict[str, Any], str, str], dict[str, Any] | None]] = {
        "codeparrot/apps": _adapt_apps_grpo,
        "BAAI/TACO": _adapt_taco_grpo,
        "deepmind/code_contests": _adapt_codecontests_grpo,
        "open-r1/codeforces": _adapt_codeforces_grpo,
    }

    if source_name == "internal/ml-research-eval-train":
        rows = _load_internal_eval_examples(bucket_name)[:target_count]
        return rows, {"target_count": target_count, "materialized_count": len(rows), "rows_scanned": len(rows)}

    handler = handlers.get(source_name)
    if handler is None:
        return [], {"target_count": target_count, "materialized_count": 0, "skipped_reason": "no handler"}

    examples: list[dict[str, Any]] = []
    attempted = 0
    for repo_id, file_name in _resolve_source_files(source_name, max_files=config.max_files_per_source, cache_dir=config.hf_cache_dir):
        for row in _iter_file_rows(repo_id, file_name, cache_dir=config.hf_cache_dir, batch_size=config.pyarrow_batch_size):
            attempted += 1
            example = handler(row, source_name, bucket_name)
            if not example or example["task_id"] in seen_ids:
                continue
            seen_ids.add(example["task_id"])
            examples.append(example)
            if len(examples) >= target_count:
                return examples, {"target_count": target_count, "materialized_count": len(examples), "rows_scanned": attempted}
    return examples, {"target_count": target_count, "materialized_count": len(examples), "rows_scanned": attempted}


def _split_rows(rows: list[dict[str, Any]], *, validation_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from .filtering import stable_split

    shaped_rows = [{"repository_identity": row.get("repository_identity") or row.get("task_id") or row.get("id"), **row} for row in rows]
    return stable_split(shaped_rows, validation_fraction=validation_fraction)


def materialize(config_path: str | Path) -> dict[str, Any]:
    config = load_materialization_config(config_path)
    manifest = load_manifest_spec(config.manifest_path, frozen=Path(config.manifest_path).suffix == ".json")
    validation = validate_manifest_spec(manifest, frozen=Path(config.manifest_path).suffix == ".json")
    if not validation.ok:
        raise ValueError("\n".join(validation.errors))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    materialized_rows: list[dict[str, Any]] = []
    source_summary: dict[str, Any] = {}
    skipped_sources: dict[str, str] = {}

    for bucket in manifest["buckets"]:
        bucket_name = bucket["name"]
        bucket_target = max(1, round(config.total_examples * float(bucket["share"])))
        sources = [source["name"] for source in bucket["sources"]]
        source_target = max(1, round(bucket_target / max(1, len(sources))))

        for source_name in sources:
            try:
                if config.task_type == "sft":
                    rows, summary = _materialize_sft_source(source_name, bucket_name, source_target, config, seen_ids)
                else:
                    rows, summary = _materialize_grpo_source(source_name, bucket_name, source_target, config, seen_ids)
            except GatedRepoError as exc:
                rows = []
                summary = {"target_count": source_target, "materialized_count": 0, "skipped_reason": str(exc)}
            except Exception as exc:
                rows = []
                summary = {"target_count": source_target, "materialized_count": 0, "skipped_reason": f"{type(exc).__name__}: {exc}"}

            materialized_rows.extend(rows)
            source_summary[source_name] = summary
            if summary.get("skipped_reason"):
                skipped_sources[source_name] = summary["skipped_reason"]

    train_rows, validation_rows = _split_rows(materialized_rows, validation_fraction=config.validation_fraction)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)

    resolved_manifest = {
        "config_path": str(config_path),
        "manifest_path": config.manifest_path,
        "task_type": config.task_type,
        "total_materialized": len(materialized_rows),
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "source_summary": source_summary,
        "skipped_sources": skipped_sources,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
    }
    write_json(output_dir / "resolved_manifest.json", resolved_manifest)
    return resolved_manifest
