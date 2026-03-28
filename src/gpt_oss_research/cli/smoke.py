from __future__ import annotations

import json
from pathlib import Path

from ..filtering import filter_records, stable_split
from ..internal_eval import run_eval
from ..io import load_jsonl, write_json, write_jsonl
from ..manifests import build_frozen_manifest, load_manifest_spec, validate_manifest_spec
from ..training.grpo import run_grpo
from ..training.sft import run_sft


def main() -> None:
    root = Path.cwd()
    artifacts_dir = root / "artifacts" / "smoke"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = artifacts_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_payloads = {}
    for name in ("sft_starting_mix", "grpo_starting_mix"):
        spec_path = root / "configs" / "datasets" / f"{name}.yaml"
        spec = load_manifest_spec(spec_path)
        validation = validate_manifest_spec(spec)
        if not validation.ok:
            raise SystemExit(f"{name} failed validation: {validation.errors}")
        relative_spec_path = spec_path.relative_to(root)
        manifest = build_frozen_manifest(spec, source_path=relative_spec_path)
        manifest_path = manifest_dir / f"{name}.manifest.json"
        write_json(manifest_path, manifest)
        manifest_payloads[name] = manifest_path

    sample_records = load_jsonl(root / "data" / "samples" / "broad_code_candidates.jsonl")
    filtered, summary = filter_records(sample_records)
    write_jsonl(artifacts_dir / "filtered_code.jsonl", filtered)
    write_json(artifacts_dir / "filter_summary.json", summary)
    train_rows, val_rows = stable_split(filtered)
    write_jsonl(artifacts_dir / "train.jsonl", train_rows)
    write_jsonl(artifacts_dir / "val.jsonl", val_rows)

    eval_report = run_eval(
        tasks_root=root / "eval" / "ml_research_eval" / "tasks",
        use_reference=True,
        output_path=artifacts_dir / "internal_eval_report.json",
    )
    sft_plan = run_sft(root / "configs" / "training" / "sft_gpt_oss_20b_lora.yaml", dry_run=True)
    grpo_plan = run_grpo(root / "configs" / "training" / "grpo_gpt_oss_20b_lora.yaml", dry_run=True)

    payload = {
        "manifests": {name: str(path) for name, path in manifest_payloads.items()},
        "filter_summary": summary,
        "split_summary": {"train": len(train_rows), "validation": len(val_rows)},
        "internal_eval": {
            "task_count": eval_report["task_count"],
            "passed": eval_report["passed"],
            "failed": eval_report["failed"],
            "pass_rate": eval_report["pass_rate"],
        },
        "sft": sft_plan,
        "grpo": grpo_plan,
    }
    write_json(artifacts_dir / "smoke_summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
