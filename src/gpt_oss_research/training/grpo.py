from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..execution import run_python_io_cases
from .common import TrainingPlan, build_target_parameters, load_training_config, materialized_dataset_summary, validate_training_manifest


def build_grpo_plan(config_path: str | Path) -> TrainingPlan:
    config = load_training_config(config_path)
    manifest_path = config["data"]["manifest_path"]
    validate_training_manifest(manifest_path)

    target_parameters = build_target_parameters(config["adapter"]["expert_targeting"])
    warnings: list[str] = []
    if config["training"].get("use_vllm", False):
        warnings.append("use_vllm is enabled; verify GPU memory headroom before running on the 2x96 GB setup")
    else:
        warnings.append("use_vllm is disabled; this matches the practical colocated/no-server default in AGENTS.md")

    dataset_summary = materialized_dataset_summary(config["data"].get("materialized_dataset_path"))
    if not dataset_summary["present"]:
        warnings.append("materialized_dataset_path is missing; GRPO can only proceed after dataset materialization")

    details = {
        "output_dir": config["output"]["output_dir"],
        "report_dir": config["output"]["report_dir"],
        "reward": config["reward"],
        "adapter": {
            "lora_r": config["adapter"]["r"],
            "lora_alpha": config["adapter"]["lora_alpha"],
            "lora_dropout": config["adapter"]["lora_dropout"],
            "target_modules": config["adapter"]["target_modules"],
        },
        "training": config["training"],
        "dataset_summary": dataset_summary,
    }
    return TrainingPlan(
        config_path=str(config_path),
        experiment_name=config["experiment_name"],
        model_name_or_path=config["model"]["name_or_path"],
        teacher_model=config["model"]["teacher_model"],
        manifest_path=manifest_path,
        target_parameters=target_parameters,
        warnings=warnings,
        details=details,
    )


def _extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for message in completion:
            if isinstance(message, dict):
                parts.append(str(message.get("content", "")))
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return str(completion)


def make_internal_eval_reward(tasks_root: str | Path):
    root = Path(tasks_root)

    def reward_func(completions, checker_type, tests, task_id=None, **kwargs):
        rewards: list[float] = []
        task_ids = task_id or [None] * len(completions)
        for completion, current_checker, test_blob, current_task_id in zip(completions, checker_type, tests, task_ids):
            completion_text = _extract_completion_text(completion)
            if current_checker == "internal_eval_task":
                resolved_task_id = (test_blob or {}).get("task_id") or current_task_id
                task_dir = root / str(resolved_task_id).replace("internal_eval:", "")
                tests_path = task_dir / "tests.py"
                if not tests_path.exists():
                    rewards.append(0.0)
                    continue
                with tempfile.TemporaryDirectory(prefix=f"reward-{resolved_task_id}-") as temp_dir_name:
                    temp_dir = Path(temp_dir_name)
                    (temp_dir / "solution.py").write_text(completion_text, encoding="utf-8")
                    shutil.copy2(tests_path, temp_dir / "test_solution.py")
                    result = subprocess.run(
                        ["python", "-m", "pytest", "-q", "test_solution.py"],
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                rewards.append(1.0 if result.returncode == 0 else 0.0)
                continue

            if current_checker == "python_io_tests":
                io_result = run_python_io_cases(
                    completion_text,
                    list((test_blob or {}).get("cases", [])),
                    timeout_sec=float((test_blob or {}).get("timeout_sec", 2.0)),
                )
                rewards.append(float(io_result["score"]))
                continue

            rewards.append(0.0)

        return rewards

    return reward_func


def make_internal_eval_reward_legacy(tasks_root: str | Path):
    root = Path(tasks_root)

    def reward_func(completions, task_id, **kwargs):
        rewards: list[float] = []
        for completion, current_task_id in zip(completions, task_id):
            task_dir = root / str(current_task_id)
            tests_path = task_dir / "tests.py"
            if not tests_path.exists():
                rewards.append(0.0)
                continue
            with tempfile.TemporaryDirectory(prefix=f"reward-{current_task_id}-") as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                completion_text = _extract_completion_text(completion)
                (temp_dir / "solution.py").write_text(completion_text, encoding="utf-8")
                shutil.copy2(tests_path, temp_dir / "test_solution.py")
                result = subprocess.run(
                    ["python", "-m", "pytest", "-q", "test_solution.py"],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            rewards.append(1.0 if result.returncode == 0 else 0.0)
        return rewards

    return reward_func


def run_grpo(config_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    plan = build_grpo_plan(config_path)
    plan_payload = {
        "config_path": plan.config_path,
        "experiment_name": plan.experiment_name,
        "model_name_or_path": plan.model_name_or_path,
        "teacher_model": plan.teacher_model,
        "manifest_path": plan.manifest_path,
        "target_parameters": plan.target_parameters,
        "warnings": plan.warnings,
        "details": plan.details,
    }
    if dry_run:
        return {"mode": "dry_run", "plan": plan_payload}

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from trl import GRPOConfig, GRPOTrainer

    config = load_training_config(config_path)
    dataset_path = config["data"]["materialized_dataset_path"]
    if not dataset_path or not Path(dataset_path).exists():
        raise FileNotFoundError("materialized_dataset_path must exist for a real GRPO run")

    dataset = load_dataset("json", data_files={"train": dataset_path})["train"]
    reward_func = make_internal_eval_reward(config["reward"]["tasks_root"])
    peft_config = LoraConfig(
        r=config["adapter"]["r"],
        lora_alpha=config["adapter"]["lora_alpha"],
        lora_dropout=config["adapter"]["lora_dropout"],
        target_modules=config["adapter"]["target_modules"],
        target_parameters=plan.target_parameters,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name_or_path"])
    quantization_config = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name_or_path"],
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
    )
    trainer = GRPOTrainer(
        model=model,
        args=GRPOConfig(**config["training"]),
        reward_funcs=reward_func,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    return {
        "mode": "train",
        "plan": plan_payload,
        "train_result": train_result.metrics,
    }
