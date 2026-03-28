from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import write_json
from ..reporting import current_git_sha, utc_now_iso
from .common import TrainingPlan, build_target_parameters, load_training_config, materialized_dataset_summary, validate_training_manifest


def build_sft_plan(config_path: str | Path) -> TrainingPlan:
    config = load_training_config(config_path)
    manifest_path = config["data"]["manifest_path"]
    validate_training_manifest(manifest_path)

    target_parameters = build_target_parameters(config["adapter"]["expert_targeting"])
    warnings: list[str] = []
    if not target_parameters:
        warnings.append("no expert target parameters were produced; this would violate the intended MoE targeting strategy")

    dataset_summary = materialized_dataset_summary(config["data"].get("materialized_dataset_path"))
    if not dataset_summary["present"]:
        warnings.append("materialized_dataset_path is missing; training can only proceed after dataset materialization")

    details = {
        "output_dir": config["output"]["output_dir"],
        "report_dir": config["output"]["report_dir"],
        "model_load": config["model_load"],
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


def run_sft(config_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    plan = build_sft_plan(config_path)
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
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from trl import SFTConfig, SFTTrainer

    config = load_training_config(config_path)
    dataset_path = config["data"]["materialized_dataset_path"]
    if not dataset_path or not Path(dataset_path).exists():
        raise FileNotFoundError("materialized_dataset_path must exist for a real SFT run")

    quantization_config = None
    if config["model_load"].get("quantization") == "mxfp4":
        quantization_config = Mxfp4Config(dequantize=bool(config["model_load"].get("dequantize", True)))

    model_kwargs = {
        "attn_implementation": config["model_load"]["attn_implementation"],
        "torch_dtype": torch.bfloat16 if config["model_load"]["torch_dtype"] == "bfloat16" else torch.float16,
        "quantization_config": quantization_config,
        "use_cache": bool(config["model_load"].get("use_cache", False)),
        "device_map": config["model_load"].get("device_map", "auto"),
    }

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name_or_path"])
    model = AutoModelForCausalLM.from_pretrained(config["model"]["name_or_path"], **model_kwargs)
    peft_config = LoraConfig(
        r=config["adapter"]["r"],
        lora_alpha=config["adapter"]["lora_alpha"],
        lora_dropout=config["adapter"]["lora_dropout"],
        target_modules=config["adapter"]["target_modules"],
        target_parameters=plan.target_parameters,
    )
    peft_model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files={"train": dataset_path})
    train_dataset = dataset["train"]

    def format_messages(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_dataset = train_dataset.map(format_messages, remove_columns=train_dataset.column_names)
    trainer = SFTTrainer(
        model=peft_model,
        args=SFTConfig(**config["training"]),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(config["output"]["output_dir"])
    report_path = Path(config["output"]["output_dir"]) / "train_report.json"
    report = {
        "generated_at_utc": utc_now_iso(),
        "git_sha": current_git_sha(),
        "task_type": "sft",
        "experiment_name": config["experiment_name"],
        "config_path": str(config_path),
        "model_name_or_path": config["model"]["name_or_path"],
        "teacher_model": config["model"]["teacher_model"],
        "manifest_path": config["data"]["manifest_path"],
        "dataset_summary": plan.details["dataset_summary"],
        "adapter": {
            "type": config["adapter"]["type"],
            "r": config["adapter"]["r"],
            "lora_alpha": config["adapter"]["lora_alpha"],
            "lora_dropout": config["adapter"]["lora_dropout"],
            "target_modules": config["adapter"]["target_modules"],
            "target_parameters": plan.target_parameters,
        },
        "training": config["training"],
        "output_dir": config["output"]["output_dir"],
        "train_result": train_result.metrics,
    }
    write_json(report_path, report)
    return {
        "mode": "train",
        "plan": plan_payload,
        "train_result": train_result.metrics,
        "report_path": str(report_path),
    }
