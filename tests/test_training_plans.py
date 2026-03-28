from pathlib import Path

from gpt_oss_research.training.grpo import build_grpo_plan
from gpt_oss_research.training.sft import build_sft_plan


ROOT = Path(__file__).resolve().parents[1]


def test_build_sft_plan():
    plan = build_sft_plan(ROOT / "configs" / "training" / "sft_gpt_oss_20b_lora.yaml")
    assert plan.experiment_name == "gpt_oss_20b_sft_baseline_v1"
    assert plan.target_parameters == [
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj",
    ]
    assert not plan.warnings or "materialized_dataset_path" not in " ".join(plan.warnings)


def test_build_grpo_plan():
    plan = build_grpo_plan(ROOT / "configs" / "training" / "grpo_gpt_oss_20b_lora.yaml")
    assert plan.experiment_name == "gpt_oss_20b_grpo_baseline_v1"
    assert plan.details["reward"]["type"] == "internal_eval_pass"

