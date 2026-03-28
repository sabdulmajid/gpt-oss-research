from pathlib import Path

from gpt_oss_research.internal_eval import discover_tasks, run_eval


ROOT = Path(__file__).resolve().parents[1]


def test_discover_tasks():
    tasks = discover_tasks(ROOT / "eval" / "ml_research_eval" / "tasks")
    task_ids = {task.task_id for task in tasks}
    assert len(tasks) >= 50
    assert {
        "accelerate_prepare",
        "autograd_clip_grad",
        "data_collator_pad_labels",
        "shift_labels_for_causal_lm",
        "transformer_masked_mean_pool",
        "training_detach_metrics",
    }.issubset(task_ids)
    assert {task.benchmark_bucket for task in tasks} == {
        "debugging_performance",
        "tensor_api",
        "training_pipeline",
        "transformers_accelerate_fsdp_mixed_precision",
    }


def test_reference_solutions_pass():
    report = run_eval(tasks_root=ROOT / "eval" / "ml_research_eval" / "tasks", use_reference=True)
    assert report["task_count"] >= 50
    assert report["failed"] == 0
    assert set(report["bucket_summary"]) == {
        "debugging_performance",
        "tensor_api",
        "training_pipeline",
        "transformers_accelerate_fsdp_mixed_precision",
    }
