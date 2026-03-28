from pathlib import Path

from gpt_oss_research.internal_eval import discover_tasks, run_eval


ROOT = Path(__file__).resolve().parents[1]


def test_discover_tasks():
    tasks = discover_tasks(ROOT / "eval" / "ml_research_eval" / "tasks")
    assert {task.task_id for task in tasks} == {
        "autograd_clip_grad",
        "data_collator_pad_labels",
        "transformer_masked_mean_pool",
        "training_detach_metrics",
    }


def test_reference_solutions_pass():
    report = run_eval(tasks_root=ROOT / "eval" / "ml_research_eval" / "tasks", use_reference=True)
    assert report["task_count"] == 4
    assert report["failed"] == 0

