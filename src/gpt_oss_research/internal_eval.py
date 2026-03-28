from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import load_yaml, write_json


@dataclass(slots=True)
class EvalTask:
    task_id: str
    topic: str
    difficulty: str
    entrypoint: str
    timeout_sec: int
    path: Path

    @property
    def prompt_path(self) -> Path:
        return self.path / "prompt.md"

    @property
    def reference_solution_path(self) -> Path:
        return self.path / "reference_solution.py"

    @property
    def tests_path(self) -> Path:
        return self.path / "tests.py"


def discover_tasks(tasks_root: str | Path) -> list[EvalTask]:
    root = Path(tasks_root)
    tasks: list[EvalTask] = []
    for metadata_path in sorted(root.glob("*/metadata.yaml")):
        metadata = load_yaml(metadata_path)
        task = EvalTask(
            task_id=metadata["task_id"],
            topic=metadata["topic"],
            difficulty=metadata["difficulty"],
            entrypoint=metadata["entrypoint"],
            timeout_sec=int(metadata.get("timeout_sec", 30)),
            path=metadata_path.parent,
        )
        tasks.append(task)
    return tasks


def evaluate_solution(task: EvalTask, solution_path: str | Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"{task.task_id}-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        shutil.copy2(Path(solution_path), temp_dir / "solution.py")
        shutil.copy2(task.tests_path, temp_dir / "test_solution.py")

        command = ["python", "-m", "pytest", "-q", "test_solution.py"]
        result = subprocess.run(
            command,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=task.timeout_sec,
            check=False,
        )
        return {
            "task_id": task.task_id,
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


def run_eval(
    *,
    tasks_root: str | Path,
    solutions_dir: str | Path | None = None,
    use_reference: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    tasks = discover_tasks(tasks_root)
    if not tasks:
        raise ValueError(f"no tasks found under {tasks_root}")
    if not use_reference and solutions_dir is None:
        raise ValueError("either use_reference must be true or solutions_dir must be provided")

    results = []
    for task in tasks:
        if use_reference:
            solution_path = task.reference_solution_path
        else:
            candidate_path = Path(solutions_dir) / f"{task.task_id}.py"
            if not candidate_path.exists():
                results.append(
                    {
                        "task_id": task.task_id,
                        "passed": False,
                        "returncode": None,
                        "stdout": "",
                        "stderr": f"missing candidate solution: {candidate_path}",
                    }
                )
                continue
            solution_path = candidate_path
        results.append(evaluate_solution(task, solution_path))

    passed = sum(1 for result in results if result["passed"])
    report = {
        "tasks_root": str(tasks_root),
        "task_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results),
        "results": results,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report

