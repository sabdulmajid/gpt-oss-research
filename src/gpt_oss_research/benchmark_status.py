from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/pub7/neel2/gpt-oss-research")
SESSION_NAME = "benchmark-pipeline"
LOG_PATH = ROOT / "artifacts" / "logs" / "benchmark_pipeline_v1.log"
BASE_REPORT = ROOT / "artifacts" / "eval" / "internal_eval_base_51_v1.json"
SFT_ADAPTER = ROOT / "artifacts" / "checkpoints" / "sft" / "gpt_oss_20b_sft_benchmark_v1" / "adapter_model.safetensors"
SFT_REPORT = ROOT / "artifacts" / "eval" / "internal_eval_sft_benchmark_v1.json"
GRPO_ADAPTER = ROOT / "artifacts" / "checkpoints" / "grpo" / "gpt_oss_20b_grpo_benchmark_v1" / "adapter_model.safetensors"
GRPO_REPORT = ROOT / "artifacts" / "eval" / "internal_eval_grpo_benchmark_v1.json"
SUMMARY_REPORT = ROOT / "reports" / "experiments" / "benchmark_pipeline_v1.md"


def _run_command(command: list[str]) -> tuple[int | None, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return None, ""
    return result.returncode, result.stdout.strip()


def benchmark_status() -> dict[str, Any]:
    session_rc, _ = _run_command(["tmux", "has-session", "-t", SESSION_NAME])
    active = session_rc == 0
    log_lines: list[str] = []
    if LOG_PATH.exists():
        log_lines = LOG_PATH.read_text(encoding="utf-8").splitlines()[-10:]

    artifacts = {
        "base_report": BASE_REPORT.exists(),
        "sft_adapter": SFT_ADAPTER.exists(),
        "sft_report": SFT_REPORT.exists(),
        "grpo_adapter": GRPO_ADAPTER.exists(),
        "grpo_report": GRPO_REPORT.exists(),
        "summary_report": SUMMARY_REPORT.exists(),
    }

    status = "idle"
    if artifacts["summary_report"]:
        status = "completed"
    elif active and log_lines and "CUDA not ready" in log_lines[-1]:
        status = "waiting_for_cuda"
    elif active:
        status = "running"

    return {
        "status": status,
        "tmux_session": {
            "name": SESSION_NAME,
            "active": active,
        },
        "log_path": str(LOG_PATH),
        "log_tail": log_lines,
        "artifacts": artifacts,
    }
