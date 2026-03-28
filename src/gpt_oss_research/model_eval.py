from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

from .internal_eval import discover_tasks, evaluate_solution
from .io import write_json
from .reporting import current_git_sha, utc_now_iso


def _extract_code(text: str) -> str:
    code_block = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    final_match = re.search(r"assistantfinal\s*(?:code)?(.*)", text, flags=re.DOTALL)
    if final_match:
        return final_match.group(1).strip()
    return text.strip()


def _load_model(model_path: str, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        attn_implementation="eager",
        quantization_config=Mxfp4Config(dequantize=True),
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    return model, tokenizer


def run_model_eval(
    *,
    model_path: str,
    tasks_root: str | Path,
    output_path: str | Path,
    adapter_path: str | None = None,
    max_new_tokens: int = 384,
) -> dict[str, Any]:
    model, tokenizer = _load_model(model_path, adapter_path=adapter_path)
    tasks = discover_tasks(tasks_root)
    output_dir = Path(output_path).resolve().parent
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for task in tasks:
        prompt = task.prompt_path.read_text(encoding="utf-8").strip()
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(generated[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True)
        code = _extract_code(generated_text)
        candidate_path = candidates_dir / f"{task.task_id}.py"
        candidate_path.write_text(code, encoding="utf-8")
        eval_result = evaluate_solution(task, candidate_path)
        eval_result["benchmark_bucket"] = task.benchmark_bucket
        eval_result["difficulty"] = task.difficulty
        eval_result["topic"] = task.topic
        eval_result["raw_generation"] = generated_text
        results.append(eval_result)

    passed = sum(1 for result in results if result["passed"])
    bucket_summary: dict[str, dict[str, int]] = {}
    for task, result in zip(tasks, results):
        summary = bucket_summary.setdefault(task.benchmark_bucket, {"total": 0, "passed": 0, "failed": 0})
        summary["total"] += 1
        if result["passed"]:
            summary["passed"] += 1
        else:
            summary["failed"] += 1
    report = {
        "generated_at_utc": utc_now_iso(),
        "git_sha": current_git_sha(),
        "model_path": model_path,
        "adapter_path": adapter_path,
        "decoding": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        },
        "task_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results),
        "bucket_summary": bucket_summary,
        "results": results,
        "candidates_dir": str(candidates_dir),
    }
    write_json(output_path, report)
    return report
