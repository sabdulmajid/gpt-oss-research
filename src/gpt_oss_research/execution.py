from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any


def normalize_output(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return "\n".join(line.rstrip() for line in stripped.splitlines())


def run_python_io_cases(
    code: str,
    cases: list[dict[str, str]],
    *,
    timeout_sec: float = 2.0,
) -> dict[str, Any]:
    if not cases:
        return {"passed": False, "score": 0.0, "total": 0, "results": [], "error": "no test cases"}

    with tempfile.TemporaryDirectory(prefix="python-io-tests-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        solution_path = temp_dir / "solution.py"
        solution_path.write_text(code, encoding="utf-8")

        case_results = []
        passed = 0
        for index, case in enumerate(cases):
            expected = normalize_output(str(case.get("output", "")))
            process = subprocess.run(
                ["python", str(solution_path)],
                input=str(case.get("input", "")),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            actual = normalize_output(process.stdout)
            case_passed = process.returncode == 0 and actual == expected
            if case_passed:
                passed += 1
            case_results.append(
                {
                    "case_index": index,
                    "passed": case_passed,
                    "returncode": process.returncode,
                    "expected": expected,
                    "actual": actual,
                    "stderr": process.stderr,
                }
            )

    return {
        "passed": passed == len(cases),
        "score": passed / len(cases),
        "total": len(cases),
        "results": case_results,
    }

