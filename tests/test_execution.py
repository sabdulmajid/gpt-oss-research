from gpt_oss_research.execution import run_python_io_cases


def test_run_python_io_cases_passes_simple_program():
    result = run_python_io_cases(
        "n = int(input())\nprint(n + 1)\n",
        [{"input": "2\n", "output": "3\n"}],
    )
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_run_python_io_cases_reports_failure():
    result = run_python_io_cases(
        "print('wrong')\n",
        [{"input": "", "output": "right\n"}],
    )
    assert result["passed"] is False
    assert result["score"] == 0.0

