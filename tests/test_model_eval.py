from gpt_oss_research.model_eval import _extract_code


def test_extract_code_prefers_code_block():
    text = "finalHere you go\n```python\nprint('ok')\n```"
    assert _extract_code(text) == "print('ok')"


def test_extract_code_falls_back_to_final_channel():
    text = "analysisblah assistantfinal def add(a, b):\n    return a + b"
    assert _extract_code(text).startswith("def add")
