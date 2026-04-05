from pathlib import Path

from gpt_oss_research.io import load_json, load_yaml


def test_load_yaml_expands_env_and_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n"
        "  path: ${MODEL_PATH:-openai/gpt-oss-20b}\n"
        "cache: ${HF_CACHE_DIR:-~/.cache/huggingface/hub}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_PATH", "/models/gpt-oss-20b")

    config = load_yaml(config_path)

    assert config["model"]["path"] == "/models/gpt-oss-20b"
    assert config["cache"].endswith("/.cache/huggingface/hub")


def test_load_json_expands_env(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"teacher_model": "${TEACHER_PATH:-openai/gpt-oss-120b}"}\n', encoding="utf-8")
    monkeypatch.setenv("TEACHER_PATH", "/models/gpt-oss-120b")

    manifest = load_json(manifest_path)

    assert manifest["teacher_model"] == "/models/gpt-oss-120b"
