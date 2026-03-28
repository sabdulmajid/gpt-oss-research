from pathlib import Path

from gpt_oss_research.manifests import build_frozen_manifest, load_manifest_spec, validate_manifest_spec


ROOT = Path(__file__).resolve().parents[1]


def test_yaml_manifests_validate():
    for name in ("sft_starting_mix.yaml", "grpo_starting_mix.yaml"):
        spec = load_manifest_spec(ROOT / "configs" / "datasets" / name)
        result = validate_manifest_spec(spec)
        assert result.ok, result.errors


def test_frozen_manifests_validate():
    for name in ("sft_starting_mix.manifest.json", "grpo_starting_mix.manifest.json"):
        spec = load_manifest_spec(ROOT / "data" / "manifests" / name, frozen=True)
        result = validate_manifest_spec(spec, frozen=True)
        assert result.ok, result.errors


def test_manifest_builder_adds_frozen_timestamp():
    spec = load_manifest_spec(ROOT / "configs" / "datasets" / "sft_starting_mix.yaml")
    manifest = build_frozen_manifest(spec, source_path="configs/datasets/sft_starting_mix.yaml")
    assert "frozen_at_utc" in manifest
    assert manifest["buckets"][0]["share_pct"] == 35.0

