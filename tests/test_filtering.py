from pathlib import Path

from gpt_oss_research.filtering import filter_records, stable_split
from gpt_oss_research.io import load_jsonl


ROOT = Path(__file__).resolve().parents[1]


def test_filter_records_keeps_only_relevant_python():
    records = load_jsonl(ROOT / "data" / "samples" / "broad_code_candidates.jsonl")
    kept, summary = filter_records(records)
    kept_paths = {record["path"] for record in kept}
    assert kept_paths == {"src/dataset.py", "trainer.py"}
    assert summary["kept_records"] == 2
    assert summary["reject_reasons"]["exact_duplicate"] == 1
    assert summary["reject_reasons"]["missing_ml_signals"] == 1


def test_stable_split_groups_by_repo():
    records = load_jsonl(ROOT / "data" / "samples" / "broad_code_candidates.jsonl")
    kept, _ = filter_records(records)
    train, validation = stable_split(kept, validation_fraction=0.5, seed="fixed")
    repos = {}
    for split_name, rows in (("train", train), ("validation", validation)):
        for row in rows:
            repo = row["repository_identity"]
            repos.setdefault(repo, set()).add(split_name)
    assert all(len(split_names) == 1 for split_names in repos.values())

