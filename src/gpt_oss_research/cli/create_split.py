from __future__ import annotations

import argparse
import json

from ..filtering import stable_split
from ..io import load_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stable repo-aware train/validation split.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", default="gpt-oss-research-v1")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    train, validation = stable_split(records, validation_fraction=args.validation_fraction, seed=args.seed)
    write_jsonl(args.train_output, train)
    write_jsonl(args.val_output, validation)
    print(json.dumps({"train": len(train), "validation": len(validation)}, indent=2))


if __name__ == "__main__":
    main()

