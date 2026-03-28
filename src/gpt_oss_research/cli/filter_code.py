from __future__ import annotations

import argparse
import json

from ..filtering import filter_records
from ..io import load_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter broad-code samples into a PyTorch/ML slice.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    kept, summary = filter_records(records)
    write_jsonl(args.output, kept)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

