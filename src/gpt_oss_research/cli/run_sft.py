from __future__ import annotations

import argparse
import json

from ..training.sft import run_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run or launch the SFT training path.")
    parser.add_argument("config_path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_sft(args.config_path, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

