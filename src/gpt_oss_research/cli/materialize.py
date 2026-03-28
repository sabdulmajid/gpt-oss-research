from __future__ import annotations

import argparse
import json

from ..materialize import materialize


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize an SFT or GRPO dataset slice from a manifest.")
    parser.add_argument("config_path")
    args = parser.parse_args()
    report = materialize(args.config_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

