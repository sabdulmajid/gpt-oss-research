from __future__ import annotations

import argparse
import json

from ..benchmark_status import benchmark_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the current benchmark pipeline status.")
    parser.parse_args()
    print(json.dumps(benchmark_status(), indent=2))


if __name__ == "__main__":
    main()
