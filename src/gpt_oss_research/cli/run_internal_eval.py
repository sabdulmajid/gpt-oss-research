from __future__ import annotations

import argparse
import json

from ..internal_eval import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the internal ML Research Eval harness.")
    parser.add_argument("--tasks-root", required=True)
    parser.add_argument("--solutions-dir")
    parser.add_argument("--use-reference", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = run_eval(
        tasks_root=args.tasks_root,
        solutions_dir=args.solutions_dir,
        use_reference=args.use_reference,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

