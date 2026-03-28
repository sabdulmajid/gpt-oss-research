from __future__ import annotations

import argparse
import json

from ..model_eval import run_model_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model on the internal ML Research Eval tasks.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tasks-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()
    report = run_model_eval(
        model_path=args.model_path,
        tasks_root=args.tasks_root,
        output_path=args.output,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

