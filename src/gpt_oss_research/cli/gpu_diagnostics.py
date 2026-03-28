from __future__ import annotations

import argparse
import json

from ..gpu_diag import collect_gpu_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GPU/driver readiness for benchmark runs.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--summary-line", action="store_true")
    parser.add_argument("--ready-check", action="store_true")
    args = parser.parse_args()

    diagnostics = collect_gpu_diagnostics()

    if args.ready_check:
        raise SystemExit(0 if diagnostics["ready"] else 1)
    if args.summary_line:
        print(diagnostics["summary"])
        return
    if args.summary:
        payload = {
            "ready": diagnostics["ready"],
            "kernel_release": diagnostics["kernel_release"],
            "summary": diagnostics["summary"],
            "recommended_actions": diagnostics["recommended_actions"],
        }
        print(json.dumps(payload, indent=2))
        return
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
