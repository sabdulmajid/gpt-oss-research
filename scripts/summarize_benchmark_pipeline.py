from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_report(path: str | Path) -> dict | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a markdown benchmark summary for base/SFT/GRPO internal eval reports.")
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--sft-report", required=True)
    parser.add_argument("--grpo-report", required=True)
    parser.add_argument("--sft-train-report")
    parser.add_argument("--grpo-train-report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reports = {
        "base": load_report(args.base_report),
        "sft": load_report(args.sft_report),
        "grpo": load_report(args.grpo_report),
    }
    train_reports = {
        "sft": load_report(args.sft_train_report) if args.sft_train_report else None,
        "grpo": load_report(args.grpo_train_report) if args.grpo_train_report else None,
    }

    output_lines = [
        "---",
        "experiment_name: benchmark_pipeline_v1",
        "status: generated",
        f"base_report: {args.base_report}",
        f"sft_report: {args.sft_report}",
        f"grpo_report: {args.grpo_report}",
        f"sft_train_report: {args.sft_train_report or 'missing'}",
        f"grpo_train_report: {args.grpo_train_report or 'missing'}",
        "---",
        "",
        "# Internal Eval Summary",
        "",
    ]

    for label, report in reports.items():
        if report is None:
            output_lines.append(f"- `{label}`: missing report")
            continue
        output_lines.append(
            f"- `{label}`: {report['passed']}/{report['task_count']} pass@1 "
            f"({report['pass_rate']:.3f})"
        )

    output_lines.extend(
        [
            "",
            "# Run Metadata",
            "",
            f"- base model: `{reports['base']['model_path']}`" if reports["base"] else "- base model: missing report",
            (
                f"- base decoding: `{json.dumps(reports['base']['decoding'], sort_keys=True)}`"
                if reports["base"] and reports["base"].get("decoding")
                else "- base decoding: missing report"
            ),
        ]
    )
    if train_reports["sft"] is not None:
        output_lines.extend(
            [
                (
                    f"- sft adapter: `{train_reports['sft']['adapter']['type']}` "
                    f"(r={train_reports['sft']['adapter']['r']})"
                ),
                f"- sft manifest: `{train_reports['sft']['manifest_path']}`",
                f"- sft dataset rows: `{train_reports['sft']['dataset_summary']['row_count']}`",
                f"- sft seed: `{train_reports['sft']['training'].get('seed', 'unset')}`",
                f"- sft max_steps: `{train_reports['sft']['training'].get('max_steps', 'unset')}`",
            ]
        )
    else:
        output_lines.append("- sft metadata: missing train report")
    if train_reports["grpo"] is not None:
        output_lines.extend(
            [
                (
                    f"- grpo adapter: `{train_reports['grpo']['adapter']['type']}` "
                    f"(r={train_reports['grpo']['adapter']['r']})"
                ),
                f"- grpo manifest: `{train_reports['grpo']['manifest_path']}`",
                f"- grpo dataset rows: `{train_reports['grpo']['dataset_summary']['row_count']}`",
                f"- grpo seed: `{train_reports['grpo']['training'].get('seed', 'unset')}`",
                f"- grpo max_steps: `{train_reports['grpo']['training'].get('max_steps', 'unset')}`",
                f"- grpo reward: `{train_reports['grpo']['reward']['type']}`",
            ]
        )
    else:
        output_lines.append("- grpo metadata: missing train report")

    output_lines.extend(["", "# Bucket Summary", ""])
    for label in ("base", "sft", "grpo"):
        report = reports[label]
        if report is None:
            output_lines.append(f"- `{label}`: missing report")
            continue
        output_lines.append(f"- `{label}` buckets: `{json.dumps(report.get('bucket_summary', {}), sort_keys=True)}`")

    if reports["base"] and reports["sft"]:
        delta = reports["sft"]["pass_rate"] - reports["base"]["pass_rate"]
        output_lines.extend(
            [
                "",
                "# Deltas",
                "",
                f"- `sft - base` delta: {delta:.3f}",
            ]
        )
    if reports["sft"] and reports["grpo"]:
        delta = reports["grpo"]["pass_rate"] - reports["sft"]["pass_rate"]
        output_lines.append(f"- `grpo - sft` delta: {delta:.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
