---
experiment_name: benchmark_materialization_v1
date_utc: 2026-03-28
status: completed
sft_config: configs/materialization/sft_benchmark_v1.yaml
grpo_config: configs/materialization/grpo_benchmark_v1.yaml
sft_manifest: data/manifests/sft_starting_mix.manifest.json
grpo_manifest: data/manifests/grpo_starting_mix.manifest.json
sft_resolved_manifest: /pub7/neel2/gpt-oss-research/artifacts/materialized/sft_benchmark_v1/resolved_manifest.json
grpo_resolved_manifest: /pub7/neel2/gpt-oss-research/artifacts/materialized/grpo_benchmark_v1/resolved_manifest.json
---

# Summary

The larger benchmark corpora for the first serious `gpt-oss-20b` SFT and GRPO runs are now frozen on disk.

- SFT benchmark slice: `977` total, `938` train, `39` validation
- GRPO benchmark slice: `383` total, `363` train, `20` validation

These are real materialized datasets, not planned counts.

# SFT Materialization Notes

The SFT slice underfilled its requested size because several sources were inaccessible or unsuitable for direct SFT targets in the current environment:

- `bigcode/the-stack-v2-dedup` is gated on this host
- `open-r1/codeforces` does not ship direct reference solutions for SFT targets
- `BigTimeCoderSean/pytorch-issues` provides issue text but not grounded resolutions for direct SFT targets

The largest realized SFT sources in the current benchmark slice are:

- `codeparrot/github-code`: `269`
- `codeparrot/apps`: `115`
- `BAAI/TACO`: `115`
- `deepmind/code_contests`: `115`
- `shrinath-suresh/stack_overflow_pytorch`: `102`
- `shrinath-suresh/pytorch-discuss-tutorial-1000`: `102`
- `suvadityamuk/huggingface-transformers-code-dataset`: `77`
- `nvidia/OpenCodeReasoning`: `77`

The HF training-stack slice from filtered GitHub repos is still thin at `5` examples. That is a real gap to improve in the next data pass.

# GRPO Materialization Notes

The GRPO slice is cleaner and fully verifiable, which is consistent with `AGENTS.md`.

Realized source counts:

- `open-r1/codeforces`: `128`
- `deepmind/code_contests`: `102`
- `BAAI/TACO`: `51`
- `codeparrot/apps`: `51`
- `internal/ml-research-eval-train`: `51`

The internal ML research bucket underfilled because the current internal eval suite has `51` trainable tasks available. Expanding the internal eval suite directly expands this GRPO source.

# Claims

The materialization pipeline is working, reproducible, and large enough to support the first serious base-vs-SFT-vs-GRPO benchmark run.

No quality claim about the data mixture should be made yet. The mixture is still a benchmarked hypothesis, not a proven optimum.
