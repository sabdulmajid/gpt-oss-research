# gpt-oss Research

This repository builds and evaluates a `gpt-oss` model specialized for ML research and PyTorch-heavy coding. The focus is not generic coding uplift. The focus is better performance on:

- PyTorch implementation
- debugging of training loops, autograd, mixed precision, distributed training, and data pipelines
- ML systems reasoning
- executable code correctness

## Current Snapshot

- Manifest-driven SFT and GRPO data pipelines are implemented and validated.
- The internal `ML Research Eval` suite contains `51` executable tasks and passes `51/51` with reference solutions.
- The first larger frozen corpora are materialized:
  - SFT benchmark slice: `977` total, `938` train, `39` validation
  - GRPO benchmark slice: `383` total, `363` train, `20` validation
- A short end-to-end `gpt-oss-20b` SFT pilot completed successfully.
- On the early 4-task pilot eval, base `gpt-oss-20b` scored `1/4` and the pilot adapter scored `3/4`.
- The larger base-vs-SFT-vs-GRPO benchmark pipeline is implemented but was not completed on the original host because of an NVIDIA driver outage.

## Repository Scope

The repo already includes:

- frozen dataset manifests and dataset-hypothesis configs
- dataset filtering, materialization, validation, and split tooling
- SFT and GRPO training entrypoints for `gpt-oss-20b`
- an internal ML research benchmark harness with executable tests
- experiment reports, metadata templates, and benchmark summaries
- GPU diagnostics and benchmark-run control scripts

## Read First

- [Project strategy](docs/project_strategy.md)
- [Setup and portability](docs/setup.md)
- [Current project status](reports/status/2026-04-04_project_status.md)
- [GPU recovery notes](docs/ops/gpu_recovery.md)

## Status

This is a real research starting point, not a finished benchmark claim.

What is validated:

- the repo installs cleanly
- smoke validation passes
- the internal benchmark harness is real and executable
- the benchmark corpora are frozen on disk
- the training and evaluation paths are wired together coherently

What is still open:

- a completed large benchmark run for base vs SFT vs GRPO on the 51-task internal suite
- external holdout evaluation
- ablations that prove whether the current mixture is actually the right one

## References

- OpenAI `gpt-oss` fine-tuning cookbook: https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers
- OpenAI `gpt-oss-20b` model page: https://developers.openai.com/api/docs/models/gpt-oss-20b
- OpenAI `gpt-oss-120b` model page: https://developers.openai.com/api/docs/models/gpt-oss-120b
- Hugging Face TRL GRPO docs: https://huggingface.co/docs/trl/v0.19.1/grpo_trainer
- Hugging Face Transformers `gpt_oss` docs: https://huggingface.co/docs/transformers/en/model_doc/gpt_oss
