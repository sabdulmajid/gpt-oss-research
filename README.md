# gpt-oss Research

This repository builds and measures a `gpt-oss` model specialized for ML research and PyTorch-heavy coding. The target is not generic coding uplift. The target is stronger performance on:

- PyTorch implementation
- debugging of training loops, autograd, mixed precision, distributed training, and data pipelines
- ML systems reasoning
- executable code correctness

`AGENTS.md` is the operating contract for the project. It defines the hardware reality, data policy, benchmark rules, anti-leakage requirements, and the SFT-first research direction.

## Research Direction

The project follows the strategy in `AGENTS.md`.

- Primary trainable model: `gpt-oss-20b`
- Primary teacher/reference model: `gpt-oss-120b`
- Order of work: SFT baseline first, then GRPO on verifiable tasks, then benchmark base vs SFT vs SFT+GRPO in the same harness

This is driven by the actual machine budget: `2 x RTX PRO Blackwell 6000` with `96 GB` VRAM each. That is strong enough for iterative `gpt-oss-20b` LoRA research. It is not evidence that `gpt-oss-120b` GRPO is a practical default on this hardware, so the repository does not pretend otherwise.

## Current Milestone

The repository now has a real first benchmark-ready milestone:

- frozen SFT and GRPO manifests that encode the current research hypotheses
- a real materialization pipeline for SFT and GRPO corpora
- a 51-task internal `ML Research Eval` suite with executable tests and bucket metadata
- trainable SFT and GRPO entrypoints for `gpt-oss-20b`
- benchmark configs, reporting paths, and an unattended benchmark pipeline

This is no longer a scaffold. It is the first coherent end-to-end research baseline.

## Validated State

What has been validated in-repo:

- internal reference eval passes `51/51`
- larger benchmark corpora are frozen on disk:
  - SFT benchmark slice: `977` examples total, `938` train, `39` validation
  - GRPO benchmark slice: `383` examples total, `363` train, `20` validation
- the broader validation slice passes: `12` pytest checks
- SFT and GRPO benchmark configs dry-run successfully with concrete plans
- GRPO now supports initializing from the best SFT adapter and saving a real adapter artifact

What previously worked before the current GPU outage:

- a short `gpt-oss-20b` SFT pilot adapter trained successfully
- on the earlier 4-task pilot eval, base `gpt-oss-20b` scored `1/4` and the pilot SFT adapter scored `3/4`

## What Is Still Not Proven

The project is not “done” by the standards in `AGENTS.md`, because the larger benchmark pipeline has not completed yet on the current 51-task suite.

Unproven items remain:

- no completed large benchmark run yet for base vs SFT vs GRPO on the 51-task internal eval
- no external holdout benchmark report yet
- no claim that the current SFT or GRPO mixtures are optimal
- no claim that GRPO improves verified tasks without hurting ML-research behavior

The immediate blocker is infrastructure, not repository structure: the current host has no working NVIDIA driver handshake, no `/dev/nvidia*` devices, and PyTorch reports zero CUDA devices. The benchmark pipeline is set up to resume automatically once CUDA is healthy again.

Operational recovery notes live in [docs/ops/gpu_recovery.md](/pub7/neel2/gpt-oss-research/docs/ops/gpu_recovery.md).

## What This Repository Already Demonstrates

This work establishes the practical foundation for real `gpt-oss` research:

- the data strategy is encoded as manifests instead of vague mixture claims
- the internal ML benchmark is executable and non-trivial
- training and evaluation paths are wired to the same benchmark discipline
- larger runs can now be launched with saved manifests, configs, and report artifacts

## Next Execution Step

The next meaningful result is not another scaffold change. It is a benchmarked run:

1. restore CUDA on the node or move execution to a healthy GPU host
2. run the benchmark pipeline for base `gpt-oss-20b`, SFT benchmark `v1`, and GRPO benchmark `v1`
3. write the first honest benchmark report from stored outputs
4. expand from the 51-task internal suite toward the `200-500` task target in `AGENTS.md`

## References

- OpenAI `gpt-oss` fine-tuning cookbook: https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers
- OpenAI `gpt-oss-20b` model page: https://developers.openai.com/api/docs/models/gpt-oss-20b
- OpenAI `gpt-oss-120b` model page: https://developers.openai.com/api/docs/models/gpt-oss-120b
- Hugging Face TRL GRPO docs: https://huggingface.co/docs/trl/v0.19.1/grpo_trainer
- Hugging Face Transformers `gpt_oss` docs: https://huggingface.co/docs/transformers/en/model_doc/gpt_oss
