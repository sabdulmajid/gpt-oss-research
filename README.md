# gpt-oss Research

This repository is building a `gpt-oss` model that is genuinely useful for ML research and PyTorch-heavy coding. The target is not generic coding uplift. The target is better performance on:

- PyTorch implementation
- debugging of training loops, autograd, data pipelines, mixed precision, and distributed training
- ML systems reasoning
- executable code correctness

`AGENTS.md` is the operating contract for the project. It defines the research direction, hardware constraints, dataset policy, benchmark rules, and anti-leakage requirements.

## Project Strategy

The current strategy is intentionally conservative.

- Primary trainable model: `gpt-oss-20b`
- Primary teacher/reference model: `gpt-oss-120b`
- Order of work: SFT first, benchmark honestly, then explore GRPO on verifiable tasks

This is driven by the actual hardware:

- 2x RTX PRO Blackwell 6000
- 96 GB VRAM each
- 192 GB total VRAM

That hardware is well suited to iterative `gpt-oss-20b` LoRA-style research. It is not evidence that `gpt-oss-120b` GRPO is practical, so this repository does not treat `120b` GRPO as the default path.

## First Milestone

The first usable milestone is now in place. It gives the project a real end-to-end starting point instead of a plan on paper.

It includes:

- frozen dataset manifests for the starting SFT and GRPO hypotheses
- a filtering and splitting pipeline for building a PyTorch/ML slice from broad code corpora
- a small but real internal `ML Research Eval` harness with executable tests
- initial SFT and GRPO configs for `gpt-oss-20b`
- report templates and validation logic that enforce exact logging and benchmark discipline
- a smoke-validation path that checks the pipeline without launching an expensive training run

The emphasis is deliberate: measurement discipline first, training claims later.

## Current Status

What works today:

- the project installs cleanly as a Python package
- manifest generation and validation run successfully
- sample code filtering and repo-aware splitting run successfully
- the internal eval harness executes real task tests
- SFT and GRPO configs dry-run into concrete, inspectable plans

What is not done yet:

- no real SFT training run has been completed
- no benchmark improvement claim exists
- the internal `ML Research Eval` is still a seed suite, not the full target benchmark
- external training corpora still need to be materialized from the dataset manifests

This repository should therefore be read as a serious research bootstrap, not as evidence of model improvement.

## Key Artifacts

The most important project files are:

- dataset hypotheses: `configs/datasets/sft_starting_mix.yaml` and `configs/datasets/grpo_starting_mix.yaml`
- train configs: `configs/training/sft_gpt_oss_20b_lora.yaml` and `configs/training/grpo_gpt_oss_20b_lora.yaml`
- internal benchmark seed: `eval/ml_research_eval/`
- benchmark/report contract: `reports/templates/benchmark_report.md`

The current milestone can be validated with a package install plus the smoke path:

```bash
python -m pip install -e .[dev]
make smoke
```

## What This Repository Is Trying To Prove

The central research question is whether a PyTorch- and ML-systems-heavy fine-tuning mixture can move `gpt-oss-20b` in the direction that matters:

- stronger PyTorch coding
- stronger debugging on real training issues
- stronger performance on an internal ML research benchmark

The starting SFT and GRPO mixtures in this repository are hypotheses. They are not presented as optimal, and they are not presented as validated until the benchmark harness proves it.

## Next Steps

The next phase should stay narrow and empirical:

1. Materialize the first real SFT dataset slice from the frozen manifests.
2. Expand the internal `ML Research Eval` from the current seed suite toward the planned 200-500 task range.
3. Run the first honest baseline comparison: base `gpt-oss-20b` versus SFT on the same evaluation harness.
4. Only after the SFT baseline is stable, wire GRPO from the best SFT checkpoint onto verifiable tasks.

## References

This repository is aligned with the current primary documentation for `gpt-oss` fine-tuning and GRPO:

- OpenAI `gpt-oss` fine-tuning cookbook: https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers
- OpenAI `gpt-oss-20b` model page: https://developers.openai.com/api/docs/models/gpt-oss-20b
- Hugging Face TRL GRPO docs: https://huggingface.co/docs/trl/v0.19.1/grpo_trainer
- Hugging Face Transformers `gpt_oss` docs: https://huggingface.co/docs/transformers/en/model_doc/gpt_oss
