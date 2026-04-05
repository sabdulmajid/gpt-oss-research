# Project Strategy

## Objective

The goal of this project is to build a `gpt-oss` model that is genuinely useful for ML research and PyTorch-heavy coding.

Target behaviors:

- strong PyTorch implementation ability
- strong debugging ability for training loops, autograd, mixed precision, distributed training, and data pipelines
- strong ML systems reasoning
- strong executable correctness on coding tasks

## Model Strategy

- Primary trainable model: `gpt-oss-20b`
- Primary teacher/reference model: `gpt-oss-120b`
- Order of work: SFT first, then benchmark honestly, then test GRPO on verifiable tasks

This project treats `gpt-oss-120b` as a teacher and reference model unless there is hard evidence that direct training is practical on the target hardware.

## Data Strategy

The training mix is intentionally biased toward ML-research-relevant behavior:

- filtered PyTorch and ML repository code
- verified executable coding tasks
- PyTorch debugging and issue-style examples
- Hugging Face training-stack code
- a small capped share of synthetic reasoning data

The current SFT and GRPO mixtures are hypotheses, not proven optima.

## Benchmark Discipline

The repo is built around a strict benchmark contract:

- do not train on benchmark test sets
- do not mix evaluation and training splits casually
- do not claim improvement without stored outputs
- do not use cherry-picked prompts as evidence
- compare base, SFT, and GRPO in the same harness

The internal `ML Research Eval` benchmark is mandatory because public coding benchmarks alone do not prove ML-research usefulness.

## Current Benchmark Plan

The first serious experiment ladder is:

1. base `gpt-oss-20b`
2. `gpt-oss-20b` SFT with the current recommended mix
3. PyTorch-heavier and PyTorch-lighter SFT ablations
4. best SFT checkpoint plus GRPO

The benchmark suite should eventually include:

- the internal `ML Research Eval`
- APPS holdout
- TACO holdout
- CodeContests holdout
- Codeforces holdout

## Hardware Reality

The original development machine was a `2 x 96 GB` Blackwell workstation. That is strong enough for iterative `gpt-oss-20b` LoRA-style work, but it should not be used as evidence that `gpt-oss-120b` GRPO is practical by default.
