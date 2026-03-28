# GPT-OSS Research Agent Guide

## Mission

This repository exists to build, measure, and document a `gpt-oss`-based model that is genuinely useful as an ML research and PyTorch-heavy coding model.

The target behavior is not "general coding but kind of better." The target behavior is:

- strong PyTorch implementation ability
- strong debugging ability for training loops, autograd, data pipelines, mixed precision, distributed training, and Transformer internals
- strong ML systems reasoning
- strong code correctness on executable tasks

The research direction starts with supervised fine-tuning (SFT). GRPO is a follow-on direction to test once the SFT baseline is clean, stable, and benchmarked.

This file is the operating contract for future agents working in this repo.

## Non-Negotiable Rules

1. Do not claim the model is "better" unless benchmark results prove it.
2. Do not use cherry-picked prompts as evidence.
3. Do not hallucinate dataset quality, benchmark wins, or hardware feasibility.
4. Do not train on benchmark test sets.
5. Do not mix training and evaluation splits casually.
6. Do not treat `gpt-oss-120b` GRPO on current hardware as the default plan.
7. Always write down the exact data mixture, exact config, exact base model, exact adapter strategy, and exact benchmark outputs.
8. Every claim in this repo should be backed by logs, manifests, or citations.

## Current Hardware Reality

Current local hardware:

- 2x NVIDIA RTX PRO Blackwell 6000
- 96 GB VRAM each
- 192 GB total VRAM across 2 GPUs

Important implication:

- `gpt-oss-120b` is reasonable for inference and teacher-model usage on this machine.
- `gpt-oss-120b` is not the default GRPO training target on this machine.
- GRPO adds online generation, trainable state, activations, optimizer state, and reward evaluation overhead. That is substantially harder than inference.
- The practical GRPO target for this hardware is `gpt-oss-20b` with LoRA/QLoRA-style training, while `gpt-oss-120b` should be treated as a teacher, evaluator, or synthetic-data generator unless experiments prove otherwise.

Do not promise `120b` GRPO feasibility on `2x96 GB` without hard experimental evidence.

## Model Strategy

Use a two-track strategy.

### Track A: Trainable model

Primary trainable model:

- `gpt-oss-20b`

Reason:

- It is materially more realistic for SFT and GRPO on current hardware.
- OpenAI's fine-tuning cookbook example is centered on `gpt-oss-20b`.
- This is the right place to iterate quickly, run ablations, and get benchmark movement without wasting weeks on a hardware mismatch.

### Track B: Teacher / reference model

Primary teacher/reference model:

- `gpt-oss-120b`

Use it for:

- generating high-quality synthetic traces and critiques
- producing hard PyTorch debugging examples
- evaluator-style comparisons
- candidate-answer reranking
- internal data generation for ML research tasks

Do not confuse "teacher model exists" with "teacher model must be the train target."

## Training Philosophy

The correct order is:

1. build a clean SFT baseline
2. benchmark it honestly
3. build a verifiable GRPO task set
4. run GRPO only after the SFT baseline is stable
5. compare SFT-only vs SFT+GRPO with the same evaluation harness

GRPO is not a substitute for missing data quality.

If the SFT baseline is noisy, overfit, or poorly specialized for ML research work, GRPO will not rescue it.

## Required Data Format

`gpt-oss` models are trained for the Harmony-style response format. Training data should preserve a structured conversational layout rather than flattening everything into generic plain text.

At a minimum:

- preserve the user request
- preserve the expected final answer
- preserve any reasoning or stepwise structure only where appropriate
- keep execution-grounded coding tasks in a format that supports testing

When producing assistant outputs for training:

- prefer exact code or exact diffs
- prefer explicit error analysis
- prefer grounded debugging steps
- avoid fluffy narration

## Fine-Tuning Approach

### SFT first

Default SFT approach:

- PEFT / LoRA-based tuning on `gpt-oss-20b`
- preserve specialization toward ML research, PyTorch, Transformers, training systems, and code correctness

Key design requirement:

- Because `gpt-oss` is an MoE model, adapter targeting should respect the expert projection structure rather than assuming a generic dense-transformer target list.

### GRPO second

Default GRPO approach:

- run GRPO only on verifiable tasks
- reward execution success, not style alone
- use executable tests/checkers whenever possible
- keep GRPO datasets narrower and cleaner than SFT datasets

For this hardware:

- do not assume a separate vLLM server node
- prefer smaller train targets and colocated or non-vLLM generation paths as needed
- expect `gpt-oss-20b` to be the practical GRPO target

## What This Repo Must Build

This repo should eventually contain:

- dataset manifests
- dataset cleaning and filtering scripts
- SFT configs
- GRPO configs
- benchmark harnesses
- evaluation reports
- experiment logs
- ablation summaries

No experiment should be considered real unless it has:

- a reproducible config
- a frozen data manifest
- an exact git SHA
- saved benchmark outputs

## Dataset Strategy

The model we want is not a generic LeetCode model. It should be a PyTorch and ML research model.

That means the training mix should intentionally overweight:

- PyTorch code
- HF Transformers code
- ML infra code
- issue/debugging examples
- executable coding tasks

and intentionally underweight:

- generic web text
- non-executable prose-only coding data
- algorithm-only data that does not transfer to ML systems work

## Recommended Starting SFT Mix

This is the recommended starting mix to test first. It is not a claimed optimum. It is the first serious hypothesis to benchmark.

| Bucket | Share | Purpose | Primary sources |
|---|---:|---|---|
| Filtered PyTorch / ML repo code | 35% | core specialization | `bigcode/the-stack-v2-dedup`, `codeparrot/github-code`, filtered to PyTorch / HF / ML infra |
| Verified coding tasks | 30% | correctness and structured problem solving | `APPS`, `TACO`, `CodeContests`, `open-r1/codeforces` |
| PyTorch QA / issues / tutorials | 20% | debugging and API fluency | `stack_overflow_pytorch`, `pytorch-discuss-tutorial-1000`, `pytorch-issues` |
| HF / Transformers / model-training code | 10% | research workflow fluency | `huggingface-transformers-code-dataset` plus filtered Transformers repos |
| Synthetic reasoning for code | 5% | carefully capped reasoning augmentation | `nvidia/OpenCodeReasoning` only as a small supplement |

This mix is intentionally conservative on synthetic reasoning data. Synthetic data can help, but it should not dominate until it proves itself in ablations.

## Recommended Starting GRPO Mix

GRPO data should be narrower and more testable than SFT data.

Recommended GRPO mix to test first:

| Bucket | Share | Purpose |
|---|---:|---|
| Internal PyTorch research tasks with executable tests | 35% | direct specialization toward the actual target behavior |
| `open-r1/codeforces` | 25% | strong verifiable coding and algorithmic rigor |
| `deepmind/code_contests` | 20% | robust multi-solution competitive programming tasks |
| `BAAI/TACO` | 10% | natural-language-to-Python synthesis with tests |
| `APPS` | 10% | execution-grounded coding tasks |

Do not use GRPO on datasets that cannot be checked automatically.

## Exact Dataset Guidance

### Core broad-code sources

1. `bigcode/the-stack-v2-dedup`
   - Use as a source of Python and ML repository code.
   - Filter aggressively.
   - Keep only files and repos relevant to PyTorch, Transformers, distributed training, CUDA, data loading, evaluation, and model training workflows.

2. `codeparrot/github-code`
   - Use as a second broad code source.
   - It is useful because it carries repository metadata and language metadata.
   - Filter to Python and ML/PyTorch-heavy repos.

### Verified coding task sources

1. `codeparrot/apps`
   - Strong for executable Python problem-solving.

2. `BAAI/TACO`
   - Good natural-language-to-Python training with tests.

3. `deepmind/code_contests`
   - Good for correctness-focused training and evaluation.

4. `open-r1/codeforces`
   - Very strong for verifiable tasks with official tests/checkers.

5. `nvidia/OpenCodeReasoning`
   - Use carefully and cap the mixture share.
   - Good for reasoning-style coding traces, but it should not replace verified task data.

### PyTorch / ML-research-specific sources

1. `shrinath-suresh/stack_overflow_pytorch`
   - Good for question/answer debugging behavior.
   - Good for API usage, common pitfalls, and error diagnosis.

2. `shrinath-suresh/pytorch-discuss-tutorial-1000`
   - Small but useful as a specialist supplement.

3. `suvadityamuk/huggingface-transformers-code-dataset`
   - Useful for HF training stack fluency.

4. `kye/all-pytorch-code`
   - Worth inspecting, but do not trust blindly.
   - Must be syntax-checked and deduplicated before training.

5. `BigTimeCoderSean/pytorch-issues`
   - Small, but useful for debugging-style examples.

### Important truth about PyTorch-specific public data

There is no single large, clean, canonical public "PyTorch research model" dataset that is sufficient on its own.

Therefore:

- use PyTorch-specific public datasets as specialist data
- build a larger custom PyTorch slice from broad code corpora
- use verified executable tasks to preserve correctness

## How To Build the Custom PyTorch Slice

When filtering broad code corpora, keep Python files and repos matching signals like:

- `import torch`
- `from torch`
- `torch.nn`
- `torch.utils.data`
- `torch.distributed`
- `torch.cuda`
- `torch.autograd`
- `torch.compile`
- `torchvision`
- `torchaudio`
- `transformers`
- `accelerate`
- `datasets`
- `deepspeed`
- `lightning`
- `pytorch_lightning`
- `xformers`
- `flash_attn`
- `triton`

Prefer files and repos with paths or names like:

- `train.py`
- `trainer.py`
- `finetune.py`
- `dataset.py`
- `dataloader.py`
- `modeling_*.py`
- `fsdp`
- `ddp`
- `amp`
- `examples/`
- `tutorials/`
- `benchmarks/`

Reject aggressively:

- vendored dependencies
- generated notebooks with no real code value
- minified or auto-generated code
- non-Python code for the current phase
- duplicate files across forks

## Data Cleaning Requirements

Before any training run:

1. Deduplicate by exact file hash and near-duplicate similarity.
2. Parse Python syntax and drop invalid files for SFT corpora unless the task is explicitly about repair.
3. Normalize licenses and provenance metadata.
4. Strip benchmark contamination where applicable.
5. Remove trivial boilerplate and pure config dumps where they do not teach useful behavior.
6. Separate train, validation, and benchmark holdouts by task source and repository identity.

## Benchmarks That Must Exist

No "this works" claim is allowed without a benchmark suite.

### External benchmarks

Use a mix of:

- LiveCodeBench
- APPS holdout
- TACO holdout
- CodeContests holdout
- Codeforces holdout

Optional secondary benchmark:

- SWE-bench Verified, but only if repository-level code editing is a target behavior and never as sole proof

### Internal benchmark: mandatory

Build an internal `ML Research Eval` suite. This is mandatory because public coding benchmarks alone do not prove PyTorch or ML research usefulness.

Target composition:

- 25% PyTorch tensor/autograd/API tasks
- 25% model-training pipeline tasks
- 25% Transformers / Accelerate / FSDP / mixed-precision tasks
- 25% debugging and performance tasks

Each task should have:

- a prompt
- a reference solution or checker
- executable unit tests
- difficulty label
- topic label

Suggested internal eval size:

- 200 to 500 tasks minimum

## Benchmark Proof Rules

No BS rules:

1. Do not report wins from one prompt.
2. Do not report wins without the baseline model in the same harness.
3. Do not compare across mismatched prompts or mismatched decoding settings.
4. Do not compare across different evaluation sets and call it an improvement.
5. Do not use benchmark test samples in SFT or GRPO training.
6. Do not call the model an "ML research model" unless it improves on the internal `ML Research Eval`.

For every benchmark report, log:

- base model
- adapter type and rank
- sequence length
- train tokens or train examples
- exact dataset mixture
- reward functions
- decoding settings
- seed
- pass@1 or exact evaluation metric

## Required Ablations

The first serious experiment set should include:

1. Base `gpt-oss-20b`
2. `gpt-oss-20b` SFT with the recommended SFT mix
3. `gpt-oss-20b` SFT with less PyTorch-heavy weighting
4. `gpt-oss-20b` SFT with more PyTorch-heavy weighting
5. Best SFT checkpoint plus GRPO

The result should show:

- whether PyTorch-heavy specialization actually helps
- whether GRPO improves verified coding without harming ML research tasks
- whether synthetic reasoning data helps or hurts

## What Agents Must Do

If asked to move this project forward, agents should:

1. preserve the mission: build a real ML research coding model
2. favor measured progress over grand claims
3. start with SFT before GRPO
4. bias training toward `gpt-oss-20b` on this hardware
5. use `gpt-oss-120b` as teacher/reference unless strong evidence justifies a direct train attempt
6. create dataset manifests before launching large runs
7. create benchmark harnesses before claiming success
8. write experiment reports after every meaningful run

If asked to train `gpt-oss-120b` with GRPO directly on the current 2x96 GB setup, agents should push back and explain why that is not the default practical path.

## What Agents Must Not Do

Agents must not:

- claim "best mix" as a proven fact without ablations
- claim benchmark improvement without stored outputs
- mix evaluation data into training
- overindex on synthetic reasoning traces
- optimize only for competitive-programming style tasks
- ignore PyTorch- and ML-systems-specific evals

## Deliverables Expected In This Repo

As the repo grows, it should contain at least:

- `data/` for manifests and filtering scripts
- `configs/` for SFT and GRPO configs
- `eval/` for benchmark harnesses
- `reports/` for experiment writeups
- `artifacts/` or documented paths for adapters and logs

## Source Links

Official `gpt-oss` references:

- OpenAI `gpt-oss-120b` model page: https://developers.openai.com/api/docs/models/gpt-oss-120b
- OpenAI cookbook, fine-tuning with Transformers: https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers
- TRL GRPO docs: https://huggingface.co/docs/trl/v0.19.1/grpo_trainer

Primary dataset references:

- The Stack v2 dedup: https://huggingface.co/datasets/bigcode/the-stack-v2-dedup
- GitHub Code: https://huggingface.co/datasets/codeparrot/github-code
- APPS: https://huggingface.co/datasets/codeparrot/apps
- TACO: https://huggingface.co/datasets/BAAI/TACO
- CodeContests: https://huggingface.co/datasets/deepmind/code_contests
- Open-R1 Codeforces: https://huggingface.co/datasets/open-r1/codeforces
- NVIDIA OpenCodeReasoning: https://huggingface.co/datasets/nvidia/OpenCodeReasoning
- Stack Overflow PyTorch: https://huggingface.co/datasets/shrinath-suresh/stack_overflow_pytorch
- PyTorch Discuss Tutorial 1000: https://huggingface.co/datasets/shrinath-suresh/pytorch-discuss-tutorial-1000
- Hugging Face Transformers code dataset: https://huggingface.co/datasets/suvadityamuk/huggingface-transformers-code-dataset
- All PyTorch code: https://huggingface.co/datasets/kye/all-pytorch-code
- PyTorch issues: https://huggingface.co/datasets/BigTimeCoderSean/pytorch-issues
