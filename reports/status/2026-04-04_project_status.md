---
date_utc: 2026-04-04
status: current_snapshot
---

# Project Status

This report summarizes what has been implemented so far, what has been validated, and what remains open.

## What Is Implemented

- dataset hypotheses and frozen manifests for the initial SFT and GRPO mixes
- dataset materialization, filtering, validation, and split tooling
- SFT and GRPO training entrypoints for `gpt-oss-20b`
- a 51-task internal `ML Research Eval` harness with executable tests
- benchmark reporting templates and experiment reports
- GPU diagnostics and benchmark pipeline control scripts

## Validated Results

- internal reference suite: `51/51` pass
- larger SFT benchmark corpus: `977` total, `938` train, `39` validation
- larger GRPO benchmark corpus: `383` total, `363` train, `20` validation
- smoke validation passes
- pytest validation passes

## Completed Experiment

One short end-to-end SFT pilot completed on `gpt-oss-20b`.

Measured result on the early pilot eval:

- base `gpt-oss-20b`: `1/4`
- SFT pilot adapter: `3/4`

This is useful evidence that the pipeline is real, but it is not a broad benchmark claim.

## What Is Not Done Yet

- no completed large base-vs-SFT-vs-GRPO benchmark run on the 51-task internal suite
- no external holdout benchmark results yet
- no ablation results yet for alternate mixture weightings
- no benchmarked GRPO result yet

## Current Blocker

The original workstation lost its NVIDIA driver/kernel match during unattended Ubuntu upgrades, so the larger benchmark pipeline could not complete there. The repo now includes diagnostics and recovery documentation, and it is ready to be pulled onto a healthy cluster or workstation.

## Recommended Next Step

Run the benchmark pipeline on a healthy GPU machine:

1. materialize the benchmark corpora if needed
2. run base `gpt-oss-20b` on the 51-task internal eval
3. run the SFT benchmark config and re-evaluate
4. run the GRPO benchmark config from the best SFT adapter and re-evaluate
5. publish the first full benchmark report with stored outputs
