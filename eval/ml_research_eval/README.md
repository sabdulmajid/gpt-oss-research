# Internal ML Research Eval

This is the internal benchmark path required by `AGENTS.md`.

Current scope:

- 51 CPU-executable tasks with prompts, reference solutions, and tests
- PyTorch- and ML-research-heavy function implementation and debugging tasks
- explicit benchmark-bucket metadata for composition tracking

Current benchmark buckets:

- `tensor_api`
- `training_pipeline`
- `transformers_accelerate_fsdp_mixed_precision`
- `debugging_performance`

This is still below the long-term 200-500 task target from `AGENTS.md`, but it is no longer a seed stub. It is the first serious internal eval slice that can support real base-vs-SFT-vs-GRPO comparisons.
