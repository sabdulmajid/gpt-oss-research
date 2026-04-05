# Setup And Portability

This repository is now portable by default:

- artifact paths are relative to the repository
- model and cache locations can be configured with environment variables
- benchmark control scripts derive the repository root automatically

## Recommended Environment Variables

Set these before running larger experiments on a new cluster or workstation:

```bash
export GPT_OSS_20B_PATH=/path/to/gpt-oss-20b
export GPT_OSS_120B_PATH=/path/to/gpt-oss-120b
export HF_CACHE_DIR=/path/to/huggingface/hub
```

Defaults if unset:

- `GPT_OSS_20B_PATH` -> `openai/gpt-oss-20b`
- `GPT_OSS_120B_PATH` -> `openai/gpt-oss-120b`
- `HF_CACHE_DIR` -> `~/.cache/huggingface/hub`

## Basic Bring-Up

```bash
python -m pip install -e .[dev]
make smoke
```

## Typical Workflow

Materialize the benchmark corpora:

```bash
make materialize-sft-benchmark
make materialize-grpo-benchmark
```

Check GPU readiness:

```bash
make gpu-diagnose
```

Start the benchmark runner:

```bash
make benchmark-start
make benchmark-status
```

Tail the benchmark log:

```bash
make benchmark-tail
```

## Notes

- The benchmark runner waits for CUDA automatically if the node is not ready yet.
- All generated checkpoints, eval outputs, and logs stay under `artifacts/`.
- Historical reports under `reports/` use repository-relative paths where possible.
