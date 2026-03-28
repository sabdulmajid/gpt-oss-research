---
experiment_name: ml_research_eval_reference_51_v1
date_utc: 2026-03-28
status: completed
suite_path: eval/ml_research_eval/tasks
task_count: 51
bucket_summary:
  tensor_api: 13
  training_pipeline: 13
  transformers_accelerate_fsdp_mixed_precision: 13
  debugging_performance: 12
reference_pass_rate: 1.0
reference_failed: 0
output_json: /pub7/neel2/gpt-oss-research/artifacts/eval/internal_eval_reference_51_v1.json
notes:
  - This is a reference-suite validation report, not a model benchmark result.
  - GPU-backed model reruns were blocked at report time by an NVIDIA driver outage on the node.
---

# Summary

The internal `ML Research Eval` harness is now a real 51-task benchmark slice rather than a seed suite.

Every task in the current suite has:

- `metadata.yaml`
- `prompt.md`
- `reference_solution.py`
- `tests.py`

The full reference run passed `51/51`.

# Composition

The current suite is intentionally bucketed so future reports can track whether comparisons are moving the right kinds of behavior:

- `tensor_api`: 13
- `training_pipeline`: 13
- `transformers_accelerate_fsdp_mixed_precision`: 13
- `debugging_performance`: 12

This is still below the long-term `200-500` task target in `AGENTS.md`, but it is large enough to stop treating the internal eval as a toy harness.

# Validation

Validation completed with:

- `python scripts/run_internal_eval.py --tasks-root eval/ml_research_eval/tasks --use-reference --output artifacts/eval/internal_eval_reference_51_v1.json`
- `python -m pytest -q tests/test_internal_eval.py`

At the time of this report, the benchmark harness was healthy but the GPU node was not. The node had no active NVIDIA kernel module and no `/dev/nvidia*` devices, so larger base/SFT/GRPO model reruns were not launched from this host state.
