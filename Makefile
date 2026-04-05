PYTHON ?= python
MODEL_20B_PATH ?= openai/gpt-oss-20b

.PHONY: manifests validate filter-sample split-sample internal-eval sft-dry-run grpo-dry-run smoke test materialize-sft materialize-grpo materialize-sft-benchmark materialize-grpo-benchmark model-eval benchmark-pipeline gpu-diagnose benchmark-start benchmark-status benchmark-tail

manifests:
	$(PYTHON) scripts/build_manifest.py configs/datasets/sft_starting_mix.yaml data/manifests/sft_starting_mix.manifest.json
	$(PYTHON) scripts/build_manifest.py configs/datasets/grpo_starting_mix.yaml data/manifests/grpo_starting_mix.manifest.json

validate:
	$(PYTHON) scripts/validate_manifest.py configs/datasets/sft_starting_mix.yaml
	$(PYTHON) scripts/validate_manifest.py configs/datasets/grpo_starting_mix.yaml
	$(PYTHON) scripts/validate_manifest.py data/manifests/sft_starting_mix.manifest.json --frozen
	$(PYTHON) scripts/validate_manifest.py data/manifests/grpo_starting_mix.manifest.json --frozen

filter-sample:
	$(PYTHON) scripts/filter_code_slice.py \
		--input data/samples/broad_code_candidates.jsonl \
		--output artifacts/smoke/filtered_code.jsonl \
		--summary-output artifacts/smoke/filter_summary.json

split-sample:
	$(PYTHON) scripts/create_split.py \
		--input artifacts/smoke/filtered_code.jsonl \
		--train-output artifacts/smoke/train.jsonl \
		--val-output artifacts/smoke/val.jsonl

internal-eval:
	$(PYTHON) scripts/run_internal_eval.py \
		--tasks-root eval/ml_research_eval/tasks \
		--use-reference \
		--output artifacts/smoke/internal_eval_report.json

sft-dry-run:
	$(PYTHON) scripts/run_sft.py configs/training/sft_gpt_oss_20b_lora.yaml --dry-run

grpo-dry-run:
	$(PYTHON) scripts/run_grpo.py configs/training/grpo_gpt_oss_20b_lora.yaml --dry-run

materialize-sft:
	$(PYTHON) scripts/materialize_datasets.py configs/materialization/sft_real_pilot.yaml

materialize-grpo:
	$(PYTHON) scripts/materialize_datasets.py configs/materialization/grpo_real_pilot.yaml

materialize-sft-benchmark:
	$(PYTHON) scripts/materialize_datasets.py configs/materialization/sft_benchmark_v1.yaml

materialize-grpo-benchmark:
	$(PYTHON) scripts/materialize_datasets.py configs/materialization/grpo_benchmark_v1.yaml

model-eval:
	$(PYTHON) scripts/run_model_eval.py \
		--model-path $(MODEL_20B_PATH) \
		--tasks-root eval/ml_research_eval/tasks \
		--output artifacts/eval/internal_eval_base_report.json

benchmark-pipeline:
	bash scripts/run_benchmark_pipeline.sh

gpu-diagnose:
	$(PYTHON) scripts/gpu_diagnostics.py --summary

benchmark-start:
	bash scripts/start_benchmark_pipeline_tmux.sh

benchmark-status:
	$(PYTHON) scripts/benchmark_status.py

benchmark-tail:
	tail -n 50 artifacts/logs/benchmark_pipeline_v1.log

smoke:
	$(PYTHON) scripts/smoke_test.py

test:
	pytest
