# Reports

This directory stores:

- experiment reports under `reports/experiments/`
- project status snapshots under `reports/status/`
- reusable templates under `reports/templates/`

Every meaningful experiment should produce a report under `reports/experiments/` using the template in `reports/templates/benchmark_report.md`.

Minimum requirements:

- exact base model
- exact adapter strategy
- exact dataset manifest path
- exact git SHA
- exact seed
- exact decoding settings
- saved benchmark outputs
