# GPU Recovery

This repository now includes a benchmark pipeline that can wait for CUDA and continue automatically. If the node loses the NVIDIA driver stack, the benchmark runner will not make progress until the host is repaired.

## Confirmed Failure Mode On This Host

The current host state is:

- running kernel: `6.8.0-87-generic`
- installed NVIDIA kernel module package: `linux-modules-nvidia-575-open-6.8.0-79-generic`
- missing matching NVIDIA module for `6.8.0-87-generic`
- `nvidia-smi` fails
- `/dev/nvidia*` is absent
- PyTorch reports zero CUDA devices

This is an OS-level mismatch, not a Python or repository bug.

## What You Can Do Without Root

You can diagnose and monitor the state:

```bash
python scripts/gpu_diagnostics.py --summary
make benchmark-status
tail -f artifacts/logs/benchmark_pipeline_v1.log
```

You cannot install kernel modules, load unsigned modules into the running kernel, or reboot into a different kernel without elevated privileges.

## Fast Recovery

If the machine still has `6.8.0-79-generic` installed, booting back into that kernel should restore the currently installed NVIDIA module set quickly.

Use the GRUB advanced boot menu and select:

- `Ubuntu, with Linux 6.8.0-79-generic`

## Durable Recovery

Upgrade the NVIDIA module metapackage so it matches the current generic kernel line:

```bash
sudo apt-get update
sudo apt-get install --yes linux-modules-nvidia-575-open-generic nvidia-driver-575-open
sudo reboot
```

That should install the `6.8.0-87` NVIDIA module package for the currently running generic kernel series.

## Post-Recovery Validation

After reboot:

```bash
nvidia-smi
python scripts/gpu_diagnostics.py --summary
make benchmark-status
```

Expected healthy signals:

- `nvidia-smi` succeeds
- `/dev/nvidia0` and related device nodes exist
- `torch.cuda.is_available()` is true
- the benchmark pipeline transitions from `waiting_for_cuda` to `running`

## Benchmark Runner

The benchmark runner is managed with `tmux`:

```bash
make benchmark-start
make benchmark-status
make benchmark-tail
```

The default session name is `benchmark-pipeline`.
