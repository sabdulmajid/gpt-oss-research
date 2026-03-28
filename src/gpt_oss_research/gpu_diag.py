from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


INSTALLED_MODULE_PACKAGE_RE = re.compile(
    r"^ii\s+(linux-modules-nvidia-[^\s]+-(?P<kernel>\d+\.\d+\.\d+-\d+-generic))\s+"
)
INSTALLED_DRIVER_PACKAGE_RE = re.compile(r"^ii\s+(nvidia-driver-(?P<variant>[^\s]+))\s+")


def _run_command(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def parse_installed_nvidia_module_packages(dpkg_output: str) -> list[dict[str, str]]:
    packages: list[dict[str, str]] = []
    for line in dpkg_output.splitlines():
        match = INSTALLED_MODULE_PACKAGE_RE.match(line)
        if match:
            packages.append(
                {
                    "package": match.group(1),
                    "kernel_release": match.group("kernel"),
                }
            )
    return packages


def parse_installed_nvidia_driver_packages(dpkg_output: str) -> list[dict[str, str]]:
    packages: list[dict[str, str]] = []
    for line in dpkg_output.splitlines():
        match = INSTALLED_DRIVER_PACKAGE_RE.match(line)
        if match:
            packages.append(
                {
                    "package": match.group(1),
                    "variant": match.group("variant"),
                }
            )
    return packages


def summarize_gpu_state(diagnostics: dict[str, Any]) -> str:
    if diagnostics["ready"]:
        return (
            f"CUDA ready on kernel {diagnostics['kernel_release']}; "
            f"torch sees {diagnostics['torch_device_count']} device(s)"
        )

    installed_kernels = [item["kernel_release"] for item in diagnostics["installed_nvidia_module_packages"]]
    reasons: list[str] = [f"kernel={diagnostics['kernel_release']}"]
    if not diagnostics["nvidia_smi_ok"]:
        reasons.append("nvidia-smi failed")
    if not diagnostics["device_nodes"]:
        reasons.append("no /dev/nvidia*")
    if diagnostics["torch_cuda_available"] is False:
        reasons.append("torch cuda unavailable")
    if installed_kernels and diagnostics["kernel_release"] not in installed_kernels:
        reasons.append(f"installed NVIDIA modules only for {installed_kernels}")
    if diagnostics["suggested_module_meta_package"]:
        reasons.append(f"suggested root fix: apt install {diagnostics['suggested_module_meta_package']}")
    return "; ".join(reasons)


def recommended_actions(diagnostics: dict[str, Any]) -> list[str]:
    if diagnostics["ready"]:
        return ["No recovery action needed."]

    actions: list[str] = []
    installed_kernels = [item["kernel_release"] for item in diagnostics["installed_nvidia_module_packages"]]
    if installed_kernels:
        actions.append(
            "Quick recovery: reboot into a kernel that already has matching NVIDIA modules installed, "
            f"for example `{installed_kernels[-1]}` if it remains in GRUB."
        )

    if diagnostics["suggested_module_meta_package"] and diagnostics["driver_meta_package"]:
        actions.append(
            "Durable recovery: as root, run "
            f"`sudo apt-get update && sudo apt-get install --yes "
            f"{diagnostics['suggested_module_meta_package']} {diagnostics['driver_meta_package']} && sudo reboot`."
        )
    else:
        actions.append("Durable recovery requires root to install the matching NVIDIA kernel modules and reboot.")

    actions.append("After recovery, validate with `nvidia-smi`, `python scripts/gpu_diagnostics.py --summary`, and `make benchmark-status`.")
    return actions


def collect_gpu_diagnostics() -> dict[str, Any]:
    kernel_release = _run_command(["uname", "-r"])["stdout"].strip()
    dpkg_result = _run_command(["dpkg", "-l"])
    module_packages = parse_installed_nvidia_module_packages(dpkg_result["stdout"])
    driver_packages = parse_installed_nvidia_driver_packages(dpkg_result["stdout"])
    driver_meta_package = driver_packages[0]["package"] if driver_packages else None
    driver_variant = driver_packages[0]["variant"] if driver_packages else None
    suggested_module_meta_package = f"linux-modules-nvidia-{driver_variant}-generic" if driver_variant else None

    nvidia_smi = _run_command(["nvidia-smi"])
    lsmod = _run_command(["lsmod"])
    loaded_modules = []
    if lsmod["returncode"] == 0:
        loaded_modules = [
            line.split()[0]
            for line in lsmod["stdout"].splitlines()[1:]
            if line and (line.startswith("nvidia") or line.startswith("nouveau"))
        ]

    device_nodes = sorted(str(path) for path in Path("/dev").glob("nvidia*"))
    torch_probe = _run_command(
        [
            "python",
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'cuda_available': bool(torch.cuda.is_available()), "
                "'device_count': int(torch.cuda.device_count())}))"
            ),
        ]
    )
    torch_cuda_available = False
    torch_device_count = 0
    if torch_probe["returncode"] == 0 and torch_probe["stdout"].strip():
        import json

        payload = json.loads(torch_probe["stdout"])
        torch_cuda_available = bool(payload["cuda_available"])
        torch_device_count = int(payload["device_count"])

    ready = bool(nvidia_smi["returncode"] == 0 and torch_cuda_available and torch_device_count > 0 and device_nodes)

    diagnostics = {
        "ready": ready,
        "kernel_release": kernel_release,
        "device_nodes": device_nodes,
        "loaded_modules": loaded_modules,
        "nvidia_smi_ok": nvidia_smi["returncode"] == 0,
        "nvidia_smi_returncode": nvidia_smi["returncode"],
        "nvidia_smi_stderr": nvidia_smi["stderr"].strip(),
        "torch_cuda_available": torch_cuda_available,
        "torch_device_count": torch_device_count,
        "installed_nvidia_module_packages": module_packages,
        "installed_nvidia_driver_packages": driver_packages,
        "driver_meta_package": driver_meta_package,
        "suggested_module_meta_package": suggested_module_meta_package,
    }
    diagnostics["summary"] = summarize_gpu_state(diagnostics)
    diagnostics["recommended_actions"] = recommended_actions(diagnostics)
    return diagnostics
