from gpt_oss_research.gpu_diag import (
    parse_installed_nvidia_driver_packages,
    parse_installed_nvidia_module_packages,
    summarize_gpu_state,
)


def test_parse_installed_nvidia_module_packages():
    dpkg_output = """
ii  linux-modules-nvidia-575-open-6.8.0-79-generic 6.8.0-79.79 amd64
rc  linux-modules-nvidia-575-open-6.8.0-78-generic 6.8.0-78.78 amd64
ii  linux-image-6.8.0-87-generic 6.8.0-87.88 amd64
"""
    packages = parse_installed_nvidia_module_packages(dpkg_output)
    assert packages == [
        {
            "package": "linux-modules-nvidia-575-open-6.8.0-79-generic",
            "kernel_release": "6.8.0-79-generic",
        }
    ]


def test_parse_installed_nvidia_driver_packages():
    dpkg_output = """
ii  nvidia-driver-575-open 575.64.03-0ubuntu0.24.04.1 amd64
ii  nvidia-utils-575 575.64.03-0ubuntu0.24.04.1 amd64
"""
    packages = parse_installed_nvidia_driver_packages(dpkg_output)
    assert packages == [
        {
            "package": "nvidia-driver-575-open",
            "variant": "575-open",
        }
    ]


def test_summarize_gpu_state_reports_kernel_mismatch():
    diagnostics = {
        "ready": False,
        "kernel_release": "6.8.0-87-generic",
        "device_nodes": [],
        "nvidia_smi_ok": False,
        "torch_cuda_available": False,
        "installed_nvidia_module_packages": [
            {
                "package": "linux-modules-nvidia-575-open-6.8.0-79-generic",
                "kernel_release": "6.8.0-79-generic",
            }
        ],
        "suggested_module_meta_package": "linux-modules-nvidia-575-open-generic",
    }
    summary = summarize_gpu_state(diagnostics)
    assert "6.8.0-87-generic" in summary
    assert "6.8.0-79-generic" in summary
    assert "linux-modules-nvidia-575-open-generic" in summary
