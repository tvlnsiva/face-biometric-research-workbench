"""Automated tests for PyTorch CPU and CUDA GPU tensor/model inference."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def _make_probe_net() -> nn.Module:
    torch.manual_seed(101)
    net = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 16),
    )
    net.eval()
    return net


def test_pytorch_cpu_inference() -> None:
    net = _make_probe_net().to("cpu")
    x = torch.randn(2, 3, 64, 64, device="cpu")
    with torch.inference_mode():
        out = net(x)
    assert out.device.type == "cpu"
    assert tuple(out.shape) == (2, 16)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU runtime is not available on this host",
)
def test_pytorch_gpu_inference_executes_on_cuda() -> None:
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    cpu_net = _make_probe_net()
    gpu_net = cpu_net.to("cuda:0")

    x_cpu = torch.randn(2, 3, 64, 64, device="cpu")
    x_gpu = x_cpu.to("cuda:0")

    with torch.inference_mode():
        out_gpu = gpu_net(x_gpu)
        torch.cuda.synchronize(0)
        out_cpu = cpu_net.to("cpu")(x_cpu)

    assert out_gpu.is_cuda, "Output tensor is not resident in CUDA memory"
    assert out_gpu.device.type == "cuda"
    assert torch.cuda.max_memory_allocated(0) > 0, "Zero CUDA memory allocated during forward pass"
    assert tuple(out_gpu.shape) == (2, 16)
    assert torch.allclose(out_gpu.cpu(), out_cpu, atol=1e-4, rtol=1e-4)
