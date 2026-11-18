"""AdaFace IR-50 / IR-SE50 50-layer Improved Residual Face Recognition Backbone (Milestone P1.1).

Architecture Reference & Attribution:
- Paper: Minchul Kim, Anil K. Jain, Xiaoming Liu, "AdaFace: Quality Adaptive Margin for Face
  Recognition", IEEE/CVF CVPR 2022 (arXiv:2204.00964).
- Upstream Repositories:
  * https://github.com/mk-minchul/AdaFace (`net.py`, MIT License)
  * https://huggingface.co/minchul/cvlface_adaface_ir50_webface4m (`models/iresnet/model.py`,
    commit `60a65befbcf7e19284c4f3ac730f56867ed29594`)
- Backbone Classification: `IR-50` (50-layer Improved Residual Network for 112x112 face crops,
  distinct from standard ImageNet bottleneck ResNet-50):
  * Stem: `Conv2d(3, 64, 3x3, stride=1, padding=1, bias=False) -> BatchNorm2d(64) -> PReLU(64)`
  * Stage layout: `[3, 4, 14, 3]` = 24 `BasicBlockIR` residual units (`24 * 2 = 48` conv layers)
    with channels `[64, 128, 256, 512]`, `MaxPool2d(1, stride)` shortcut when `in_c == depth`
    else `Conv1x1 + BatchNorm2d`.
  * Output head: `BatchNorm2d(512) -> Dropout(0.4) -> Flatten -> Linear(25088, 512) -> BatchNorm1d(512, affine=False)`
  * Total parameterized weight layers: `1 (stem) + 48 (body) + 1 (FC) = 50` layers (`43,585,600` params).
  * Forward output: Raw pre-L2-normalization 512-D embedding tensor `[batch, 512]`. Downstream
    L2 normalization is performed explicitly by the recognition adapter / pipeline.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    MaxPool2d,
    Module,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
)


class Flatten(Module):
    """Contiguous-safe tensor flatten (`[B, C, H, W] -> [B, C*H*W]`) compatible with ONNX export."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, 1)


class SEModule(Module):
    """Squeeze-and-Excitation channel attention module used only in `ir_se` variants."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BasicBlockIR(Module):
    """2-convolution Improved Residual unit (`BN-Conv3x3-BN-PReLU-Conv3x3-BN`) for IR-18..IR-100."""

    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer: Module = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class BasicBlockIRSE(BasicBlockIR):
    """Improved Residual unit with Squeeze-and-Excitation (`IR-SE50`)."""

    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


_BlockSpec = namedtuple("_BlockSpec", ["in_channel", "depth", "stride"])


def _get_block(in_channel: int, depth: int, num_units: int, stride: int = 2) -> list[_BlockSpec]:
    return [_BlockSpec(in_channel, depth, stride)] + [
        _BlockSpec(depth, depth, 1) for _ in range(num_units - 1)
    ]


def _get_blocks(num_layers: int) -> list[list[_BlockSpec]]:
    if num_layers == 50:
        return [
            _get_block(in_channel=64, depth=64, num_units=3),
            _get_block(in_channel=64, depth=128, num_units=4),
            _get_block(in_channel=128, depth=256, num_units=14),
            _get_block(in_channel=256, depth=512, num_units=3),
        ]
    raise ValueError(f"Unsupported num_layers={num_layers}; Milestone P1.1 implements 50-layer IR-50.")


class AdaFaceIRBackbone(Module):
    """50-layer Improved Residual (`IR-50` / `IR-SE50`) face recognition backbone."""

    def __init__(
        self,
        input_size: tuple[int, int] = (112, 112),
        num_layers: int = 50,
        mode: str = "ir",
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        if tuple(input_size) not in {(112, 112), (224, 224)}:
            raise ValueError(f"input_size must be (112, 112) or (224, 224); got {input_size}")
        if num_layers != 50:
            raise ValueError(f"Expected num_layers=50; got {num_layers}")
        if mode not in {"ir", "ir_se"}:
            raise ValueError(f"mode must be 'ir' or 'ir_se'; got '{mode}'")

        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.num_layers = num_layers
        self.mode = mode
        self.output_dim = int(output_dim)

        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )

        unit_module = BasicBlockIR if mode == "ir" else BasicBlockIRSE
        output_channel = 512
        spatial = 7 if self.input_size[0] == 112 else 14

        self.output_layer = Sequential(
            BatchNorm2d(output_channel),
            Dropout(0.4),
            Flatten(),
            Linear(output_channel * spatial * spatial, self.output_dim),
            BatchNorm1d(self.output_dim, affine=False),
        )

        modules: list[Module] = []
        for block in _get_blocks(num_layers):
            for spec in block:
                modules.append(unit_module(spec.in_channel, spec.depth, spec.stride))
        self.body = Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x


def build_ir50_backbone(
    input_size: tuple[int, int] = (112, 112),
    output_dim: int = 512,
    mode: str = "ir",
) -> AdaFaceIRBackbone:
    """Instantiate an `IR-50` (`mode='ir'`) or `IR-SE50` (`mode='ir_se'`) PyTorch module."""
    return AdaFaceIRBackbone(
        input_size=input_size,
        num_layers=50,
        mode=mode,
        output_dim=output_dim,
    )


def extract_backbone_state_dict(raw_checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Normalize official CVLFace (`net.*`) or AdaFace Lightning (`model.*`) state dicts."""
    if not isinstance(raw_checkpoint, dict):
        raise ValueError(f"Expected checkpoint state_dict mapping; got {type(raw_checkpoint).__name__}")

    state = raw_checkpoint.get("state_dict", raw_checkpoint)
    if not isinstance(state, dict) or len(state) == 0:
        raise ValueError("Checkpoint state_dict is empty or malformed.")

    # AdaFace Lightning `.ckpt` includes `head.kernel`; filter to `model.*` backbone keys when present
    if any(k.startswith("model.") for k in state):
        cleaned = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    elif any(k.startswith("net.") for k in state):
        cleaned = {(k[len("net.") :] if k.startswith("net.") else k): v for k, v in state.items()}
    else:
        cleaned = dict(state)

    return cleaned


def load_adaface_ir50_checkpoint(
    checkpoint_path: str | Path,
    input_size: tuple[int, int] = (112, 112),
    output_dim: int = 512,
    mode: str = "ir",
    strict: bool = True,
) -> AdaFaceIRBackbone:
    """Load an official AdaFace / CVLFace IR-50 PyTorch checkpoint onto CPU in eval mode."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"AdaFace IR-50 checkpoint not found: {path}")

    # Official CVLFace `model.pt` stores `state_dict` tensors cleanly; try weights_only=True first,
    # falling back to weights_only=False only if the upstream checkpoint embeds an OmegaConf dict.
    try:
        raw = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001
        raw = torch.load(str(path), map_location="cpu", weights_only=False)

    clean_sd = extract_backbone_state_dict(raw)
    model = build_ir50_backbone(input_size=input_size, output_dim=output_dim, mode=mode)
    model.load_state_dict(clean_sd, strict=strict)
    model.eval()
    return model
