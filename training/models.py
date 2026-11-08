"""Teacher and student networks. Needs torch.

Both are U-Nets that output a delta (contract section 3). The student is limited to what section 8 allows: plain
``Conv``, ``LeakyRelu``, ``Resize`` (linear, half pixel), ``Concat``, ``Add``, ``InstanceNormalization``, ``Tanh``. The teacher
may also use residual blocks and self-attention, and is never exported.

The last layer starts at exactly zero, so an untrained model changes nothing, and zero stays representable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def _activation(kind: str) -> nn.Module:
    # LeakyReLU by default. Measured on the stand-in pairs: a plain ReLU student without normalization went dead under the
    # zero-pulling region loss (the output became one constant) and stayed there. Both are allowed ops (contract section 8).
    return nn.ReLU() if kind == "relu" else nn.LeakyReLU(0.1)


def _norm(kind: str, channels: int) -> nn.Module:
    # Instance normalization only: BatchNorm depends on a batch that the stateless runtime does not have.
    return nn.InstanceNorm2d(channels, affine=True) if kind == "instance" else nn.Identity()


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str, residual: bool, activation: str = "leaky") -> None:
        super().__init__()
        self.act = _activation(activation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = _norm(norm, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = _norm(norm, out_channels)
        self.skip: nn.Module | None = None
        if residual:
            self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        if self.skip is not None:
            h = h + self.skip(x)
        return self.act(h)


class SelfAttention(nn.Module):
    """Teacher only: whole-face context at the bottleneck."""

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return x + self.scale * attended.transpose(1, 2).reshape(b, c, h, w)


class DeltaUNet(nn.Module):
    """Input [N, 3, S, S] in [0, 1], BGR. Output [N, 3, S, S] in [-1, 1], BGR: the change to add."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        channels = [config.base * 2**i for i in range(config.depth + 1)]

        def stack(in_channels: int, out_channels: int) -> nn.Sequential:
            blocks: list[nn.Module] = [ConvBlock(in_channels, out_channels, config.norm, config.residual, config.activation)]
            blocks += [
                ConvBlock(out_channels, out_channels, config.norm, config.residual, config.activation)
                for _ in range(config.blocks - 1)
            ]
            return nn.Sequential(*blocks)

        self.first = stack(3, channels[0])
        self.down = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels[i], channels[i], 3, stride=2, padding=1), _activation(config.activation), stack(channels[i], channels[i + 1]))
            for i in range(config.depth)
        )
        self.middle: nn.Module = SelfAttention(channels[-1]) if config.attention else nn.Identity()
        self.up = nn.ModuleList(stack(channels[i + 1] + channels[i], channels[i]) for i in reversed(range(config.depth)))
        self.last = nn.Conv2d(channels[0], 3, 3, padding=1)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = self.first(x)
        for down in self.down:
            skips.append(h)
            h = down(h)
        h = self.middle(h)
        for up in self.up:
            h = F.interpolate(h, scale_factor=2.0, mode="bilinear", align_corners=False)
            h = up(torch.cat([h, skips.pop()], dim=1))
        return torch.tanh(self.last(h))


class Float32Edges(nn.Module):
    """float16 inside, float32 at the graph's edges (contract section 8)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.half()).float()


def weights_megabytes(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 1e6
