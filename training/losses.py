"""Losses. Needs torch. All tensors are NCHW, BGR, as the model sees them.

Must also run under ``torch-directml`` (torch 2.4.1): no FFT and no complex numbers anywhere in this file (contract 1.7).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig

_EPS = 1e-6


def bgr_to_lab(image: torch.Tensor) -> torch.Tensor:
    """sRGB-encoded BGR in [0, 1] to Lab (D65), scaled so that L runs over [0, 1] and a, b over about [-1, 1]."""
    clipped = image.clamp(0.0, 1.0)
    linear = torch.where(clipped > 0.04045, ((clipped.clamp(min=0.04045) + 0.055) / 1.055) ** 2.4, clipped / 12.92)
    b, g, r = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    # sRGB to XYZ, each row divided by the white point. Spelled out: this also has to run on torch-directml.
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > 0.008856, t.clamp(min=0.008856) ** (1.0 / 3.0), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    return torch.cat([1.16 * fy - 0.16, 5.0 * (fx - fy), 2.0 * (fy - fz)], dim=1)


def luminance(image: torch.Tensor) -> torch.Tensor:
    return 0.114 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.299 * image[:, 2:3]


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the mask. An empty mask gives 0, not NaN."""
    return (values * mask).sum() / (mask.sum() * values.shape[1] + _EPS)


def lab_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((bgr_to_lab(a) - bgr_to_lab(b)).abs(), mask)


def gradient_l1(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Luminance gradients at full and half size. Keeps lines (a liner, a lip edge) from being averaged away. No weights."""
    total = a.new_zeros(())
    x, y, m = luminance(a), luminance(b), mask
    for _ in range(2):
        for dim in (2, 3):
            dx, dy = x.diff(dim=dim), y.diff(dim=dim)
            weight = m.narrow(dim, 0, m.shape[dim] - 1)
            total = total + masked_mean((dx - dy).abs(), weight)
        x, y, m = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2), F.avg_pool2d(m, 2)
    return total


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """``image`` sampled at each pixel plus ``flow`` (pixels, [N, 2, H, W]). The same as ``samples.warp_by_flow``."""
    _, _, h, w = image.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    grid_x = 2.0 * (xs + 0.5 + flow[:, 0]) / w - 1.0
    grid_y = 2.0 * (ys + 0.5 + flow[:, 1]) / h - 1.0
    return F.grid_sample(image, torch.stack([grid_x, grid_y], dim=3), mode="bilinear", padding_mode="border", align_corners=False)


def temporal_loss(
    delta: torch.Tensor,
    next_delta: torch.Tensor,
    target: torch.Tensor,
    next_target: torch.Tensor,
    flow: torch.Tensor,
    confidence: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """The delta must change from frame to frame no more than the target's does, plus a small pull towards not changing.

    The crop already aligns the frames; ``flow`` is what motion remains. Where the flow failed, the pixel counts less.
    """
    moved = (warp(delta, flow) - next_delta).abs()
    allowed = (warp(target, flow) - next_target).abs()
    return masked_mean(F.relu(moved - allowed) + 0.1 * moved, mask * confidence)


def low_frequency_loss(delta: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Identity guard: fine detail beyond what the target has is what changes a face's features. Only the excess counts."""

    def detail(x: torch.Tensor) -> torch.Tensor:
        return (x - F.avg_pool2d(x, 5, stride=1, padding=2)).abs()

    return masked_mean(F.relu(detail(delta) - detail(target)), mask)


def magnitude_loss(delta: torch.Tensor, max_abs_delta: float, mask: torch.Tensor) -> torch.Tensor:
    """Identity guard: mean size of the delta, and strongly anything the runtime's clamp would cut."""
    return masked_mean(delta.abs() + 10.0 * F.relu(delta.abs() - max_abs_delta), mask)


def region_loss(delta: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Where the runtime's weight is 0 the delta must already be 0: a model must not lean on the mask (contract section 10).

    Per pixel of the crop, not per pixel of the forbidden area: otherwise the few pixels of a closed mouth would pull as
    hard as the whole background and drown everything else out. The loss weight then says how much more a pixel counts here.
    """
    return (delta.abs().mean(dim=1, keepdim=True) * (1.0 - weight)).mean()


def identity_loss(embedder: nn.Module, output: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Optional. Cosine distance between embeddings of the face with and without the delta. RGB, as such models expect."""
    with torch.no_grad():
        wanted = F.normalize(embedder(reference.flip(1)), dim=1)
    return (1.0 - (F.normalize(embedder(output.clamp(0.0, 1.0).flip(1)), dim=1) * wanted).sum(dim=1)).mean()


class PerceptualLoss(nn.Module):
    """Optional: VGG16 features, with weights from a local file only. Nothing is ever downloaded here.

    Whether a given weights file may be used is for its source's licence, to be checked by whoever passes it (see
    docs/plan-b-training.md). A loss weight above 0 without a file is an error, not a silent switch to something else.
    """

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)
    _LAYERS = (3, 8, 15, 22)

    def __init__(self, weights_path: str) -> None:
        super().__init__()
        from torchvision.models import vgg16

        network = vgg16(weights=None)
        network.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.features = network.features[: self._LAYERS[-1] + 1].eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self._STD).view(1, 3, 1, 1))

    def _activations(self, image: torch.Tensor) -> list[torch.Tensor]:
        x = (image.flip(1).clamp(0.0, 1.0) - self.mean) / self.std
        out = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self._LAYERS:
                out.append(x)
        return out

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # The mask is applied to the image: masked pixels show the target, so they give no difference.
        blended = output * mask + target * (1.0 - mask)
        with torch.no_grad():
            wanted = self._activations(target)
        return sum(F.l1_loss(a, b) for a, b in zip(self._activations(blended), wanted))


@dataclass
class Batch:
    """A training batch on the device. Two frames: t and t + 1."""

    input: torch.Tensor  # [N, 2, 3, S, S]
    delta: torch.Tensor  # [N, 2, 3, S, S], the clean target
    weight: torch.Tensor  # [N, 2, 1, S, S]
    flow: torch.Tensor  # [N, 2, S, S]
    flow_confidence: torch.Tensor  # [N, 1, S, S]
    has_next: torch.Tensor  # [N]


def compute_losses(
    prediction: torch.Tensor,
    batch: Batch,
    config: LossConfig,
    window: torch.Tensor,
    max_abs_delta: float,
    strengths: torch.Tensor,
    teacher_delta: torch.Tensor | None = None,
    perceptual: nn.Module | None = None,
    embedder: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Every term, weighted, plus ``"total"``. ``prediction`` is [2N, 3, S, S], frames t then t + 1 interleaved by pair.

    ``window`` is the runtime's edge window [1, 1, S, S]; ``strengths`` [N, 1, 1, 1] the random slider position.
    """
    n, _, c, s, _ = batch.input.shape
    x = batch.input.reshape(2 * n, c, s, s)
    target = batch.delta.reshape(2 * n, c, s, s)
    weight = batch.weight.reshape(2 * n, 1, s, s)
    mask = weight * window
    terms: dict[str, torch.Tensor] = {}

    def term(name: str, factor: float, compute) -> None:
        if factor > 0.0:
            terms[name] = factor * compute()

    term("lab", config.lab, lambda: lab_l1(x + prediction, x + target, mask))
    strength = strengths.repeat_interleave(2, dim=0)
    term("lab_partial", config.lab_partial, lambda: lab_l1(x + strength * prediction, x + strength * target, mask))
    term("detail", config.detail, lambda: gradient_l1(x + prediction, x + target, mask))
    term("region", config.region, lambda: region_loss(prediction, weight))
    term("magnitude", config.magnitude, lambda: magnitude_loss(prediction, max_abs_delta, mask))
    term("low_frequency", config.low_frequency, lambda: low_frequency_loss(prediction, target, mask))

    paired = batch.has_next.view(n, 1, 1, 1)
    if float(batch.has_next.sum()) > 0.0:
        first, second = prediction[0::2], prediction[1::2]
        first_target, second_target = target[0::2], target[1::2]
        next_mask = mask[1::2] * paired
        term(
            "temporal",
            config.temporal,
            lambda: temporal_loss(first, second, first_target, second_target, batch.flow, batch.flow_confidence, next_mask),
        )
        # The runtime draws the newest finished delta, normally the previous frame's, through the current landmarks.
        term("lag", config.lag, lambda: masked_mean((first - second_target).abs(), next_mask))

    if perceptual is not None:
        term("perceptual", config.perceptual, lambda: perceptual(x + prediction, x + target, mask))
    if embedder is not None:
        term("identity", config.identity, lambda: identity_loss(embedder, x + prediction, x))
    if teacher_delta is not None:
        term("distill", config.distill, lambda: masked_mean((prediction - teacher_delta).abs(), mask))

    terms["total"] = sum(terms.values())
    return terms
