"""Registers the learned finishing stage. The code lives in ``makeup_cam.neural`` (see its docstring)."""

from __future__ import annotations

from ..neural.stage import NeuralFinishEffect
from .base import register_effect

NeuralEffect = register_effect(NeuralFinishEffect)

__all__ = ["NeuralEffect", "NeuralFinishEffect"]
