"""Checks, clamps, windows and smooths raw model outputs (``docs/dev/contract-plan-b.md`` sections 3, 5 and 7.5).

In its own module because the worker process runs it: there it shares no GIL with the camera thread. At 512 it takes about
1.2 ms, and next to the classic effects that cost them about 0.5 ms per frame when it ran in the app's process.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import geometry

# Contract section 7.5 (provisional): a mean change this large over the crop's center is not a finish.
MAX_MEAN_ABS_DELTA = 0.15


class DeltaFilter:
    """OpenCV rather than numpy: it takes a third of the CPU this way."""

    def __init__(self, size: int, max_abs_delta: float, temporal_alpha: float) -> None:
        self._max = float(max_abs_delta)
        self._alpha = float(temporal_alpha)
        window = geometry.edge_window(size)
        self._window = cv2.merge((window, window, window))
        low = size // 4
        self._center = (slice(low, size - low), slice(low, size - low))
        self._center_values = 3 * (size - 2 * low) ** 2
        self._smoothed: np.ndarray | None = None

    def reset(self) -> None:
        self._smoothed = None

    def __call__(self, raw: np.ndarray) -> np.ndarray | None:
        """``raw``: 3 x S x S. Returns a new S x S x 3 array, or None when the output is not acceptable."""
        delta = cv2.merge((raw[0], raw[1], raw[2]))  # also the copy out of the shared block
        if not all(math.isfinite(total) for total in cv2.sumElems(delta)[:3]):
            return None
        if cv2.norm(delta[self._center], cv2.NORM_L1) / self._center_values > MAX_MEAN_ABS_DELTA:
            return None
        limit = self._max
        cv2.threshold(delta, limit, limit, cv2.THRESH_TRUNC, dst=delta)
        cv2.max(delta, (-limit, -limit, -limit, 0.0), dst=delta)
        cv2.multiply(delta, self._window, dst=delta)
        if self._smoothed is not None and self._alpha < 1.0:
            delta = cv2.addWeighted(delta, self._alpha, self._smoothed, 1.0 - self._alpha, 0.0)
        self._smoothed = delta
        return delta
