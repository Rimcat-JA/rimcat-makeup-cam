"""BGR to NV12, byte for byte what pyvirtualcam's own conversion gives (contract section 2.3).

    Y = (66 R + 130 G + 26 B + 0x1000) >> 8
    2x2 mean per channel: (((a + c + 1) >> 1) + ((b + d + 1) >> 1) + 1) >> 1
    U = (112 B - 74 G - 38 R + 0x8000) >> 8, V = (112 R - 94 G - 18 B + 0x8000) >> 8

``test_nv12_same_as_pyvirtualcam`` compares the bytes in the queue. A pyvirtualcam update that changes its conversion
fails that test, and only then is this formula changed.
"""

from __future__ import annotations

import cv2
import numpy as np

_T66 = (np.arange(256, dtype=np.uint16) * 66 + 0x1000).reshape(256, 1)
_T130 = (np.arange(256, dtype=np.uint16) * 130).reshape(256, 1)
_T26 = (np.arange(256, dtype=np.uint16) * 26).reshape(256, 1)


def _half(channel: np.ndarray) -> np.ndarray:
    """The mean of each 2 x 2 block, as uint16."""
    total = cv2.add(channel[0::2], channel[1::2], dtype=cv2.CV_16U)
    total = (total + 1) >> 1
    return (total[:, 0::2] + total[:, 1::2] + 1) >> 1


def to_nv12(bgr: np.ndarray, y: np.ndarray, uv: np.ndarray) -> None:
    """Write ``bgr`` (even width and height) into the planes ``y`` (H x W) and ``uv`` (H/2 x W)."""
    b, g, r = cv2.split(bgr)
    total = cv2.add(cv2.add(cv2.LUT(r, _T66), cv2.LUT(g, _T130)), cv2.LUT(b, _T26))
    y[:] = total.view(np.uint8)[:, 1::2]  # the high byte is the >> 8
    rb, gb, bb = _half(r), _half(g), _half(b)
    uv[:, 0::2] = (112 * bb + 0x8000 - 74 * gb - 38 * rb) >> 8
    uv[:, 1::2] = (112 * rb + 0x8000 - 94 * gb - 18 * bb) >> 8


class NV12Buffer:
    """One NV12 frame in private memory, touched so that no conversion ever faults a page in."""

    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.data = np.zeros(width * height * 3 // 2, np.uint8)
        self.y = self.data[: width * height].reshape(height, width)
        self.uv = self.data[width * height :].reshape(height // 2, width)

    def fill(self, bgr: np.ndarray) -> None:
        to_nv12(bgr, self.y, self.uv)
