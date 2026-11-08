"""What a model may change, in crop space, and how the runtime applies a delta. Needs numpy and cv2.

The weight comes from the runtime's own ``geometry.edit_weight``, so that the rule learned and the rule enforced cannot
drift apart. At runtime it is rasterized in the frame and the delta warped there; here the same polygons are rasterized
in the crop, which is what the contract asks for (section 10).
"""

from __future__ import annotations

import cv2
import numpy as np

from makeup_cam import regions
from makeup_cam.neural import geometry


def landmarks_to_crop(landmarks: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """(N, 2) continuous frame coordinates, as in ``regions``, to continuous crop coordinates.

    ``crop_matrix`` indexes pixel centers (OpenCV); ``regions`` treats pixel i as covering [i, i + 1). So the matrix is
    undone by half a pixel on each side of the map.
    """
    a = matrix[:, :2]
    points = np.asarray(landmarks, dtype=np.float64)
    return ((points - 0.5) @ a.T + matrix[:, 2] + 0.5).astype(np.float32)


def crop_weight(
    landmarks: np.ndarray,
    matrix: np.ndarray,
    size: int,
    frame_mask: np.ndarray | None = None,
    may_edit_eyes: bool = False,
) -> np.ndarray:
    """float32 S x S in [0, 1]: section 5's weight for this crop. ``frame_mask`` is float32 at the frame's size, the
    union of the hair and face-skin masks, or None when only the face oval is known."""
    crop_mask = None
    if frame_mask is not None:
        crop_mask = cv2.warpAffine(frame_mask, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return geometry.edit_weight(
        landmarks_to_crop(landmarks, matrix), regions.Roi(0, 0, size, size), crop_mask, None, may_edit_eyes=may_edit_eyes
    )


def apply_delta(
    crop: np.ndarray,
    delta: np.ndarray,
    weight: np.ndarray,
    strength: float = 1.0,
    max_abs_delta: float = 0.25,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """The contract's formula (section 3) in crop space, for S x S x 3 arrays. Not clipped: the pipeline clips later."""
    if window is None:
        window = geometry.edge_window(crop.shape[0])
    applied = np.clip(delta, -max_abs_delta, max_abs_delta) * (window * weight * strength)[..., None]
    return crop + applied
