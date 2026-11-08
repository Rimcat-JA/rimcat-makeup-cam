"""The aligned crop of Plan B (``docs/dev/contract-plan-b.md`` sections 4 and 5).

Every part (runtime, evaluation, training) uses this one definition and imports it from here. The
module imports only numpy, cv2 and ``makeup_cam.regions``, so it also works in the training environment.
``crop_matrix`` and ``extract_crop`` are the normative code of the contract: change them only through it.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .. import regions

CROP_DEFINITION = 1
SIDE_PER_FACE_SCALE = 4.5   # crop side = this x regions.face_scale
CENTER_BELOW_EYES = 0.4     # crop center below the eye midpoint, x face_scale
ALLOWED_SIZES = (256, 384, 512)

# Edge window: fades over this share of the crop's side from each edge.
EDGE_WINDOW_SHARE = 0.08

# How sure a mask must be before a model may draw there: (weight 0, full weight). MediaPipe's masks hold about 0.01 to 0.017
# everywhere, and their soft rim reaches onto the wall and the collar, so a mask's low values must not count. Provisional,
# chosen on stills and CREMA-D: 0.3 leaves the wall out and the hair in. A frame's masks may also be a few frames old.
MASK_SURE = (0.3, 0.7)

_MOTION_POINTS = sorted(set(regions.LIPS_OUTER) | set(regions.FACE_OVAL) | set(regions.side_indices("left").eye) | set(regions.side_indices("right").eye))
_MOTION_RANK = int(0.9 * len(_MOTION_POINTS))

# Soft gates (provisional, contract section 5): (zero strength, full strength).
GATE_FACE_SCALE = (35.0, 60.0)  # face_scale in frame pixels
GATE_OUTSIDE = (0.15, 0.0)  # share of FACE_OVAL points outside the frame
GATE_TURN = (0.35, 0.6)  # min(dL, dR) / max(dL, dR), nose tip to each eye center


_PYRAMID_MARGIN = 8  # x 2^level pixels of the frame
_ONE_PIXEL = np.ones((3, 3), np.uint8)
_CUT_MARGIN_PX = 3
_CUT_SHIFT = 4
_CUT_FIXED = 1 << _CUT_SHIFT


def crop_matrix(landmarks: np.ndarray, size: int) -> np.ndarray:
    """2x3 float64: frame pixel index -> crop pixel index, for cv2.warpAffine."""
    right = regions.eye_center(landmarks, "right").astype(np.float64)
    left = regions.eye_center(landmarks, "left").astype(np.float64)
    u = (left - right) / np.linalg.norm(left - right)  # along the eyes: image right for an upright face
    v = np.array([-u[1], u[0]])                        # image down for an upright face
    scale = float(regions.face_scale(landmarks))
    center = (left + right) / 2 + CENTER_BELOW_EYES * scale * v
    k = size / (SIDE_PER_FACE_SCALE * scale)
    a = np.array([u, v]) * k
    t = size / 2 - a @ center                 # continuous frame coordinates -> continuous crop coordinates
    return np.column_stack([a, t + a.sum(axis=1) * 0.5 - 0.5])  # OpenCV indexes pixel centers (regions.py: continuous)


def extract_crop(frame: np.ndarray, landmarks: np.ndarray, size: int) -> np.ndarray:
    """The crop as float32 S x S x 3 BGR. ``frame`` is float32 BGR in [0, 1]."""
    m = crop_matrix(landmarks, size)
    reduction = SIDE_PER_FACE_SCALE * float(regions.face_scale(landmarks)) / size
    level = 0
    while reduction / 2**level > 1.5:           # never sample more than 1.5x sparser: the crop would alias and flicker
        level += 1
    image = frame
    if level > 0:
        step = 2**level
        # Only what the crop can reach: the rest of the frame costs time on the camera thread and adds nothing. Starting on a
        # multiple of ``step`` keeps every half-size pixel where it is in the whole frame's pyramid, and the margin is further
        # than the filter of all levels reaches. The result is the same but for warpAffine's rounding (at most 0.003 of a level).
        roi = crop_roi(m, size, frame.shape)
        if roi is not None:
            margin = _PYRAMID_MARGIN * step
            x0, y0 = max(roi.x0 - margin, 0) // step * step, max(roi.y0 - margin, 0) // step * step
            image = frame[y0 : roi.y1 + margin, x0 : roi.x1 + margin]
            m[:, 2] += m[:, :2] @ (x0, y0)
        for _ in range(level):
            image = cv2.pyrDown(image)
        m[:, :2] *= step                        # cv2.pyrDown: half-size pixel i is centered on full-size pixel 2i
    return cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def edge_window(size: int) -> np.ndarray:
    """float32 S x S: reaches about 0 at the crop's border, so the border never shows (contract section 5)."""
    distance = np.minimum(np.arange(size) + 0.5, size - np.arange(size) - 0.5) / (EDGE_WINDOW_SHARE * size)
    t = np.clip(distance, 0.0, 1.0)
    w = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    return np.outer(w, w)


def crop_roi(matrix: np.ndarray, size: int, shape: tuple[int, ...]) -> regions.Roi | None:
    """The part of a frame of ``shape`` (H, W, ...) that a crop with this ``crop_matrix`` covers, or None."""
    inverse = cv2.invertAffineTransform(matrix)
    corners = np.array([[-0.5, -0.5], [size - 0.5, -0.5], [size - 0.5, size - 0.5], [-0.5, size - 0.5]])
    frame_points = corners @ inverse[:, :2].T + inverse[:, 2]
    if not np.isfinite(frame_points).all():
        return None
    x0, y0 = frame_points.min(axis=0)
    x1, y1 = frame_points.max(axis=0)
    roi = regions.Roi(
        max(0, math.floor(x0)), max(0, math.floor(y0)), min(shape[1], math.ceil(x1) + 1), min(shape[0], math.ceil(y1) + 1)
    )
    return roi if roi.w > 0 and roi.h > 0 else None


def edit_weight(
    landmarks: np.ndarray,
    roi: regions.Roi,
    hair_mask: np.ndarray | None = None,
    face_skin_mask: np.ndarray | None = None,
    may_edit_eyes: bool = False,
    mask_trust: float = 1.0,
) -> np.ndarray:
    """float32 ROI-sized: what a model may change, whatever it outputs (contract section 5).

    Exactly 0 inside the mouth opening (grown by 0.01 x face scale), inside the eye openings unless
    ``may_edit_eyes``, and outside the face oval and what the two masks are sure of (``allowed_area``).
    ``mask_trust`` (0..1) scales what the masks add to the oval: see ``MaskTrust``.

    "Exactly 0" holds for every pixel a polygon touches, also the ones it only partly covers: ``expand_polygon``
    grows radially, so a thin mouth or eye hardly grows across its height, and its anti-aliased edge would let a model
    change the teeth's rim. The anti-aliased step moves one pixel out instead, so the edge stays soft.
    """
    s = float(regions.face_scale(landmarks))
    weight = allowed_area(landmarks, roi, hair_mask, face_skin_mask, mask_trust)
    _cut_out(weight, roi, regions.expand_polygon(landmarks[regions.LIPS_INNER], 0.01 * s))
    if not may_edit_eyes:
        for side in regions.SIDES:
            _cut_out(weight, roi, regions.expand_polygon(landmarks[regions.side_indices(side).eye], 0.005 * s))
    return weight


def allowed_area(
    landmarks: np.ndarray,
    roi: regions.Roi,
    hair_mask: np.ndarray | None = None,
    face_skin_mask: np.ndarray | None = None,
    mask_trust: float = 1.0,
) -> np.ndarray:
    """float32 ROI-sized, a new array: the face oval and what the masks are sure of. Exactly 0 everywhere else."""
    s = float(regions.face_scale(landmarks))
    area = _face_oval(landmarks, roi, 0.03 * s, 0.05 * s)
    masks = [mask[roi.slices] for mask in (hair_mask, face_skin_mask) if mask is not None]  # none: the oval alone decides
    if not masks or mask_trust <= 0.0:
        return area
    # "Sure" only rises with the mask, so the larger of two masks gives the larger of the two results: half the work.
    part = masks[0] if len(masks) == 1 else cv2.max(masks[0], masks[1])
    low, high = MASK_SURE
    gain = 1.0 / (high - low)
    sure = cv2.addWeighted(part, gain, part, 0.0, -low * gain)
    cv2.threshold(sure, 1.0, 1.0, cv2.THRESH_TRUNC, dst=sure)
    cv2.threshold(sure, 0.0, 0.0, cv2.THRESH_TOZERO, dst=sure)
    if mask_trust < 1.0:
        sure = cv2.addWeighted(sure, mask_trust, sure, 0.0, 0.0)
    return cv2.max(area, sure, dst=area)


def _face_oval(landmarks: np.ndarray, roi: regions.Roi, expand_px: float, feather_px: float) -> np.ndarray:
    """``regions.face_oval_mask``, to the bit, made on the oval's own box. That function clips its whole ROI with numpy, which
    holds the GIL against the classic effect next door for 0.25 ms on a large face; the crop's ROI is four times the oval's box."""
    points = regions.expand_polygon(landmarks[regions.FACE_OVAL], expand_px)
    box = regions.roi_around(points, (roi.y1, roi.x1), pad=0.0, margin_px=regions._FEATHER_REACH * max(feather_px, 0.0) + 1)
    area = np.zeros((roi.h, roi.w), np.float32)
    if box is None or box.x1 <= roi.x0 or box.y1 <= roi.y0:
        return area
    box = regions.Roi(max(box.x0, roi.x0), max(box.y0, roi.y0), box.x1, box.y1)
    area[box.y0 - roi.y0 : box.y1 - roi.y0, box.x0 - roi.x0 : box.x1 - roi.x0] = regions.polygon_mask(points, box, feather_px)
    return area


def _cut_out(weight: np.ndarray, roi: regions.Roi, polygon: np.ndarray) -> None:
    """In place on ``weight``: exactly 0 on every pixel the polygon touches, and its soft edge one pixel further out.

    Works on the polygon's own small rectangle. Few Python steps on purpose: this runs next to the classic effects and shares
    the GIL with them. ``cv2.fillPoly`` also draws the outline, so a thin polygon does not vanish, and regions counts a pixel i
    as covering [i, i + 1): the one-pixel dilation takes in every pixel the polygon touches.
    """
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    left, top = max(int(x0) - _CUT_MARGIN_PX, roi.x0), max(int(y0) - _CUT_MARGIN_PX, roi.y0)
    right, bottom = min(int(x1) + _CUT_MARGIN_PX + 1, roi.x1), min(int(y1) + _CUT_MARGIN_PX + 1, roi.y1)
    if right <= left or bottom <= top:
        return
    canvas = np.zeros((bottom - top, right - left), np.uint8)
    # Pixel centers: continuous x covers pixel x - 0.5 in cv2's terms.
    points = np.round((polygon - (left + 0.5, top + 0.5)) * _CUT_FIXED).astype(np.int32)
    cv2.fillPoly(canvas, [points], 255, cv2.LINE_8, _CUT_SHIFT)
    touched = cv2.dilate(canvas, _ONE_PIXEL)
    keep = cv2.blur(cv2.subtract(255, touched), (3, 3))  # the soft edge, one pixel further out
    keep = cv2.min(keep, cv2.subtract(255, touched))
    view = weight[top - roi.y0 : bottom - roi.y0, left - roi.x0 : right - roi.x0]
    cv2.multiply(view, keep.astype(np.float32), dst=view, scale=1.0 / 255.0)


def motion_points(landmarks: np.ndarray) -> np.ndarray:
    """Face parts in crop coordinates, in units of the face scale. Where these differ between two frames, a delta of the one
    does not fit the other: the crop follows the eyes, so a pan costs nothing, while a nod or a word moves the rest against them.
    """
    matrix = crop_matrix(landmarks, 1)
    return (landmarks[_MOTION_POINTS].astype(np.float64) @ matrix[:, :2].T + matrix[:, 2]) * SIDE_PER_FACE_SCALE


def paste_drift(now: np.ndarray, then: np.ndarray) -> float:
    """How far most face parts are from where they were: the 90th percentile of the ``motion_points`` distances."""
    distances = np.hypot(*(now - then).T)
    return float(np.partition(distances, _MOTION_RANK)[_MOTION_RANK])


def usable_landmarks(landmarks: np.ndarray, shape: tuple[int, ...]) -> bool:
    """False for points no real face gives: rasterizing those stalls ``cv2``, and they would paste anywhere."""
    if not np.isfinite(landmarks).all():
        return False
    scale = float(regions.face_scale(landmarks))
    return 1.0 <= scale <= 2.0 * max(shape[0], shape[1])


def soft_gate(landmarks: np.ndarray, shape: tuple[int, ...]) -> float:
    """0..1: how much of the stage's strength this face allows (contract section 5, provisional).

    Small, partly outside or turned faces fade instead of switching. 0 for landmarks that are not usable.
    """
    if not usable_landmarks(landmarks, shape):
        return 0.0
    size = _ramp(float(regions.face_scale(landmarks)), *GATE_FACE_SCALE)
    if size <= 0.0:
        return 0.0
    oval = landmarks[regions.FACE_OVAL]
    inside = (oval[:, 0] >= 0) & (oval[:, 0] < shape[1]) & (oval[:, 1] >= 0) & (oval[:, 1] < shape[0])
    outside = _ramp(1.0 - float(np.count_nonzero(inside)) / len(oval), *GATE_OUTSIDE)
    nose = landmarks[regions.NOSE_TIP]
    to_right = float(np.linalg.norm(regions.eye_center(landmarks, "right") - nose))
    to_left = float(np.linalg.norm(regions.eye_center(landmarks, "left") - nose))
    turn = _ramp(min(to_right, to_left) / max(to_right, to_left, 1e-6), *GATE_TURN)
    return size * outside * turn


def _ramp(value: float, zero_at: float, one_at: float) -> float:
    """Smooth 0 -> 1 between the two values, in either direction."""
    t = min(max((value - zero_at) / (one_at - zero_at), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)
