"""The aligned crop and the weight of what a model may change (contract sections 4 and 5)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from makeup_cam import regions
from makeup_cam.neural import geometry


def landmarks_with(points: dict[int, tuple[float, float]]) -> np.ndarray:
    landmarks = np.zeros((478, 2), np.float32)
    for index, point in points.items():
        landmarks[index] = point
    return landmarks


@pytest.mark.parametrize(
    ("points", "size", "expected"),
    [
        (
            {33: (550, 360), 133: (610, 360), 362: (670, 360), 263: (730, 360), 2: (640, 430)},
            512,
            [[0.948148, 0, -350.840741], [0, 0.948148, -130.87037]],
        ),
        (
            {
                33: (493.6408, 381.246),
                133: (564.5469, 393.7487),
                362: (635.4531, 406.2513),
                263: (706.3593, 418.754),
                2: (585.4136, 482.7238),
            },
            256,
            [[0.38906, 0.068602, -133.147926], [-0.068602, 0.38906, -9.558188]],
        ),
    ],
)
def test_crop_matrix_matches_the_contracts_golden_values(points, size, expected) -> None:
    np.testing.assert_allclose(geometry.crop_matrix(landmarks_with(points), size), expected, atol=1e-3)


@pytest.mark.parametrize("size", [256, 512])
def test_a_crop_samples_exactly_where_the_matrix_says(face_landmarks_720p: np.ndarray, size: int) -> None:
    """On a ramp image every crop pixel's value tells where it was sampled, with or without the pyramid."""
    ys, xs = np.mgrid[0:720, 0:1280].astype(np.float32)
    ramp = np.dstack([xs / 1280.0, ys / 720.0, np.zeros_like(xs)])
    crop = geometry.extract_crop(ramp, face_landmarks_720p, size)
    inverse = geometry.cv2.invertAffineTransform(geometry.crop_matrix(face_landmarks_720p, size))
    cy, cx = np.mgrid[size // 4 : 3 * size // 4 : 16, size // 4 : 3 * size // 4 : 16]
    source = inverse[:, :2] @ np.stack([cx.ravel(), cy.ravel()]) + inverse[:, 2:]
    np.testing.assert_allclose(crop[cy, cx, 0].ravel() * 1280.0, source[0], atol=0.02)
    np.testing.assert_allclose(crop[cy, cx, 1].ravel() * 720.0, source[1], atol=0.02)


def test_the_crop_does_not_use_lips_chin_or_iris(face_landmarks_720p: np.ndarray) -> None:
    moved = face_landmarks_720p.copy()
    moved[regions.LIPS_INNER] += (3, 9)
    moved[regions.LIPS_OUTER] += (3, 9)
    moved[152] += (5, 20)  # chin
    moved[468:478] += (4, 4)  # irises
    np.testing.assert_array_equal(geometry.crop_matrix(moved, 512), geometry.crop_matrix(face_landmarks_720p, 512))


def test_edge_window_fades_to_the_border_and_leaves_the_middle_alone() -> None:
    window = geometry.edge_window(256)
    assert window.dtype == np.float32 and window.shape == (256, 256)
    assert window[128, 128] == 1.0
    assert window[0, 128] < 0.002 and window[128, 0] < 0.002 and window[255, 128] < 0.002
    assert window[128, int(0.08 * 256) + 1] == 1.0


def test_crop_roi_covers_the_crop_and_stays_in_the_frame(face_landmarks_720p: np.ndarray) -> None:
    matrix = geometry.crop_matrix(face_landmarks_720p, 256)
    roi = geometry.crop_roi(matrix, 256, (720, 1280, 3))
    assert roi is not None and 0 <= roi.x0 < roi.x1 <= 1280 and 0 <= roi.y0 < roi.y1 <= 720
    far = face_landmarks_720p + (5000, 0)
    assert geometry.crop_roi(geometry.crop_matrix(far, 256), 256, (720, 1280, 3)) is None


def test_weight_is_exactly_zero_in_the_mouth_eyes_and_outside(face_landmarks_720p: np.ndarray) -> None:
    landmarks = face_landmarks_720p
    roi = regions.Roi(0, 0, 1280, 720)
    weight = geometry.edit_weight(landmarks, roi)
    assert np.all(weight[regions.polygon_mask(landmarks[regions.LIPS_INNER], roi) > 0.0] == 0.0)
    for side in regions.SIDES:
        assert np.all(weight[regions.eye_opening_mask(landmarks, side, roi) > 0.0] == 0.0)
    assert weight[5, 5] == 0.0 and weight[715, 1275] == 0.0
    cheek = regions.cheek_center(landmarks, "left").astype(int)
    assert weight[cheek[1], cheek[0]] == pytest.approx(1.0)
    # With leave to edit the eyes, only the mouth stays out.
    free = geometry.edit_weight(landmarks, roi, may_edit_eyes=True)
    center = regions.eye_center(landmarks, "left").astype(int)
    assert free[center[1], center[0]] > 0.9 and weight[center[1], center[0]] == 0.0


def test_masks_widen_what_may_change_but_never_open_the_mouth(face_landmarks_720p: np.ndarray) -> None:
    landmarks = face_landmarks_720p
    roi = regions.Roi(0, 0, 1280, 720)
    hair = np.ones((720, 1280), np.float32)
    weight = geometry.edit_weight(landmarks, roi, hair_mask=hair)
    assert weight[5, 5] == 1.0
    assert np.all(weight[regions.polygon_mask(landmarks[regions.LIPS_INNER], roi) > 0.0] == 0.0)


def test_soft_gate_is_full_for_a_normal_face(face_landmarks_720p: np.ndarray) -> None:
    assert geometry.soft_gate(face_landmarks_720p, (720, 1280, 3)) == pytest.approx(1.0, abs=0.02)


def test_soft_gate_fades_small_faces_and_faces_leaving_the_frame(face_landmarks_720p: np.ndarray) -> None:
    landmarks = face_landmarks_720p
    center = landmarks.mean(axis=0)
    scale = regions.face_scale(landmarks)

    def scaled(to: float) -> np.ndarray:
        return center + (landmarks - center) * (to / scale)

    assert geometry.soft_gate(scaled(34.9), (720, 1280, 3)) == 0.0
    assert 0.0 < geometry.soft_gate(scaled(47.0), (720, 1280, 3)) < 1.0
    assert geometry.soft_gate(scaled(61.0), (720, 1280, 3)) == pytest.approx(1.0, abs=0.02)
    outside = landmarks - (landmarks[regions.FACE_OVAL, 0].min() + regions.face_scale(landmarks) * 1.5, 0)
    assert geometry.soft_gate(outside, (720, 1280, 3)) == 0.0


def test_soft_gate_fades_a_turned_head(face_landmarks_720p: np.ndarray) -> None:
    turned = face_landmarks_720p.copy()
    left = regions.eye_center(turned, "left")
    # The nose tip right next to one eye: as far as a head turns.
    turned[regions.NOSE_TIP] = left + (5.0, 5.0)
    assert geometry.soft_gate(turned, (720, 1280, 3)) == 0.0


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1e9])
def test_unusable_landmarks_are_recognized(face_landmarks_720p: np.ndarray, bad: float) -> None:
    broken = face_landmarks_720p.copy()
    broken[33] = (bad, bad)
    assert not geometry.usable_landmarks(broken, (720, 1280, 3))
    assert geometry.soft_gate(broken, (720, 1280, 3)) == 0.0
    assert geometry.usable_landmarks(face_landmarks_720p, (720, 1280, 3))


def _floor_mask(value: float = 0.0166, shape: tuple[int, int] = (720, 1280)) -> np.ndarray:
    """What MediaPipe's masks hold away from the person: a small value everywhere."""
    return np.full(shape, value, np.float32)


def test_a_masks_floor_gives_no_weight_at_all(face_landmarks_720p: np.ndarray) -> None:
    roi = regions.Roi(0, 0, 1280, 720)
    with_floor = geometry.edit_weight(face_landmarks_720p, roi, _floor_mask(), _floor_mask())
    oval_only = geometry.edit_weight(face_landmarks_720p, roi)
    assert np.array_equal(with_floor, oval_only)


def test_a_masks_soft_rim_gives_no_weight_until_the_mask_is_sure(face_landmarks_720p: np.ndarray) -> None:
    low, high = geometry.MASK_SURE
    roi = regions.Roi(0, 0, 1280, 720)
    ramp = np.tile(np.linspace(0.0, 1.0, 1280, dtype=np.float32), (720, 1))
    weight = geometry.edit_weight(face_landmarks_720p, roi, ramp)
    y = 10  # above the face: only the mask counts here
    assert face_landmarks_720p[:, 1].min() > 40
    assert np.all(weight[y][ramp[y] <= low] == 0.0)
    assert np.all(weight[y][ramp[y] >= high] == 1.0)
    middle = (ramp[y] > low) & (ramp[y] < high)
    assert np.all(np.diff(weight[y][middle]) >= 0.0)
    np.testing.assert_allclose(weight[y][middle], (ramp[y][middle] - low) / (high - low), atol=1e-5)


@pytest.mark.parametrize("seed", range(6))
def test_every_pixel_a_polygon_touches_has_weight_zero_whatever_the_pose(face_landmarks_720p: np.ndarray, seed: int) -> None:
    """Rotated, scaled, moved by fractions of a pixel, with the mouth nearly shut: the rim of the teeth stays exact."""
    rng = np.random.default_rng(seed)
    center = face_landmarks_720p.mean(axis=0)
    angle, scale = rng.uniform(-0.4, 0.4), rng.uniform(0.7, 1.6)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)
    landmarks = (face_landmarks_720p - center) @ rotation.T * scale + center + rng.uniform(-3.0, 3.0, 2).astype(np.float32)
    if seed % 2:  # a closed mouth: a thin polygon
        lips = regions.LIPS_INNER
        middle = landmarks[lips, 1].mean()
        landmarks[lips, 1] = middle + (landmarks[lips, 1] - middle) * 0.02
    roi = regions.Roi(0, 0, 1280, 720)
    s = regions.face_scale(landmarks)
    weight = geometry.edit_weight(landmarks, roi)
    mouth = regions.polygon_mask(regions.expand_polygon(landmarks[regions.LIPS_INNER], 0.01 * s), roi)
    assert np.all(weight[mouth > 0.0] == 0.0)
    for side in regions.SIDES:
        eye = regions.polygon_mask(regions.expand_polygon(landmarks[regions.side_indices(side).eye], 0.005 * s), roi)
        assert np.all(weight[eye > 0.0] == 0.0)


def test_the_weight_rises_softly_beside_a_cut_and_a_cut_near_the_roi_edge_is_safe(face_landmarks_720p: np.ndarray) -> None:
    roi = regions.Roi(0, 0, 1280, 720)
    weight = geometry.edit_weight(face_landmarks_720p, roi)
    x, y = face_landmarks_720p[regions.LIPS_INNER].mean(axis=0).astype(int)
    column = weight[y:y + 200, x]
    assert column[0] == 0.0
    assert np.abs(np.diff(column)).max() < 0.7  # one pixel of edge, not a hard cut from 0 to 1
    # A crop that ends inside the mouth: no error, and the part it has is cut out.
    half = regions.Roi(0, 0, 1280, int(y))
    assert geometry.edit_weight(face_landmarks_720p, half).shape == (int(y), 1280)
    outside = regions.Roi(0, 0, 10, 10)
    assert geometry.edit_weight(face_landmarks_720p, outside).shape == (10, 10)


def test_motion_points_ignore_a_pan_and_a_zoom_and_see_a_nod(face_landmarks_720p: np.ndarray) -> None:
    now = geometry.motion_points(face_landmarks_720p)
    center = face_landmarks_720p.mean(axis=0)
    panned = geometry.motion_points(face_landmarks_720p + np.array([17.0, -9.0], np.float32))
    zoomed = geometry.motion_points(center + (face_landmarks_720p - center) * 1.2)
    assert geometry.paste_drift(now, panned) < 1e-3
    assert geometry.paste_drift(now, zoomed) < 0.01  # half a pixel of the frame, against limits of 0.05 and up
    s = regions.face_scale(face_landmarks_720p)
    nodded = face_landmarks_720p.copy()
    below = nodded[:, 1] > regions.eye_center(face_landmarks_720p, "left")[1] + 0.2 * s
    nodded[below, 1] += 0.1 * s
    assert 0.04 < geometry.paste_drift(now, geometry.motion_points(nodded)) < 0.15


# ---- what the stage's cheaper paths must not change ----------------------------------------------------------------------


def _crop_from_the_whole_frame(frame: np.ndarray, landmarks: np.ndarray, size: int) -> np.ndarray:
    m = geometry.crop_matrix(landmarks, size)
    reduction = geometry.SIDE_PER_FACE_SCALE * float(regions.face_scale(landmarks)) / size
    image, level = frame, 0
    while reduction / 2**level > 1.5:
        image, level = cv2.pyrDown(image), level + 1
    m[:, :2] *= 2**level
    return cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


@pytest.mark.parametrize("size", [256, 384, 512])
@pytest.mark.parametrize(
    ("scale", "angle", "shift"),
    [(1.0, 0.0, (0, 0)), (2.0, 0.0, (0, 0)), (3.0, 20.0, (40, -30)), (1.5, -35.0, (-500, 200)), (2.5, 10.0, (560, -300)), (2.2, 0.0, (0, 330)), (4.0, 45.0, (301, 77))],
)
def test_the_crop_from_the_frames_part_is_the_one_from_the_whole_frame(
    face_landmarks_720p: np.ndarray, size: int, scale: float, angle: float, shift: tuple[int, int]
) -> None:
    rng = np.random.default_rng(3)
    frame = cv2.GaussianBlur(rng.random((720, 1281, 3), np.float32), (0, 0), 1.0)  # an odd width too
    center = face_landmarks_720p.mean(axis=0)
    turn = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, scale)
    landmarks = (face_landmarks_720p @ turn[:, :2].T + turn[:, 2] + np.array(shift)).astype(np.float32)
    # Not to the bit: warpAffine rounds its fixed-point coordinates a little differently. On noise this is 0.003 of an 8-bit level.
    difference = np.abs(geometry.extract_crop(frame, landmarks, size) - _crop_from_the_whole_frame(frame, landmarks, size))
    assert difference.max() < 2e-5


def test_the_oval_made_on_its_own_box_equals_the_one_from_regions(face_landmarks_720p: np.ndarray) -> None:
    s = float(regions.face_scale(face_landmarks_720p))
    for shift in ((0, 0), (500, 0), (-500, -200), (2000, 0)):
        landmarks = face_landmarks_720p + np.array(shift, np.float32)
        roi = regions.Roi(100, 50, 900, 700)
        np.testing.assert_array_equal(geometry._face_oval(landmarks, roi, 0.03 * s, 0.05 * s), regions.face_oval_mask(landmarks, roi, expand_px=0.03 * s, feather_px=0.05 * s))


def test_the_two_masks_together_give_what_each_gives_alone(face_landmarks_720p: np.ndarray) -> None:
    rng = np.random.default_rng(4)
    hair, skin = (cv2.GaussianBlur(rng.random((720, 1280), np.float32), (0, 0), 8.0) * 4.0 - 1.5 for _ in range(2))
    hair, skin = np.clip(hair, 0.0, 1.0), np.clip(skin, 0.0, 1.0)
    roi = regions.Roi(200, 0, 1100, 720)
    alone = np.maximum(geometry.allowed_area(face_landmarks_720p, roi, hair), geometry.allowed_area(face_landmarks_720p, roi, None, skin))
    np.testing.assert_array_equal(geometry.allowed_area(face_landmarks_720p, roi, hair, skin), alone)


def test_how_much_the_masks_add_follows_the_trust(face_landmarks_720p: np.ndarray) -> None:
    mask = np.ones((720, 1280), np.float32)
    roi = regions.Roi(100, 0, 1200, 720)
    oval = geometry.allowed_area(face_landmarks_720p, roi)
    outside = oval == 0.0
    assert outside.any()
    np.testing.assert_array_equal(geometry.allowed_area(face_landmarks_720p, roi, mask, mask, 0.0), oval)
    assert np.allclose(geometry.allowed_area(face_landmarks_720p, roi, mask, mask, 0.25)[outside], 0.25)
    assert np.array_equal(geometry.allowed_area(face_landmarks_720p, roi, mask, mask, 1.0), geometry.allowed_area(face_landmarks_720p, roi, mask, mask))
    full = geometry.edit_weight(face_landmarks_720p, roi, mask, mask, False, 0.0)
    np.testing.assert_array_equal(full, geometry.edit_weight(face_landmarks_720p, roi))
