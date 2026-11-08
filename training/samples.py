"""One training sample: a pair of neighbouring frames, cropped, masked, with webcam conditions. Needs numpy and cv2.

Sampling is stateless: a sample is fully determined by (seed, step, position in the batch). Resuming therefore continues
exactly where a run stopped, with any number of loader workers, and nothing has to be remembered but the step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from makeup_cam import regions
from makeup_cam.neural import geometry

from .augment import Photometric, degrade
from .config import AugmentConfig, DataConfig
from .layout import Sequence
from .masks import crop_weight


def read_image(path) -> np.ndarray:
    """float32 BGR in [0, 1]. Through a byte buffer: cv2.imread cannot open non-ASCII paths on Windows."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"画像を読めません: {getattr(path, 'name', path)}")
    return image.astype(np.float32) / 255.0


def read_mask(path) -> np.ndarray:
    mask = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise OSError(f"マスクを読めません: {getattr(path, 'name', path)}")
    return mask.astype(np.float32) / 255.0


def jitter_landmarks(landmarks: np.ndarray, shift: np.ndarray, degrees: float, scale: float) -> np.ndarray:
    """A small similarity about the eye midpoint, standing in for imprecise tracking. ``shift`` is in face scales."""
    right = regions.eye_center(landmarks, "right")
    left = regions.eye_center(landmarks, "left")
    center = (left + right) / 2
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle) * (1.0 + scale), np.sin(angle) * (1.0 + scale)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (landmarks - center) @ rotation.T + center + shift.astype(np.float32) * float(regions.face_scale(landmarks))


@dataclass(frozen=True)
class View:
    """Clean crops of one frame."""

    input: np.ndarray  # S x S x 3
    target: np.ndarray
    weight: np.ndarray  # S x S, what the runtime would let through


def crop_view(
    sequence: Sequence, landmarks: np.ndarray, jittered: np.ndarray, index: int, size: int, may_edit_eyes: bool = False
) -> View:
    """``landmarks`` are the frame's own; ``jittered`` place the crop."""
    frame_input = read_image(sequence.input_path(index))
    frame_target = read_image(sequence.target_path(index))
    if frame_input.shape != frame_target.shape:
        raise OSError(f"{sequence.name}: {index:06d} の input と target の大きさが違います")
    frame_mask = read_mask(sequence.mask_path(index)) if sequence.has_masks else None
    matrix = geometry.crop_matrix(jittered, size)
    return View(
        input=geometry.extract_crop(frame_input, jittered, size),
        target=geometry.extract_crop(frame_target, jittered, size),
        weight=crop_weight(landmarks, matrix, size, frame_mask, may_edit_eyes),
    )


def backward_flow(next_crop: np.ndarray, crop: np.ndarray) -> np.ndarray:
    """S x S x 2, pixels: ``next_crop[y, x]`` is about ``crop[y + f_y, x + f_x]``. Motion left over after the crop."""
    a = cv2.cvtColor((next_crop * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor((crop * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 2, 5, 1.2, 0)


def warp_by_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """``image`` sampled at each pixel plus its flow."""
    size = flow.shape[0]
    x, y = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
    return cv2.remap(image, x + flow[..., 0], y + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def flow_confidence(next_crop: np.ndarray, warped: np.ndarray) -> np.ndarray:
    """S x S in (0, 1]: low where the flow did not explain the next frame (a blink, the mouth opening)."""
    error = np.abs(next_crop - warped).mean(axis=2)
    return np.exp(-error / 0.05).astype(np.float32)


def _chw(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image.transpose(2, 0, 1))


@dataclass
class SampleBuilder:
    """Builds samples from a list of sequences. Picklable, so it can live in loader workers."""

    sequences: list[Sequence]
    data: DataConfig
    augment: AugmentConfig
    max_abs_delta: float = 0.25
    may_edit_eyes: bool = False
    seed: int = 1
    _landmarks: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _groups: dict[str, list[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.sequences:
            raise ValueError("no sequences")
        for i, sequence in enumerate(self.sequences):
            self._groups.setdefault(sequence.group, []).append(i)

    def landmarks(self, sequence: Sequence) -> np.ndarray:
        if sequence.name not in self._landmarks:
            self._landmarks[sequence.name] = sequence.landmarks()
        return self._landmarks[sequence.name]

    def pick(self, rng: np.random.Generator) -> tuple[Sequence, int]:
        """Each person equally often, whatever the length of their recordings."""
        groups = sorted(self._groups)
        members = self._groups[groups[int(rng.integers(len(groups)))]]
        sequence = self.sequences[members[int(rng.integers(len(members)))]]
        return sequence, int(rng.integers(sequence.frames))

    def build(self, step: int, position: int) -> dict[str, np.ndarray]:
        """The sample at this position of this step's batch. Arrays are float32, channels first, [2, ...] for two frames."""
        rng = np.random.default_rng([self.seed, step, position])
        sequence, t = self.pick(rng)
        return self.build_at(sequence, t, rng)

    def build_at(self, sequence: Sequence, t: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        size = self.data.size
        landmarks = self.landmarks(sequence)
        has_next = sequence.consecutive and t + 1 < sequence.frames
        t1 = t + 1 if has_next else t

        # One jitter for both frames: the crop's alignment errors move slowly at runtime.
        shift = rng.normal(0.0, self.data.jitter_shift, 2)
        degrees = float(rng.normal(0.0, self.data.jitter_degrees))
        scale = float(rng.normal(0.0, self.data.jitter_scale))
        views = []
        for index in (t, t1):
            jittered = jitter_landmarks(landmarks[index], shift, degrees, scale)
            views.append(crop_view(sequence, landmarks[index], jittered, index, size, self.may_edit_eyes))

        photometric = Photometric.sample(rng, self.augment) if self.augment.enabled else Photometric.identity()
        light = photometric.field(size)
        clean = [photometric.apply(view.input, light) for view in views]
        targets = [photometric.apply(view.target, light) for view in views]
        deltas = [np.clip(target - crop, -self.max_abs_delta, self.max_abs_delta) for target, crop in zip(targets, clean)]
        inputs = [degrade(crop, rng, self.augment) for crop in clean] if self.augment.enabled else clean

        if has_next and self.data.temporal_warp == "flow":
            flow = backward_flow(clean[1], clean[0])
            confidence = flow_confidence(clean[1], warp_by_flow(clean[0], flow))
        else:
            flow = np.zeros((size, size, 2), dtype=np.float32)
            confidence = np.ones((size, size), dtype=np.float32)
        return {
            "input": np.stack([_chw(crop) for crop in inputs]),
            "delta": np.stack([_chw(delta) for delta in deltas]),
            "weight": np.stack([view.weight[None] for view in views]),
            "flow": _chw(flow),
            "flow_confidence": confidence[None],
            "has_next": np.float32(1.0 if has_next else 0.0),
        }
