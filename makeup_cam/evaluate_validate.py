"""Does the evaluation tool see what it should? Injected defects and clean runs on all clips.

``python -m makeup_cam.evaluate_validate --out DIR``: for every clip, runs the real classic pipeline, a well-behaved
imitation of a model (written against the contract's paste formula, no learned weights), and a list of injected
defects, measures each with ``makeup_cam.evaluate`` and stores the per-frame series and metrics under ``DIR``.
``--summarize`` then gives false alarms and misses per metric. The crude face paste below exists only so that the identity
guards can be shown to fire. Everything is written under ``DIR``, which must not be tracked by git.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import evaluate as ev
from . import regions
from .neural import geometry
from .presets import load_preset

logger = logging.getLogger(__name__)

ACTORS = ("1002", "1012", "1013", "1029", "1046", "1060", "1072", "1073", "1074", "1076", "1082")
TAKES = ("DFA_NEU_XX", "IEO_HAP_HI", "ITS_HAP_XX", "TIE_SAD_XX")
MODEL_CROP = 256
TEMPORAL_ALPHA = 0.6


@dataclass
class Rendered:
    """A clip through the real pipeline."""

    bare: list[np.ndarray]
    classic: list[np.ndarray]
    stage: list[np.ndarray | None]
    hair: list[np.ndarray | None]
    skin: list[np.ndarray | None]


class GlitchTracker:
    """Wraps the real tracker: adds a jump on some frames, or a constant offset. For drawing with bad landmarks."""

    def __init__(self, inner: Any, jumps: dict[int, np.ndarray] | None = None, offset: np.ndarray | None = None) -> None:
        self._inner = inner
        self._jumps = jumps or {}
        self._offset = offset
        self._frame = -1

    def detect(self, frame: np.ndarray, ts_ms: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        self._frame += 1
        points, z = self._inner.detect(frame, ts_ms)
        if points is None:
            return points, z
        if self._offset is not None:
            points = points + self._offset * regions.face_scale(points)
        jump = self._jumps.get(self._frame)
        if jump is not None:
            points = points + jump * regions.face_scale(points)
        return points, z

    def set_smoothing(self, smoothing: float) -> None:
        self._inner.set_smoothing(smoothing)

    def close(self) -> None:
        self._inner.close()


def render(frames: list[np.ndarray], preset: str, fps: float, tracker_wrap: Callable[[Any], Any] | None = None) -> Rendered:
    """Frames through the real classic pipeline, with the masks and landmarks the stage would get."""
    from .pipeline import MakeupPipeline

    settings = ev.file_settings(load_preset(preset))
    probe = ev.StageProbe()
    tracker = ev.default_tracker_factory(settings.general.landmark_smoothing)
    if tracker_wrap is not None:
        tracker = tracker_wrap(tracker)
    segmenter = ev.default_segmenter(settings.general.segmentation_interval)
    pipeline = MakeupPipeline(settings, tracker=tracker, segmenter=segmenter, effects=ev.builtin_effects([probe], without_neural=True))
    out = Rendered([], [], [], [], [])
    try:
        for index, frame in enumerate(frames):
            result = pipeline.process(frame, round(index * 1000.0 / fps))
            out.bare.append(frame)
            out.classic.append(result.output)
            out.stage.append(probe.landmarks)
            out.hair.append(probe.hair_mask)
            out.skin.append(probe.face_skin_mask)
    finally:
        pipeline.close()
    return out


def build_refs(data: Rendered, reference: list[np.ndarray], fps: float, same_as_bare: bool) -> list[ev.FrameRef]:
    """The reference side of a clip, with its own raw landmark tracks."""
    track_bare = ev.default_tracker_factory(0.0)
    track_ref = None if same_as_bare else ev.default_tracker_factory(0.0)
    refs = []
    try:
        for index, (bare, ref) in enumerate(zip(data.bare, reference)):
            ts = round(index * 1000.0 / fps)
            raw_bare = track_bare.detect(bare, ts)[0]
            raw_ref = raw_bare if track_ref is None else track_ref.detect(ref, ts)[0]
            refs.append(ev.make_frame_ref(index, bare, ref, data.stage[index], raw_bare, raw_ref, data.hair[index], data.skin[index]))
    finally:
        track_bare.close()
        if track_ref is not None:
            track_ref.close()
    return refs


def measure(name: str, mode: str, data: Rendered, reference: list[np.ndarray], refs: list[ev.FrameRef], outputs: list[np.ndarray], fps: float, track: bool = True) -> ev.ClipResult:
    """``track=False``: without looking for landmarks in the output, for defects those measures are not meant for: far faster."""
    tracker = ev.default_tracker_factory(0.0) if track else None

    def track_output(frame: np.ndarray, index: int) -> np.ndarray | None:
        return tracker.detect(frame, round(index * 1000.0 / fps))[0]

    analyzer = ev.ClipAnalyzer(name, fps, mode, track_output=track_output if track else None)
    try:
        for ref, bare, rf, out in zip(refs, data.bare, reference, outputs):
            analyzer.feed(ref, bare, rf, out)
    finally:
        if tracker is not None:
            tracker.close()
    return analyzer.finish()


# --------------------------------------------------------------------------
# A well-behaved imitation of a model (no learned weights): the runtime's formula, age 1, smoothing
# --------------------------------------------------------------------------


def paste_delta(frame: np.ndarray, delta: np.ndarray, landmarks: np.ndarray, zones: ev.Zones) -> np.ndarray:
    """``frame + delta * weight``, through the current landmarks, as the stage does it."""
    roi = zones.roi
    matrix = geometry.crop_matrix(landmarks, delta.shape[0])
    inverse = cv2.invertAffineTransform(matrix)
    inverse[0, 2] -= roi.x0
    inverse[1, 2] -= roi.y0
    warped = cv2.warpAffine(delta, inverse, (roi.w, roi.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    out = frame.copy()
    view = out[roi.slices].astype(np.float32) / 255.0 + warped * zones.weight[..., None]
    moved = zones.weight > 0.0
    converted = cv2.convertScaleAbs(np.clip(view, 0.0, 1.0), alpha=255.0)
    target = out[roi.slices]
    target[moved] = converted[moved]
    return out


def _tint_pattern(size: int = MODEL_CROP) -> np.ndarray:
    """A fixed, smooth change: a little warmer everywhere, a little more on the cheeks."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    cheeks = np.exp(-(((xs - 0.32) ** 2 + (ys - 0.48) ** 2) / 0.006)) + np.exp(-(((xs - 0.68) ** 2 + (ys - 0.48) ** 2) / 0.006))
    delta = np.zeros((size, size, 3), np.float32)
    delta[..., 0] = -0.006 - 0.008 * cheeks
    delta[..., 1] = 0.002 - 0.004 * cheeks
    delta[..., 2] = 0.008 + 0.016 * cheeks
    return delta * geometry.edge_window(size)[..., None]


def imitate_model(data: Rendered, zones: list[ev.Zones | None], kind: str) -> list[np.ndarray]:
    """``tint``: a fixed change. ``smooth``: blur the crop a little: depends on the picture, so it moves with it."""
    window = geometry.edge_window(MODEL_CROP)[..., None]
    tint = _tint_pattern()
    smoothed: np.ndarray | None = None
    pending: np.ndarray | None = None
    outputs = []
    for classic, stage, zone in zip(data.classic, data.stage, zones):
        if stage is None or zone is None:
            smoothed, pending = None, None
            outputs.append(classic)
            continue
        if kind == "tint":
            delta = tint
        else:
            # Age 1: this frame is drawn with the delta made from the last one.
            delta = pending if pending is not None else np.zeros((MODEL_CROP, MODEL_CROP, 3), np.float32)
            crop = geometry.extract_crop(classic.astype(np.float32) / 255.0, stage, MODEL_CROP)
            raw = (cv2.GaussianBlur(crop, (0, 0), 2.0) - crop) * 0.5 * window
            smoothed = raw if smoothed is None else smoothed + TEMPORAL_ALPHA * (raw - smoothed)
            pending = np.clip(smoothed, -0.25, 0.25)
        outputs.append(paste_delta(classic, delta, stage, zone))
    return outputs


# --------------------------------------------------------------------------
# Injected defects
# --------------------------------------------------------------------------


def _soft_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Zero-mean noise of standard deviation 1, with features of about ``sigma`` pixels."""
    h, w = shape
    down = max(int(sigma), 1)
    small = rng.standard_normal((h // down + 2, w // down + 2), np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return field / max(float(field.std()), 1e-6)


def _add_levels(frame: np.ndarray, change: np.ndarray) -> np.ndarray:
    if change.ndim == 2:
        change = change[..., None]
    return np.clip(frame.astype(np.float32) + change + 0.5, 0, 255).astype(np.uint8)


def _face_mask(zone: ev.Zones, shape: tuple[int, int]) -> np.ndarray:
    full = np.zeros(shape, np.float32)
    full[zone.roi.slices] = zone.weight * (zone.allowed_soft > 0.5)
    return full


def shimmer_face(outs: list[np.ndarray], zones: list[ev.Zones | None], level: float, seed: int, sigma: float = 2.0) -> list[np.ndarray]:
    """New noise on the face in every frame."""
    rng = np.random.default_rng(seed)
    result = []
    for out, zone in zip(outs, zones):
        if zone is None:
            result.append(out)
            continue
        mask = _face_mask(zone, out.shape[:2])
        result.append(_add_levels(out, _soft_noise(rng, out.shape[:2], sigma) * level * mask))
    return result


def pump_face(outs: list[np.ndarray], zones: list[ev.Zones | None], level: float) -> list[np.ndarray]:
    """The face gets brighter and darker in turn: a flicker no block average hides."""
    result = []
    for index, (out, zone) in enumerate(zip(outs, zones)):
        if zone is None:
            result.append(out)
            continue
        sign = 1.0 if index % 2 == 0 else -1.0
        result.append(_add_levels(out, sign * level * _face_mask(zone, out.shape[:2])))
    return result


def shimmer_background(outs: list[np.ndarray], refs: list[ev.FrameRef], level: float, seed: int) -> list[np.ndarray]:
    """New noise on the far background in every frame."""
    rng = np.random.default_rng(seed)
    result = []
    for out, ref in zip(outs, refs):
        if ref.far is None:
            result.append(out)
            continue
        mask = cv2.resize(ref.far.astype(np.float32), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR)
        result.append(_add_levels(out, _soft_noise(rng, out.shape[:2], 3.0) * level * mask))
    return result


def _cheek_patch(lm: np.ndarray, shape: tuple[int, int], level: float, side: str = "left") -> np.ndarray:
    """A soft patch of color on one cheek."""
    s = regions.face_scale(lm)
    center = regions.cheek_center(lm, side)
    h, w = shape
    x0, y0 = max(int(center[0] - s), 0), max(int(center[1] - s), 0)
    x1, y1 = min(int(center[0] + s), w), min(int(center[1] + s), h)
    patch = np.zeros((h, w, 3), np.float32)
    if x1 <= x0 or y1 <= y0:
        return patch
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    bump = np.exp(-(((xs - center[0]) ** 2 + (ys - center[1]) ** 2) / (2 * (0.22 * s) ** 2)))
    patch[y0:y1, x0:x1, 0] = -0.6 * level * bump
    patch[y0:y1, x0:x1, 1] = -0.3 * level * bump
    patch[y0:y1, x0:x1, 2] = level * bump
    return patch


def _through_weight(change: np.ndarray, zone: ev.Zones | None) -> np.ndarray:
    """What the stage's weight leaves of a change, so that the exact checks do not find it by accident."""
    if zone is None:
        return np.zeros_like(change)
    full = np.zeros(change.shape[:2], np.float32)
    full[zone.roi.slices] = zone.weight
    return change * full[..., None]


def patch_lag(outs: list[np.ndarray], data: Rendered, zones: list[ev.Zones | None], level: float, lag: int) -> list[np.ndarray]:
    """A pasted patch placed where the cheek was ``lag`` frames ago."""
    result = []
    for index, out in enumerate(outs):
        lm = data.stage[max(index - lag, 0)]
        result.append(out if lm is None else _add_levels(out, _through_weight(_cheek_patch(lm, out.shape[:2], level), zones[index])))
    return result


def paint_lag(outs: list[np.ndarray], reference: list[np.ndarray], lag: int) -> list[np.ndarray]:
    """The whole applied change stays in the frame where it was ``lag`` frames ago."""
    result = []
    for index, rf in enumerate(reference):
        old = max(index - lag, 0)
        delta = outs[old].astype(np.float32) - reference[old].astype(np.float32)
        result.append(np.clip(rf.astype(np.float32) + delta, 0, 255).astype(np.uint8))
    return result


def paint_shift(outs: list[np.ndarray], reference: list[np.ndarray], stage: list[np.ndarray | None], amount: float) -> list[np.ndarray]:
    """The whole applied change lands ``amount`` x face scale to the side and down: makeup off target."""
    result = []
    for out, rf, lm in zip(outs, reference, stage):
        if lm is None:
            result.append(out)
            continue
        s = regions.face_scale(lm)
        delta = out.astype(np.float32) - rf.astype(np.float32)
        moved = cv2.warpAffine(delta, np.float32([[1, 0, amount * s * 0.6], [0, 1, amount * s * 0.8]]), (out.shape[1], out.shape[0]), flags=cv2.INTER_LINEAR)
        result.append(np.clip(rf.astype(np.float32) + moved, 0, 255).astype(np.uint8))
    return result


def teeth_warp(outs: list[np.ndarray], stage: list[np.ndarray | None], pixels: float) -> list[np.ndarray]:
    """Moves the picture inside the open mouth: teeth change shape, the lips stay."""
    result = []
    for out, lm in zip(outs, stage):
        if lm is None or ev.mouth_gap(lm) <= ev.OPEN_MOUTH:
            result.append(out)
            continue
        s = regions.face_scale(lm)
        roi = regions.roi_around(lm[regions.LIPS_OUTER], out.shape, pad=0.0, margin_px=0.1 * s)
        if roi is None:
            result.append(out)
            continue
        ys, xs = np.mgrid[0 : roi.h, 0 : roi.w].astype(np.float32)
        period = max(0.25 * s, 4.0)
        map_x = xs + pixels * np.sin(ys * 2 * np.pi / period)
        map_y = ys + pixels * np.sin(xs * 2 * np.pi / period)
        moved = cv2.remap(out[roi.slices], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        inner = regions.polygon_mask(regions.expand_polygon(lm[regions.LIPS_INNER], -0.01 * s), roi, feather_px=1.0)[..., None]
        copy = out.copy()
        copy[roi.slices] = np.clip(moved.astype(np.float32) * inner + out[roi.slices].astype(np.float32) * (1.0 - inner) + 0.5, 0, 255).astype(np.uint8)
        result.append(copy)
    return result


def teeth_tint(outs: list[np.ndarray], stage: list[np.ndarray | None], level: float) -> list[np.ndarray]:
    """Changes the color inside the mouth only: no shape, so only the exact check can see it."""
    result = []
    for out, lm in zip(outs, stage):
        if lm is None:
            result.append(out)
            continue
        s = regions.face_scale(lm)
        roi = regions.roi_around(lm[regions.LIPS_INNER], out.shape, pad=0.0, margin_px=0.05 * s)
        if roi is None:
            result.append(out)
            continue
        inner = regions.polygon_mask(lm[regions.LIPS_INNER], roi)[..., None]
        copy = out.copy()
        copy[roi.slices] = np.clip(out[roi.slices].astype(np.float32) + level * inner + 0.5, 0, 255).astype(np.uint8)
        result.append(copy)
    return result


def eye_warp(outs: list[np.ndarray], stage: list[np.ndarray | None], pixels: float) -> list[np.ndarray]:
    """Shifts the picture inside the eye openings: another gaze, another iris position."""
    result = []
    for out, lm in zip(outs, stage):
        if lm is None:
            result.append(out)
            continue
        copy = out.copy()
        for side in regions.SIDES:
            indices = regions.side_indices(side).eye
            roi = regions.roi_around(lm[indices], out.shape, pad=0.1, margin_px=pixels + 2)
            if roi is None:
                continue
            moved = cv2.warpAffine(out[roi.slices], np.float32([[1, 0, pixels], [0, 1, 0]]), (roi.w, roi.h), borderMode=cv2.BORDER_REPLICATE)
            mask = regions.eye_opening_mask(lm, side, roi)[..., None]
            copy[roi.slices] = np.clip(moved.astype(np.float32) * mask + copy[roi.slices].astype(np.float32) * (1.0 - mask) + 0.5, 0, 255).astype(np.uint8)
        result.append(copy)
    return result


def background_patch(outs: list[np.ndarray], refs: list[ev.FrameRef], level: float) -> list[np.ndarray]:
    """A soft patch on the wall, away from the person, staying in one place."""
    first = next((ref for ref in refs if ref.far is not None and ref.far.any()), None)
    if first is None:
        return outs
    distance = cv2.distanceTransform(first.far.astype(np.uint8), cv2.DIST_L2, 3)  # at half size
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    radius = max(float(distance[y, x]) * 2.0 * 0.7, 8.0)
    h, w = outs[0].shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    bump = level * np.exp(-(((xs - 2 * x) ** 2 + (ys - 2 * y) ** 2) / (2 * radius**2)))
    return [_add_levels(out, bump) for out in outs]


def background_shift(outs: list[np.ndarray], refs: list[ev.FrameRef], level: float) -> list[np.ndarray]:
    """The whole far background a little brighter."""
    result = []
    for out, ref in zip(outs, refs):
        if ref.far is None:
            result.append(out)
            continue
        mask = cv2.resize(ref.far.astype(np.float32), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR)
        result.append(_add_levels(out, level * mask))
    return result


def spike(outs: list[np.ndarray], data: Rendered, zones: list[ev.Zones | None], level: float, frames: tuple[int, ...]) -> list[np.ndarray]:
    """A blob on the cheek for single frames."""
    result = list(outs)
    for frame_index in frames:
        if frame_index < len(result) and data.stage[frame_index] is not None:
            change = _cheek_patch(data.stage[frame_index], result[frame_index].shape[:2], level)
            result[frame_index] = _add_levels(result[frame_index], _through_weight(change, zones[frame_index]))
    return result


def mouth_delay(outs: list[np.ndarray], stage: list[np.ndarray | None], lag: int) -> list[np.ndarray]:
    """The lower face comes from ``lag`` frames ago: the lips run late."""
    result = []
    for index, (out, lm) in enumerate(zip(outs, stage)):
        old = max(index - lag, 0)
        if lm is None:
            result.append(out)
            continue
        s = regions.face_scale(lm)
        roi = regions.roi_around(lm[regions.LIPS_OUTER], out.shape, pad=0.0, margin_px=0.35 * s)
        if roi is None:
            result.append(out)
            continue
        ys, xs = np.mgrid[0 : roi.h, 0 : roi.w].astype(np.float32)
        cx, cy = roi.w / 2, roi.h / 2
        mask = np.clip(1.6 - np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2) * 1.6, 0.0, 1.0)[..., None]
        copy = out.copy()
        copy[roi.slices] = (outs[old][roi.slices].astype(np.float32) * mask + out[roi.slices].astype(np.float32) * (1.0 - mask) + 0.5).astype(np.uint8)
        result.append(copy)
    return result


def _through_zone(classic: np.ndarray, target: np.ndarray, zone: ev.Zones | None) -> np.ndarray:
    """Only what the runtime's weight lets through, so that the bit-exact checks cannot find it by accident."""
    out = classic.copy()
    if zone is None:
        return out
    weight = zone.weight[..., None]
    mixed = classic[zone.roi.slices].astype(np.float32) * (1.0 - weight) + target[zone.roi.slices].astype(np.float32) * weight
    out[zone.roi.slices] = np.clip(mixed + 0.5, 0, 255).astype(np.uint8)
    return out


def _warp_field(image: np.ndarray, field: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    height, width = image.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    dx, dy = field(xs, ys)
    return cv2.remap(image, (xs + dx).astype(np.float32), (ys + dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _each_face(outs: list[np.ndarray], stage: list[np.ndarray | None], zones: list[ev.Zones | None], make: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> list[np.ndarray]:
    return [out if lm is None else _through_zone(out, make(out, lm), zone) for out, lm, zone in zip(outs, stage, zones)]


def nose_wider(outs: list[np.ndarray], stage: list[np.ndarray | None], zones: list[ev.Zones | None], amount: float) -> list[np.ndarray]:
    """Content moved, no landmark asked: a model that makes the nose wider through a delta."""
    def make(out: np.ndarray, lm: np.ndarray) -> np.ndarray:
        s, center = regions.face_scale(lm), lm[regions.NOSE_TIP]
        return _warp_field(out, lambda xs, ys: (-(xs - center[0]) * amount * np.exp(-((xs - center[0]) ** 2 + (ys - center[1]) ** 2) / (0.45 * s) ** 2), 0.0 * xs))

    return _each_face(outs, stage, zones, make)


def jaw_slimmer(outs: list[np.ndarray], stage: list[np.ndarray | None], zones: list[ev.Zones | None], amount: float) -> list[np.ndarray]:
    def make(out: np.ndarray, lm: np.ndarray) -> np.ndarray:
        s, center = regions.face_scale(lm), lm[regions.NOSE_TIP]
        return _warp_field(out, lambda xs, ys: ((xs - center[0]) * amount * np.clip((ys - center[1]) / s, 0.0, 1.2), 0.0 * xs))

    return _each_face(outs, stage, zones, make)


def eyes_larger(outs: list[np.ndarray], stage: list[np.ndarray | None], zones: list[ev.Zones | None], amount: float) -> list[np.ndarray]:
    def make(out: np.ndarray, lm: np.ndarray) -> np.ndarray:
        s = regions.face_scale(lm)
        centers = [regions.eye_center(lm, side) for side in regions.SIDES]

        def field(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            dx, dy = 0.0 * xs, 0.0 * xs
            for center in centers:
                falloff = np.exp(-((xs - center[0]) ** 2 + (ys - center[1]) ** 2) / (0.5 * s) ** 2)
                dx, dy = dx - (xs - center[0]) * amount * falloff, dy - (ys - center[1]) * amount * falloff
            return dx, dy

        return _warp_field(out, field)

    return _each_face(outs, stage, zones, make)


def seam_across(outs: list[np.ndarray], stage: list[np.ndarray | None], zones: list[ev.Zones | None], level: float) -> list[np.ndarray]:
    """A hard step across the face at the nose: above it, everything is ``level`` brighter."""
    def make(out: np.ndarray, lm: np.ndarray) -> np.ndarray:
        target = out.astype(np.float32)
        target[: int(lm[regions.NOSE_TIP][1])] += level
        return np.clip(target, 0, 255).astype(np.uint8)

    return _each_face(outs, stage, zones, make)


def grid_on_face(outs: list[np.ndarray], stage: list[np.ndarray | None], zones: list[ev.Zones | None], level: float) -> list[np.ndarray]:
    """A grid fixed to the face in crop space: the classic artefact of a transposed convolution."""
    def make(out: np.ndarray, lm: np.ndarray) -> np.ndarray:
        grid = (np.indices((256, 256)).sum(axis=0) // 8 % 2).astype(np.float32) * 2.0 - 1.0
        inverse = cv2.invertAffineTransform(geometry.crop_matrix(lm, 256))
        delta = cv2.warpAffine(grid, inverse, (out.shape[1], out.shape[0]), flags=cv2.INTER_LINEAR)
        return np.clip(out.astype(np.float32) + delta[..., None] * level, 0, 255).astype(np.uint8)

    return _each_face(outs, stage, zones, make)


def one_side_cast(outs: list[np.ndarray], stage: list[np.ndarray | None], zones: list[ev.Zones | None], level: float) -> list[np.ndarray]:
    def make(out: np.ndarray, lm: np.ndarray) -> np.ndarray:
        target = out.astype(np.float32)
        target[:, : int(lm[regions.NOSE_TIP][0]), 2] += level
        return np.clip(target, 0, 255).astype(np.uint8)

    return _each_face(outs, stage, zones, make)


def whole_cast(outs: list[np.ndarray], stage: list[np.ndarray | None], zones: list[ev.Zones | None], level: float) -> list[np.ndarray]:
    def make(out: np.ndarray, lm: np.ndarray) -> np.ndarray:
        target = out.astype(np.float32)
        target[..., 2] += level
        return np.clip(target, 0, 255).astype(np.uint8)

    return _each_face(outs, stage, zones, make)


def paint_on_rim(outs: list[np.ndarray], zones: list[ev.Zones | None], level: float) -> list[np.ndarray]:
    """Paint where a mask is unsure and the face is not: wall, collar, shoulders, through the weight."""
    painted = []
    for out, zone in zip(outs, zones):
        result = out.copy()
        if zone is not None:
            rim = (zone.weight > 0.02) & (zone.weight < 0.6)
            view = result[zone.roi.slices].astype(np.float32)
            view[rim] += (level * zone.weight[rim])[..., None]
            result[zone.roi.slices] = np.clip(view + 0.5, 0, 255).astype(np.uint8)
        painted.append(result)
    return painted


def frame_locked_patch(outs: list[np.ndarray], stage: list[np.ndarray | None], zones: list[ev.Zones | None], level: float) -> list[np.ndarray]:
    """A soft patch fixed to the frame, not to the face. It can only be told from makeup when the face moves."""
    first = next((lm for lm in stage if lm is not None), None)
    if first is None:
        return outs
    s = regions.face_scale(first)
    center = first[regions.NOSE_TIP] + np.array([0.5 * s, 0.3 * s])
    ys, xs = np.mgrid[0 : outs[0].shape[0], 0 : outs[0].shape[1]].astype(np.float32)
    bump = np.exp(-((xs - center[0]) ** 2 + (ys - center[1]) ** 2) / (0.3 * s) ** 2)

    def make(out: np.ndarray, lm: np.ndarray) -> np.ndarray:
        target = out.astype(np.float32)
        target[..., 2] += level * bump
        target[..., 0] -= 0.5 * level * bump
        return np.clip(target, 0, 255).astype(np.uint8)

    return _each_face(outs, stage, zones, make)


def delay_all(outs: list[np.ndarray], lag: int) -> list[np.ndarray]:
    return [outs[max(index - lag, 0)] for index in range(len(outs))]


def freeze(outs: list[np.ndarray], start: int, length: int) -> list[np.ndarray]:
    result = list(outs)
    for index in range(start + 1, min(start + length, len(result))):
        result[index] = result[start]
    return result


def heavy(outs: list[np.ndarray], reference: list[np.ndarray], factor: float) -> list[np.ndarray]:
    """The applied change, multiplied: a look that is merely too strong."""
    return [np.clip(rf.astype(np.float32) + (out.astype(np.float32) - rf.astype(np.float32)) * factor, 0, 255).astype(np.uint8) for out, rf in zip(outs, reference)]


# ---- The identity guards need another person's face. Crude on purpose, only ever measured, never saved.


def _track_all(frames: list[np.ndarray], fps: float) -> list[np.ndarray | None]:
    tracker = ev.default_tracker_factory(0.0)
    try:
        return [tracker.detect(frame, round(index * 1000.0 / fps))[0] for index, frame in enumerate(frames)]
    finally:
        tracker.close()


def _match_colors(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    good = mask > 0.5
    if not good.any():
        return source
    for channel in range(3):
        sa, sb = a[..., channel][good], b[..., channel][good]
        a[..., channel] = (a[..., channel] - sa.mean()) * (sb.std() / max(sa.std(), 1.0)) + sb.mean()
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


_SWAP_ANCHORS = [33, 133, 362, 263, 1, 2, 61, 291]
_SWAP_POINTS = sorted(set(regions.FACE_OVAL) | set(regions.LIPS_OUTER) | {33, 133, 362, 263, 159, 145, 386, 374, 70, 105, 107, 300, 334, 336, 168, 6, 197, 1, 2, 98, 327})


def swap_similarity(outs: list[np.ndarray], donor: list[np.ndarray], donor_lm: list[np.ndarray | None], raw: list[np.ndarray | None]) -> list[np.ndarray]:
    """Another person's face aligned by a similarity transform: the face's proportions change."""
    result = []
    for index, (out, lm) in enumerate(zip(outs, raw)):
        d = index % len(donor)
        if lm is None or donor_lm[d] is None:
            result.append(out)
            continue
        matrix = cv2.estimateAffinePartial2D(donor_lm[d][_SWAP_ANCHORS].astype(np.float32), lm[_SWAP_ANCHORS].astype(np.float32))[0]
        if matrix is None:
            result.append(out)
            continue
        warped = cv2.warpAffine(donor[d], matrix, (out.shape[1], out.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        result.append(_blend_face(out, warped, lm))
    return result


def swap_warped(outs: list[np.ndarray], donor: list[np.ndarray], donor_lm: list[np.ndarray | None], raw: list[np.ndarray | None]) -> list[np.ndarray]:
    """Another person's face warped onto this one's landmarks: the proportions stay, the face does not."""
    result = []
    for index, (out, lm) in enumerate(zip(outs, raw)):
        d = index % len(donor)
        if lm is None or donor_lm[d] is None:
            result.append(out)
            continue
        result.append(_blend_face(out, _piecewise_warp(donor[d], donor_lm[d][_SWAP_POINTS], lm[_SWAP_POINTS], out.shape), lm))
    return result


def _piecewise_warp(source: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    from scipy.spatial import Delaunay

    out = np.zeros(shape, np.uint8)
    for tri in Delaunay(dst_pts).simplices:
        s_tri, d_tri = src_pts[tri].astype(np.float32), dst_pts[tri].astype(np.float32)
        x0, y0 = np.floor(d_tri.min(axis=0)).astype(int)
        x1, y1 = np.ceil(d_tri.max(axis=0)).astype(int) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, shape[1]), min(y1, shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        matrix = cv2.getAffineTransform(s_tri, d_tri - np.float32([x0, y0]))
        patch = cv2.warpAffine(source, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.round(d_tri - np.float32([x0, y0])).astype(np.int32), 1)
        region = out[y0:y1, x0:x1]
        region[mask > 0] = patch[mask > 0]
    return out


def _blend_face(out: np.ndarray, warped: np.ndarray, lm: np.ndarray) -> np.ndarray:
    s = regions.face_scale(lm)
    roi = regions.Roi(0, 0, out.shape[1], out.shape[0])
    mask = regions.polygon_mask(regions.expand_polygon(lm[regions.FACE_OVAL], -0.12 * s), roi, feather_px=0.06 * s)
    matched = _match_colors(warped, out, mask)
    return np.clip(matched.astype(np.float32) * mask[..., None] + out.astype(np.float32) * (1.0 - mask[..., None]) + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# One clip, all scenarios
# --------------------------------------------------------------------------


def clip_name(actor: str, take: str) -> str:
    return f"{actor}_{take}"


def donor_of(actor: str, take: str, seed_offset: int = 5) -> str:
    """A different actor's clip of the same take."""
    index = ACTORS.index(actor)
    return clip_name(ACTORS[(index + seed_offset) % len(ACTORS)], take)


def run_clip(name: str, vids: Path, out_dir: Path, frames_limit: int | None = None) -> dict[str, Any]:
    cv2.setNumThreads(2)
    path = vids / f"{name}.flv"
    fps = ev.file_fps(path)
    frames = list(ev.read_frames(path, frames_limit))
    actor, take = name[:4], name[5:]
    seed = int(actor) * 7 + TAKES.index(take)
    store: dict[str, dict[str, float | None]] = {}
    series: dict[str, dict[str, np.ndarray]] = {}
    events: dict[str, list[dict[str, Any]]] = {}

    def record(key: str, result: ev.ClipResult) -> None:
        store[key] = {k: ev._rounded(v) for k, v in result.metrics.items()}
        events[key] = [e.as_dict() for e in result.events]
        series[key] = result.series

    # ---- Reference: the bare input. Output: the classic presets.
    natural = render(frames, "natural", fps)
    refs_input = build_refs(natural, natural.bare, fps, same_as_bare=True)
    record("input/clean_natural", measure(name, ev.MODE_INPUT, natural, natural.bare, refs_input, natural.classic, fps))
    for preset in ("polished", "strong"):
        other = render(frames, preset, fps)
        record(f"input/clean_{preset}", measure(name, ev.MODE_INPUT, other, other.bare, build_refs(other, other.bare, fps, True), other.classic, fps))
        del other

    zones_input = [ref.zones for ref in refs_input]
    raw = [ref.raw_bare for ref in refs_input]
    donor_frames = list(ev.read_frames(vids / f"{donor_of(actor, take)}.flv", frames_limit))
    donor_lm = _track_all(donor_frames, fps)

    # Severity sweeps only on a quarter of the clips (one take per actor, in turn): they are slow, and say how far detection reaches.
    sweep = (ACTORS.index(actor) + TAKES.index(take)) % 4 == 0

    def run_input(key: str, outputs: list[np.ndarray], track: bool = True) -> None:
        record(f"input/{key}", measure(name, ev.MODE_INPUT, natural, natural.bare, refs_input, outputs, fps, track))

    classic = natural.classic
    run_input("shimmer_face_2", shimmer_face(classic, zones_input, 2.0, seed), False)
    run_input("pump_face_2", pump_face(classic, zones_input, 2.0), False)
    run_input("shimmer_bg_2", shimmer_background(classic, refs_input, 2.0, seed), False)
    run_input("teeth_warp_2", teeth_warp(classic, natural.stage, 2.0), False)
    run_input("eye_warp_2", eye_warp(classic, natural.stage, 2.0), False)
    run_input("background_patch_8", background_patch(classic, refs_input, 8.0), False)
    run_input("background_shift_3", background_shift(classic, refs_input, 3.0), False)
    run_input("paint_shift_0.06", paint_shift(classic, natural.bare, natural.stage, 0.06))
    run_input("paint_shift_0.12", paint_shift(classic, natural.bare, natural.stage, 0.12))
    run_input("spike_24", spike(classic, natural, zones_input, 24.0, (30, 50)), False)
    run_input("mouth_delay_2", mouth_delay(classic, natural.stage, 2))
    run_input("swap_similarity", swap_similarity(classic, donor_frames, donor_lm, raw))
    run_input("swap_warped", swap_warped(classic, donor_frames, donor_lm, raw))
    if sweep:
        run_input("shimmer_face_4", shimmer_face(classic, zones_input, 4.0, seed), False)
        run_input("spike_12", spike(classic, natural, zones_input, 12.0, (30, 50)), False)
        run_input("delay_all_2", delay_all(classic, 2))
        run_input("freeze_6", freeze(classic, 30, 6), False)

    # ---- Drawing with bad landmarks, through the real effects.
    rng = np.random.default_rng(seed)
    jump_frames = {}
    for frame_index in (30, 50):
        direction = rng.normal(0.0, 1.0, 2).astype(np.float32)
        jump_frames[frame_index] = direction / max(float(np.linalg.norm(direction)), 1e-6) * 0.08
    glitch = render(frames, "natural", fps, tracker_wrap=lambda t: GlitchTracker(t, jumps=jump_frames))
    record("input/tracker_jump", measure(name, ev.MODE_INPUT, glitch, glitch.bare, build_refs(glitch, glitch.bare, fps, True), glitch.classic, fps))
    del glitch
    offset = render(frames, "natural", fps, tracker_wrap=lambda t: GlitchTracker(t, offset=np.array([0.06, 0.08], np.float32)))
    record("input/tracker_offset", measure(name, ev.MODE_INPUT, offset, offset.bare, build_refs(offset, offset.bare, fps, True), offset.classic, fps))
    del offset

    # ---- Reference: the classic output. Output: an imitation of a well-behaved model.
    refs_classic = build_refs(natural, natural.classic, fps, same_as_bare=False)
    zones = [ref.zones for ref in refs_classic]
    clean = imitate_model(natural, zones, "tint")

    def run_classic(key: str, outputs: list[np.ndarray], track: bool = True) -> None:
        record(f"classic/{key}", measure(name, ev.MODE_CLASSIC, natural, natural.classic, refs_classic, outputs, fps, track))

    run_classic("clean_tint", clean)
    run_classic("clean_smooth", imitate_model(natural, zones, "smooth"))
    run_classic("shimmer_face_2", shimmer_face(clean, zones, 2.0, seed), False)
    run_classic("pump_face_2", pump_face(clean, zones, 2.0), False)
    run_classic("shimmer_bg_2", shimmer_background(clean, refs_classic, 2.0, seed), False)
    run_classic("teeth_warp_1", teeth_warp(clean, natural.stage, 1.0), False)
    run_classic("teeth_tint_3", teeth_tint(clean, natural.stage, 3.0), False)
    run_classic("eye_warp_2", eye_warp(clean, natural.stage, 2.0), False)
    run_classic("background_patch_4", background_patch(clean, refs_classic, 4.0), False)
    run_classic("background_shift_2", background_shift(clean, refs_classic, 2.0), False)
    run_classic("spike_12", spike(clean, natural, zones, 12.0, (30, 50)), False)
    run_classic("mouth_delay_1", mouth_delay(clean, natural.stage, 1))
    run_classic("freeze_6", freeze(clean, 30, 6), False)
    run_classic("heavy_20x", heavy(clean, natural.classic, 20.0), False)
    run_classic("warp_nose_15", nose_wider(clean, natural.stage, zones, 0.15), False)
    run_classic("warp_nose_30", nose_wider(clean, natural.stage, zones, 0.30), False)
    run_classic("warp_jaw_20", jaw_slimmer(clean, natural.stage, zones, 0.20), False)
    run_classic("warp_eyes_10", eyes_larger(clean, natural.stage, zones, 0.10), False)
    run_classic("seam_10", seam_across(clean, natural.stage, zones, 10.0), False)
    run_classic("seam_20", seam_across(clean, natural.stage, zones, 20.0), False)
    run_classic("grid_5", grid_on_face(clean, natural.stage, zones, 5.0), False)
    run_classic("side_cast_8", one_side_cast(clean, natural.stage, zones, 8.0), False)
    run_classic("skin_cast_25", whole_cast(clean, natural.stage, zones, 25.0), False)
    run_classic("rim_paint_20", paint_on_rim(clean, zones, 20.0), False)
    run_classic("swap_similarity", swap_similarity(clean, donor_frames, donor_lm, raw))
    run_classic("swap_warped", swap_warped(clean, donor_frames, donor_lm, raw))
    if sweep:
        run_classic("clean_noop", natural.classic)
        run_classic("shimmer_face_1", shimmer_face(clean, zones, 1.0, seed), False)
        run_classic("shimmer_face_4", shimmer_face(clean, zones, 4.0, seed), False)
        run_classic("pump_face_1", pump_face(clean, zones, 1.0), False)
        run_classic("shimmer_bg_1", shimmer_background(clean, refs_classic, 1.0, seed), False)
        run_classic("teeth_warp_2", teeth_warp(clean, natural.stage, 2.0), False)
        run_classic("teeth_warp_4", teeth_warp(clean, natural.stage, 4.0), False)
        run_classic("eye_warp_1", eye_warp(clean, natural.stage, 1.0), False)
        run_classic("background_patch_8", background_patch(clean, refs_classic, 8.0), False)
        run_classic("spike_6", spike(clean, natural, zones, 6.0, (30, 50)), False)
        run_classic("spike_24", spike(clean, natural, zones, 24.0, (30, 50)), False)
        run_classic("mouth_delay_2", mouth_delay(clean, natural.stage, 2))
        run_classic("mouth_delay_3", mouth_delay(clean, natural.stage, 3))
        run_classic("delay_all_1", delay_all(clean, 1))
        run_classic("delay_all_3", delay_all(clean, 3))
    # What the closures above hold is not needed any more: the next part is the largest.
    refs_classic = refs_input = natural = None
    del zones, clean, zones_input, classic

    _run_swaying(name, frames, fps, sweep, record)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {f"{key}|{k}": v.astype(np.float32) for key, per in series.items() for k, v in per.items()}
    np.savez_compressed(out_dir / f"{name}.npz", **arrays)
    summary = {"clip": name, "fps": fps, "frames": len(frames), "metrics": store, "events": events}
    (out_dir / f"{name}.json").write_text(json.dumps(summary), encoding="utf-8")
    return {"clip": name, "frames": len(frames)}


def _run_swaying(name: str, frames: list[np.ndarray], fps: float, sweep: bool, record: Callable[[str, ev.ClipResult], None]) -> None:
    """The same clip, swaying: the lag detectors need motion."""
    moving = render(list(ev.sway_frames(frames)), "natural", fps)
    refs_pan = build_refs(moving, moving.bare, fps, True)
    record("pan/clean_natural", measure(name, ev.MODE_INPUT, moving, moving.bare, refs_pan, moving.classic, fps))
    for lag in (1, 2) + ((3, 5) if sweep else ()):
        record(f"pan/paint_lag_{lag}", measure(name, ev.MODE_INPUT, moving, moving.bare, refs_pan, paint_lag(moving.classic, moving.bare, lag), fps, False))
    refs_pan_c = build_refs(moving, moving.classic, fps, False)
    zones_pan = [ref.zones for ref in refs_pan_c]
    clean_pan = imitate_model(moving, zones_pan, "tint")
    record("pan_classic/clean_tint", measure(name, ev.MODE_CLASSIC, moving, moving.classic, refs_pan_c, clean_pan, fps))
    record("pan_classic/clean_smooth", measure(name, ev.MODE_CLASSIC, moving, moving.classic, refs_pan_c, imitate_model(moving, zones_pan, "smooth"), fps))
    for lag in (2,) + ((1, 3, 5) if sweep else ()):
        record(f"pan_classic/patch_lag_{lag}", measure(name, ev.MODE_CLASSIC, moving, moving.classic, refs_pan_c, patch_lag(clean_pan, moving, zones_pan, 12.0, lag), fps, False))
    record("pan_classic/frame_patch_8", measure(name, ev.MODE_CLASSIC, moving, moving.classic, refs_pan_c, frame_locked_patch(clean_pan, moving.stage, zones_pan, 8.0), fps, False))
    del moving, refs_pan, refs_pan_c, zones_pan, clean_pan
    # A nod and a bow: the clean runs must stay clean.
    nodding = render(list(ev.nod_frames(frames)), "natural", fps)
    record("nod/clean_natural", measure(name, ev.MODE_INPUT, nodding, nodding.bare, build_refs(nodding, nodding.bare, fps, True), nodding.classic, fps))
    refs_nod = build_refs(nodding, nodding.classic, fps, False)
    zones_nod = [ref.zones for ref in refs_nod]
    record("nod_classic/clean_tint", measure(name, ev.MODE_CLASSIC, nodding, nodding.classic, refs_nod, imitate_model(nodding, zones_nod, "tint"), fps))
    record("nod_classic/clean_smooth", measure(name, ev.MODE_CLASSIC, nodding, nodding.classic, refs_nod, imitate_model(nodding, zones_nod, "smooth"), fps))


def rerun_swaying(name: str, vids: Path, out_dir: Path) -> dict[str, Any]:
    """Only the swaying scenarios, merged into what is stored: when only the lag measures changed."""
    cv2.setNumThreads(2)
    path = vids / f"{name}.flv"
    fps = ev.file_fps(path)
    frames = list(ev.read_frames(path))
    actor, take = name[:4], name[5:]
    sweep = (ACTORS.index(actor) + TAKES.index(take)) % 4 == 0
    stored = json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))
    with np.load(out_dir / f"{name}.npz") as old:
        arrays = {key: old[key] for key in old.files}

    def record(key: str, result: ev.ClipResult) -> None:
        stored["metrics"][key] = {k: ev._rounded(v) for k, v in result.metrics.items()}
        stored["events"][key] = [e.as_dict() for e in result.events]
        for series_key in [k for k in arrays if k.startswith(f"{key}|")]:
            del arrays[series_key]
        arrays.update({f"{key}|{k}": v.astype(np.float32) for k, v in result.series.items()})

    _run_swaying(name, frames, fps, sweep, record)
    np.savez_compressed(out_dir / f"{name}.npz", **arrays)
    (out_dir / f"{name}.json").write_text(json.dumps(stored), encoding="utf-8")
    return {"clip": name, "frames": len(frames)}


def identity_clip(name: str, vids: Path, model: Path) -> dict[str, Any]:
    """An optional embedding model on one clip: cosine with the bare frame, for clean presets and for another face."""
    cv2.setNumThreads(2)
    logging.disable(logging.WARNING)
    embedder = ev.SFaceEmbedder(model)
    path = vids / f"{name}.flv"
    fps = ev.file_fps(path)
    frames = list(ev.read_frames(path))
    actor, take = name[:4], name[5:]
    donor_frames = list(ev.read_frames(vids / f"{donor_of(actor, take)}.flv"))
    donor_lm = _track_all(donor_frames, fps)
    raw = _track_all(frames, fps)
    natural = render(frames, "natural", fps)
    strong = render(frames, "strong", fps)
    cases = {
        "natural": natural.classic,
        "strong": strong.classic,
        "swap_similarity": swap_similarity(natural.classic, donor_frames, donor_lm, raw),
        "swap_warped": swap_warped(natural.classic, donor_frames, donor_lm, raw),
    }
    out: dict[str, Any] = {"clip": name}
    for key, outputs in cases.items():
        tracked = _track_all(outputs, fps)
        values = []
        for index in range(ev.SETTLE_FRAMES, len(frames), 4):
            if raw[index] is None or tracked[index] is None:
                continue
            a, b = embedder.embed(frames[index], raw[index]), embedder.embed(outputs[index], tracked[index])
            if a is not None and b is not None:
                values.append(float(a @ b))
        out[key] = {"p01": float(np.percentile(values, 1)), "median": float(np.median(values)), "n": len(values)} if values else None
    return out


def _worker(args: tuple[str, str, str, int | None, bool]) -> dict[str, Any]:
    name, vids, out, limit, swaying_only = args
    logging.disable(logging.WARNING)
    # No timing is measured here, so these workers may also use the second hardware threads the pipeline leaves free.
    from . import pipeline

    pipeline._use_one_thread_per_core = lambda: None
    started = time.perf_counter()
    result = rerun_swaying(name, Path(vids), Path(out)) if swaying_only else run_clip(name, Path(vids), Path(out), limit)
    result["seconds"] = round(time.perf_counter() - started, 1)
    return result


# --------------------------------------------------------------------------
# False alarms and misses
# --------------------------------------------------------------------------

# Which detectors each scenario is meant for. "event:kind" is a breakdown event, anything else a metric.
_TEETH = ("mouth_interior_max", "mouth_inner_edge_corr", "mouth_band_edge_corr", "event:teeth_altered")
_EYES = ("eye_interior_max", "eye_edge_corr", "event:eyes_altered")
_BACKGROUND = ("outside_max", "background_change", "event:background_changed")
_SWAP = ("identity_structure_p01", "landmark_shift_p99", "identity_geometry_p99", "event:identity_shift")
_LATENCY = ("mouth_latency_frames",)
_LAG = ("paint_lag_frames", "paint_lag_local_frames", "event:paint_behind")
EXPECTED: dict[str, tuple[str, ...]] = {
    "shimmer_face": ("flicker_ratio", "delta_flicker"),
    "pump_face": ("flicker_ratio", "flicker_block_ratio", "delta_flicker"),
    "shimmer_bg": ("flicker_outside_ratio", *_BACKGROUND),
    "teeth_warp": _TEETH,
    "teeth_tint": ("mouth_interior_max",),
    "eye_warp": _EYES,
    "background_patch": _BACKGROUND,
    "background_shift": _BACKGROUND,
    "spike": ("event:spike",),
    "mouth_delay": _LATENCY,
    "delay_all": _LATENCY,
    "freeze": ("event:frozen",),
    "heavy": ("delta_abs_p99",),
    "swap_similarity": _SWAP,
    "swap_warped": _SWAP,
    "patch_lag": _LAG,
    "frame_patch": ("frame_locked_ratio", *_LAG),
    "warp_": ("warp_flow_px",),
    "seam_": ("seam_edge",),
    "grid_": ("seam_edge",),
    "side_cast": ("side_cast",),
    "skin_cast": ("skin_cast",),
    "rim_paint": ("rim_change",),
    "paint_lag": _LAG,
    "paint_shift": ("paint_offset", "event:off_target"),
    "tracker_jump": ("event:landmark_jump",),
    "tracker_offset": ("paint_offset", "event:off_target"),
}
# Detectors that do not apply in a mode: they are not counted, for false alarms or for detection.
_NOT_IN_INPUT_MODE = {"mouth_interior_max", "eye_interior_max", "outside_max", "delta_abs_p99", "paint_lag_local_frames", "warp_flow_px", "seam_edge", "side_cast", "skin_cast", "rim_change", "frame_locked_ratio", "paste_drift_p99"}
# Only a running model and the clock can show these: nothing here measures them. Not by ``modes``: the metrics only
# meant for the classic reference have the same modes.
_LIVE_ONLY = {"all_kernels_dml", "rtt_p95_ms", "rtt_p99_ms", "stage_p95_ms", "fps_loss", "miss_pct", "vram_mb", "strength_zero_identical", "strength_monotonic", "fade_drill"}
_NOT_CLIP_METRICS = _LIVE_ONLY | {"breaks_per_30min", "identity_cosine_p01", "paste_drift_p99"}


def expected_for(scenario: str) -> tuple[str, ...]:
    kind = scenario.split("/", 1)[1]
    for prefix, detectors in EXPECTED.items():
        if kind.startswith(prefix):
            return detectors
    return ()


def metric_fires(key: str, value: float | None, mode: str = ev.MODE_CLASSIC) -> bool:
    """Would this clip's value miss the limit that applies to a single clip?"""
    threshold = ev.THRESHOLD_BY_KEY[key]
    median_limit, worst_limit = threshold.limits(mode)
    if value is None or median_limit is None or mode not in threshold.modes:
        return False
    limit = worst_limit if worst_limit is not None else median_limit
    return value > limit if threshold.op == "le" else value < limit


def replay_events(series: dict[str, np.ndarray], mode: str, rules: dict[str, float], settle: int = ev.SETTLE_FRAMES) -> dict[str, int]:
    """Events a stored clip gives under other rules: the same code as live."""
    state = ev.FindingState()
    recorder = ev.EventRecorder("replay", None)
    frames = len(next(iter(series.values())))
    for index in range(settle, frames):
        sample = {key: float(values[index]) for key, values in series.items()}
        for kind, peak in ev.frame_findings(sample, state, rules, mode):
            recorder.add(kind, index, peak)
    counts: dict[str, int] = {}
    for event in recorder.events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    return counts


def scenario_mode(scenario: str) -> str:
    return ev.MODE_CLASSIC if scenario.startswith(("classic/", "pan_classic/")) else ev.MODE_INPUT


def load_results(folder: Path, rules: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
    """scenario -> clip -> {"metrics": ..., "events": {kind: count}}. With ``rules``, events come from replaying the series."""
    table: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("*.json")):
        if not path.with_suffix(".npz").is_file():
            continue  # the summary itself
        data = json.loads(path.read_text(encoding="utf-8"))
        arrays = np.load(path.with_suffix(".npz")) if rules is not None else None
        for scenario, metrics in data["metrics"].items():
            if arrays is None:
                counts: dict[str, int] = {}
                for event in data["events"][scenario]:
                    counts[event["kind"]] = counts.get(event["kind"], 0) + 1
            else:
                series = {name.split("|", 1)[1]: arrays[name].astype(np.float64) for name in arrays.files if name.startswith(f"{scenario}|")}
                counts = replay_events(series, scenario_mode(scenario), rules)
                # Metrics come from the same series, under the current code. The lag's fit is not in them: its stored value is.
                moving = metrics.get("paint_lag_motion") or 0.0
                lag = metrics.get("paint_lag_frames")
                stored = (float("nan") if lag is None or moving < ev.LagEstimator.MIN_FRAMES else lag, moving)
                local = metrics.get("paint_lag_local_frames")
                metrics = {key: ev._rounded(value) for key, value in ev.clip_metrics(series, ev.SETTLE_FRAMES, stored, scenario_mode(scenario), float("nan") if local is None else local).items()}
            table.setdefault(scenario, {})[data["clip"]] = {"metrics": metrics, "events": counts}
    return table


def _clip_fires(detector: str, clip: dict[str, Any], mode: str = ev.MODE_CLASSIC) -> bool:
    if detector.startswith("event:"):
        return clip["events"].get(detector[6:], 0) > 0
    return metric_fires(detector, clip["metrics"].get(detector), mode)


def summarize(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per scenario: share of clips each detector fires on. Clean: false alarms. Defect: misses of all its detectors."""
    metric_keys = [t.key for t in ev.THRESHOLDS if t.value is not None and t.key not in _NOT_CLIP_METRICS]
    event_kinds = ["spike", "off_target", "landmark_jump", "teeth_altered", "eyes_altered", "background_changed", "paint_behind", "face_lost", "frozen", "identity_shift"]
    out: dict[str, Any] = {"clean": {}, "defects": {}}
    for scenario, clips in sorted(table.items()):
        mode = scenario_mode(scenario)
        expected = [d for d in expected_for(scenario) if mode == ev.MODE_CLASSIC or d not in _NOT_IN_INPUT_MODE]
        if "clean" in scenario:
            fires = {}
            for key in metric_keys:
                if mode == ev.MODE_INPUT and key in _NOT_IN_INPUT_MODE:
                    continue
                count = sum(metric_fires(key, clip["metrics"].get(key), mode) for clip in clips.values())
                if count:
                    fires[key] = count
            for kind in event_kinds:
                count = sum(clip["events"].get(kind, 0) > 0 for clip in clips.values())
                if count:
                    fires["event:" + kind] = count
            any_alarm = sum(any(_clip_fires(k, clip, mode) for k in fires) for clip in clips.values())
            total_events = sum(sum(clip["events"].values()) for clip in clips.values())
            out["clean"][scenario] = {"clips": len(clips), "clips_with_any_alarm": any_alarm, "fires": fires, "events": total_events}
        else:
            detail = {d: sum(_clip_fires(d, clip, mode) for clip in clips.values()) for d in expected}
            caught = sum(any(_clip_fires(d, clip, mode) for d in expected) for clip in clips.values())
            out["defects"][scenario] = {"clips": len(clips), "caught": caught, "missed": len(clips) - caught, "by_detector": detail}
    return out


def per_detector(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per metric and event kind, and per mode: false alarms on clean clips, and how often it fires on the defects meant for it."""
    metric_keys = [t.key for t in ev.THRESHOLDS if t.value is not None and t.key not in _NOT_CLIP_METRICS]
    event_kinds = [f"event:{kind}" for kind in ("spike", "off_target", "landmark_jump", "teeth_altered", "eyes_altered", "background_changed", "paint_behind", "frozen", "identity_shift")]
    out: dict[str, Any] = {}
    for name, modes in (("classic", ("classic/", "pan_classic/")), ("input", ("input/", "pan/"))):
        mode = ev.MODE_CLASSIC if name == "classic" else ev.MODE_INPUT
        for detector in metric_keys + event_kinds:
            if not detector.startswith("event:") and mode not in ev.THRESHOLD_BY_KEY[detector].modes:
                continue
            row: dict[str, Any] = {"clean_clips": 0, "false_alarms": 0, "defect_clips": 0, "fired": 0, "defects": {}}
            for scenario, clips in table.items():
                if not scenario.startswith(modes):
                    continue
                if "clean" in scenario:
                    measured = [c for c in clips.values() if detector.startswith("event:") or c["metrics"].get(detector) is not None]
                    row["clean_clips"] += len(measured)
                    row["false_alarms"] += sum(_clip_fires(detector, c, mode) for c in measured)
                elif detector in expected_for(scenario):
                    fired = sum(_clip_fires(detector, c, mode) for c in clips.values())
                    row["defect_clips"] += len(clips)
                    row["fired"] += fired
                    row["defects"][scenario.split("/", 1)[1]] = f"{fired}/{len(clips)}"
            out[f"{detector}|{name}"] = row
    return out


def distributions(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """For choosing thresholds: per metric, the clean range and the range under each defect meant for it."""
    out: dict[str, Any] = {}
    keys = [t.key for t in ev.THRESHOLDS if t.value is not None and t.key not in _NOT_CLIP_METRICS] + ["background_block_max"]
    for scenario, clips in sorted(table.items()):
        per = {}
        for key in keys:
            values = [clip["metrics"].get(key) for clip in clips.values()]
            good = np.array([v for v in values if v is not None], np.float64)
            if good.size:
                per[key] = {"min": float(good.min()), "p05": float(np.percentile(good, 5)), "median": float(np.median(good)), "p95": float(np.percentile(good, 95)), "max": float(good.max()), "n": int(good.size)}
        out[scenario] = per
    return out


def split_check(table: dict[str, dict[str, Any]], margin: float = 1.15) -> dict[str, Any]:
    """Thresholds set from half the actors: how do they do on the other half? A check on overfitting to these 44 clips.

    The limit is the worst clean value of the first half, with a margin. Reported: clean clips of the second half over
    it (false alarms) and defect clips of the second half inside it (misses for that metric alone).
    """
    first, second = set(ACTORS[::2]), set(ACTORS[1::2])
    results: dict[str, Any] = {}
    for threshold in ev.THRESHOLDS:
        if threshold.value is None or threshold.key in _NOT_CLIP_METRICS:
            continue
        for mode_prefixes, mode_name in ((("classic/", "pan_classic/"), "classic"), (("input/", "pan/"), "input")):
            if mode_name == "input" and threshold.key in _NOT_IN_INPUT_MODE:
                continue
            clean = {name: clips for name, clips in table.items() if "clean" in name and name.startswith(mode_prefixes)}
            values_a = [c["metrics"].get(threshold.key) for clips in clean.values() for clip, c in clips.items() if clip[:4] in first]
            values_a = [v for v in values_a if v is not None]
            if not values_a:
                continue
            high = threshold.op == "le"
            base = max(values_a) if high else min(values_a)
            limit = base * margin + (0.0 if base else 1e-6) if high else base - abs(base) * (margin - 1.0)
            values_b = [c["metrics"].get(threshold.key) for clips in clean.values() for clip, c in clips.items() if clip[:4] in second]
            values_b = [v for v in values_b if v is not None]
            alarms = sum((v > limit) if high else (v < limit) for v in values_b)
            detail = {"limit_from_first_half": limit, "clean_second_half": len(values_b), "false_alarms_second_half": int(alarms), "defects": {}}
            for scenario, clips in table.items():
                if "clean" in scenario or not scenario.startswith(mode_prefixes) or threshold.key not in expected_for(scenario):
                    continue
                values = [c["metrics"].get(threshold.key) for clip, c in clips.items() if clip[:4] in second]
                values = [v for v in values if v is not None]
                if values:
                    missed = sum((v <= limit) if high else (v >= limit) for v in values)
                    detail["defects"][scenario] = {"clips": len(values), "missed": int(missed)}
            results[f"{threshold.key}|{mode_name}"] = detail
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Injected defects and clean runs for the evaluation tool")
    parser.add_argument("--summarize", action="store_true", help="only sum up what is under --out, into summary.json")
    parser.add_argument("--replay", action="store_true", help="with --summarize: events from the stored series, under the current rules")
    parser.add_argument("--vids", type=Path, default=Path(".test-media/testvids"))
    parser.add_argument("--out", type=Path, required=True, help="folder for series and metrics (not tracked by git)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--clips", nargs="*", default=None, help="clip names; default: all 44")
    parser.add_argument("--frames", type=int, default=None, help="only this many frames per clip (a quick try)")
    parser.add_argument("--force", action="store_true", help="measure clips again that already have results")
    parser.add_argument("--swaying-only", action="store_true", help="measure only the swaying scenarios again, into the stored results")
    parser.add_argument("--identity-model", type=Path, help="only try this optional embedding model on clean and swapped clips")
    args = parser.parse_args(argv)
    if args.summarize:
        table = load_results(args.out, ev.EVENT_RULES if args.replay else None)
        summary = {"clips": len({clip for clips in table.values() for clip in clips}), "summary": summarize(table), "per_detector": per_detector(table), "split_check": split_check(table), "distributions": distributions(table)}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(json.dumps(summary["summary"], indent=0)[:200000])
        return 0
    names = args.clips or [clip_name(a, t) for a in ACTORS for t in TAKES]
    if args.identity_model is not None:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(identity_clip, names, [args.vids] * len(names), [args.identity_model] * len(names)))
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "identity_model.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        for key in ("natural", "strong", "swap_similarity", "swap_warped"):
            values = [row[key]["p01"] for row in rows if row[key]]
            print(key, "p01 over clips: min", round(min(values), 3), "median", round(float(np.median(values)), 3), "max", round(max(values), 3))
        return 0
    todo = [n for n in names if args.force or args.swaying_only or not (args.out / f"{n}.json").is_file()]
    print(f"{len(todo)} clips to measure, {len(names) - len(todo)} done", flush=True)
    if args.workers <= 1:
        for name in todo:
            print(_worker((name, str(args.vids), str(args.out), args.frames, args.swaying_only)), flush=True)
        return 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, (n, str(args.vids), str(args.out), args.frames, args.swaying_only)): n for n in todo}
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:  # the others go on; the clip stays missing and is measured next time
                print(f"{futures[future]} failed: {exc!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
