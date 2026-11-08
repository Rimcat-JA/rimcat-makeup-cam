"""The evaluation tool, with fake tracker, segmenter and model: no real model, no media, no GPU needed."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from neural_fakes import FakeBackend, uniform_delta, write_model_folder

from makeup_cam import evaluate as ev
from makeup_cam import evaluate_validate as validate
from makeup_cam import regions
from makeup_cam.config import MakeupSettings
from makeup_cam.neural import geometry, model_files, worker
from makeup_cam.presets import load_preset
from makeup_cam.segmentation import SegMasks

H, W = 720, 1280
DATA = Path(__file__).resolve().parent / "data" / "face_landmarks_720p.npy"
FRAMES = 44


def _open_mouth(landmarks: np.ndarray) -> np.ndarray:
    """The test face with its mouth open: the lower lip moved down."""
    out = landmarks.copy()
    s = regions.face_scale(landmarks)
    lower = [p for p in set(regions.LIPS_LOWER_INNER) | set(regions.LIPS_LOWER_OUTER) if p not in (78, 308, 61, 291)]
    out[lower, 1] += 0.2 * s
    return out


@pytest.fixture(autouse=True)
def short_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev, "REALTIME_WARMUP_S", 0.3)


@pytest.fixture(scope="module")
def base_landmarks() -> np.ndarray:
    return _open_mouth(np.load(DATA).astype(np.float32))


@pytest.fixture(scope="module")
def picture() -> np.ndarray:
    """Texture with edges everywhere, so that every region has something to compare."""
    rng = np.random.default_rng(5)
    coarse = cv2.resize(rng.integers(40, 215, (45, 80, 3), dtype=np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.normal(0.0, 30.0, (H, W, 3)).astype(np.float32), (0, 0), 1.5)
    return np.clip(coarse.astype(np.float32) + fine, 0, 255).astype(np.uint8)


class Clip:
    """Frames and the landmarks that go with them. The picture sways, with a little new noise per frame."""

    def __init__(self, picture: np.ndarray, landmarks: np.ndarray, frames: int = FRAMES, sway: float = 0.0, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.frames: list[np.ndarray] = []
        self.landmarks: list[np.ndarray] = []
        for index in range(frames):
            dx = sway * np.sin(2 * np.pi * index / 24.0)
            moved = cv2.warpAffine(picture, np.float32([[1, 0, dx], [0, 1, 0]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
            noisy = moved.astype(np.float32) + rng.normal(0.0, 1.0, moved.shape).astype(np.float32)
            self.frames.append(np.clip(noisy, 0, 255).astype(np.uint8))
            self.landmarks.append(landmarks + np.array([dx, 0.0], np.float32))


class FakeTracker:
    """Answers by call number. One per use, as the real ones are."""

    def __init__(self, landmarks: Sequence[np.ndarray]) -> None:
        self._landmarks = landmarks
        self._call = -1

    def detect(self, frame: np.ndarray, ts_ms: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        self._call += 1
        points = self._landmarks[self._call % len(self._landmarks)].copy()
        return points, np.zeros(len(points), np.float32)

    def set_smoothing(self, smoothing: float) -> None:
        pass

    def close(self) -> None:
        pass


class FakeSegmenter:
    def update(self, frame: np.ndarray, ts_ms: int, frame_index: int) -> SegMasks:
        return SegMasks()

    def set_interval(self, interval: int) -> None:
        pass

    def close(self) -> None:
        pass


class DrillBackend(FakeBackend):
    """Fails like the real one: a kill ends it, a drill makes the next job fail."""

    def __init__(self, spec: Any, clock: Any, model: Any = None) -> None:
        super().__init__(spec, clock, model)
        self._next: int | None = None

    def kill(self) -> None:
        super().kill()
        self._set_failure("killed", None)

    def drill(self, code: int) -> None:
        self._next = code

    def finish(self, raw: np.ndarray | None = None) -> None:
        code, self._next = self._next, None
        if code == worker.DRILL_RUN_ERROR:
            self.pending = None
            self._set_failure("run error", worker.CODE_RUN_ERROR)
            return
        if code == worker.DRILL_BAD_OUTPUT:
            raw = np.full((3, self.size, self.size), np.nan, np.float32)
        super().finish(raw)


def fake_tools(landmarks: Sequence[np.ndarray], delta: float | None = 0.03) -> ev.Tools:
    model = None if delta is None else (lambda crop: uniform_delta(crop.shape[0], delta))
    return ev.Tools(
        tracker_factory=lambda smoothing: FakeTracker(landmarks),
        segmenter_factory=lambda interval, model_dir=None, background=False: FakeSegmenter(),
        backend_factory=(lambda spec: DrillBackend(spec, time.perf_counter, model)),
    )


def write_video(path: Path, frames: Sequence[np.ndarray]) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.CAP_FFMPEG, cv2.VideoWriter.fourcc(*"FFV1"), 30.0, (frames[0].shape[1], frames[0].shape[0]))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def analyze(clip: Clip, outputs: Sequence[np.ndarray], mode: str = ev.MODE_CLASSIC, reference: Sequence[np.ndarray] | None = None, settle: int = 3, raw_out: Sequence[np.ndarray] | None = None) -> ev.ClipResult:
    """The analyzer on frames in memory, with landmarks known."""
    reference = clip.frames if reference is None else reference
    calls = {"n": -1}

    def track(frame: np.ndarray, index: int) -> np.ndarray | None:
        calls["n"] += 1
        return (clip.landmarks if raw_out is None else raw_out)[calls["n"]]

    analyzer = ev.ClipAnalyzer("synthetic", 30.0, mode, track_output=track, settle_frames=settle)
    for index, (bare, ref, out, lm) in enumerate(zip(clip.frames, reference, outputs, clip.landmarks)):
        analyzer.feed(ev.make_frame_ref(index, bare, ref, lm, lm, lm, None, None), bare, ref, out)
    return analyzer.finish()


def kinds(result: ev.ClipResult) -> set[str]:
    return {event.kind for event in result.events}


@pytest.fixture(scope="module")
def clip(picture: np.ndarray, base_landmarks: np.ndarray) -> Clip:
    return Clip(picture, base_landmarks)


@pytest.fixture(scope="module")
def zones(clip: Clip) -> list[ev.Zones]:
    return [ev.build_zones(lm, None, None, W, H) for lm in clip.landmarks]


# ---- small measures --------------------------------------------------------------------------------------------------


def test_clips_are_brought_to_the_evaluation_size() -> None:
    frame = np.full((360, 480, 3), 200, np.uint8)
    out = ev.bring_to_size(frame)
    assert out.shape == (720, 1280, 3)
    assert out[360, 160, 0] == 200 and out[360, 159, 0] == 0  # 960 wide, black bars of 160
    same = np.zeros((720, 1280, 3), np.uint8)
    assert ev.bring_to_size(same) is same


def test_edge_correlation_sees_a_changed_shape_but_not_a_changed_color(picture: np.ndarray) -> None:
    gray = cv2.cvtColor(picture[100:200, 100:200], cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = np.ones(gray.shape, bool)
    assert ev.edge_correlation(gray, gray, mask) == pytest.approx(1.0)
    assert ev.edge_correlation(gray, gray + 10.0, mask) == pytest.approx(1.0)
    moved = cv2.warpAffine(gray, np.float32([[1, 0, 3], [0, 1, 0]]), (100, 100), borderMode=cv2.BORDER_REPLICATE)
    assert ev.edge_correlation(gray, moved, mask) < 0.6
    assert np.isnan(ev.edge_correlation(gray, gray, np.zeros(gray.shape, bool)))


def test_similarity_residual_ignores_pose_and_sees_proportions(base_landmarks: np.ndarray) -> None:
    points = base_landmarks[:100].astype(np.float64)
    angle = 0.2
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    assert ev.similarity_residual(points, points @ rotation.T * 1.3 + 40.0) == pytest.approx(0.0, abs=1e-6)
    wider = points * np.array([1.2, 1.0])
    assert ev.similarity_residual(points, wider) > 1.0


@pytest.mark.parametrize("lag", [0, 1, 2, 4])
def test_latency_is_found_from_two_mouth_series(lag: int) -> None:
    rng = np.random.default_rng(3)
    signal = cv2.GaussianBlur(rng.normal(0, 1, (1, 120)).astype(np.float32), (0, 0), 2.0).ravel().astype(np.float64)
    late = np.concatenate([np.full(lag, signal[0]), signal[: len(signal) - lag]])
    found, correlation = ev.estimate_latency(signal, late)
    assert found == pytest.approx(lag, abs=0.25)
    assert correlation > 0.8


def test_latency_says_nothing_without_motion() -> None:
    assert np.isnan(ev.estimate_latency(np.zeros(60), np.zeros(60))[0])
    assert np.isnan(ev.estimate_latency(np.zeros(5), np.zeros(5))[0])


def test_zones_are_the_runtime_weight(base_landmarks: np.ndarray) -> None:
    mask = np.zeros((H, W), np.float32)
    mask[50:120, 300:900] = 0.8
    made = ev.build_zones(base_landmarks, mask, None, W, H)
    assert np.array_equal(made.weight, geometry.edit_weight(base_landmarks, made.roi, mask, None))
    assert made.mouth_exact.any() and made.eyes_exact.any()
    assert (made.weight[made.mouth_exact] == 0.0).all()
    assert (made.weight[made.eyes_exact] == 0.0).all()
    assert made.mouth_band is not None and made.mouth_inner is not None


def test_a_closed_mouth_has_no_mouth_measures() -> None:
    landmarks = np.load(DATA).astype(np.float32)
    closed = landmarks.copy()
    closed[regions.LIPS_INNER_BOTTOM] = closed[regions.LIPS_INNER_TOP]
    assert ev.build_zones(closed, None, None, W, H).mouth_band is None


# ---- quiet on clean ------------------------------------------------------------------------------------------------


def test_an_output_equal_to_the_reference_is_perfect(clip: Clip) -> None:
    result = analyze(clip, clip.frames)
    m = result.metrics
    assert m["flicker_ratio"] == pytest.approx(1.0)
    assert m["flicker_outside_ratio"] == pytest.approx(1.0)
    assert m["delta_flicker"] == 0.0
    assert m["mouth_interior_max"] == 0.0 and m["eye_interior_max"] == 0.0 and m["outside_max"] == 0.0
    assert m["mouth_band_edge_corr"] == pytest.approx(1.0)
    assert m["landmark_shift_p99"] == 0.0
    assert m["identity_geometry_p99"] == pytest.approx(0.0, abs=1e-6)
    assert result.events == []


def test_a_change_the_stage_may_make_raises_no_alarm(clip: Clip, zones: list[ev.Zones]) -> None:
    delta = np.full((256, 256, 3), 0.008, np.float32) * geometry.edge_window(256)[..., None]
    outputs = [validate.paste_delta(f, delta, lm, z) for f, lm, z in zip(clip.frames, clip.landmarks, zones)]
    result = analyze(clip, outputs)
    assert result.metrics["mouth_interior_max"] == 0.0
    assert result.metrics["eye_interior_max"] == 0.0
    assert result.metrics["outside_max"] == 0.0
    assert result.metrics["delta_abs_p99"] > 0.005
    assert result.metrics["flicker_ratio"] < ev.THRESHOLD_BY_KEY["flicker_ratio"].value
    assert result.events == []


# ---- fires on defects ------------------------------------------------------------------------------------------------


def test_shimmer_on_the_face_fires(clip: Clip, zones: list[ev.Zones]) -> None:
    result = analyze(clip, validate.shimmer_face(clip.frames, zones, 4.0, 9))
    assert result.metrics["flicker_ratio"] > ev.THRESHOLD_BY_KEY["flicker_ratio"].worst
    assert result.metrics["delta_flicker"] > 1.0


def test_a_pumping_face_fires_the_block_measure(clip: Clip, zones: list[ev.Zones]) -> None:
    result = analyze(clip, validate.pump_face(clip.frames, zones, 2.0))
    assert result.metrics["flicker_block_ratio"] > 2.0


def test_shimmer_on_the_background_fires(clip: Clip) -> None:
    refs = [ev.make_frame_ref(i, f, f, lm, lm, lm, None, None) for i, (f, lm) in enumerate(zip(clip.frames, clip.landmarks))]
    result = analyze(clip, validate.shimmer_background(clip.frames, refs, 4.0, 9))
    assert result.metrics["flicker_outside_ratio"] > 1.5
    assert result.metrics["outside_max"] > 0.0
    assert "background_changed" in kinds(result)


def test_a_patch_on_the_wall_fires(clip: Clip) -> None:
    refs = [ev.make_frame_ref(i, f, f, lm, lm, lm, None, None) for i, (f, lm) in enumerate(zip(clip.frames, clip.landmarks))]
    result = analyze(clip, validate.background_patch(clip.frames, refs, 8.0))
    assert result.metrics["outside_max"] > 3.0
    assert result.metrics["background_block_max"] > 3.0
    assert "background_changed" in kinds(result)


def test_teeth_changing_shape_fire(clip: Clip) -> None:
    result = analyze(clip, validate.teeth_warp(clip.frames, clip.landmarks, 2.0))
    assert result.metrics["mouth_interior_max"] > 0.0
    assert result.metrics["mouth_inner_edge_corr"] < 0.9
    assert "teeth_altered" in kinds(result)


def test_teeth_changing_color_fire_the_exact_check_only(clip: Clip) -> None:
    result = analyze(clip, validate.teeth_tint(clip.frames, clip.landmarks, 3.0))
    assert result.metrics["mouth_interior_max"] > 0.0
    assert result.metrics["mouth_inner_edge_corr"] > 0.9
    assert "teeth_altered" in kinds(result)


def test_the_crop_for_measuring_is_the_contracts_crop(clip: Clip) -> None:
    frame, landmarks = clip.frames[0], clip.landmarks[0]
    normative = geometry.extract_crop(frame.astype(np.float32), landmarks, ev.ANALYSIS_CROP)
    fast = ev._crop_levels(frame, landmarks)
    inner = (slice(16, -16), slice(16, -16))
    assert np.abs(fast[inner] - normative[inner]).max() < 2.0  # only the blur near the part's edge may differ


def test_the_exact_checks_are_only_for_the_classic_reference(clip: Clip) -> None:
    result = analyze(clip, validate.teeth_tint(clip.frames, clip.landmarks, 3.0), mode=ev.MODE_INPUT)
    assert "teeth_altered" not in kinds(result)  # whitening against the bare input is not a defect


def test_eyes_changing_fire(clip: Clip) -> None:
    result = analyze(clip, validate.eye_warp(clip.frames, clip.landmarks, 3.0))
    assert result.metrics["eye_interior_max"] > 0.0
    assert result.metrics["eye_edge_corr"] < 0.9
    assert "eyes_altered" in kinds(result)


def test_a_one_frame_blob_is_a_spike_with_its_frame_number(clip: Clip, zones: list[ev.Zones]) -> None:
    data = validate.Rendered(clip.frames, clip.frames, clip.landmarks, [None] * FRAMES, [None] * FRAMES)
    result = analyze(clip, validate.spike(clip.frames, data, zones, 24.0, (25,)))
    spikes = [event for event in result.events if event.kind == "spike"]
    assert spikes and spikes[0].start in (25, 26)
    assert spikes[0].peak > ev.EVENT_RULES["spike_floor"]


def test_a_frozen_output_is_found(clip: Clip) -> None:
    sway = Clip(clip.frames[0], clip.landmarks[0], sway=10.0)
    result = analyze(sway, validate.freeze(sway.frames, 15, 10))
    assert "frozen" in kinds(result)


def test_a_late_mouth_is_found() -> None:
    gaps = 0.12 + 0.08 * np.sin(np.arange(60) * 2 * np.pi / 13.0)
    points = np.load(DATA).astype(np.float32)
    s = regions.face_scale(points)

    def at(gap: float) -> np.ndarray:
        out = points.copy()
        out[regions.LIPS_INNER_BOTTOM] = out[regions.LIPS_INNER_TOP] + np.array([0.0, gap * s], np.float32)
        return out

    bare = [at(g) for g in gaps]
    late = [at(g) for g in np.concatenate([np.full(3, gaps[0]), gaps[:-3]])]
    frame = np.zeros((H, W, 3), np.uint8)
    analyzer = ev.ClipAnalyzer("synthetic", 30.0, ev.MODE_INPUT, track_output=lambda f, i: late[i], settle_frames=3)
    for index, lm in enumerate(bare):
        analyzer.feed(ev.make_frame_ref(index, frame, frame, lm, lm, lm, None, None), frame, frame, frame)
    assert analyzer.finish().metrics["mouth_latency_frames"] == pytest.approx(3.0, abs=0.3)


def test_landmarks_used_for_drawing_that_jump_are_found(clip: Clip) -> None:
    analyzer = ev.ClipAnalyzer("synthetic", 30.0, ev.MODE_INPUT, track_output=lambda f, i: clip.landmarks[i], settle_frames=3)
    s = regions.face_scale(clip.landmarks[0])
    for index, (frame, raw) in enumerate(zip(clip.frames, clip.landmarks)):
        stage = raw + np.array([0.1 * s, 0.0], np.float32) if index == 25 else raw
        analyzer.feed(ev.make_frame_ref(index, frame, frame, stage, raw, raw, None, None), frame, frame, frame)
    result = analyzer.finish()
    assert "landmark_jump" in kinds(result)


def test_output_landmarks_far_from_the_reference_are_an_identity_shift(clip: Clip) -> None:
    other = [lm * np.array([1.12, 0.9], np.float32) for lm in clip.landmarks]
    result = analyze(clip, clip.frames, raw_out=other)
    assert result.metrics["identity_geometry_p99"] > ev.THRESHOLD_BY_KEY["identity_geometry_p99"].value
    assert result.metrics["landmark_shift_p99"] > ev.THRESHOLD_BY_KEY["landmark_shift_p99"].value
    assert "identity_shift" in kinds(result)


def test_another_face_is_found_by_structure_even_with_the_same_landmarks(clip: Clip, picture: np.ndarray) -> None:
    other = np.roll(picture, 37, axis=0)
    data = [cv2.addWeighted(other, 1.0, f, 0.0, 0.0) for f in clip.frames]
    outputs = [validate._blend_face(f, d, lm) for f, d, lm in zip(clip.frames, data, clip.landmarks)]
    result = analyze(clip, outputs)  # the output's landmarks are the input's: geometry alone cannot see this
    assert result.metrics["identity_geometry_p99"] == pytest.approx(0.0, abs=1e-6)
    assert result.metrics["identity_structure_p01"] < 0.5
    assert "identity_shift" in kinds(result)


def _moving_pattern(lag: int) -> ev.ClipResult:
    """A patch of change attached to a moving head, drawn where the head was ``lag`` frames ago."""
    rng = np.random.default_rng(2)
    pattern = cv2.GaussianBlur(rng.normal(0.0, 60.0, (H, W, 3)).astype(np.float32), (0, 0), 12.0)
    base = np.full((H, W, 3), 128, np.uint8)
    landmarks = np.load(DATA).astype(np.float32)
    analyzer = ev.ClipAnalyzer("synthetic", 30.0, ev.MODE_CLASSIC, settle_frames=3)
    for index in range(70):
        dx = 40.0 * np.sin(2 * np.pi * index / 40.0)
        drawn = 40.0 * np.sin(2 * np.pi * (index - lag) / 40.0)
        lm = landmarks + np.array([dx, 0.0], np.float32)
        moved = cv2.warpAffine(pattern, np.float32([[1, 0, drawn], [0, 1, 0]]), (W, H))
        out = np.clip(base.astype(np.float32) + moved, 0, 255).astype(np.uint8)
        analyzer.feed(ev.make_frame_ref(index, base, base, lm, lm, lm, None, None), base, base, out)
    return analyzer.finish()


def test_paint_that_follows_the_face_has_no_lag() -> None:
    assert abs(_moving_pattern(0).metrics["paint_lag_frames"]) < 0.35


def test_paint_left_behind_is_measured_in_frames() -> None:
    result = _moving_pattern(3)
    assert result.metrics["paint_lag_frames"] == pytest.approx(3.0, abs=1.0)
    assert "paint_behind" in kinds(result)


def test_lag_says_nothing_without_motion(clip: Clip) -> None:
    assert np.isnan(analyze(clip, clip.frames).metrics["paint_lag_frames"])
    assert np.isnan(analyze(clip, clip.frames).metrics["paint_lag_local_frames"])


def _moving_patch(lag: int, level: float = 12.0) -> ev.ClipResult:
    """A broad change that follows the face, plus one small patch on a cheek that comes ``lag`` frames late."""
    landmarks = np.load(DATA).astype(np.float32)
    rng = np.random.default_rng(2)
    broad = cv2.GaussianBlur(rng.normal(0.0, 60.0, (H, W, 3)).astype(np.float32), (0, 0), 30.0) * 8.0
    base = np.full((H, W, 3), 128, np.uint8)
    analyzer = ev.ClipAnalyzer("synthetic", 30.0, ev.MODE_CLASSIC, settle_frames=3)
    for index in range(70):
        shift = np.array([40.0 * np.sin(2 * np.pi * index / 40.0), 0.0], np.float32)
        lm = landmarks + shift
        old = landmarks + np.array([40.0 * np.sin(2 * np.pi * (index - lag) / 40.0), 0.0], np.float32)
        moved = cv2.warpAffine(broad, np.float32([[1, 0, shift[0]], [0, 1, 0]]), (W, H))
        out = np.clip(base.astype(np.float32) + moved + validate._cheek_patch(old, (H, W), level), 0, 255).astype(np.uint8)
        analyzer.feed(ev.make_frame_ref(index, base, base, lm, lm, lm, None, None), base, base, out)
    return analyzer.finish()


def test_a_small_patch_that_lags_shows_in_the_local_fit() -> None:
    late = _moving_patch(3).metrics
    on_time = _moving_patch(0).metrics
    limit = ev.THRESHOLD_BY_KEY["paint_lag_local_frames"].worst
    assert late["paint_lag_local_frames"] > limit
    assert on_time["paint_lag_local_frames"] < limit
    assert late["paint_lag_local_frames"] > late["paint_lag_frames"]  # the fit over the whole face sees less of it


def test_with_no_pattern_at_all_nothing_can_lag() -> None:
    landmarks = np.load(DATA).astype(np.float32)
    base = np.full((H, W, 3), 128, np.uint8)
    analyzer = ev.ClipAnalyzer("synthetic", 30.0, ev.MODE_CLASSIC, settle_frames=3)
    for index in range(70):
        lm = landmarks + np.array([40.0 * np.sin(2 * np.pi * index / 40.0), 0.0], np.float32)
        analyzer.feed(ev.make_frame_ref(index, base, base, lm, lm, lm, None, None), base, base, base)
    assert analyzer.finish().metrics["paint_lag_local_frames"] == 0.0


def test_lip_color_off_the_lips_is_measured(clip: Clip, zones: list[ev.Zones]) -> None:
    def painted(offset: float) -> list[np.ndarray]:
        out = []
        for frame, lm in zip(clip.frames, clip.landmarks):
            s = regions.face_scale(lm)
            roi = regions.Roi(0, 0, W, H)
            mask = regions.polygon_mask(lm[regions.LIPS_OUTER] + np.array([0.0, offset * s], np.float32), roi)
            change = np.zeros((H, W, 3), np.float32)
            change[..., 2] = 40.0 * mask
            change[..., 1] = -20.0 * mask
            out.append(np.clip(frame.astype(np.float32) + change, 0, 255).astype(np.uint8))
        return out

    on_target = analyze(clip, painted(0.0), mode=ev.MODE_INPUT).metrics["paint_offset"]
    off_target = analyze(clip, painted(0.4), mode=ev.MODE_INPUT)
    assert on_target < 0.1
    assert off_target.metrics["paint_offset"] > 0.3
    assert "off_target" in kinds(off_target)


def test_events_are_merged_and_saved_with_their_frames(clip: Clip, tmp_path: Path) -> None:
    analyzer = ev.ClipAnalyzer("synthetic", 30.0, ev.MODE_CLASSIC, track_output=None, crops_dir=tmp_path / "events", settle_frames=3)
    outputs = validate.teeth_tint(clip.frames, clip.landmarks, 3.0)
    for index, (bare, out, lm) in enumerate(zip(clip.frames, outputs, clip.landmarks)):
        analyzer.feed(ev.make_frame_ref(index, bare, bare, lm, lm, lm, None, None), bare, bare, out)
    result = analyzer.finish()
    teeth = [event for event in result.events if event.kind == "teeth_altered"]
    assert len(teeth) == 1
    assert teeth[0].start == 3 and teeth[0].end == FRAMES - 1
    saved = list((tmp_path / "events" / "synthetic").glob("teeth_altered_*.png"))
    assert len(saved) == 1
    assert teeth[0].crops == ["events/synthetic/" + saved[0].name]
    assert cv2.imdecode(np.fromfile(saved[0], np.uint8), cv2.IMREAD_COLOR).shape[1] > 320 * 3


def test_replaying_stored_series_gives_the_live_events(clip: Clip) -> None:
    result = analyze(clip, validate.teeth_warp(clip.frames, clip.landmarks, 2.0), settle=3)
    counts = validate.replay_events(result.series, ev.MODE_CLASSIC, ev.EVENT_RULES, settle=3)
    live: dict[str, int] = {}
    for event in result.events:
        live[event.kind] = live.get(event.kind, 0) + 1
    assert counts == live


# ---- verdict ------------------------------------------------------------------------------------------------------


def _good_metrics(mode: str = ev.MODE_INPUT) -> dict[str, dict[str, Any]]:
    metrics = {}
    for threshold in ev.THRESHOLDS:
        metrics[threshold.key] = ev._entry(threshold, 1.0, 1.0, "clip", ev.STATUS_MEASURED, True)
        if mode not in threshold.modes:
            metrics[threshold.key].update(status=ev.STATUS_NOT_APPLICABLE, passed=None)
    return metrics


def test_all_good_is_a_pass() -> None:
    assert ev.decide_verdict(_good_metrics(), 5000)[0] == "pass"


def test_one_miss_is_a_fail_whatever_else_is_missing() -> None:
    metrics = _good_metrics()
    metrics["flicker_ratio"].update(passed=False)
    metrics["identity_cosine_p01"].update(status=ev.STATUS_NOT_RUN, passed=None)
    assert ev.decide_verdict(metrics, 5000)[0] == "fail"


def test_a_metric_with_no_data_never_counts_as_passed() -> None:
    metrics = _good_metrics()
    metrics["breaks_per_30min"].update(status=ev.STATUS_NOT_RUN, passed=None)
    assert ev.decide_verdict(metrics, 5000)[0] == "incomplete"


def test_too_few_frames_cannot_pass() -> None:
    verdict, notes = ev.decide_verdict(_good_metrics(), 10)
    assert verdict == "incomplete"
    assert notes


def test_the_detector_never_takes_part_in_the_verdict() -> None:
    metrics = _good_metrics()
    metrics["deepfake_margin"].update(status=ev.STATUS_NOT_RUN, passed=None)
    assert ev.decide_verdict(metrics, 5000)[0] == "pass"
    threshold = ev.THRESHOLD_BY_KEY["deepfake_margin"]
    assert threshold.value is None and threshold.required is False


def test_a_detector_score_is_shown_for_review_only(tmp_path: Path) -> None:
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"output_score": 0.9, "reference_score": 0.1, "detector": "x"}), encoding="utf-8")
    report = ev.read_detector_report(path)
    result = ev.ClipResult("c", 100, 100, 30.0, {}, [], {})
    metrics, _ = ev.build_metrics(ev.MODE_INPUT, [result], detector=report)
    entry = metrics["deepfake_margin"]
    assert entry["value"] == pytest.approx(0.8)
    assert entry["status"] == ev.STATUS_INFORMATION
    assert entry["passed"] is None and entry["threshold"] is None


@pytest.mark.parametrize("content", ["not json", "{}", '{"output_score": "x", "reference_score": 1}', '{"output_score": NaN, "reference_score": 1}'])
def test_a_bad_detector_file_is_an_error_in_japanese(tmp_path: Path, content: str) -> None:
    path = tmp_path / "scores.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ev.EvaluationError, match="検出器"):
        ev.read_detector_report(path)


def _clip_with_events(count: int, seconds: float = 100.0) -> ev.ClipResult:
    events = [ev.ClipEvent("c", "spike", i, i, 1.0) for i in range(count)]
    return ev.ClipResult("c", int(seconds * 30), int(seconds * 30), 30.0, {}, events, {})


def test_a_short_run_cannot_show_a_low_rate() -> None:
    entry, _ = ev._breaks([_clip_with_events(1)], None)
    assert entry["status"] == ev.STATUS_NOT_RUN and entry["passed"] is None


def test_a_short_run_can_show_a_high_rate() -> None:
    entry, _ = ev._breaks([_clip_with_events(2)], None)
    assert entry["status"] == ev.STATUS_MEASURED and entry["passed"] is False


def test_a_long_run_measures_the_rate() -> None:
    soak = {"minutes": 60.0, "event_counts": {"spike": 1}, "events": [ev.ClipEvent("soak", "spike", 1, 1, 1.0)]}
    entry, events = ev._breaks([_clip_with_events(0)], soak)
    assert entry["status"] == ev.STATUS_MEASURED and entry["passed"] is True
    assert entry["value"] == pytest.approx(1 / (60.0 + 100.0 / 60.0) * 30.0, abs=0.001)
    assert len(events) == 1


def test_a_clean_long_run_does_not_excuse_what_the_quality_pass_found() -> None:
    soak = {"minutes": 30.0, "event_counts": {}, "events": []}
    entry, events = ev._breaks([_clip_with_events(8)], soak)
    assert entry["status"] == ev.STATUS_MEASURED and entry["passed"] is False
    assert entry["quality_pass_events"] == 8 and len(events) == 8
    one, _ = ev._breaks([_clip_with_events(1)], soak)
    assert one["passed"] is True
    assert one["value"] == pytest.approx(1 / (30.0 + 100.0 / 60.0) * 30.0, abs=0.01)


def test_the_cosine_is_held_against_the_bare_frame_with_and_without_a_model() -> None:
    def clip_with(out: float, ref: float) -> ev.ClipResult:
        return ev.ClipResult("c", 100, 100, 30.0, {"identity_cosine_p01": out, "identity_cosine_ref_p01": ref}, [], {})

    limit = ev.THRESHOLD_BY_KEY["identity_cosine_p01"].value
    assert limit > 0.519  # another face scores at best this much with SFace
    for mode in (ev.MODE_INPUT, ev.MODE_CLASSIC):
        assert ev._identity_cosine([clip_with(limit + 0.05, 1.0)], mode)["passed"] is True
        assert ev._identity_cosine([clip_with(limit - 0.05, 1.0)], mode)["passed"] is False
        assert ev._identity_cosine([clip_with(limit + 0.05, 1.0)], mode)["threshold"] == limit
    # Added makeup moves the cosine as such: a stand-in that imitates natural from off lost 0.028 to 0.059.
    entry = ev._identity_cosine([clip_with(0.94, 1.0)], ev.MODE_CLASSIC)
    assert entry["passed"] is True and entry["below_classic_worst"] == pytest.approx(-0.06)


def test_the_worst_clip_counts() -> None:
    threshold = ev.THRESHOLD_BY_KEY["flicker_ratio"]
    clips = [ev.ClipResult(f"c{i}", 100, 100, 30.0, {"flicker_ratio": 1.0}, [], {}) for i in range(10)]
    clips.append(ev.ClipResult("bad", 100, 100, 30.0, {"flicker_ratio": threshold.worst + 0.5}, [], {}))
    entry = ev.aggregate_clip_metric(threshold, clips)
    assert entry["value"] == pytest.approx(1.0)
    assert entry["worst_clip"] == "bad"
    assert entry["passed"] is False


def test_every_contract_metric_has_a_threshold() -> None:
    contract = {
        "flicker_ratio", "mouth_interior_max", "mouth_band_edge_corr", "eye_interior_max", "outside_max", "landmark_shift_p99",
        "identity_cosine_p01", "delta_abs_p99", "breaks_per_30min", "deepfake_margin", "all_kernels_dml", "rtt_p95_ms", "rtt_p99_ms",
        "stage_p95_ms", "fps_loss", "miss_pct", "vram_mb", "strength_zero_identical", "strength_monotonic", "fade_drill",
    }
    assert contract <= set(ev.THRESHOLD_BY_KEY)
    assert all(threshold.basis for threshold in ev.THRESHOLDS)


# ---- the whole tool, with fakes -------------------------------------------------------------------------------------


@pytest.fixture
def video(tmp_path: Path, clip: Clip) -> Path:
    return write_video(tmp_path / "1234_FAKE_CLIP.avi", clip.frames[: ev.SETTLE_FRAMES + 8])


def _args(**changes: Any) -> Any:
    parser = ev.build_parser()
    args = parser.parse_args(["--no-sway"])  # the fake tracker does not follow a sway
    for key, value in changes.items():
        setattr(args, key, value)
    return args


def test_a_run_without_a_model_measures_the_preset_against_the_input(tmp_path: Path, video: Path, clip: Clip) -> None:
    out = tmp_path / "out"
    report = ev.run_evaluation(_args(clips=[str(video)], out=out, preset="off"), fake_tools(clip.landmarks))
    assert report["conditions"]["mode"] == ev.MODE_INPUT
    assert report["model"] is None and report["eligible_for_calls"] is False
    assert report["metrics"]["mouth_interior_max"]["status"] == ev.STATUS_NOT_APPLICABLE
    assert report["metrics"]["fade_drill"]["status"] == ev.STATUS_NOT_APPLICABLE
    assert report["metrics"]["flicker_ratio"]["value"] == pytest.approx(1.0)
    assert (out / "evaluation.json").is_file() and (out / "report_ja.md").is_file()


def test_a_run_without_a_model_or_a_folder_is_refused(video: Path, clip: Clip) -> None:
    with pytest.raises(ev.EvaluationError, match="--out"):
        ev.run_evaluation(_args(clips=[str(video)]), fake_tools(clip.landmarks))


def test_only_the_top_of_the_slider_can_pass() -> None:
    model = {"name": "m", "version": "1", "sha256": "x", "stand_in": False}
    clips = [ev.ClipResult("c", 2000, 2000, 30.0, {}, [], {})]
    metrics = _good_metrics(ev.MODE_CLASSIC)
    full = ev.build_report(model, ev.MODE_CLASSIC, "natural", clips, metrics, [], None, None, [], 1.0)
    assert full["verdict"] == "pass" and full["eligible_for_calls"] is True
    assert full["conditions"]["strength"] == 1.0
    weak = ev.build_report(model, ev.MODE_CLASSIC, "natural", clips, metrics, [], None, None, [], 0.5)
    assert weak["verdict"] == "incomplete" and weak["eligible_for_calls"] is False
    assert weak["conditions"]["strength"] == 0.5
    assert any("強さ" in note for note in weak["notes"])


@pytest.mark.parametrize("flags", [["--strength", "1.5"], ["--strength", "-0.1"], ["--frames-per-clip", "0"], ["--fps-seconds", "0"], ["--soak-minutes", "-1"]])
def test_values_that_would_measure_nothing_are_refused(tmp_path: Path, video: Path, capsys: pytest.CaptureFixture[str], flags: list[str]) -> None:
    assert ev.main(["--clips", str(video), "--out", str(tmp_path / "out"), *flags]) == ev.EXIT_ERROR
    assert flags[0] in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_crops_of_faces_cannot_land_in_a_tracked_folder(tmp_path: Path, video: Path, clip: Clip) -> None:
    """A throwaway repository of the test's own."""
    if subprocess.run(["git", "--version"], capture_output=True, check=False).returncode != 0:
        pytest.skip("no git")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".gitignore").write_text("kept-out/\n", encoding="utf-8")
    assert ev.out_dir_is_tracked(repo / "results") is True
    assert ev.out_dir_is_tracked(repo / "kept-out") is False
    assert ev.out_dir_is_tracked(tmp_path / "elsewhere") is False
    with pytest.raises(ev.EvaluationError, match="git"):
        ev.run_evaluation(_args(clips=[str(video)], out=repo / "results", preset="off"), fake_tools(clip.landmarks))
    assert not (repo / "results").exists()


def test_a_rendered_video_is_measured_against_a_paired_one(tmp_path: Path, video: Path, clip: Clip) -> None:
    rendered = write_video(tmp_path / "rendered.avi", validate.teeth_warp(clip.frames, clip.landmarks, 3.0))
    report = ev.run_evaluation(
        _args(clips=[str(video)], rendered=rendered, reference=video, out=tmp_path / "out", preset="off"), fake_tools(clip.landmarks)
    )
    assert report["conditions"]["mode"] == ev.MODE_PAIRED
    assert report["metrics"]["paired_abs"]["status"] == ev.STATUS_INFORMATION
    assert report["metrics"]["paired_abs"]["passed"] is None
    assert report["metrics"]["mouth_inner_edge_corr"]["passed"] is False
    assert report["verdict"] == "fail"


@pytest.fixture(scope="module")
def full_model_run(tmp_path_factory: pytest.TempPathFactory, clip: Clip) -> tuple[dict[str, Any], Path]:
    """One whole run with a fake model: quality pass, strength checks, fade drill."""
    folder = tmp_path_factory.mktemp("full")
    video = write_video(folder / "1234_FAKE_CLIP.avi", clip.frames)
    model = write_model_folder(folder / "model", size=256)
    return ev.run_evaluation(_args(clips=[str(video)], model=model, preset="off"), fake_tools(clip.landmarks, delta=0.012)), model


@pytest.fixture
def quality_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The quality pass alone: the slower checks of the running model have tests of their own."""
    monkeypatch.setattr(ev, "run_strength_checks", lambda *args, **kwargs: {"zero_identical": True, "monotonic": True, "mean_change": {}})
    monkeypatch.setattr(ev, "run_fade_drill", lambda *args, **kwargs: {"ok": True, "cases": {}})


def test_a_run_with_a_model_covers_the_runtime(full_model_run: tuple[dict[str, Any], Path]) -> None:
    report, _model = full_model_run
    metrics = report["metrics"]
    assert report["conditions"]["mode"] == ev.MODE_CLASSIC
    assert metrics["mouth_interior_max"]["value"] == 0.0
    assert metrics["eye_interior_max"]["value"] == 0.0
    assert metrics["outside_max"]["value"] == 0.0
    assert metrics["delta_abs_p99"]["value"] > 0.0
    assert metrics["strength_zero_identical"]["value"] is True
    assert metrics["strength_monotonic"]["value"] is True, metrics["strength_monotonic"]
    assert metrics["fade_drill"]["value"] is True, metrics["fade_drill"]
    assert set(metrics["fade_drill"]["cases"]) == {"kill", "run_error", "bad_output"}
    assert metrics["rtt_p95_ms"]["status"] == ev.STATUS_NOT_RUN  # only with --measure-fps
    assert report["verdict"] == "incomplete"


def test_a_stand_in_is_never_eligible_and_the_metadata_is_written(full_model_run: tuple[dict[str, Any], Path]) -> None:
    report, model = full_model_run
    assert report["eligible_for_calls"] is False
    assert any("代用品" in note for note in report["notes"])
    meta = json.loads((model / "model.json").read_text(encoding="utf-8"))
    evaluation = meta["evaluation"]
    assert evaluation["verdict"] == report["verdict"]
    assert evaluation["model_sha256"] == meta["sha256"]
    assert evaluation["report_sha256"] == hashlib.sha256((model / "evaluation.json").read_bytes()).hexdigest()
    assert meta["name"] == "fake"  # the rest of the file is kept


def test_a_model_without_a_pass_cannot_get_past_the_gate(tmp_path: Path, video: Path, clip: Clip, quality_only: None) -> None:
    model = write_model_folder(tmp_path / "model", size=256, stand_in=False)
    report = ev.run_evaluation(_args(clips=[str(video)], model=model, preset="off"), fake_tools(clip.landmarks))
    assert report["verdict"] != "pass"
    with pytest.raises(model_files.ModelRefused):
        model_files.check_gate(model_files.load_spec(model))


def test_a_passing_report_written_by_the_tool_opens_the_gate(tmp_path: Path) -> None:
    model = write_model_folder(tmp_path / "model", stand_in=False)
    summary = {"name": "fake", "version": "0.0.0", "sha256": json.loads((model / "model.json").read_text(encoding="utf-8"))["sha256"], "stand_in": False}
    report = {
        "verdict": "pass", "model": summary, "date": "2026-11-06T00:00:00", "eligible_for_calls": True, "notes": [], "events": [], "metrics": _good_metrics(),
        "conditions": {"mode": ev.MODE_CLASSIC, "preset": "natural", "clips": [], "frames": 0, "frame_size": [1280, 720], "soak_minutes": 30, "pretrained": []},
    }
    ev.write_report(report, tmp_path / "out", model, update_metadata=True)
    model_files.check_gate(model_files.load_spec(model))


def test_no_update_leaves_the_model_folder_alone(tmp_path: Path, video: Path, clip: Clip, quality_only: None) -> None:
    model = write_model_folder(tmp_path / "model", size=256)
    before = (model / "model.json").read_bytes()
    ev.run_evaluation(_args(clips=[str(video)], model=model, out=tmp_path / "out", preset="off", no_update_metadata=True), fake_tools(clip.landmarks))
    assert (model / "model.json").read_bytes() == before
    assert not (model / "evaluation.json").exists()
    assert (tmp_path / "out" / "evaluation.json").is_file()


def test_a_changed_model_file_is_refused(tmp_path: Path, video: Path, clip: Clip) -> None:
    model = write_model_folder(tmp_path / "model", size=256)
    (model / "model.onnx").write_bytes(b"another file")
    with pytest.raises(ev.EvaluationError, match="モデル"):
        ev.run_evaluation(_args(clips=[str(video)], model=model, preset="off"), fake_tools(clip.landmarks))


def test_a_heavy_model_fails_on_size_and_the_stage_still_keeps_mouth_and_eyes(tmp_path: Path, video: Path, clip: Clip, quality_only: None, monkeypatch: pytest.MonkeyPatch) -> None:
    limit = dataclasses.replace(ev.THRESHOLD_BY_KEY["delta_abs_p99"], value=0.05)
    monkeypatch.setitem(ev.THRESHOLD_BY_KEY, "delta_abs_p99", limit)
    model = write_model_folder(tmp_path / "model", size=256)
    report = ev.run_evaluation(_args(clips=[str(video)], model=model, preset="off"), fake_tools(clip.landmarks, delta=0.12))
    assert report["metrics"]["mouth_interior_max"]["value"] == 0.0
    assert report["metrics"]["eye_interior_max"]["value"] == 0.0
    assert report["metrics"]["delta_abs_p99"]["passed"] is False
    assert report["verdict"] == "fail"


def test_a_model_whose_every_result_is_rejected_fails_and_the_tool_still_finishes(tmp_path: Path, video: Path, clip: Clip) -> None:
    model = write_model_folder(tmp_path / "model", size=256)
    report = ev.run_evaluation(_args(clips=[str(video)], model=model, preset="off"), fake_tools(clip.landmarks, delta=0.3))
    assert report["metrics"]["fade_drill"]["value"] is False
    assert report["verdict"] == "fail"
    assert report["event_counts"].get("worker_restart", 0) > 0
    assert report["event_counts"].get("health_fall", 0) > 0


def test_the_report_shows_every_threshold_in_japanese_and_no_paths(tmp_path: Path, video: Path, clip: Clip) -> None:
    out = tmp_path / "out"
    ev.run_evaluation(_args(clips=[str(video)], out=out, preset="off"), fake_tools(clip.landmarks))
    text = (out / "report_ja.md").read_text(encoding="utf-8")
    for threshold in ev.THRESHOLDS:
        assert threshold.label in text
        assert f"`{threshold.key}`" in text
        assert ev._format_limit(threshold, ev.MODE_INPUT) in text
    assert "判定:" in text
    assert "隠す" in text  # the note on what the tool is not for
    raw = (out / "evaluation.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["conditions"]["clips"] == ["1234_FAKE_CLIP"]
    assert str(tmp_path) not in raw and str(tmp_path).replace("\\", "\\\\") not in raw
    assert data["thresholds"]["flicker_ratio"]["value"] == ev.THRESHOLD_BY_KEY["flicker_ratio"].value


def test_the_report_shows_the_rules_for_breakdowns_too(tmp_path: Path, video: Path, clip: Clip) -> None:
    out = tmp_path / "out"
    ev.run_evaluation(_args(clips=[str(video)], out=out, preset="off"), fake_tools(clip.landmarks))
    text = (out / "report_ja.md").read_text(encoding="utf-8")
    assert set(ev.EVENT_RULES) == set(ev.EVENT_RULE_LABELS)
    for key, label in ev.EVENT_RULE_LABELS.items():
        assert label in text and f"`{key}`" in text


def test_real_time_numbers_are_in_the_report_also_without_a_model(tmp_path: Path, video: Path, clip: Clip, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    args = _args(clips=[str(video)], out=out, preset="off", measure_fps=True, fps_seconds=1.0)
    report = ev.run_evaluation(args, fake_tools(clip.landmarks))
    assert report["realtime"]["classic_fps"] > 5.0
    text = (out / "report_ja.md").read_text(encoding="utf-8")
    assert "実際の時間での計測" in text and "混み具合" in text
    assert f"{report['realtime']['classic_fps']:.2f} fps" in text
    ev.print_summary(report, out)
    assert f"{report['realtime']['classic_fps']:.2f} fps" in capsys.readouterr().out


def test_a_gpu_check_that_fails_leaves_no_paths_in_its_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Failed:
        returncode = 1
        stdout = ""
        stderr = 'File "C:/Users/somebody/model.py", line 1'

    monkeypatch.setattr(ev.subprocess, "run", lambda *args, **kwargs: Failed())
    assert ev.check_kernels(tmp_path / "model.onnx", 64) == {"available": False}


def test_exit_codes(tmp_path: Path, video: Path, clip: Clip, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    assert ev.main(["--clips", str(tmp_path / "none*.flv"), "--out", str(tmp_path)]) == ev.EXIT_ERROR
    assert "動画" in capsys.readouterr().err
    for verdict, code in (("pass", ev.EXIT_PASS), ("fail", ev.EXIT_FAIL), ("incomplete", ev.EXIT_INCOMPLETE)):
        report = {"verdict": verdict, "metrics": _good_metrics(), "events": [], "notes": []}
        monkeypatch.setattr(ev, "run_evaluation", lambda args: report)
        assert ev.main(["--clips", str(video), "--out", str(tmp_path)]) == code
        assert ev.VERDICT_LABELS[verdict] in capsys.readouterr().out


def test_the_swayed_clips_are_at_most_the_number_given_and_evenly_spread() -> None:
    paths = [Path(f"{index:02d}.flv") for index in range(44)]
    chosen = ev.sway_choice(paths)
    assert len(chosen) == ev.SWAY_CLIPS and len(set(chosen)) == ev.SWAY_CLIPS
    assert chosen[0] == paths[0]
    assert chosen[-1].stem >= "36"
    few = paths[:3]
    assert ev.sway_choice(few) == few


def test_swaying_moves_the_frame_and_not_its_size(clip: Clip) -> None:
    swayed = list(ev.sway_frames(clip.frames[:12]))
    assert len(swayed) == 12 and swayed[5].shape == clip.frames[5].shape
    assert np.array_equal(swayed[0], clip.frames[0])
    assert not np.array_equal(swayed[5], clip.frames[5])


class FakeEmbedder:
    """Tells frames apart by their mean color: enough for the bookkeeping around an embedding."""

    def embed(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
        vector = np.array(cv2.mean(frame)[:3], np.float64) - 100.0
        return vector / max(float(np.linalg.norm(vector)), 1e-6)


def test_an_identity_model_is_optional_and_named_with_its_licence(tmp_path: Path, video: Path, clip: Clip) -> None:
    tools = dataclasses.replace(fake_tools(clip.landmarks), embedder=FakeEmbedder())
    report = ev.run_evaluation(_args(clips=[str(video)], out=tmp_path / "out", preset="off", identity_model=Path("some-model.onnx"), identity_licence="Apache-2.0"), tools)
    assert report["conditions"]["pretrained"] == [{"name": "some-model", "licence": "Apache-2.0"}]
    assert report["metrics"]["identity_cosine_p01"]["status"] == ev.STATUS_MEASURED
    without = ev.run_evaluation(_args(clips=[str(video)], out=tmp_path / "out2", preset="off"), fake_tools(clip.landmarks))
    assert without["metrics"]["identity_cosine_p01"]["status"] == ev.STATUS_NOT_RUN
    assert without["conditions"]["pretrained"] == []


def test_a_broken_identity_model_does_not_stop_the_evaluation(clip: Clip) -> None:
    class Broken:
        def embed(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
            raise RuntimeError("broken")

    analyzer = ev.ClipAnalyzer("synthetic", 30.0, ev.MODE_INPUT, track_output=lambda f, i: clip.landmarks[i], embedder=ev._CosineProbe(Broken()), settle_frames=3)
    for index, (frame, lm) in enumerate(zip(clip.frames[:8], clip.landmarks)):
        analyzer.feed(ev.make_frame_ref(index, frame, frame, lm, lm, lm, None, None), frame, frame, frame)
    assert np.isnan(analyzer.finish().metrics["identity_cosine_p01"])


def test_a_slow_pipeline_shows_in_the_real_frame_rate(video: Path, clip: Clip) -> None:
    class Slow(ev.StageProbe):
        def apply(self, frame: np.ndarray, ctx: Any, settings: Any) -> np.ndarray:
            time.sleep(0.06)
            return frame

    frames = clip.frames[:10]
    settings = ev._without_model(load_preset("off"))
    slow = ev.run_realtime(frames, settings, 30.0, 1.5, fake_tools(clip.landmarks), probe=Slow())
    assert 5.0 < slow.fps_out < 20.0


def test_the_tool_needs_no_onnxruntime_and_no_face_model_to_import() -> None:
    assert "onnxruntime" not in Path(ev.__file__).read_text(encoding="utf-8").split("_KERNEL_SCRIPT")[0]


def test_the_probe_changes_no_pixel(clip: Clip) -> None:
    from makeup_cam.pipeline import MakeupPipeline

    settings = ev.file_settings(load_preset("natural"))

    def run(extra: list[Any]) -> list[np.ndarray]:
        pipeline = MakeupPipeline(settings, tracker=FakeTracker(clip.landmarks), segmenter=FakeSegmenter(), effects=ev.builtin_effects(extra, without_neural=True))
        outputs = [pipeline.process(frame, index * 33).output for index, frame in enumerate(clip.frames[:8])]
        pipeline.close()
        return outputs

    probe = ev.StageProbe()
    with_probe = run([probe])
    without = run([])
    assert all(np.array_equal(a, b) for a, b in zip(with_probe, without))
    assert np.array_equal(probe.classic, with_probe[-1])


def test_the_classic_output_comes_from_just_before_the_stage(tmp_path: Path, clip: Clip) -> None:
    from makeup_cam.pipeline import MakeupPipeline

    settings = ev._with_model(load_preset("off"), 1.0)
    probe = ev.StageProbe()
    stage = ev.build_stage(write_model_folder(tmp_path / "m"), fake_tools(clip.landmarks))
    pipeline = MakeupPipeline(settings, tracker=FakeTracker(clip.landmarks), segmenter=FakeSegmenter(), effects=ev.builtin_effects([probe], neural=stage))
    last = None
    for index, frame in enumerate(clip.frames[:30]):
        last = pipeline.process(frame, index * 33)
        if index == 0:
            ev.wait_until_serving(stage)
        ev.wait_for_stage(stage)
    assert stage.health > 0.5
    assert np.array_equal(probe.classic, clip.frames[29])
    assert not np.array_equal(last.output, probe.classic)
    pipeline.close()


def test_soak_counts_events_and_keeps_going(tmp_path: Path, video: Path, clip: Clip) -> None:
    options = ev.RunOptions(MakeupSettings.from_dict(load_preset("off").to_dict()), out_dir=tmp_path)
    result = ev.run_soak([video], options, fake_tools(clip.landmarks), minutes=4.0 / 60.0)
    assert result["minutes"] == pytest.approx(4.0 / 60.0)
    assert result["frames_seen"] > 10
    assert result["fps_out"] > 0.0


def test_measure_realtime_without_a_model(video: Path, clip: Clip) -> None:
    options = ev.RunOptions(MakeupSettings.from_dict(load_preset("off").to_dict()))
    result = ev.measure_realtime([video], options, fake_tools(clip.landmarks), seconds=1.0)
    assert result["classic_fps"] > 5.0
    assert "fps_loss" not in result


def test_the_clock_is_read_on_the_slowest_clip_and_a_middling_one() -> None:
    paths = [Path(f"{name}.flv") for name in ("fast", "middle", "slow", "other")]

    def result(name: str, ms: float, scale: float) -> ev.ClipResult:
        return ev.ClipResult(name, 10, 10, 30.0, {"classic_render_ms": ms, "face_scale_median": scale}, [], {})

    # The largest face is not the slowest clip: that is what let a model pass that loses frames elsewhere.
    chosen = ev.realtime_clips(paths, [result("fast", 20.0, 148.0), result("middle", 25.0, 100.0), result("slow", 31.0, 120.0), result("other", 26.0, 110.0), result("slow+sway", 90.0, 120.0)])
    assert chosen[0] == Path("slow.flv")
    assert chosen[1] in (Path("middle.flv"), Path("other.flv"))
    assert ev.realtime_clips(paths, []) == paths[:1]


def test_a_quality_pass_records_how_long_the_classic_effects_took(tmp_path: Path, video: Path, clip: Clip) -> None:
    options = ev.RunOptions(MakeupSettings.from_dict(load_preset("natural").to_dict()))
    result = ev.run_quality_clip(video, options, fake_tools(clip.landmarks))
    assert result.metrics["classic_render_ms"] > 0.0


def test_the_worst_case_of_the_clock_counts_also_at_the_budgets_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options = ev.RunOptions(MakeupSettings(), model_dir=tmp_path)
    runs: list[tuple[str, str]] = []

    def fake_case(path: Path, preset: str, settings: Any, options: Any, tools: Any, seconds: float) -> dict[str, Any]:
        runs.append((path.stem, preset))
        loss = {("a", ""): 0.1, ("b", ""): 0.2, ("a", "natural"): 2.5, ("b", "natural"): 0.3}[(path.stem, preset)]
        timing = {"median": 1.0, "p95": loss, "p99": 2.0, "max": 3.0}
        return {"clip": path.stem, "preset": preset, "seconds": seconds, "fps_in": 30.0, "classic_fps": 30.0, "classic_fps_after": 30.0, "model_fps": 30.0 - loss, "fps_loss": loss, "miss_pct": loss, "vram_mb": 100.0 * loss, "rtt_ms": timing, "stage_ms": timing, "model_process_ms": timing}

    monkeypatch.setattr(ev, "_measure_model_case", fake_case)
    out = ev.measure_realtime([Path("a.flv"), Path("b.flv")], options, ev.Tools(), 1.0, {"natural": MakeupSettings()})
    assert len(runs) == 4
    assert out["fps_loss"] == 2.5 and out["clip"] == "a" and out["preset"] == "natural"
    assert out["rtt_ms"]["p95"] == 2.5 and out["miss_pct"] == 2.5 and out["vram_mb"] == 250.0
    assert len(out["cases"]) == 4


def test_measure_realtime_with_a_model_reads_the_stage(tmp_path: Path, video: Path, clip: Clip) -> None:
    options = ev.RunOptions(MakeupSettings.from_dict(load_preset("off").to_dict()), model_dir=write_model_folder(tmp_path / "m"))
    result = ev.measure_realtime([video], options, fake_tools(clip.landmarks), seconds=1.0)
    assert result["model_fps"] > 5.0
    assert result["stage_ms"] is not None
    assert result["rtt_ms"] is not None
    assert result["miss_pct"] is not None
    metrics, _ = ev.build_metrics(ev.MODE_CLASSIC, [ev.ClipResult("c", 100, 100, 30.0, {}, [], {})], realtime=result)
    assert metrics["fps_loss"]["status"] == ev.STATUS_MEASURED
    assert metrics["vram_mb"]["status"] == ev.STATUS_NOT_RUN  # a fake has no process


# ---- validation harness ---------------------------------------------------------------------------------------------


def test_the_scenarios_all_have_their_detectors() -> None:
    assert validate.expected_for("classic/teeth_warp_2") == validate.EXPECTED["teeth_warp"]
    assert validate.expected_for("pan/paint_lag_3") == validate.EXPECTED["paint_lag"]
    assert validate.expected_for("classic/clean_tint") == ()
    for detectors in validate.EXPECTED.values():
        for detector in detectors:
            if not detector.startswith("event:"):
                assert detector in ev.THRESHOLD_BY_KEY


def test_a_donor_is_always_another_actor() -> None:
    for actor in validate.ACTORS:
        for take in validate.TAKES:
            assert validate.donor_of(actor, take)[:4] != actor
            assert validate.donor_of(actor, take)[5:] == take


def test_a_summary_counts_false_alarms_and_misses() -> None:
    bad = {"metrics": {"flicker_ratio": 9.0}, "events": {}}
    good = {"metrics": {"flicker_ratio": 1.0}, "events": {}}
    table = {
        "classic/clean_tint": {"1002_A": good, "1012_A": bad},
        "classic/shimmer_face_2": {"1002_A": bad, "1012_A": good},
    }
    summary = validate.summarize(table)
    assert summary["clean"]["classic/clean_tint"]["clips_with_any_alarm"] == 1
    assert summary["clean"]["classic/clean_tint"]["fires"] == {"flicker_ratio": 1}
    assert summary["defects"]["classic/shimmer_face_2"] == {"clips": 2, "caught": 1, "missed": 1, "by_detector": {"flicker_ratio": 1, "delta_flicker": 0}}


def test_metrics_only_for_the_classic_reference_count_as_false_alarms_too() -> None:
    """They share their modes with what only a running model can show, which must not hide them from the count."""
    bad = {"metrics": {"mouth_interior_max": 3.0, "delta_abs_p99": 0.9, "paint_lag_local_frames": 5.0}, "events": {}}
    table = {"classic/clean_tint": {"1002_A": bad}}
    assert set(validate.summarize(table)["clean"]["classic/clean_tint"]["fires"]) == {"mouth_interior_max", "delta_abs_p99", "paint_lag_local_frames"}
    rows = validate.per_detector(table)
    assert rows["mouth_interior_max|classic"]["false_alarms"] == 1
    assert "fps_loss|classic" not in rows
    assert "mouth_interior_max" in validate.distributions(table)["classic/clean_tint"]
    # Not in the input mode, where makeup changes those by design.
    assert validate.summarize({"input/clean_natural": {"1002_A": bad}})["clean"]["input/clean_natural"]["fires"] == {}


def test_the_table_per_detector_counts_both_ways() -> None:
    bad = {"metrics": {"flicker_ratio": 9.0}, "events": {"spike": 1}}
    good = {"metrics": {"flicker_ratio": 1.0}, "events": {}}
    table = {
        "classic/clean_tint": {"1002_A": good, "1012_A": bad},
        "classic/shimmer_face_2": {"1002_A": bad, "1012_A": good},
        "classic/spike_12": {"1002_A": bad},
    }
    rows = validate.per_detector(table)
    flicker = rows["flicker_ratio|classic"]
    assert (flicker["clean_clips"], flicker["false_alarms"]) == (2, 1)
    assert (flicker["defect_clips"], flicker["fired"]) == (2, 1)
    assert flicker["defects"] == {"shimmer_face_2": "1/2"}
    spike = rows["event:spike|classic"]
    assert (spike["false_alarms"], spike["fired"], spike["defect_clips"]) == (1, 1, 1)


# ---- the real models, on the public-domain test portrait ------------------------------------------------------------


def test_the_real_pipeline_on_a_real_face_stays_quiet(tmp_path: Path, face_frame_720p: np.ndarray, models_dir: Path) -> None:
    rng = np.random.default_rng(4)
    frames = []
    for index in range(ev.SETTLE_FRAMES + 10):
        dx = 6.0 * np.sin(2 * np.pi * index / 20.0)
        moved = cv2.warpAffine(face_frame_720p, np.float32([[1, 0, dx], [0, 1, 0]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        frames.append(np.clip(moved.astype(np.float32) + rng.normal(0.0, 1.0, moved.shape).astype(np.float32), 0, 255).astype(np.uint8))
    video = write_video(tmp_path / "portrait.avi", frames)
    result = ev.run_quality_clip(video, ev.RunOptions(load_preset("natural")), ev.Tools())
    assert result.measured_frames >= 8
    assert result.metrics["flicker_ratio"] < 1.25
    assert result.metrics["identity_structure_p01"] > 0.75
    assert result.metrics["landmark_shift_p99"] < 0.03
    assert result.metrics["mouth_latency_frames"] != result.metrics["mouth_latency_frames"] or result.metrics["mouth_latency_frames"] < 2.0


# ---- what holds still: a seam, a grid, a cast, moved content (review findings) ------------------------------------------


def _flat_reference() -> np.ndarray:
    rng = np.random.default_rng(3)
    return cv2.GaussianBlur(rng.normal(150.0, 12.0, (256, 256)).astype(np.float32), (0, 0), 3.0)


def _calm_mask() -> np.ndarray:
    mask = np.zeros((256, 256), np.uint8)
    mask[60:200, 60:200] = 1
    return mask


def test_a_step_in_the_change_shows_where_the_picture_has_no_edge() -> None:
    gray, mask = _flat_reference(), _calm_mask()
    smooth = np.zeros((256, 256, 3), np.float32) + 4.0
    step = smooth.copy()
    step[:128] += 20.0
    assert ev.seam_edge(smooth, gray, mask) < 0.1
    assert ev.seam_edge(step, gray, mask) > ev.THRESHOLD_BY_KEY["seam_edge"].worst


def test_a_grid_in_the_change_shows() -> None:
    gray, mask = _flat_reference(), _calm_mask()
    grid = ((np.indices((256, 256)).sum(axis=0) // 8 % 2).astype(np.float32) * 2.0 - 1.0)[..., None] * np.ones(3, np.float32) * 5.0
    assert ev.seam_edge(grid.astype(np.float32), gray, mask) > ev.THRESHOLD_BY_KEY["seam_edge"].worst


def test_a_change_that_follows_the_pictures_own_edges_is_no_seam() -> None:
    mask = _calm_mask()
    gray = np.full((256, 256), 100.0, np.float32)
    gray[:128] = 160.0
    follows = np.zeros((256, 256, 3), np.float32)
    follows[:128] = 15.0  # a quarter of the picture's own edge
    assert ev.seam_edge(follows, gray, mask) < ev.THRESHOLD_BY_KEY["seam_edge"].value


def test_moved_content_shows_as_flow_and_a_change_of_color_does_not() -> None:
    rng = np.random.default_rng(4)
    gray = cv2.GaussianBlur(rng.normal(128.0, 40.0, (256, 256)).astype(np.float32), (0, 0), 2.0)
    mask = _calm_mask()
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    same = ev.warp_flow(flow, gray, np.clip(gray + 6.0, 0, 255), mask)
    moved = ev.warp_flow(flow, gray, cv2.warpAffine(gray, np.float32([[1, 0, 2.0], [0, 1, 0]]), (256, 256), borderMode=cv2.BORDER_REPLICATE), mask)
    assert same < ev.THRESHOLD_BY_KEY["warp_flow_px"].value
    assert moved > ev.THRESHOLD_BY_KEY["warp_flow_px"].value


def test_a_cast_on_one_side_and_on_the_whole_face_show_in_the_analyzer(clip: Clip) -> None:
    clean = analyze(clip, clip.frames)
    assert clean.metrics["side_cast"] < 0.5 and clean.metrics["skin_cast"] < 0.5
    half = [frame.copy() for frame in clip.frames]
    whole = [frame.copy() for frame in clip.frames]
    for a, b in zip(half, whole):
        a[:, : a.shape[1] // 2, 2] = np.clip(a[:, : a.shape[1] // 2, 2].astype(np.int16) + 12, 0, 255).astype(np.uint8)
        b[..., 2] = np.clip(b[..., 2].astype(np.int16) + 25, 0, 255).astype(np.uint8)
    assert analyze(clip, half).metrics["side_cast"] > ev.THRESHOLD_BY_KEY["side_cast"].value
    assert analyze(clip, whole).metrics["skin_cast"] > ev.THRESHOLD_BY_KEY["skin_cast"].value


def test_a_soft_patch_fixed_to_the_frame_shows_when_the_face_moves() -> None:
    estimator = ev.LagEstimator()
    still = np.zeros((H // 4, W // 4, 3), np.float32)
    landmarks = np.load(DATA).astype(np.float32)
    s = regions.face_scale(landmarks)
    ys, xs = np.mgrid[0 : H // 4, 0 : W // 4].astype(np.float32)
    center = (landmarks[regions.NOSE_TIP] + (0.5 * s, 0.3 * s)) / 4.0
    still[..., 2] = 8.0 * np.exp(-((xs - center[0]) ** 2 + (ys - center[1]) ** 2) / (0.3 * s / 4.0) ** 2)
    inside = np.ones((ev.ANALYSIS_CROP, ev.ANALYSIS_CROP), np.float32)
    for index in range(70):
        shift = np.array([30.0 * np.sin(2 * np.pi * index / 40.0), 0.0], np.float32)
        estimator.update(still, landmarks + shift, inside, inside)
    assert estimator.moving_frames >= estimator.MIN_FRAMES
    assert estimator.clip_frame_locked() > ev.THRESHOLD_BY_KEY["frame_locked_ratio"].value

    following = ev.LagEstimator()
    for index in range(70):
        shift = 30.0 * np.sin(2 * np.pi * index / 40.0)
        moved = cv2.warpAffine(still, np.float32([[1, 0, shift / 4.0], [0, 1, 0]]), (W // 4, H // 4), flags=cv2.INTER_LINEAR)
        following.update(moved, landmarks + np.array([shift, 0.0], np.float32), inside, inside)
    assert following.clip_frame_locked() < ev.THRESHOLD_BY_KEY["frame_locked_ratio"].value


def test_the_new_spatial_checks_count_for_the_verdict() -> None:
    for key in ("warp_flow_px", "seam_edge", "side_cast", "skin_cast", "rim_change", "frame_locked_ratio", "paste_drift_p99"):
        threshold = ev.THRESHOLD_BY_KEY[key]
        assert threshold.required and ev.MODE_CLASSIC in threshold.modes and threshold.basis
        clips = [ev.ClipResult("c", 100, 100, 30.0, {key: (threshold.worst or threshold.value) * 2.0}, [], {})]
        metrics, _ = ev.build_metrics(ev.MODE_CLASSIC, clips)
        assert metrics[key]["passed"] is False


def test_paint_on_a_masks_soft_rim_is_measured_and_the_rest_of_the_change_is_not(clip: Clip) -> None:
    landmarks = clip.landmarks[0]
    hair = np.zeros((H, W), np.float32)
    hair[20:60] = 0.2  # a soft rim above the head: the weight is 0 here
    zones = ev.build_zones(landmarks, hair, None, W, H)
    assert zones.rim is not None and zones.rim.any()
    assert not (zones.rim & (zones.weight > 0.0)).any()
    assert ev.build_zones(landmarks, None, None, W, H).rim is None


def _with_block(frames: Sequence[np.ndarray], landmarks: np.ndarray, frame: int, level: int) -> list[np.ndarray]:
    """The same frames with a block on the cheek changed by ``level`` on one frame."""
    out = [image.copy() for image in frames]
    x, y = regions.cheek_center(landmarks, "left").astype(int)
    view = out[frame][y - 25 : y + 25, x - 25 : x + 25]
    view[:] = np.clip(view.astype(np.int16) + level, 0, 255).astype(np.uint8)
    return out


def test_a_spike_is_what_the_picture_did_not_do_too(clip: Clip) -> None:
    """A model's change follows a blink: 3 levels where the picture itself jumped by 26 is no break (1013_TIE_SAD_XX, a stand-in)."""
    frame = 25
    moved = _with_block(clip.frames, clip.landmarks[0], frame, 26)
    follows = _with_block(moved, clip.landmarks[0], frame, 4)
    alone = _with_block(clip.frames, clip.landmarks[0], frame, 8)
    following = analyze(clip, follows, reference=moved)
    assert not [event for event in following.events if event.kind == "spike"]
    on_its_own = analyze(clip, alone)
    assert [event for event in on_its_own.events if event.kind == "spike"]


def test_nodding_moves_the_frame_and_not_its_size(clip: Clip) -> None:
    nodded = list(ev.nod_frames(clip.frames[:12]))
    assert len(nodded) == 12 and nodded[5].shape == clip.frames[5].shape
    assert not np.array_equal(nodded[5], clip.frames[5])


def test_clips_given_twice_add_up(tmp_path: Path) -> None:
    args = ev.build_parser().parse_args(["--clips", "a.flv", "--clips", "b.flv", "c.flv"])
    assert args.clips == ["a.flv", "b.flv", "c.flv"]


def test_too_few_frames_per_clip_say_so(tmp_path: Path, video: Path, clip: Clip) -> None:
    with pytest.raises(ev.EvaluationError, match="frames-per-clip"):
        ev.run_evaluation(_args(clips=[str(video)], out=tmp_path / "o", preset="off", frames_per_clip=15), fake_tools(clip.landmarks))


def test_a_stage_that_stays_off_says_why_in_japanese_only(clip: Clip) -> None:
    class Off:
        def stats(self) -> dict[str, Any]:
            return {"phase": "off", "status": "モデルがありません", "refused_because": None}

    with pytest.raises(ev.EvaluationError) as refused:
        ev.wait_until_serving(Off())
    assert "None" not in str(refused.value) and "モデルがありません" in str(refused.value)


def test_a_refused_model_is_reported_in_japanese_only(tmp_path: Path) -> None:
    (tmp_path / "model.json").write_text("[" * 200_000, encoding="utf-8")
    with pytest.raises(ev.EvaluationError) as refused:
        ev._model_summary(tmp_path)
    assert str(refused.value) == f"モデルを読み込めません: {model_files.NOT_PASSED_MESSAGE}"


def test_the_report_says_how_far_the_code_is_from_its_commit(tmp_path: Path, video: Path, clip: Clip) -> None:
    report = ev.run_evaluation(_args(clips=[str(video)], out=tmp_path / "o", preset="off"), fake_tools(clip.landmarks))
    assert "uncommitted_files" in report["conditions"]


# ---- the clock: repeatable, and on the right clips ------------------------------------------------------------------------


def _scripted_clock(monkeypatch: pytest.MonkeyPatch, model_fps: Sequence[float], classic_fps: float = 30.0, overloads: int = 0) -> list[dict[str, Any]]:
    """``run_realtime`` that plays back a script: classic runs at ``classic_fps``, model runs at each of ``model_fps`` in turn."""
    calls: list[dict[str, Any]] = []
    left = list(model_fps)

    def fake(frames: Any, settings: Any, fps: float, seconds: float, tools: Any, stage: Any = None, **kwargs: Any) -> ev.RealtimeResult:
        calls.append({"stage": stage, **kwargs})
        if stage is None:
            return ev.RealtimeResult(seconds, fps, classic_fps, [31.0, 31.5])
        info = {"frames_by_age": {1: 100, 2: 1}, "rtt_ms": [2.0, 2.5], "stage_ms": [1.0, 1.1], "overloads": overloads, "lost_frames": 2, "thrifty_frames": 5}
        return ev.RealtimeResult(seconds, fps, left.pop(0), [32.0, 32.5], info, 120.0)

    monkeypatch.setattr(ev, "run_realtime", fake)
    monkeypatch.setattr(ev, "build_stage", lambda *args, **kwargs: object())
    monkeypatch.setattr(ev, "_realtime_frames", lambda path, options: [np.zeros((4, 4, 3), np.uint8)] * 2)
    monkeypatch.setattr(ev, "file_fps", lambda path: 30.0)
    return calls


def test_one_lucky_run_does_not_decide_the_frame_rate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Measured: the same model on the same clip read 0.40, 0.00, 0.18, 0.13, -0.02, 0.01 and 1.14 from one run to the next."""
    calls = _scripted_clock(monkeypatch, [30.0, 28.8, 29.9])
    options = ev.RunOptions(MakeupSettings(), model_dir=tmp_path)
    case = ev._measure_model_case(Path("a.flv"), "", MakeupSettings(), options, ev.Tools(), 1.0)
    assert case["fps_loss"] == pytest.approx(1.2)
    assert case["model_fps"] == pytest.approx(28.8)
    assert case["fps_loss_runs"] == pytest.approx([0.0, 1.2, 0.1])
    assert sum(1 for call in calls if call["stage"] is not None) == 3
    assert sum(1 for call in calls if call["stage"] is None) == 4  # classic on both sides of each


def test_frames_that_sent_no_job_on_purpose_are_not_late(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _scripted_clock(monkeypatch, [30.0, 30.0, 30.0])
    case = ev._measure_model_case(Path("a.flv"), "", MakeupSettings(), ev.RunOptions(MakeupSettings(), model_dir=tmp_path), ev.Tools(), 1.0)
    assert case["frames_by_age"][2] == 3 and case["thrifty_frames"] == 15
    assert case["miss_pct"] == 0.0


def test_the_clock_runs_with_what_the_app_does_after_each_frame(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _scripted_clock(monkeypatch, [30.0, 30.0, 30.0])
    ev._measure_model_case(Path("a.flv"), "", MakeupSettings(), ev.RunOptions(MakeupSettings(), model_dir=tmp_path), ev.Tools(), 1.0)
    assert calls and all(call.get("deliver") for call in calls)


def test_the_delivery_takes_its_time_on_the_output_thread(video: Path, clip: Clip, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = clip.frames[0]
    began = time.perf_counter()
    for _ in range(20):
        ev._deliver_like_the_app(frame)
    assert (time.perf_counter() - began) / 20.0 >= ev.DELIVERY_MS / 1000.0
    seen: list[int] = []
    monkeypatch.setattr(ev, "_deliver_like_the_app", lambda output: seen.append(output.shape[0]))
    frames = list(clip.frames[:4])
    ev.run_realtime(frames, ev._without_model(MakeupSettings.from_dict(load_preset("off").to_dict())), 30.0, 0.3, fake_tools(clip.landmarks))
    assert seen == []  # the long run has an analysis in its place
    ev.run_realtime(frames, ev._without_model(MakeupSettings.from_dict(load_preset("off").to_dict())), 30.0, 0.3, fake_tools(clip.landmarks), deliver=True)
    assert seen and seen[0] == frames[0].shape[0]


def test_the_rounds_can_be_chosen_and_must_be_at_least_one(tmp_path: Path, video: Path, clip: Clip) -> None:
    assert ev.build_parser().parse_args([]).fps_rounds == ev.REALTIME_ROUNDS
    with pytest.raises(ev.EvaluationError):
        ev.check_arguments(ev.build_parser().parse_args(["--fps-rounds", "0"]))


def test_a_stage_that_gave_up_to_keep_the_frame_rate_does_not_pass_the_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _scripted_clock(monkeypatch, [30.0, 30.0, 30.0], overloads=1)
    realtime = ev.measure_realtime([Path("a.flv")], ev.RunOptions(MakeupSettings(), model_dir=tmp_path), ev.Tools(), 1.0)
    assert realtime["fps_loss"] == pytest.approx(0.0) and realtime["budget_pauses"] == 3
    metrics, _ = ev.build_metrics(ev.MODE_CLASSIC, [ev.ClipResult("c", 100, 100, 30.0, {}, [], {})], realtime=realtime)
    assert metrics["fps_loss"]["passed"] is False and metrics["fps_loss"]["budget_pauses"] == 3
    verdict, _ = ev.decide_verdict(metrics, 2000)
    assert verdict == "fail"


def test_the_budgets_preset_is_read_on_the_clips_heavy_at_that_preset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A model evaluated at ``off``: all clips cost 3.1 ms there, so its quality pass cannot tell which is heavy at ``natural``."""
    paths = [tmp_path / f"{name}.avi" for name in ("a", "b", "heavy")]
    for path in paths:
        path.write_bytes(b"not read")
    costs = {"a": 3.1, "b": 3.2, "heavy": 3.0}
    natural_costs = {"a": 20.0, "b": 25.0, "heavy": 31.0}
    measured: list[tuple[str, str]] = []

    def quality(path: Path, options: Any, tools: Any, **kwargs: Any) -> ev.ClipResult:
        return ev.ClipResult(path.stem, 100, 100, 30.0, {"classic_render_ms": costs[path.stem]}, [], {})

    def cost(chosen: Sequence[Path], settings: Any, options: Any, tools: Any) -> list[ev.ClipResult]:
        return [ev.ClipResult(path.stem, 30, 30, 30.0, {"classic_render_ms": natural_costs[path.stem]}, [], {}) for path in chosen]

    def case(path: Path, preset: str, settings: Any, options: Any, tools: Any, seconds: float) -> dict[str, Any]:
        measured.append((path.stem, preset))
        timing = {"median": 1.0, "p95": 1.0, "p99": 1.0, "max": 1.0}
        return {"clip": path.stem, "preset": preset, "seconds": seconds, "fps_in": 30.0, "classic_fps": 30.0, "classic_fps_after": 30.0, "model_fps": 30.0, "fps_loss": 0.0, "miss_pct": 0.0, "vram_mb": 1.0, "rtt_ms": timing, "stage_ms": timing, "model_process_ms": timing, "classic_process_ms": timing}

    monkeypatch.setattr(ev, "run_quality_clip", quality)
    monkeypatch.setattr(ev, "render_costs", cost)
    monkeypatch.setattr(ev, "_measure_model_case", case)
    monkeypatch.setattr(ev, "run_strength_checks", lambda *args, **kwargs: {"zero_identical": True, "monotonic": True, "mean_change": []})
    monkeypatch.setattr(ev, "run_fade_drill", lambda *args, **kwargs: {"ok": True, "cases": []})
    model = write_model_folder(tmp_path / "m", training_data={"kind": "none", "input_preset": "off", "note": "test"})
    args = _args(clips=[str(path) for path in paths], model=model, out=tmp_path / "out", measure_fps=True, fps_seconds=1.0)
    ev.run_evaluation(args, ev.Tools(backend_factory=lambda spec: None))
    assert ("heavy", "natural") in measured
    assert {clip for clip, preset in measured if preset == ""} >= {"b"}  # the evaluated preset keeps its own clips


def test_the_cost_of_each_clip_is_measured_at_the_given_settings(video: Path, clip: Clip) -> None:
    natural = MakeupSettings.from_dict(load_preset("natural").to_dict())
    off = MakeupSettings.from_dict(load_preset("off").to_dict())
    options = ev.RunOptions(off)
    tools = fake_tools(clip.landmarks)
    heavy = ev.render_costs([video], natural, options, tools)
    light = ev.render_costs([video], off, options, tools)
    assert heavy[0].name == video.stem and heavy[0].metrics["classic_render_ms"] > light[0].metrics["classic_render_ms"] >= 0.0
    assert ev.realtime_clips([video], heavy) == [video]
