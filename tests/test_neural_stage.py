"""The stage as an effect, with a fake backend and a fake clock: no GPU, no ONNX package, no process."""

from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from neural_fakes import FakeBackend, uniform_delta, write_model_folder

from makeup_cam import regions
from makeup_cam.config import NeuralSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.neural import geometry, stage, worker
from makeup_cam.neural.backend import BackendError
from makeup_cam.neural.model_files import ModelSpec, load_spec
from makeup_cam.neural.stage import NeuralFinishEffect, _weighted_delta

FRAME_S = 1.0 / 30.0
SIZE = 256


class Rig:
    """A stage run frame by frame: the control thread's step runs between frames, on a fake clock."""

    def __init__(self, tmp_path: Path, landmarks: np.ndarray, *, folder: Path | None = None, model: Any = None, **kwargs: Any) -> None:
        self.now = 100.0
        self.landmarks = landmarks
        self.backends: list[FakeBackend] = []
        self.model = model
        self.prepare_backend: Any = None
        self.hold_ahead = False  # True: a test keeps the helper thread from finishing
        self.effect = NeuralFinishEffect(
            model_dir=folder or write_model_folder(tmp_path / "model", size=SIZE),
            allow_unevaluated=True,
            backend_factory=self._new_backend,
            clock=lambda: self.now,
            control_thread=False,
            **kwargs,
        )
        self.section = NeuralSettings(enabled=True, strength=1.0)
        self.classic = np.full((720, 1280, 3), 0.5, np.float32)

    def _new_backend(self, spec: ModelSpec) -> FakeBackend:
        backend = FakeBackend(spec, lambda: self.now, self.model)
        if self.prepare_backend is not None:
            self.prepare_backend(backend)
        self.backends.append(backend)
        return backend

    @property
    def backend(self) -> FakeBackend:
        return self.backends[-1]

    def control(self) -> None:
        self.effect._control_step(self.now)

    def frame(
        self,
        *,
        intensity: float = 1.0,
        landmarks: np.ndarray | None = None,
        control: bool = True,
        ahead: bool = False,
        ahead_landmarks: np.ndarray | None = None,
        **masks: Any,
    ) -> np.ndarray:
        image = self.classic.copy()
        lm = self.landmarks if landmarks is None else landmarks
        ctx = FrameContext(height=720, width=1280, landmarks=lm, intensity=intensity, **masks)
        if ahead or ahead_landmarks is not None:
            early = ctx if ahead_landmarks is None else FrameContext(height=720, width=1280, landmarks=ahead_landmarks, **masks)
            self.effect.start_ahead(early, self.section)
            job = self.effect._ahead
            deadline = time.perf_counter() + 5.0
            while job is not None and not job.done and not self.hold_ahead and time.perf_counter() < deadline:
                time.sleep(0.0005)
        out = self.effect.apply(image, ctx, self.section)
        if control:
            self.control()
        self.now += FRAME_S
        return out

    def frames(self, count: int, **kwargs: Any) -> np.ndarray:
        out = self.classic
        for _ in range(count):
            out = self.frame(**kwargs)
        return out

    def changed(self, out: np.ndarray) -> float:
        return float(np.abs(out - self.classic).max())


@pytest.fixture
def face_landmarks_720p(face_landmarks_720p: np.ndarray) -> np.ndarray:
    """The portrait made larger (its face is just under the soft gate's full size), so that health reaches 1."""
    center = face_landmarks_720p.mean(axis=0)
    larger = center + (face_landmarks_720p - center) * 1.5
    assert geometry.soft_gate(larger, (720, 1280, 3)) == 1.0
    return larger


@pytest.fixture
def rig(tmp_path: Path, face_landmarks_720p: np.ndarray) -> Rig:
    return Rig(tmp_path, face_landmarks_720p)


def running(rig: Rig) -> Rig:
    """Worker up, delta there, health full."""
    for _ in range(40):
        rig.frame()
        if rig.effect.health == 1.0:
            return rig
    raise AssertionError(f"health only reached {rig.effect.health}")


# ---- nothing to do, nothing done --------------------------------------------------------------


def test_strength_zero_starts_nothing_and_changes_nothing(rig: Rig) -> None:
    rig.section.strength = 0.0
    before = threading.active_count()
    out = rig.frames(5)
    assert np.array_equal(out, rig.classic)
    assert rig.backends == [] and rig.effect.status_message is None
    assert threading.active_count() == before


def test_strength_zero_on_a_running_stage_sends_no_jobs_and_is_bit_exact(rig: Rig) -> None:
    running(rig)
    rig.section.strength = 0.0
    sent = len(rig.backend.jobs)
    out = rig.frames(5)
    assert np.array_equal(out, rig.classic)
    assert len(rig.backend.jobs) == sent


def test_no_model_is_a_bit_exact_no_op_with_a_clear_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, face_landmarks_720p: np.ndarray) -> None:
    monkeypatch.setenv("AI_MAKEUP_CAM_NEURAL_DIR", str(tmp_path))
    monkeypatch.setattr(stage, "_onnxruntime_available", lambda: True)
    effect = NeuralFinishEffect(control_thread=False)
    image = np.full((720, 1280, 3), 0.5, np.float32)
    expected = image.copy()
    ctx = FrameContext(height=720, width=1280, landmarks=face_landmarks_720p)
    section = NeuralSettings(enabled=True, strength=1.0)
    for _ in range(3):
        assert np.array_equal(effect.apply(image, ctx, section), expected)
        effect._control_step(time.perf_counter())
    assert effect.status_message == "モデルがありません"
    assert effect.backend is None


def test_without_onnxruntime_nothing_starts_and_the_status_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, face_landmarks_720p: np.ndarray
) -> None:
    monkeypatch.setattr(stage, "_onnxruntime_available", lambda: False)
    monkeypatch.setattr(stage.ProcessBackend, "start", lambda self: pytest.fail("a worker was started"))
    effect = NeuralFinishEffect(model_dir=write_model_folder(tmp_path / "m"), allow_unevaluated=True, control_thread=False)
    image = np.full((720, 1280, 3), 0.5, np.float32)
    expected = image.copy()
    ctx = FrameContext(height=720, width=1280, landmarks=face_landmarks_720p)
    assert np.array_equal(effect.apply(image, ctx, NeuralSettings(enabled=True, strength=1.0)), expected)
    effect._control_step(time.perf_counter())
    assert "onnxruntime" in effect.status_message
    assert effect.backend is None


def test_a_stand_in_is_refused_for_calls(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    effect = NeuralFinishEffect(
        model_dir=write_model_folder(tmp_path / "m"), backend_factory=lambda spec: pytest.fail("started"), control_thread=False
    )
    image = np.full((720, 1280, 3), 0.5, np.float32)
    effect.apply(image, FrameContext(height=720, width=1280, landmarks=face_landmarks_720p), NeuralSettings(enabled=True, strength=1.0))
    effect._control_step(time.perf_counter())
    assert effect.status_message == "評価に合格したモデルがありません"


def test_the_control_thread_does_not_start_before_the_stage_is_wanted(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    effect = NeuralFinishEffect(model_dir=write_model_folder(tmp_path / "m"), backend_factory=lambda spec: pytest.fail("started"))
    image = np.full((720, 1280, 3), 0.5, np.float32)
    ctx = FrameContext(height=720, width=1280, landmarks=face_landmarks_720p)
    effect.apply(image, ctx, NeuralSettings(enabled=True, strength=0.0))
    assert effect._control is None
    effect.close()


# ---- strength -----------------------------------------------------------------------------------


def test_health_rises_over_half_a_second(rig: Rig) -> None:
    rig.frame()  # the worker starts behind this frame
    rig.frame()  # first job
    healths = []
    for _ in range(16):
        rig.frame()
        healths.append(rig.effect.health)
    assert healths[0] == pytest.approx(2.0 / 15.0)
    assert np.all(np.diff(healths)[:12] > 0) and healths[-1] == 1.0
    assert np.diff(healths).max() == pytest.approx(1.0 / 15.0)


def _cheek_change(rig: Rig, out: np.ndarray) -> float:
    x, y = regions.cheek_center(rig.landmarks, "left").astype(int)
    return float(out[y, x, 0] - rig.classic[y, x, 0])


def test_strength_and_intensity_multiply(rig: Rig) -> None:
    running(rig)
    full = _cheek_change(rig, rig.frame())
    assert full == pytest.approx(0.04, abs=1e-4)
    rig.section.strength = 0.5
    assert _cheek_change(rig, rig.frame()) == pytest.approx(0.02, abs=1e-4)
    assert _cheek_change(rig, rig.frame(intensity=0.5)) == pytest.approx(0.01, abs=1e-4)
    assert np.array_equal(rig.frame(intensity=0.0), rig.classic)


def test_only_the_face_changes_and_mouth_and_eyes_stay_bit_exact(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    rig = Rig(tmp_path, face_landmarks_720p, model=lambda crop: uniform_delta(crop.shape[0], 0.14))
    out = running(rig).frame()
    roi = regions.Roi(0, 0, 1280, 720)
    keep_out = regions.polygon_mask(regions.expand_polygon(rig.landmarks[regions.LIPS_INNER], 0.01 * regions.face_scale(rig.landmarks)), roi)
    assert np.array_equal(out[keep_out >= 1.0], rig.classic[keep_out >= 1.0])
    for side in regions.SIDES:
        eye = regions.eye_opening_mask(rig.landmarks, side, roi)
        assert np.array_equal(out[eye >= 1.0], rig.classic[eye >= 1.0])
    allowed = regions.face_oval_mask(rig.landmarks, roi, expand_px=0.03 * regions.face_scale(rig.landmarks), feather_px=0.05 * regions.face_scale(rig.landmarks))
    assert np.array_equal(out[allowed <= 0.0], rig.classic[allowed <= 0.0])
    assert rig.changed(out) > 0.1


def test_may_edit_eyes_from_the_metadata_opens_the_eyes_only(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    folder = write_model_folder(tmp_path / "eyes", size=SIZE, may_edit={"eyes": True})
    rig = Rig(tmp_path, face_landmarks_720p, folder=folder)
    out = running(rig).frame()
    x, y = regions.eye_center(rig.landmarks, "left").astype(int)
    assert out[y, x, 0] != rig.classic[y, x, 0]
    x, y = rig.landmarks[regions.LIPS_INNER].mean(axis=0).astype(int)
    assert out[y, x, 0] == rig.classic[y, x, 0]


def test_the_delta_is_clamped_whatever_the_model_says(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    def spikes(crop: np.ndarray) -> np.ndarray:
        raw = np.zeros((3, crop.shape[0], crop.shape[0]), np.float32)
        raw[:, 120:136, 90:106] = 1.0  # a small square: the mean over the center stays low
        return raw

    rig = Rig(tmp_path, face_landmarks_720p, model=spikes)
    out = running(rig).frame()
    assert 0.2 < rig.changed(out) <= 0.25 + 1e-5


# ---- the paste made ahead on the helper thread ---------------------------------------------------


def _textured_delta(crop: np.ndarray) -> np.ndarray:
    grid = np.linspace(-0.1, 0.1, crop.shape[0], dtype=np.float32)
    return np.stack([np.outer(grid, grid) * 10.0, np.outer(np.abs(grid), grid), np.tile(grid, (crop.shape[0], 1))])


def _two_rigs(tmp_path: Path, landmarks: np.ndarray) -> tuple[Rig, Rig]:
    made = [Rig(tmp_path / name, landmarks, folder=write_model_folder(tmp_path / name, size=SIZE), model=_textured_delta) for name in ("a", "b")]
    for rig in made:
        rig.classic = np.random.default_rng(5).random((720, 1280, 3), dtype=np.float32)
    return made[0], made[1]


def test_a_paste_made_ahead_equals_one_made_on_the_camera_thread_bit_for_bit(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    inline, ahead = _two_rigs(tmp_path, face_landmarks_720p)
    hair = np.zeros((720, 1280), np.float32)
    hair[50:200, 400:900] = 0.7
    for index in range(30):
        moved = face_landmarks_720p + index * 0.75
        expected = inline.frame(landmarks=moved, hair_mask=hair)
        out = ahead.frame(landmarks=moved, hair_mask=hair, ahead=True)
        assert np.array_equal(out, expected), index
    assert ahead.effect.pastes_ahead >= 10
    assert inline.effect.pastes_ahead == 0
    assert ahead.changed(out) > 0.0
    ahead.effect.close()


def test_a_paste_that_is_not_ready_never_makes_the_frame_wait(tmp_path: Path, face_landmarks_720p: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    inline, ahead = _two_rigs(tmp_path, face_landmarks_720p)
    for rig in (inline, ahead):
        running(rig)
    release = threading.Event()
    real_weight = geometry.edit_weight

    def stuck_on_the_helper(*args: Any, **kwargs: Any) -> np.ndarray:
        if threading.current_thread().name == "neural-paste":
            release.wait(10.0)
        return real_weight(*args, **kwargs)

    monkeypatch.setattr(geometry, "edit_weight", stuck_on_the_helper)
    ahead.hold_ahead = True
    before = ahead.effect.pastes_inline
    started = time.perf_counter()
    out = ahead.frame(ahead=True)
    elapsed = time.perf_counter() - started
    release.set()
    assert elapsed < 1.0
    assert ahead.effect.pastes_inline == before + 1
    assert np.array_equal(out, inline.frame())
    ahead.effect.close()


def test_a_paste_made_for_other_landmarks_is_not_used(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    inline, ahead = _two_rigs(tmp_path, face_landmarks_720p)
    for rig in (inline, ahead):
        running(rig)
    moved = face_landmarks_720p + 30.0
    out = ahead.frame(landmarks=moved, ahead_landmarks=face_landmarks_720p)
    assert np.array_equal(out, inline.frame(landmarks=moved))
    ahead.effect.close()


def test_a_failure_on_the_helper_thread_costs_nothing(tmp_path: Path, face_landmarks_720p: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    inline, ahead = _two_rigs(tmp_path, face_landmarks_720p)
    for rig in (inline, ahead):
        running(rig)
    real_weight = geometry.edit_weight

    def fails_on_the_helper(*args: Any, **kwargs: Any) -> np.ndarray:
        if threading.current_thread().name == "neural-paste":
            raise RuntimeError("drill")
        return real_weight(*args, **kwargs)

    monkeypatch.setattr(geometry, "edit_weight", fails_on_the_helper)
    before = ahead.effect.pastes_inline
    assert np.array_equal(ahead.frame(ahead=True), inline.frame())
    assert ahead.effect.pastes_inline == before + 1
    ahead.effect.close()


def test_nothing_is_prepared_when_nothing_will_be_pasted(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    rig = Rig(tmp_path, face_landmarks_720p)
    rig.frame(ahead=True)  # health is still 0
    assert rig.effect._ahead_thread._thread is None
    running(rig)
    rig.section = NeuralSettings(enabled=True, strength=0.0)
    rig.frame(ahead=True)
    assert rig.effect._ahead_thread._thread is None
    rig.section = NeuralSettings(enabled=True, strength=1.0)
    running(rig)
    rig.frame(ahead=True, landmarks=None)
    assert rig.effect._ahead_thread._thread is not None
    rig.effect.close()


def test_reset_throws_away_a_paste_made_ahead(rig: Rig) -> None:
    running(rig)
    ctx = FrameContext(height=720, width=1280, landmarks=rig.landmarks)
    rig.effect.start_ahead(ctx, rig.section)
    job = rig.effect._ahead
    rig.effect.reset()
    assert rig.effect._ahead is None and job.abandoned
    before = rig.effect.pastes_ahead
    out = rig.frame()
    assert rig.effect.pastes_ahead == before
    assert rig.changed(out) <= 0.04 * stage.HEALTH_RISE + 1e-6  # risen from 0, with the new result
    rig.effect.close()


def test_a_late_result_reaches_a_paste_made_ahead(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    """The helper waits a moment for the job in flight, so that the frame shows age 1."""
    rig = Rig(tmp_path, face_landmarks_720p, model=_textured_delta)
    running(rig)
    rig.backend.hold = True
    rig.frame()  # this frame's job is now in flight
    before = dict(rig.effect.frames_by_age)
    threading.Timer(0.003, rig.backend.finish).start()
    rig.frame(ahead=True)
    assert rig.effect.frames_by_age.get(1, 0) == before.get(1, 0) + 1
    rig.effect.close()


def test_masks_are_asked_for_only_by_a_stage_that_will_use_them(rig: Rig) -> None:
    effect = rig.effect
    assert not effect.wants_masks(rig.section)  # not started
    running(rig)
    assert effect.wants_masks(rig.section)
    assert not effect.wants_masks(NeuralSettings(enabled=True, strength=0.0))
    effect.close()
    assert not effect.wants_masks(rig.section)


def test_a_delta_of_another_size_is_never_pasted(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    rig = Rig(tmp_path, face_landmarks_720p)
    running(rig)
    rig.effect._spec = dataclasses.replace(rig.effect._spec, size=512)
    rig.backend.hold = True  # no new results: only the old delta is there
    rig.frame(control=False)
    assert rig.changed(rig.frame(control=False)) == 0.0


def test_the_crop_is_taken_from_the_frame_with_the_contracts_geometry(rig: Rig) -> None:
    rig.classic = np.random.default_rng(0).random((720, 1280, 3), dtype=np.float32)
    rig.frame()
    rig.backend.hold = True
    rig.frame(control=False)
    crop, _, _ = rig.backend.pending
    np.testing.assert_array_equal(crop, geometry.extract_crop(rig.classic, rig.landmarks, SIZE))


# ---- results that are late ----------------------------------------------------------------------


def _center_of_change(out: np.ndarray, classic: np.ndarray) -> np.ndarray:
    change = np.abs(out - classic).sum(axis=2)
    ys, xs = np.nonzero(change > change.max() * 0.5)
    return np.array([xs.mean(), ys.mean()])


def test_a_late_result_follows_the_face_through_the_current_landmarks(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    def dot(crop: np.ndarray) -> np.ndarray:
        raw = np.zeros((3, crop.shape[0], crop.shape[0]), np.float32)
        raw[:, 128:144, 128:144] = 0.2  # on the cheek-side of the crop's middle: inside the face oval
        return raw

    rig = Rig(tmp_path, face_landmarks_720p, model=dot)
    running(rig)
    rig.backend.hold = True
    here = _center_of_change(rig.frame(), rig.classic)
    assert rig.backend.pending is not None
    moved = face_landmarks_720p + (31.0, 17.0)
    there = _center_of_change(rig.frame(landmarks=moved), rig.classic)
    np.testing.assert_allclose(there - here, (31.0, 17.0), atol=1.5)
    assert rig.effect.health == 1.0


def test_a_late_result_is_reused_for_ages_two_and_three_then_the_stage_fades(rig: Rig) -> None:
    running(rig)
    rig.backend.hold = True
    rig.frame()  # the job of this frame stays in flight; the delta is from the frame before: age 1
    for _ in range(2):  # ages 2 and 3
        out = rig.frame()
        assert rig.effect.health == 1.0
        assert rig.changed(out) > 0.03
    healths = []
    for _ in range(11):  # age 4 and older
        out = rig.frame()
        healths.append(rig.effect.health)
    assert healths[0] == pytest.approx(0.9)
    assert healths[9] == pytest.approx(0.0, abs=1e-9)
    assert np.array_equal(out, rig.classic)  # classic, bit for bit


def test_the_fade_comes_back_when_results_return(rig: Rig) -> None:
    running(rig)
    rig.backend.hold = True
    rig.frames(16)
    assert rig.effect.health == 0.0
    rig.backend.finish()
    rig.backend.hold = False
    running(rig)
    assert rig.effect.health_falls == 1


def test_a_frame_never_waits_for_a_job_in_flight(rig: Rig) -> None:
    running(rig)
    rig.backend.hold = True
    rig.frame()
    jobs = len(rig.backend.jobs)
    started = time.perf_counter()
    rig.frames(5)
    assert (time.perf_counter() - started) / 5 < 0.5  # no blocking anywhere near the 1 s of a stuck job
    assert len(rig.backend.jobs) == jobs  # at most one job in flight


def test_smoothing_follows_the_contracts_formula(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    folder = write_model_folder(
        tmp_path / "smooth",
        size=SIZE,
        output={"name": "delta", "kind": "delta", "layout": "NCHW", "colors": "BGR", "dtype": "float32", "temporal_alpha": 0.5},
    )
    values = iter([0.0, 0.1, 0.1])
    rig = Rig(tmp_path, face_landmarks_720p, folder=folder, model=lambda crop: uniform_delta(crop.shape[0], next(values)))
    rig.frame()
    rig.frame()
    rig.frame()
    assert rig.backend.latest().delta[128, 128, 0] == pytest.approx(0.05)
    rig.frame()
    assert rig.backend.latest().delta[128, 128, 0] == pytest.approx(0.075)


# ---- bad output ---------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_result_that_is_not_finite_is_discarded(tmp_path: Path, face_landmarks_720p: np.ndarray, bad: float) -> None:
    good = lambda crop: uniform_delta(crop.shape[0])  # noqa: E731
    rig = Rig(tmp_path, face_landmarks_720p, model=good)
    running(rig)
    newest = rig.backend.latest()

    def broken(crop: np.ndarray) -> np.ndarray:
        raw = uniform_delta(crop.shape[0])
        raw[0, 3, 3] = bad
        return raw

    rig.backend.model = broken
    rig.frame()
    assert rig.backend.latest() is newest
    assert rig.backend.rejected_results == 1


def test_a_result_that_changes_too_much_is_discarded(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    rig = Rig(tmp_path, face_landmarks_720p)
    running(rig)
    rig.backend.model = lambda crop: uniform_delta(crop.shape[0], 0.16)
    rig.frame()
    assert rig.backend.rejected_results == 1


def test_three_bad_results_in_a_row_lead_to_a_restart(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    rig = Rig(tmp_path, face_landmarks_720p)
    running(rig)
    rig.backend.model = lambda crop: uniform_delta(crop.shape[0], np.nan)
    rig.frames(3)
    assert rig.backends[0].killed == 1
    assert rig.effect.backend is None
    assert rig.effect.stats()["phase"] == "waiting"


# ---- failures and restarts ----------------------------------------------------------------------


def fail_worker(rig: Rig, why: str = "the worker process ended", code: int | None = None) -> None:
    rig.backend.failure = (why, code)


def test_after_a_failure_the_last_delta_fades_out_and_reaches_classic_without_a_jump(rig: Rig) -> None:
    running(rig)
    fail_worker(rig)
    changes = []
    for _ in range(11):
        changes.append(_cheek_change(rig, rig.frame()))
    assert np.all(np.diff(changes[1:]) < 0.0)
    assert np.abs(np.diff(changes)).max() <= 0.04 * 0.1 + 1e-4  # one step of the fall, no more
    assert np.array_equal(rig.frame(), rig.classic)


def test_restart_waits_5_30_and_120_seconds_then_stops(rig: Rig) -> None:
    for number, wait in enumerate(stage.RESTART_WAITS_S):
        running(rig)
        assert len(rig.backends) == number + 1
        fail_worker(rig)
        rig.frame()
        assert rig.effect.stats()["phase"] == "waiting"
        rig.now += wait - 1.0
        rig.control()
        assert rig.effect.stats()["phase"] == "waiting", "too early"
        rig.now += 1.1
        rig.control()
        assert rig.effect.stats()["phase"] == "running"
    assert rig.effect.restarts == 3
    running(rig)
    fail_worker(rig)
    rig.frame()
    assert rig.effect.stats()["phase"] == "failed"
    assert rig.effect.status_message == "エラーのため止めました。もう一度オンにすると再開します"
    rig.now += 3600.0
    rig.frames(3)
    assert len(rig.backends) == 4  # stays stopped


def test_failures_older_than_ten_minutes_are_forgotten(rig: Rig) -> None:
    for _ in range(6):
        running(rig)
        fail_worker(rig)
        rig.frame()
        rig.now += stage.RESTART_WINDOW_S + 1.0
        rig.control()
        assert rig.effect.stats()["phase"] == "running"
    assert rig.effect.restarts == 6


def _use_up_restarts(rig: Rig) -> None:
    for wait in (*stage.RESTART_WAITS_S, None):
        running(rig)
        fail_worker(rig)
        rig.frame()
        if wait is not None:
            rig.now += wait + 0.1
            rig.control()
    assert rig.effect.stats()["phase"] == "failed"


def test_switching_the_strength_off_and_on_gives_a_stopped_stage_another_chance(rig: Rig) -> None:
    _use_up_restarts(rig)
    rig.section.strength = 0.0
    rig.frames(2)
    assert rig.effect.stats()["phase"] == "failed"
    rig.section = NeuralSettings(enabled=True, strength=0.7)
    rig.frames(2)
    assert rig.effect.stats()["phase"] == "running"
    assert rig.effect.restarts == 3  # a fresh start, and the count of failures starts over


def test_switching_the_section_off_and_on_gives_a_stopped_stage_another_chance(rig: Rig) -> None:
    _use_up_restarts(rig)
    rig.now += 1.0  # apply is not called while the section is off
    rig.section = NeuralSettings(enabled=True, strength=1.0)
    rig.frames(2)
    assert rig.effect.stats()["phase"] == "running"


def test_a_lost_face_alone_does_not_restart_a_stopped_stage(rig: Rig) -> None:
    _use_up_restarts(rig)
    rig.now += 5.0
    rig.frames(2)  # the same section object: nobody touched the settings
    assert rig.effect.stats()["phase"] == "failed"


def test_coming_back_after_a_pause_rises_from_zero_instead_of_showing_an_old_delta(rig: Rig) -> None:
    running(rig)
    rig.now += stage.RESUME_GAP_S + 0.5  # the section was off, say
    rig.backend.hold = True
    assert np.array_equal(rig.frame(), rig.classic)
    assert rig.effect.health == 0.0


def test_switching_the_strength_on_rises_from_zero(rig: Rig) -> None:
    running(rig)
    rig.section.strength = 0.0
    rig.frames(2)
    rig.section.strength = 1.0
    rig.backend.hold = True
    assert np.array_equal(rig.frame(), rig.classic)


def test_a_stuck_job_kills_the_worker_after_one_second(rig: Rig) -> None:
    running(rig)
    rig.backend.hold = True
    rig.frame()
    rig.now += 0.9
    rig.control()
    assert rig.backends[0].killed == 0
    rig.now += 0.2
    rig.control()
    assert rig.backends[0].killed == 1
    assert rig.effect.stats()["phase"] == "waiting"


def test_after_a_restart_ten_good_results_are_needed_before_health_rises(rig: Rig) -> None:
    running(rig)
    fail_worker(rig)
    rig.frames(12)
    assert rig.effect.health == 0.0
    rig.now += 5.1
    rig.control()
    assert rig.effect.stats()["phase"] == "running"
    healths = []
    for _ in range(11):
        rig.frame()
        healths.append(rig.effect.health)
    assert healths[8] == 0.0  # nine results so far
    assert healths[10] > 0.0
    rig.frames(15)
    assert rig.effect.health == 1.0


def test_no_gpu_at_the_first_start_means_off_with_a_status(rig: Rig) -> None:
    def no_gpu(backend: FakeBackend) -> None:
        backend.start_error = BackendError("session is on the CPU", worker.CODE_NOT_DML)

    rig.prepare_backend = no_gpu
    assert np.array_equal(rig.frame(), rig.classic)
    assert rig.effect.stats()["phase"] == "off"
    assert rig.effect.status_message == "GPU が使えないため、従来のメイクのまま動いています"
    rig.frames(3)
    assert len(rig.backends) == 1


def test_no_gpu_after_a_device_loss_keeps_trying(rig: Rig) -> None:
    running(rig)
    fail_worker(rig, "device lost", worker.CODE_RUN_ERROR)
    rig.frame()

    def no_gpu(backend: FakeBackend) -> None:
        backend.start_error = BackendError("session is on the CPU", worker.CODE_NOT_DML)

    rig.prepare_backend = no_gpu
    rig.now += 5.1
    rig.control()
    assert rig.effect.stats()["phase"] == "waiting"
    assert rig.effect.status_message == "GPU が使えないため、従来のメイクのまま動いています"
    rig.prepare_backend = None
    rig.now += 30.1
    rig.control()
    assert rig.effect.stats()["phase"] == "running"


@pytest.mark.parametrize("code", [worker.CODE_LOAD_ERROR, worker.CODE_BAD_MODEL])
def test_a_model_that_cannot_load_is_not_retried(rig: Rig, code: int) -> None:
    def broken(backend: FakeBackend) -> None:
        backend.start_error = BackendError("no", code)

    rig.prepare_backend = broken
    rig.frame()
    assert rig.effect.stats()["phase"] == "failed"
    rig.now += 600.0
    rig.frames(2)
    assert len(rig.backends) == 1


def test_any_exception_from_starting_counts_as_a_failure(rig: Rig) -> None:
    def broken(backend: FakeBackend) -> None:
        backend.start_error = OSError("no shared memory")  # type: ignore[assignment]

    rig.prepare_backend = broken
    rig.frame()
    assert rig.effect.stats()["phase"] == "waiting"
    assert rig.backend.closed == 1


# ---- overload -----------------------------------------------------------------------------------


def test_overload_fades_pauses_jobs_and_tries_again(rig: Rig) -> None:
    running(rig)
    rig.backend.hold = True
    for _ in range(stage.OVERLOAD_MIN_FRAMES + 10):
        rig.frame(control=False)
        rig.effect._control_step(rig.now)
        if rig.backend.pending is not None and rig.now - rig.backend.pending[2] > 2 * FRAME_S:
            rig.backend.finish()  # slow results, one job per three frames: frames without a job
    assert rig.effect.overloads == 1
    assert rig.effect.status_message == "処理が間に合わないため、一時的に弱めています"
    rig.backend.hold = False
    jobs = len(rig.backend.jobs)
    out = rig.frames(10)
    assert np.array_equal(out, rig.classic)
    assert len(rig.backend.jobs) == jobs  # no jobs during the pause
    rig.now += stage.OVERLOAD_PAUSE_S
    running(rig)
    assert rig.effect.status_message == "動作中"
    assert len(rig.backend.jobs) > jobs
    assert rig.effect._overload_pause == 2 * stage.OVERLOAD_PAUSE_S


def test_results_slower_than_a_frame_count_as_overload(rig: Rig) -> None:
    running(rig)
    for _ in range(stage.OVERLOAD_MIN_FRAMES + 5):
        rig.backend.slow_results.append(rig.now)
        rig.frame()
    assert rig.effect.overloads == 1


def test_too_much_work_on_the_camera_thread_counts_as_overload(rig: Rig) -> None:
    running(rig)
    rig.effect._stage_recent.extend([stage.STAGE_OVER_BUDGET_MS + 5.0] * 150)
    for _ in range(stage.OVERLOAD_MIN_FRAMES + 5):
        rig.effect._stage_recent.append(stage.STAGE_OVER_BUDGET_MS + 5.0)
        rig.frame()
    assert rig.effect.overloads == 1
    assert rig.effect.status_message == "処理が間に合わないため、一時的に弱めています"


def test_a_few_slow_frames_are_no_overload(rig: Rig) -> None:
    running(rig)
    for number in range(stage.OVERLOAD_MIN_FRAMES + 30):
        if number % 10 == 0:
            rig.backend.slow_results.append(rig.now)
        rig.frame()
    assert rig.effect.overloads == 0
    assert rig.effect.health == 1.0


# ---- reset, gates, broken input, closing ---------------------------------------------------------


def test_reset_drops_the_delta_and_rises_from_zero(rig: Rig) -> None:
    running(rig)
    rig.backend.hold = True
    rig.frame()  # a job is in flight
    rig.effect.reset()
    assert rig.effect.health == 0.0
    assert np.array_equal(rig.frame(), rig.classic)
    rig.backend.finish()  # from before the reset: must not come back
    assert rig.backend.latest() is None
    rig.backend.hold = False
    rig.frame()
    assert rig.backend.latest() is not None
    rig.frame()
    assert rig.effect.health == pytest.approx(2.0 / 15.0)


def test_reset_starts_the_smoothing_over(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    folder = write_model_folder(
        tmp_path / "smooth",
        size=SIZE,
        output={"name": "delta", "kind": "delta", "layout": "NCHW", "colors": "BGR", "dtype": "float32", "temporal_alpha": 0.5},
    )
    values = iter([0.0, 0.1])
    rig = Rig(tmp_path, face_landmarks_720p, folder=folder, model=lambda crop: uniform_delta(crop.shape[0], next(values)))
    rig.frames(2)
    rig.effect.reset()
    rig.frame()
    assert rig.backend.latest().delta[128, 128, 0] == pytest.approx(0.1)


def test_a_small_face_fades_and_sends_no_jobs(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    rig = Rig(tmp_path, face_landmarks_720p)
    running(rig)
    center = face_landmarks_720p.mean(axis=0)
    small = center + (face_landmarks_720p - center) * (30.0 / regions.face_scale(face_landmarks_720p))
    jobs = len(rig.backend.jobs)
    out = rig.frames(11, landmarks=small)
    assert rig.effect.health == 0.0
    assert np.array_equal(out, rig.classic)
    assert len(rig.backend.jobs) == jobs


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1e9])
def test_broken_landmarks_never_paste_and_never_raise(rig: Rig, bad: float) -> None:
    running(rig)
    broken = rig.landmarks.copy()
    broken[:, :] = bad
    out = rig.frames(3, landmarks=broken)
    assert np.array_equal(out, rig.classic)


def test_a_face_outside_the_frame_is_handled(rig: Rig) -> None:
    running(rig)
    rig.frames(3, landmarks=rig.landmarks + (5000.0, 0.0))
    out = rig.frames(12, landmarks=rig.landmarks + (5000.0, 0.0))
    assert np.array_equal(out, rig.classic)


def test_the_stage_stops_when_idle_and_starts_again_when_wanted(rig: Rig) -> None:
    running(rig)
    rig.now += stage.IDLE_STOP_S + 1.0
    rig.control()
    assert rig.backends[0].killed == 1
    assert rig.effect.backend is None
    assert rig.effect.status_message is None
    rig.frames(2)
    assert len(rig.backends) == 2
    assert rig.effect.stats()["phase"] == "running"


def test_closing_returns_at_once_and_later_frames_pass_through(rig: Rig) -> None:
    running(rig)
    backend = rig.backend
    closing = threading.Event()
    real_close = backend.close

    def slow_close() -> None:
        closing.wait(5.0)
        real_close()

    backend.close = slow_close  # type: ignore[method-assign]
    started = time.perf_counter()
    rig.effect.close()
    assert time.perf_counter() - started < 1.0
    closing.set()
    assert np.array_equal(rig.frame(control=False), rig.classic)
    deadline = time.perf_counter() + 5.0
    while backend.closed == 0 and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert backend.closed == 1


def test_stats_are_there_for_the_evaluation(rig: Rig) -> None:
    running(rig)
    stats = rig.effect.stats()
    assert sum(stats["frames_by_age"].values()) > 0
    assert stats["good_results"] > 0 and len(stats["rtt_ms"]) > 0 and len(stats["stage_ms"]) > 0
    assert stats["status"] == "動作中"


def test_the_control_thread_survives_a_bug_in_its_step(tmp_path: Path, face_landmarks_720p: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    effect = NeuralFinishEffect(model_dir=write_model_folder(tmp_path / "m"), allow_unevaluated=True, backend_factory=lambda spec: None)
    calls = []

    def buggy(now: float) -> None:
        calls.append(now)
        raise ValueError("a bug")

    monkeypatch.setattr(effect, "_control_step", buggy)
    image = np.full((720, 1280, 3), 0.5, np.float32)
    effect.apply(image, FrameContext(height=720, width=1280, landmarks=face_landmarks_720p), NeuralSettings(enabled=True, strength=1.0))
    deadline = time.perf_counter() + 5.0
    while len(calls) < 2 and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert len(calls) >= 2
    effect.close()


# ---- offline: waits for the current result --------------------------------------------------------


def test_sync_mode_shows_the_full_result_of_this_very_frame(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    folder = write_model_folder(tmp_path / "m", size=SIZE)
    effect = NeuralFinishEffect(
        model_dir=folder, allow_unevaluated=True, sync=True, backend_factory=lambda spec: FakeBackend(spec, time.perf_counter)
    )
    image = np.full((720, 1280, 3), 0.5, np.float32)
    ctx = FrameContext(height=720, width=1280, landmarks=face_landmarks_720p)
    out = effect.apply(image, ctx, NeuralSettings(enabled=True, strength=1.0))
    x, y = regions.cheek_center(face_landmarks_720p, "left").astype(int)
    assert out[y, x, 0] == pytest.approx(0.54, abs=1e-4)
    assert effect.stats()["frames_by_age"] == {0: 1}
    effect.close()


def test_sync_mode_falls_back_to_classic_when_there_is_no_model(tmp_path: Path, face_landmarks_720p: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage, "_onnxruntime_available", lambda: True)
    effect = NeuralFinishEffect(model_dir=tmp_path / "nothing", allow_unevaluated=True, sync=True)
    image = np.full((720, 1280, 3), 0.5, np.float32)
    expected = image.copy()
    ctx = FrameContext(height=720, width=1280, landmarks=face_landmarks_720p)
    assert np.array_equal(effect.apply(image, ctx, NeuralSettings(enabled=True, strength=1.0)), expected)
    assert effect.status_message == "モデルがありません"
    effect.close()



# ---- what real masks hold (review finding: a floor of about 0.017 weighted the whole crop) -----------


def test_a_masks_floor_leaves_the_background_bit_exact(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    rig = Rig(tmp_path, face_landmarks_720p, model=lambda crop: uniform_delta(crop.shape[0], 0.14))
    floor = np.full((720, 1280), 0.0166, np.float32)
    running(rig)
    out = rig.frame(hair_mask=floor, face_skin_mask=floor)
    s = regions.face_scale(rig.landmarks)
    roi = regions.Roi(0, 0, 1280, 720)
    outside = regions.face_oval_mask(rig.landmarks, roi, expand_px=0.03 * s, feather_px=0.05 * s) <= 0.0
    assert outside.any()
    assert np.array_equal(out[outside], rig.classic[outside])


def test_a_masks_floor_does_not_widen_what_is_warped(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    spec = load_spec(write_model_folder(tmp_path / "model", size=SIZE))
    floor = np.full((720, 1280), 0.0166, np.float32)
    delta = np.full((SIZE, SIZE, 3), 0.04, np.float32)
    plain = _weighted_delta(face_landmarks_720p, None, None, (720, 1280, 3), spec, lambda: delta)
    with_floor = _weighted_delta(face_landmarks_720p, floor, floor, (720, 1280, 3), spec, lambda: delta)
    assert (with_floor.y0, with_floor.y1, with_floor.x0, with_floor.x1) == (plain.y0, plain.y1, plain.x0, plain.x1)


# ---- a delta that does not fit the face (review finding: a nod moved the paste off the mouth) ------------


def _nodded(landmarks: np.ndarray, amount: float) -> np.ndarray:
    """The lower face moved against the eyes, as in a nod: the crop follows the eyes and cannot follow this."""
    s = regions.face_scale(landmarks)
    eye_line = (regions.eye_center(landmarks, "left")[1] + regions.eye_center(landmarks, "right")[1]) / 2
    moved = landmarks.copy()
    below = moved[:, 1] > eye_line + 0.2 * s
    moved[below, 1] += amount * s
    return moved


def _age_one(rig: Rig, landmarks: np.ndarray) -> np.ndarray:
    """One frame, with the result of the frame before as the newest."""
    rig.backend.hold = True
    if rig.backend.pending is not None:
        rig.backend.finish()
    return rig.frame(landmarks=landmarks)


def test_a_nod_weakens_the_paste_in_steps_and_down_to_the_classic_output(rig: Rig) -> None:
    running(rig)
    assert rig.effect.motion == 1.0
    levels = [rig.effect.motion]
    out = rig.classic
    for index in range(10):
        out = _age_one(rig, _nodded(rig.landmarks, 0.2 if index % 2 else 0.0))
        levels.append(rig.effect.motion)
    assert levels[-1] == 0.0
    assert all(a - b <= stage.MOTION_FALL + 1e-6 for a, b in zip(levels, levels[1:]))
    assert np.array_equal(out, rig.classic)
    assert rig.effect.health == 1.0  # no failure: nothing to restart, nothing counted as a fall
    assert rig.effect.health_falls == 0
    assert rig.effect.stats()["motion_weak_frames"] > 0


def test_it_comes_back_slowly_when_the_face_holds_still(rig: Rig) -> None:
    running(rig)
    for index in range(10):
        _age_one(rig, _nodded(rig.landmarks, 0.2 if index % 2 else 0.0))
    assert rig.effect.motion == 0.0
    levels = []
    for _ in range(40):
        _age_one(rig, rig.landmarks)
        levels.append(rig.effect.motion)
    assert levels[-1] == 1.0
    assert all(b - a <= stage.MOTION_RISE + 1e-6 for a, b in zip(levels, levels[1:]))


def test_a_pan_costs_nothing_because_the_crop_follows_it(rig: Rig) -> None:
    running(rig)
    for index in range(20):
        _age_one(rig, rig.landmarks + np.array([6.0 * index, 3.0 * index], np.float32))
        assert rig.effect.motion == 1.0
    assert rig.effect.stats()["motion_weak_frames"] == 0


def test_an_old_delta_on_a_still_face_stays_at_full_strength(rig: Rig) -> None:
    running(rig)
    rig.backend.hold = True
    rig.frame()  # the job in flight stays in flight
    for _ in range(2):
        rig.frame()
        assert rig.effect.motion == 1.0


def test_the_same_motion_counts_for_more_the_older_the_delta(rig: Rig) -> None:
    """Age 3 on a face moving a little each frame: too far off, where age 1 is not."""
    running(rig)
    step = 0.03
    for index in range(12):
        _age_one(rig, _nodded(rig.landmarks, step * (index % 2)))
    assert rig.effect.motion == 1.0
    rig.backend.hold = True
    rig.backend.finish() if rig.backend.pending is not None else None
    for index in range(3):
        rig.frame(landmarks=_nodded(rig.landmarks, 3 * step * (index + 1)))
    assert rig.effect.motion < 1.0


def test_a_reset_forgets_the_motion(rig: Rig) -> None:
    running(rig)
    for index in range(10):
        _age_one(rig, _nodded(rig.landmarks, 0.2 if index % 2 else 0.0))
    rig.effect.reset()
    assert rig.effect.motion == 1.0
    assert rig.effect.stats()["drift"] is not None


def test_offline_use_is_not_weakened(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    rig = Rig(tmp_path, face_landmarks_720p, sync=True)
    rig.effect._ready_or_off.set()
    for index in range(6):
        rig.frame(landmarks=_nodded(rig.landmarks, 0.2 if index % 2 else 0.0))
    assert rig.effect.motion == 1.0


# ---- a level that weakens the paste must not show by itself -------------------------------------------------------------


def _most_turned_within_a_frame(levels: list[float]) -> float:
    return max((abs(a + c - 2.0 * b) for a, b, c in zip(levels, levels[1:], levels[2:])), default=0.0)


def test_the_paste_weakens_and_recovers_without_a_corner(rig: Rig) -> None:
    """A fall that turned into a rise changed the speed by 0.32 in one frame: 3 levels on a delta of 10 to 12, for one frame.
    That was every break of a 30-minute run with a large delta."""
    running(rig)
    levels = [rig.effect.motion]
    for script in ([0.0] * 3, [0.15, 0.0] * 2, [0.0] * 4, [0.15, 0.0] * 3, [0.0] * 30, [0.3, 0.0] * 3, [0.0] * 30):
        for nod in script:
            _age_one(rig, _nodded(rig.landmarks, nod))
            levels.append(rig.effect.motion)
    assert min(levels) < 0.5  # the script did weaken it
    assert _most_turned_within_a_frame(levels) <= 0.13


def test_a_turn_of_the_level_changes_the_picture_by_little_for_one_frame(rig: Rig) -> None:
    running(rig)
    pictures, nods = [], [0.0, 0.0, 0.15, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for nod in nods:
        out = _age_one(rig, _nodded(rig.landmarks, nod))
        added = out - rig.classic
        pictures.append(float(np.abs(added).max()))
    assert max(pictures) > 0.0
    turned = _most_turned_within_a_frame(pictures)
    assert turned <= 0.13 * max(pictures) + 1e-6


def test_the_ease_reaches_its_target_exactly_and_stays_within_its_limits() -> None:
    ease = stage._Ease(stage.MOTION_FALL, stage.MOTION_RISE)
    values = [ease.value]
    for target in [0.0] * 12 + [1.0] * 30 + [0.4] * 12 + [1.0] * 20:
        values.append(ease.step(target))
    assert values[12] == 0.0 and values[42] == 1.0 and values[-1] == 1.0
    assert all(b - a <= stage.MOTION_RISE + 1e-9 and a - b <= stage.MOTION_FALL + 1e-9 for a, b in zip(values, values[1:]))
    assert _most_turned_within_a_frame(values) <= stage.EASE_ACCEL + 1e-9
    still = stage._Ease(stage.MOTION_FALL, stage.MOTION_RISE)
    assert all(still.step(1.0) == 1.0 for _ in range(5))  # untouched means exactly untouched


# ---- masks that are behind a moving face --------------------------------------------------------------------------------


def _beside_the_face(rig: Rig, landmarks: np.ndarray) -> tuple[Any, np.ndarray]:
    """The crop's rectangle, and in it what only a mask may reach: the face oval does not."""
    roi = geometry.crop_roi(geometry.crop_matrix(landmarks, 256), 256, (720, 1280, 3))
    assert roi is not None
    return roi, geometry.allowed_area(landmarks, roi) == 0.0


def test_masks_behind_a_fast_pan_leave_the_wall_alone(rig: Rig) -> None:
    """The masks are 2 to 3 frames old. In a pan their edge stays on the wall, and at full trust the model may draw there."""
    running(rig)
    mask = np.ones((720, 1280), np.float32)
    step_ms = 33
    out, shifted = rig.classic, rig.landmarks
    for index in range(12):
        shifted = rig.landmarks + np.float32([index * 14.0 - 80.0, 0.0])
        ts = 1000 + index * step_ms
        out = rig.frame(landmarks=shifted, hair_mask=mask, face_skin_mask=mask, timestamp_ms=ts, masks_timestamp_ms=ts - 3 * step_ms)
    roi, outside = _beside_the_face(rig, shifted)
    change = np.abs(out[roi.slices] - rig.classic[roi.slices]).max(axis=2)
    assert rig.effect.mask_trust == 0.0
    assert not change[outside].any()
    assert change[~outside].max() > 0.0  # the face oval follows the landmarks and stays finished


def test_masks_of_a_still_face_count_in_full_however_old(rig: Rig) -> None:
    running(rig)
    mask = np.ones((720, 1280), np.float32)
    out = rig.classic
    for index in range(12):
        ts = 1000 + index * 33
        out = rig.frame(hair_mask=mask, face_skin_mask=mask, timestamp_ms=ts, masks_timestamp_ms=ts - 99)
    assert rig.effect.mask_trust == 1.0
    roi, outside = _beside_the_face(rig, rig.landmarks)
    assert np.abs(out[roi.slices] - rig.classic[roi.slices]).max(axis=2)[outside].max() > 0.0


def test_the_trust_comes_back_after_the_face_stopped_and_starts_without_a_corner() -> None:
    trust = stage.MaskTrust()
    landmarks = np.load(Path(__file__).parent / "data" / "face_landmarks_720p.npy").astype(np.float32)
    values = []
    ts = 0
    for index in range(100):
        ts += 33
        values.append(trust.step(landmarks + np.float32([(min(index, 52) - 40) * 14.0 if index >= 40 else 0.0, 0.0]), ts, ts - 66))
    assert values[39] == 1.0 and min(values) == 0.0 and values[-1] == 1.0
    assert _most_turned_within_a_frame(values) <= stage.EASE_ACCEL + 1e-9


def test_masks_older_than_anything_kept_are_not_trusted(face_landmarks_720p: np.ndarray) -> None:
    trust = stage.MaskTrust()
    for index in range(10):
        value = trust.step(face_landmarks_720p, 10000 + index * 33, 100)
    assert value == 0.0


def test_a_source_that_does_not_say_when_is_trusted(face_landmarks_720p: np.ndarray) -> None:
    trust = stage.MaskTrust()
    assert all(trust.step(face_landmarks_720p, 33 * index, None) == 1.0 for index in range(5))


def test_trust_gives_the_same_paste_ahead_and_on_the_camera_thread(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    mask = np.ones((720, 1280), np.float32)
    first, second = _two_rigs(tmp_path, face_landmarks_720p)
    outs = []
    for rig_now, ahead in ((first, True), (second, False)):
        running(rig_now)
        out = rig_now.classic
        for index in range(6):
            ts = 1000 + index * 33
            shifted = rig_now.landmarks + np.float32([index * 5.0, 0.0])
            out = rig_now.frame(landmarks=shifted, hair_mask=mask, timestamp_ms=ts, masks_timestamp_ms=ts - 66, ahead=ahead)
        outs.append(out)
    assert 0.0 < first.effect.mask_trust < 1.0
    np.testing.assert_array_equal(outs[0], outs[1])


# ---- the frame-rate budget ------------------------------------------------------------------------------------------------


def _busy_for_the_overload_check(rig: Rig) -> None:
    running(rig)
    rig.frames(stage.OVERLOAD_MIN_FRAMES + 5)


def test_frames_the_app_never_got_to_make_the_stage_send_fewer_jobs(rig: Rig) -> None:
    _busy_for_the_overload_check(rig)
    before = len(rig.backend.jobs)
    rig.frames(20)
    assert len(rig.backend.jobs) - before == 20  # nothing lost: a job on every frame
    rig.frames(stage.BUDGET_LOST_FRAMES, skipped_frames=1)
    before = len(rig.backend.jobs)
    rig.frames(20)
    assert len(rig.backend.jobs) - before == 10
    assert rig.effect.overloads == 0 and rig.effect.health == 1.0
    assert rig.effect.stats()["thrifty_frames"] == 10
    assert rig.effect.stats()["lost_frames"] == stage.BUDGET_LOST_FRAMES


def test_a_lost_frame_or_two_change_nothing(rig: Rig) -> None:
    _busy_for_the_overload_check(rig)
    rig.frames(stage.BUDGET_LOST_FRAMES - 1, skipped_frames=1)
    before = len(rig.backend.jobs)
    rig.frames(10)
    assert len(rig.backend.jobs) - before == 10 and rig.effect.overloads == 0


def test_frames_still_lost_after_that_fade_the_stage_to_the_classic_output(rig: Rig) -> None:
    _busy_for_the_overload_check(rig)
    rig.frames(stage.BUDGET_LOST_FRAMES, skipped_frames=1)
    rig.frames(5)
    rig.frames(stage.BUDGET_PAUSE_FRAMES, skipped_frames=1)
    assert rig.effect.overloads == 1
    assert rig.effect.status_message == stage.STATUS_OVERLOAD
    before = len(rig.backend.jobs)
    out = rig.frames(20)
    assert np.array_equal(out, rig.classic) and rig.effect.health == 0.0
    assert len(rig.backend.jobs) == before
    assert rig.effect.health_falls >= 1
    rig.now += stage.OVERLOAD_PAUSE_S
    rig.frames(30)
    assert rig.effect.health == 1.0  # it tries again, still thrifty
    assert rig.effect._thrifty


def test_a_burst_too_short_to_break_the_budget_costs_no_pause(rig: Rig) -> None:
    _busy_for_the_overload_check(rig)
    rig.frames(stage.BUDGET_LOST_FRAMES, skipped_frames=1)
    rig.frames(5)
    rig.frames(stage.BUDGET_PAUSE_FRAMES - 1, skipped_frames=1)
    assert rig.effect.overloads == 0 and rig.effect._thrifty
    rig.now += stage.BUDGET_PAUSE_WINDOW_S
    rig.frames(stage.BUDGET_LOST_FRAMES, skipped_frames=1)  # those are too old by now
    assert rig.effect.overloads == 0


def test_a_job_on_every_frame_again_after_a_long_time_without_losses(rig: Rig) -> None:
    _busy_for_the_overload_check(rig)
    rig.frames(stage.BUDGET_LOST_FRAMES, skipped_frames=1)
    assert rig.effect._thrifty
    rig.now += stage.THRIFTY_HOLD_S + 1.0
    rig.frames(stage.OVERLOAD_MIN_FRAMES + 5)
    assert not rig.effect._thrifty
    before = len(rig.backend.jobs)
    rig.frames(10)
    assert len(rig.backend.jobs) - before == 10


def test_frames_lost_for_a_reason_known_to_be_elsewhere_do_not_count(tmp_path: Path, face_landmarks_720p: np.ndarray) -> None:
    rig_now = Rig(tmp_path, face_landmarks_720p)
    rig_now.effect._keep_frame_rate = False
    _busy_for_the_overload_check(rig_now)
    rig_now.frames(3 * stage.BUDGET_LOST_FRAMES, skipped_frames=1)
    assert not rig_now.effect._thrifty and rig_now.effect.overloads == 0


def test_frames_lost_before_the_stage_shows_anything_do_not_count(rig: Rig) -> None:
    rig.frames(stage.BUDGET_LOST_FRAMES, skipped_frames=1, control=False)  # nothing is running yet
    assert rig.effect.stats()["lost_frames"] == 0
