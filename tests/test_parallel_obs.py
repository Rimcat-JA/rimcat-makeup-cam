"""What the sender and the rescue write, in OBS's own queue (contract sections 2.3, 6 and 8). Skipped without OBS Virtual Camera.

Must be run after any update of OBS or pyvirtualcam: the rescue relies on a layout that nothing public promises.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from makeup_cam.presets import load_preset

if sys.platform != "win32":
    pytest.skip("OBS's queue is read here on Windows only", allow_module_level=True)

from parallel_helpers import FRAME_MS, StableMasks, StableTracker, make_frames, wait_until  # noqa: E402

from makeup_cam.parallel import engine as engine_module  # noqa: E402
from makeup_cam.parallel.clock import declared_interval  # noqa: E402
from makeup_cam.parallel.engine import FALLBACK_MESSAGE, ParallelEngine  # noqa: E402
from makeup_cam.parallel.nv12 import NV12Buffer  # noqa: E402
from makeup_cam.parallel.obs_queue import ObsQueue, read_header  # noqa: E402
from makeup_cam.camera import fit_frame  # noqa: E402
from makeup_cam.pipeline import MakeupPipeline  # noqa: E402
from makeup_cam.runner import CameraRunner  # noqa: E402

W, H = 1280, 720
PAD = 32


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path / "home"))


@pytest.fixture
def pyvirtualcam() -> Any:
    module = pytest.importorskip("pyvirtualcam")
    try:
        module.Camera(W, H, 30, print_fps=False).close()
    except Exception as exc:
        pytest.skip(f"needs OBS Virtual Camera: {exc}")
    time.sleep(0.05)
    return module


def _frames() -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 256, (H, W, 3), np.uint8) for _ in range(3)]
    frames += [np.zeros((H, W, 3), np.uint8), np.full((H, W, 3), 255, np.uint8), np.full((H, W, 3), (10, 200, 50), np.uint8)]
    return frames


def _shown(queue: ObsQueue) -> tuple[bytes, dict[str, Any]]:
    header = read_header(queue._pointer)
    offset = queue.offsets[header["read_idx"] % 3]
    return ctypes.string_at(queue._pointer + offset + PAD, W * H * 3 // 2), header


def _through_bgr(pyvirtualcam: Any, frames: list[np.ndarray]) -> list[tuple[bytes, dict[str, Any]]]:
    with pyvirtualcam.Camera(W, H, 30, fmt=pyvirtualcam.PixelFormat.BGR, print_fps=False) as camera:
        queue = _open_for_reading(camera)
        shown = []
        for frame in frames:
            camera.send(frame)
            shown.append(_shown(queue))
        queue.close()
    time.sleep(0.05)
    return shown


def _open_for_reading(camera: Any) -> ObsQueue:
    camera.send(np.zeros((H, W, 3), np.uint8) if camera.fmt.name == "BGR" else np.zeros(W * H * 3 // 2, np.uint8))
    return ObsQueue(W, H, 30)


def test_nv12_same_as_pyvirtualcam(pyvirtualcam: Any) -> None:
    frames = _frames()
    theirs = _through_bgr(pyvirtualcam, frames)
    with pyvirtualcam.Camera(W, H, 30, fmt=pyvirtualcam.PixelFormat.NV12, print_fps=False) as camera:
        queue = _open_for_reading(camera)
        buffer = NV12Buffer(W, H)
        for frame, (bytes_of_bgr, _) in zip(frames, theirs, strict=True):
            buffer.fill(frame)
            camera.send(buffer.data)
            assert _shown(queue)[0] == bytes_of_bgr
        queue.close()


def test_obs_queue_layout(pyvirtualcam: Any) -> None:
    """After ``Queue.commit`` the bytes and the header are those after ``pyvirtualcam.Camera.send`` of the same frame."""
    frames = _frames()
    theirs = _through_bgr(pyvirtualcam, frames)
    with pyvirtualcam.Camera(W, H, 30, fmt=pyvirtualcam.PixelFormat.BGR, print_fps=False) as camera:
        queue = _open_for_reading(camera)
        for frame, (bytes_of_bgr, header_of_bgr) in zip(frames, theirs, strict=True):
            queue.send(frame)
            bytes_of_ours, header_of_ours = _shown(queue)
            assert bytes_of_ours == bytes_of_bgr
            for field in ("state", "offsets", "type", "cx", "cy", "interval"):
                assert header_of_ours[field] == header_of_bgr[field], field
            assert header_of_ours["write_idx"] == header_of_ours["read_idx"]
        queue.close()


def test_the_queue_declares_the_period_of_the_grid(pyvirtualcam: Any) -> None:
    with pyvirtualcam.Camera(W, H, 30, fmt=pyvirtualcam.PixelFormat.NV12, print_fps=False) as camera:
        queue = _open_for_reading(camera)
        assert read_header(queue._pointer)["interval"] == declared_interval(30) == 333333
        queue.close()


def test_a_write_with_nv12_is_short(pyvirtualcam: Any) -> None:
    with pyvirtualcam.Camera(W, H, 30, fmt=pyvirtualcam.PixelFormat.NV12, print_fps=False) as camera:
        buffer = NV12Buffer(W, H)
        times = []
        for _ in range(60):
            started = time.perf_counter()
            camera.send(buffer.data)
            times.append(time.perf_counter() - started)
    assert float(np.median(times)) < 0.001


@pytest.mark.parametrize(
    ("width", "height", "mirror"),
    [(W, H, False), (W, H, True), (640, 360, True)],
    ids=["as-it-is", "mirrored", "mirrored-and-smaller"],
)
def test_the_queue_holds_the_output_of_the_stages(pyvirtualcam: Any, face_frame_720p: np.ndarray, width: int, height: int, mirror: bool) -> None:
    """Not ``ProcessResult.output`` but what the sender writes: the mirror and the fit are the sender's own."""
    settings = load_preset("natural")
    settings.general.mirror_output = mirror
    pipeline = MakeupPipeline(settings, tracker=StableTracker(), segmenter=StableMasks())
    engine = ParallelEngine()
    pipeline.attach_engine(engine)
    engine.slot_wait_s = 30.0
    done: list[Any] = []
    engine.start((H, W), done.append)
    try:
        assert engine.wait_ready(60.0)
        assert wait_until(lambda: engine._sender_ready.is_set(), 10.0)
        camera = engine.virtual_camera(width=width, height=height, fps=30)
        frames = make_frames(face_frame_720p, 8)
        count = 75  # past the settling, onto the grid
        for i in range(count):
            assert pipeline.process(frames[i % 8].copy(), i * FRAME_MS).in_flight
            time.sleep(1 / 30)
        assert wait_until(lambda: len(done) == count, 10.0)
        time.sleep(0.3)
        queue = ObsQueue(width, height, 30)
        header = read_header(queue._pointer)
        shown = ctypes.string_at(queue._pointer + queue.offsets[header["read_idx"] % 3] + PAD, width * height * 3 // 2)
        queue.close()
        expected = NV12Buffer(width, height)
        expected.fill(fit_frame(done[-1].output, width, height))
        assert shown == bytes(expected.data)
        camera.close()
    finally:
        pipeline.close()


class _Reader(threading.Thread):
    """Reads the camera as a call app does, until told to stop."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.frames = 0
        self.opened = threading.Event()
        self.halt = threading.Event()

    def run(self) -> None:
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        while not self.halt.is_set():
            ok, _frame = capture.read()
            if ok:
                self.frames += 1
                self.opened.set()
            else:
                time.sleep(0.002)
        capture.release()


class _Camera:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def open(self) -> None:
        pass

    def read(self) -> np.ndarray:
        time.sleep(1 / 30)
        return self.frame.copy()

    def close(self) -> None:
        pass


def test_after_a_killed_sender_a_new_sender_goes_on_on_the_queue_a_reader_holds(pyvirtualcam: Any, face_frame_720p: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "RECOVER_AFTER_S", (1.0, 20.0, 60.0))
    pipeline = MakeupPipeline(load_preset("natural"), tracker=StableTracker(), segmenter=StableMasks())
    engine = ParallelEngine()
    pipeline.attach_engine(engine)
    runner = CameraRunner(pipeline, camera_factory=lambda **_k: _Camera(face_frame_720p), output_size=(W, H))
    reader = _Reader()
    runner.start()
    try:
        assert engine.wait_ready(60.0)
        assert wait_until(lambda: runner.status.virtual_camera is not None, 10.0), runner.status
        reader.start()
        assert reader.opened.wait(15.0), "this test reads camera 0, which must be OBS Virtual Camera here"
        time.sleep(0.5)
        old = engine.child_pids["sender"]
        engine._sender.process.kill()
        assert wait_until(lambda: engine._state == engine_module._RUNNING and engine.child_pids["sender"] != old, 15.0), engine._state
        assert engine._local is None, "the sender has the queue now, and this process does not"
        assert wait_until(lambda: runner.status.processing_error is None, 5.0)
        assert runner.status.virtual_camera is not None
        count = reader.frames
        time.sleep(3.0)
        assert reader.frames - count >= 80, "the reader gets frames from the new sender"
    finally:
        reader.halt.set()
        reader.join(5.0)
        runner.stop(3.0)
        pipeline.close()


def test_a_killed_sender_with_a_reader_goes_on_through_the_rescue(pyvirtualcam: Any, face_frame_720p: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "RECOVER_AFTER_S", (60.0, 60.0, 60.0))  # the next test is about what comes after this

    class Camera:
        def open(self) -> None:
            pass

        def read(self) -> np.ndarray:
            time.sleep(1 / 30)
            return face_frame_720p.copy()

        def close(self) -> None:
            pass

    pipeline = MakeupPipeline(load_preset("natural"), tracker=StableTracker(), segmenter=StableMasks())
    engine = ParallelEngine()
    pipeline.attach_engine(engine)
    runner = CameraRunner(pipeline, camera_factory=lambda **_k: Camera(), output_size=(W, H))
    reader = _Reader()
    runner.start()
    try:
        assert engine.wait_ready(60.0)
        assert wait_until(lambda: runner.status.virtual_camera is not None, 10.0), runner.status
        reader.start()
        assert reader.opened.wait(15.0), "this test reads camera 0, which must be OBS Virtual Camera here"
        time.sleep(0.5)
        engine._sender.process.kill()
        assert wait_until(lambda: runner.status.processing_error == FALLBACK_MESSAGE, 3.0)
        assert wait_until(lambda: engine._local is not None, 3.0)
        count = reader.frames
        assert wait_until(lambda: reader.frames > count + 20, 5.0), "the reader keeps getting frames"
        assert isinstance(engine._local, ObsQueue), "with a reader on the old queue, no process can open the camera"
    finally:
        reader.halt.set()
        reader.join(5.0)
        runner.stop(3.0)
        pipeline.close()
