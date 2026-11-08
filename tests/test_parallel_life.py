"""Start, stop and failure of the pipeline of processes (contract sections 7, 8, 9 and 12.4). Real processes throughout.

Failures come from outside, with no hook in the app: a kill, a suspended process, a suspended process in short turns.
"""

from __future__ import annotations

import ast
import os
import random
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from makeup_cam import parallel
from makeup_cam.camera import VirtualCameraError
from makeup_cam.config import MakeupSettings
from makeup_cam.presets import load_preset
from makeup_cam.runner import CameraRunner, RunnerStatus

if sys.platform != "win32":
    pytest.skip("the pipeline of processes runs on Windows only", allow_module_level=True)

from parallel_helpers import (  # noqa: E402
    FRAME_MS,
    StableMasks,
    StableTracker,
    descendants,
    make_frames,
    number_of,
    process_alive,
    resume,
    suspend,
    wait_until,
)

from makeup_cam.parallel import engine as engine_module  # noqa: E402
from makeup_cam.parallel import shm  # noqa: E402
from makeup_cam.parallel.engine import (  # noqa: E402
    FALLBACK_MESSAGE,
    HANG_LIMIT_S,
    SLOW_MESSAGE,
    VCAM_REOPEN_MESSAGE,
    ParallelEngine,
)
from makeup_cam.parallel.obs_queue import ObsQueue, QueueRefused  # noqa: E402
from makeup_cam.parallel.sender_main import open_device  # noqa: E402
from makeup_cam.pipeline import MakeupPipeline, ProcessResult  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSES = ["stage0", "stage1", "stage2", "sender"]


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path / "home"))


def obs_camera_opens() -> bool:
    try:
        import pyvirtualcam

        pyvirtualcam.Camera(64, 48, 30, print_fps=False).close()
        return True
    except Exception:
        return False


class Rig:
    """A pipeline with an engine, fed by hand at the camera's rate. The virtual camera stays off."""

    def __init__(self, face: np.ndarray, frames: int, settings: MakeupSettings | None = None) -> None:
        self.frames = make_frames(face, 8)
        self.count = frames
        self.pipeline = MakeupPipeline(settings or load_preset("natural"), tracker=StableTracker(), segmenter=StableMasks())
        self.engine = ParallelEngine()
        self.pipeline.attach_engine(self.engine)
        self.seen: list[int] = []
        self.next = 0
        self.lock = threading.Lock()
        self.pids: dict[str, int] = {}

    def start(self) -> None:
        self.engine.start((720, 1280), self._collect)
        assert self.engine.wait_ready(60.0)
        self.pids = dict(self.engine.child_pids)
        assert set(self.pids) == set(PROCESSES)

    def _collect(self, result: ProcessResult) -> None:
        with self.lock:
            self.seen.append(number_of(result))

    def frame(self, number: int) -> np.ndarray:
        frame = self.frames[number % len(self.frames)].copy()
        frame[0, :, 0] = number % 256
        frame[0, :, 1] = (number >> 8) % 256
        return frame

    def feed(self, how_many: int, pace_s: float = 1.0 / 30.0) -> None:
        for _ in range(how_many):
            started = time.perf_counter()
            number = self.next
            self.next += 1
            result = self.pipeline.process(self.frame(number), number * FRAME_MS)
            if not result.in_flight:  # drawn here: the runner would call on_frame now
                self._collect(result)
            rest = pace_s - (time.perf_counter() - started)
            if rest > 0:
                time.sleep(rest)

    def close(self) -> None:
        self.pipeline.close()


@pytest.fixture
def rig(face_frame_720p: np.ndarray) -> Iterator[Rig]:
    made = Rig(face_frame_720p, 0)
    yield made
    for pid in made.pids.values():
        if process_alive(pid):
            resume(pid)
    made.close()


def assert_nothing_left(names: tuple[list[str], list[str]], pids: dict[str, int]) -> None:
    blocks, semaphores = names
    assert wait_until(lambda: not any(process_alive(pid) for pid in pids.values()), 2.0), pids
    for name in blocks:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)
    for name in semaphores:
        with pytest.raises(OSError):
            shm.Semaphore(name, create=False)


def _report(engine: ParallelEngine) -> Any:
    report = engine.report()
    assert report is not None
    return report


# ---- failure ----------------------------------------------------------------------------


@pytest.mark.parametrize("victim", PROCESSES)
def test_a_killed_process_costs_no_frame_in_flight_and_the_line_says_so(rig: Rig, victim: str) -> None:
    rig.start()
    rig.feed(random.Random(victim).randrange(40, 70))
    before = rig.next
    rig.engine._line.stages[int(victim[-1])].process.kill() if victim != "sender" else rig.engine._sender.process.kill()
    killed_at = time.perf_counter()
    assert wait_until(lambda: _report(rig.engine).processing_error == FALLBACK_MESSAGE, 2.0)
    said_after = time.perf_counter() - killed_at
    assert said_after < engine_module.WATCH_S + 0.25
    rig.feed(20)
    assert rig.next == before + 20
    assert wait_until(lambda: len(rig.seen) >= rig.next, 5.0), rig.seen[-10:]
    assert rig.seen == list(range(rig.next)), "in order, and every frame"
    assert _report(rig.engine).processing_error == FALLBACK_MESSAGE
    assert not any(process_alive(rig.pids[name]) for name in PROCESSES if name != "sender")
    assert rig.engine.stages_drawing is False


@pytest.mark.parametrize("victim", ["stage0", "stage1", "stage2", "sender"])
def test_a_hung_process_is_given_up_after_the_limit(rig: Rig, victim: str) -> None:
    rig.start()
    rig.feed(40)
    suspend(rig.pids[victim])
    hung_at = time.perf_counter()
    while _report(rig.engine).processing_error != FALLBACK_MESSAGE and time.perf_counter() - hung_at < 5.0:
        rig.feed(1)
    took = time.perf_counter() - hung_at
    assert _report(rig.engine).processing_error == FALLBACK_MESSAGE
    assert HANG_LIMIT_S * 0.9 < took < HANG_LIMIT_S + 1.0, took
    rig.feed(10)
    assert wait_until(lambda: len(rig.seen) >= 40, 3.0)
    assert rig.seen == sorted(set(rig.seen)), "in order, nothing twice"
    assert rig.seen[-1] == rig.next - 1, "later frames go on"
    if victim != "sender":
        assert not process_alive(rig.pids[victim])
    else:
        assert not process_alive(rig.pids["sender"])


def test_a_stage_slower_than_the_budget_is_no_failure(rig: Rig) -> None:
    rig.start()
    rig.feed(40)
    pid = rig.pids["stage1"]
    stop = threading.Event()

    def slow_it() -> None:
        while not stop.is_set():
            suspend(pid)
            time.sleep(0.16)  # slower than a frame can wait for a free slot, and far from a hang
            resume(pid)
            time.sleep(0.012)

    slower = threading.Thread(target=slow_it, daemon=True)
    slower.start()
    said = False
    try:
        end = time.perf_counter() + 5.0
        while time.perf_counter() < end:
            rig.feed(1)
            said = said or _report(rig.engine).processing_error == SLOW_MESSAGE
    finally:
        stop.set()
        slower.join(1.0)
        resume(pid)
    report = _report(rig.engine)
    assert report.processing_error != FALLBACK_MESSAGE
    assert rig.engine.stages_drawing
    assert said, "the status line says that frames were lost"
    assert report.lost_frames > 0
    assert process_alive(pid)
    time.sleep(0.3)
    number = rig.next
    rig.feed(30)
    assert wait_until(lambda: rig.seen and rig.seen[-1] >= number + 20, 3.0)
    assert rig.seen == sorted(set(rig.seen))
    assert wait_until(lambda: _report(rig.engine).processing_error is None, engine_module.SLOW_MESSAGE_S + 2.0)


def test_all_processes_standing_still_while_the_collector_runs_is_a_hang(rig: Rig) -> None:
    rig.start()
    rig.feed(40)
    pids = list(rig.pids.values())
    for pid in pids:
        suspend(pid)
    time.sleep(1.5)  # meanwhile the collector would see the hang: it is the one that must not stand still here
    was_fallback = _report(rig.engine).processing_error == FALLBACK_MESSAGE
    for pid in pids:
        if process_alive(pid):
            resume(pid)
    assert was_fallback, "with the collector running this is a real hang, and the test below is the difference"


def test_after_the_collector_itself_stood_still_there_is_time_to_recover(rig: Rig) -> None:
    rig.start()
    rig.feed(40)
    engine = rig.engine
    now = time.perf_counter()
    engine._last_watch = now - 2.0  # what it finds when it runs again
    engine._last_out = now - 2.0
    engine._session.flags[shm.SESSION_HEART] = shm.now() - 2.0
    engine._watch()
    assert _report(engine).processing_error != FALLBACK_MESSAGE
    assert engine.stages_drawing
    rig.feed(30)
    assert _report(engine).processing_error != FALLBACK_MESSAGE


def test_after_a_failure_the_next_start_tries_the_stages_again(rig: Rig) -> None:
    rig.start()
    rig.feed(10)
    rig.engine._line.stages[1].process.kill()
    assert wait_until(lambda: _report(rig.engine).processing_error == FALLBACK_MESSAGE, 2.0)
    rig.engine.stop()
    assert rig.engine.report() is None
    rig.start()
    rig.feed(5)
    assert _report(rig.engine).processing_error is None
    assert rig.engine.stages_drawing


def test_a_process_that_cannot_start_is_a_fallback_at_once(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shm, "child_command", lambda module, args: (["no-such-program-at-all"], dict(os.environ)))
    rig.engine.start((720, 1280), rig._collect)
    assert wait_until(lambda: _report(rig.engine).processing_error == FALLBACK_MESSAGE, 5.0)
    rig.feed(3)
    assert rig.seen == [0, 1, 2]


def test_a_stage_that_never_reports_ready_is_a_fallback(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    original = shm.child_command

    def silent_stage(module: str, args: Any) -> tuple[list[str], dict[str, str]]:
        command, env = original(module, args)
        if module.endswith("stage_main") and args["stage"] == 1:
            command = [command[0], "-c", "import time; time.sleep(120)"]
        return command, env

    monkeypatch.setattr(shm, "child_command", silent_stage)
    monkeypatch.setattr(engine_module, "READY_TIMEOUT_S", 1.5)
    rig.engine.start((720, 1280), rig._collect)
    rig.feed(2)  # in process, while the stages are starting
    assert wait_until(lambda: _report(rig.engine).processing_error == FALLBACK_MESSAGE, 8.0)
    rig.feed(2)
    assert rig.seen == [0, 1, 2, 3]
    rig.engine.stop()
    assert not descendants(os.getpid()) & _pids_now(rig)


def _pids_now(rig: Rig) -> set[int]:
    return set(rig.engine.child_pids.values())


def test_frames_are_drawn_in_process_until_the_stages_are_ready(rig: Rig) -> None:
    rig.engine.start((720, 1280), rig._collect)
    first = rig.pipeline.process(rig.frame(0), 0)
    assert not first.in_flight
    assert rig.engine.wait_ready(60.0)
    second = rig.pipeline.process(rig.frame(1), FRAME_MS)
    assert second.in_flight


def _priority_class(pid: int) -> int:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
    kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    assert handle
    value = kernel32.GetPriorityClass(handle)
    kernel32.CloseHandle(handle)
    return value


def test_all_processes_run_above_normal_and_the_main_process_returns(rig: Rig) -> None:
    above_normal, normal = 0x8000, 0x20
    assert _priority_class(os.getpid()) == normal
    rig.start()
    assert _priority_class(os.getpid()) == above_normal
    for name, pid in rig.pids.items():
        assert _priority_class(pid) == above_normal, name
    rig.engine.stop()
    assert _priority_class(os.getpid()) == normal


def test_the_output_thread_does_not_wait_for_the_spawn(rig: Rig) -> None:
    started = time.perf_counter()
    rig.engine.start((720, 1280), rig._collect)
    assert time.perf_counter() - started < 0.1
    assert rig.engine.wait_ready(60.0)


# ---- the virtual camera -----------------------------------------------------------------


def test_nv12_refused_means_bgr() -> None:
    class Backend:
        class PixelFormat:
            NV12 = "nv12"
            BGR = "bgr"

        made: list[Any] = []

        @classmethod
        def Camera(cls, _w: int, _h: int, _fps: float, *, fmt: str, print_fps: bool) -> str:  # noqa: N802
            cls.made.append(fmt)
            if fmt == "nv12":
                raise RuntimeError("not this format")
            return "camera"

    assert open_device(Backend, 1280, 720, 30) == ("camera", False)
    assert Backend.made == ["nv12", "bgr"]


def test_no_rescue_for_a_queue_with_a_wrong_header_and_the_other_string(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    tries: list[float] = []

    def refused(_w: int, _h: int, _fps: float) -> Any:
        tries.append(time.monotonic())
        raise VirtualCameraError("no camera")

    def wrong_queue(_w: int, _h: int, _fps: float) -> Any:
        raise QueueRefused("a header this code does not know")

    monkeypatch.setattr(engine_module, "_start_virtual_camera", refused)
    monkeypatch.setattr(engine_module, "ObsQueue", wrong_queue)
    monkeypatch.setattr(engine_module, "LOCAL_CAMERA_RETRY_S", 0.2)
    rig.start()
    rig.engine._camera_request = (1280, 720, 30)
    rig.feed(5)
    rig.engine._sender.process.kill()
    assert wait_until(lambda: _report(rig.engine).processing_error == FALLBACK_MESSAGE, 2.0)
    assert wait_until(lambda: _report(rig.engine).virtual_camera_error == VCAM_REOPEN_MESSAGE, 2.0)
    assert len(tries) >= engine_module.LOCAL_CAMERA_TRIES
    count = len(tries)
    end = time.perf_counter() + 0.7
    while time.perf_counter() < end:
        rig.engine._send_at_once(np.zeros((720, 1280, 3), np.uint8))
        time.sleep(0.01)
    assert len(tries) > count, "tried again"
    assert _report(rig.engine).virtual_camera_error == VCAM_REOPEN_MESSAGE
    rig.engine._camera_request = None


def test_obs_queue_refuses_a_wrong_header() -> None:
    import ctypes
    import struct

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateFileMappingW.restype = ctypes.c_void_p
    k.CreateFileMappingW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_wchar_p]
    k.MapViewOfFile.restype = ctypes.c_void_p
    k.MapViewOfFile.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t]
    k.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    name = f"mkcam-test-queue-{os.getpid()}"
    w, h = 64, 48
    size = w * h * 3 // 2 + 32
    mapping = k.CreateFileMappingW(ctypes.c_void_p(-1), None, 4, 0, 96 + 3 * size, name)
    pointer = k.MapViewOfFile(mapping, 0xF001F, 0, 0, 0)

    def header(kind: int = 0, cx: int = w, cy: int = h, interval: int = 333333, state: int = 2, spacing: int = size) -> None:
        raw = struct.pack("<3I3I3I", 0, 0, state, 96, 96 + spacing, 96 + 2 * spacing, kind, cx, cy) + b"\0" * 4 + struct.pack("<Q", interval)
        ctypes.memmove(pointer, raw, len(raw))

    try:
        header()
        queue = ObsQueue(w, h, 30, name=name)
        queue.send(np.full((h, w, 3), 128, np.uint8))
        queue.close()
        for wrong in (dict(kind=1), dict(cx=63), dict(cy=47), dict(interval=166666), dict(state=1), dict(spacing=size - 1)):
            header(**wrong)
            with pytest.raises(QueueRefused):
                ObsQueue(w, h, 30, name=name)
        with pytest.raises(OSError):
            ObsQueue(w, h, 30, name=name + "-none")
    finally:
        k.UnmapViewOfFile(pointer)
        k.CloseHandle(mapping)


# ---- the runner -------------------------------------------------------------------------


class Camera:
    """A camera on the runner's capture thread, at 30 fps. Its size can change."""

    def __init__(self, face: np.ndarray) -> None:
        self.face = face
        self.size = (720, 1280)
        self.number = 0
        self.closed = threading.Event()
        self.next_at = time.perf_counter()

    def open(self) -> None:
        self.closed.clear()

    def read(self) -> np.ndarray | None:
        rest = self.next_at - time.perf_counter()
        if rest > 0:
            time.sleep(rest)
        self.next_at = max(self.next_at + 1.0 / 30.0, time.perf_counter() - 0.1)
        frame = self.face if self.size == self.face.shape[:2] else cv2.resize(self.face, self.size[::-1])
        self.number += 1
        frame = frame.copy()
        frame[0, :, 0] = self.number % 256
        frame[0, :, 1] = (self.number >> 8) % 256
        return frame

    def close(self) -> None:
        self.closed.set()


class RunnerRig:
    def __init__(self, face: np.ndarray, *, virtual_camera: bool, **options: Any) -> None:
        self.camera = Camera(face)
        self.pipeline = MakeupPipeline(load_preset("natural"), tracker=StableTracker(), segmenter=_AnySizeMasks())
        self.engine = ParallelEngine()
        self.pipeline.attach_engine(self.engine)
        self.results: list[ProcessResult] = []
        self.statuses: list[RunnerStatus] = []
        self.runner = CameraRunner(
            self.pipeline,
            camera_factory=lambda **_kwargs: self.camera,
            virtual_camera=virtual_camera,
            on_frame=self.results.append,
            on_status=self.statuses.append,
            output_size=(1280, 720),
            **options,
        )

    def close(self) -> None:
        self.runner.stop(3.0)
        self.pipeline.close()


class _AnySizeMasks:
    """``StableMasks`` for whatever size the frame has."""

    def __init__(self) -> None:
        self._by_size: dict[tuple[int, int], StableMasks] = {}

    def update(self, frame: np.ndarray, ts_ms: int, frame_index: int) -> Any:
        return self._by_size.setdefault(frame.shape[:2], StableMasks(frame.shape[:2])).update(frame, ts_ms, frame_index)

    def set_interval(self, interval: int) -> None:
        pass

    def close(self) -> None:
        pass


def test_the_runner_runs_on_the_stages_and_the_preview_comes_back(face_frame_720p: np.ndarray) -> None:
    rig = RunnerRig(face_frame_720p, virtual_camera=False)
    rig.runner.start()
    try:
        assert rig.engine.wait_ready(60.0)
        assert wait_until(lambda: len(rig.results) > 60, 20.0)
        result = rig.results[-1]
        assert result.preview.shape == (720, 1280, 3)
        assert not result.in_flight
        assert wait_until(lambda: rig.runner.status.fps > 25.0, 5.0), rig.runner.status
        status = rig.runner.status
        assert 0.0 < status.processing_ms < 500.0
        assert status.processing_error is None
    finally:
        rig.close()


def test_a_new_frame_size_makes_a_new_block_and_new_stages(face_frame_720p: np.ndarray) -> None:
    rig = RunnerRig(face_frame_720p, virtual_camera=False)
    rig.runner.start()
    try:
        assert rig.engine.wait_ready(60.0)
        assert wait_until(lambda: len(rig.results) > 15, 20.0)
        old_pids = dict(rig.engine.child_pids)
        old_names = rig.engine.names_in_use
        rig.camera.size = (480, 640)
        assert wait_until(lambda: rig.results[-1].preview.shape == (480, 640, 3), 5.0)
        assert wait_until(lambda: rig.engine._line is not None and rig.engine._line.shape == (480, 640) and rig.engine._state == "running", 30.0)
        count = len(rig.results)
        assert wait_until(lambda: len(rig.results) > count + 30, 10.0)
        assert rig.results[-1].preview.shape == (480, 640, 3)
        for name in ("stage0", "stage1", "stage2"):
            assert not process_alive(old_pids[name])
        assert process_alive(old_pids["sender"]), "one sender for the whole session"
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=old_names[0][1])
        assert rig.runner.status.processing_error is None
    finally:
        rig.close()


def test_stop_and_start_ten_times_leaves_nothing(face_frame_720p: np.ndarray) -> None:
    rig = RunnerRig(face_frame_720p, virtual_camera=False)
    try:
        for _ in range(10):
            rig.runner.start()
            assert rig.engine.wait_ready(60.0)
            count = len(rig.results)
            assert wait_until(lambda: len(rig.results) > count + 3, 20.0)
            pids = dict(rig.engine.child_pids)
            names = rig.engine.names_in_use
            started = time.perf_counter()
            rig.runner.stop(2.0)
            assert time.perf_counter() - started < 2.0
            assert_nothing_left(names, pids)
    finally:
        rig.close()


def test_stopping_while_the_stages_start(face_frame_720p: np.ndarray) -> None:
    rig = RunnerRig(face_frame_720p, virtual_camera=False)
    try:
        before = descendants(os.getpid())
        for pause in (0.0, 0.05, 0.2):
            rig.runner.start()
            time.sleep(pause)
            rig.runner.stop(2.0)
            assert wait_until(lambda: descendants(os.getpid()) <= before, 3.0)
    finally:
        rig.close()


def test_the_virtual_camera_refused_at_the_start_and_off_and_on(face_frame_720p: np.ndarray) -> None:
    if not obs_camera_opens():
        pytest.skip("needs OBS Virtual Camera")
    import pyvirtualcam

    holder = pyvirtualcam.Camera(1280, 720, 30, print_fps=False)  # the sender's Camera() is refused while this holds it
    rig = RunnerRig(face_frame_720p, virtual_camera=True, retry_interval_s=0.2)
    try:
        rig.runner.start()
        assert wait_until(lambda: rig.runner.status.virtual_camera_error is not None, 20.0), rig.runner.status
        assert rig.engine.wait_ready(60.0)
        count = len(rig.results)
        assert wait_until(lambda: len(rig.results) > count + 10, 10.0), "the preview goes on"
        holder.close()
        assert wait_until(lambda: rig.runner.status.virtual_camera is not None, 10.0), rig.runner.status
        assert rig.runner.status.virtual_camera_error is None
        assert wait_until(lambda: rig.runner.status.fps > 25.0, 5.0)
        rig.runner.set_virtual_camera(False)
        assert wait_until(lambda: rig.runner.status.virtual_camera is None, 5.0)
        count = len(rig.results)
        assert wait_until(lambda: len(rig.results) > count + 10, 10.0)
        rig.runner.set_virtual_camera(True)
        assert wait_until(lambda: rig.runner.status.virtual_camera is not None, 5.0)
    finally:
        holder.close()
        rig.close()
    assert wait_until(obs_camera_opens, 3.0), "the camera opens again"


def test_placeholders_reach_the_camera_when_the_camera_is_lost(face_frame_720p: np.ndarray) -> None:
    if not obs_camera_opens():
        pytest.skip("needs OBS Virtual Camera")
    rig = RunnerRig(face_frame_720p, virtual_camera=True, camera_timeout_s=0.3)
    original = rig.camera.read
    stop_frames = threading.Event()

    def read() -> np.ndarray | None:
        if stop_frames.is_set():
            time.sleep(0.05)
            return None
        return original()

    rig.camera.read = read  # type: ignore[method-assign]
    try:
        rig.runner.start()
        assert rig.engine.wait_ready(60.0)
        assert wait_until(lambda: rig.runner.status.virtual_camera is not None, 10.0)
        stop_frames.set()
        assert wait_until(lambda: rig.runner.status.camera_error is not None, 3.0)
        session = rig.engine._session
        written = int(session.flags[shm.SESSION_WRITES])
        assert wait_until(lambda: int(session.flags[shm.SESSION_WRITES]) > written + 10, 3.0), "placeholders are written"
        stop_frames.clear()
        assert wait_until(lambda: rig.runner.status.camera_ok, 3.0)
    finally:
        rig.close()


# ---- the switch -------------------------------------------------------------------------


def test_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(parallel.SWITCH_ENV, "0")
    assert not parallel.enabled()
    monkeypatch.setenv(parallel.SWITCH_ENV, "1")
    assert parallel.enabled()
    assert not parallel.enabled(no_parallel=True)
    monkeypatch.delenv(parallel.SWITCH_ENV)
    assert parallel.enabled() == (parallel.physical_cores() >= parallel.MIN_PHYSICAL_CORES)
    monkeypatch.setattr(parallel, "physical_cores", lambda: parallel.MIN_PHYSICAL_CORES - 1)
    assert not parallel.enabled()
    monkeypatch.setenv(parallel.SWITCH_ENV, "1")
    assert parallel.enabled()


def test_with_the_switch_off_no_engine_and_no_child(monkeypatch: pytest.MonkeyPatch, face_frame_720p: np.ndarray) -> None:
    monkeypatch.setenv(parallel.SWITCH_ENV, "0")
    pipeline = MakeupPipeline(load_preset("natural"), tracker=StableTracker(), segmenter=StableMasks())
    assert parallel.attach_engine(pipeline) is None
    assert pipeline.engine is None
    before = descendants(os.getpid())
    result = pipeline.process(face_frame_720p, 0)
    assert not result.in_flight
    assert descendants(os.getpid()) == before
    pipeline.close()


def test_a_pipeline_without_an_engine_stays_as_it_was(face_frame_720p: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(parallel.SWITCH_ENV, "1")
    pipeline = MakeupPipeline(load_preset("natural"), tracker=StableTracker(), segmenter=StableMasks())
    before = descendants(os.getpid())
    assert not pipeline.process(face_frame_720p, 0).in_flight
    assert descendants(os.getpid()) == before
    pipeline.close()


def test_only_the_window_and_headless_pass_the_engine() -> None:
    """Analysis, evaluation and the recorder never use the stages."""
    users: set[str] = set()
    for path in (ROOT / "makeup_cam").rglob("*.py"):
        if path.parent.name == "parallel":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for call in [n for n in ast.walk(function) if isinstance(n, ast.Call)]:
                if isinstance(call.func, ast.Name) and call.func.id == "attach_engine":
                    users.add(f"{path.name}:{function.name}")
    assert users == {"__main__.py:run_headless", "gui.py:run_gui"}


def test_the_command_line_knows_no_parallel() -> None:
    from makeup_cam.__main__ import parse_args

    assert parse_args(["--headless", "--no-parallel"]).no_parallel
    assert not parse_args(["--headless"]).no_parallel


def test_the_status_line_counts_lost_frames() -> None:
    from makeup_cam.__main__ import format_status

    assert "落ちた" not in format_status(RunnerStatus(), virtual_camera_enabled=False)
    assert "落ちたフレーム 3" in format_status(RunnerStatus(lost_frames=3), virtual_camera_enabled=False)


# ---- the learned finish in stage 2 ------------------------------------------------------


def test_the_finish_runs_in_stage_2_and_its_status_reaches_the_line(face_frame_720p: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a model it says so, and it is the effect in the stage that says it: the one here never ran."""
    from makeup_cam.neural import model_files, stage

    monkeypatch.setenv(model_files.NEURAL_DIR_ENV, str(tmp_path / "no-models"))
    settings = load_preset("natural")
    settings.neural.enabled = True
    settings.neural.strength = 0.5
    rig = Rig(face_frame_720p, 0, settings)
    try:
        rig.start()
        rig.feed(10)
        assert wait_until(lambda: rig.pipeline.finishing_status() in (stage.STATUS_NO_MODEL, stage.STATUS_NO_RUNTIME), 5.0), rig.pipeline.finishing_status()
        assert next(e for e in rig.pipeline._effects if e.name == "neural").status_message is None
    finally:
        rig.close()


# ---- real runs, in processes of their own -----------------------------------------------


def _clip(path: Path, face: np.ndarray) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (1280, 720))
    for i in range(30):
        writer.write(cv2.convertScaleAbs(face, alpha=1.0, beta=float(i % 3)))
    writer.release()
    return path


_SEND = r"""
import ctypes, sys, time
kernel32 = ctypes.windll.kernel32
kernel32.FreeConsole()
if not kernel32.AttachConsole(int(sys.argv[1])):
    sys.exit(3)
handler = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(lambda _event: 1)
kernel32.SetConsoleCtrlHandler(handler, True)
if sys.argv[2] == "close":
    window = kernel32.GetConsoleWindow()
    if not window:
        sys.exit(4)
    ctypes.windll.user32.PostMessageW(window, 0x0010, 0, 0)
else:
    kernel32.GenerateConsoleCtrlEvent(0, 0)
    time.sleep(0.2)
kernel32.FreeConsole()
"""

_APP = r"""
import ctypes, sys
ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)
from makeup_cam.__main__ import main
sys.exit(main(sys.argv[1:]))
"""

_GUI_CLOSES_ITSELF = r"""
import argparse, json, sys, time
from pathlib import Path
from PySide6.QtCore import QTimer
from makeup_cam import gui
from makeup_cam.parallel import engine as engine_module
import makeup_cam.pipeline as pipeline_module

report = Path(sys.argv[2])
original = gui.MainWindow.start
made = []

def start(self):
    made.append(self)
    original(self)
    engine = self._runner._pipeline.engine

    def check():
        report.write_text(json.dumps({"pids": engine.child_pids, "names": engine.names_in_use, "state": engine._state, "at": time.perf_counter()}), encoding="utf-8")
        self.close()

    def wait():
        if engine._state == "running" and self._last_result is not None:
            QTimer.singleShot(2000, check)
        else:
            QTimer.singleShot(100, wait)

    wait()

gui.MainWindow.start = start
sys.exit(gui.run_gui(argparse.Namespace(camera_file=Path(sys.argv[1]), no_vcam=len(sys.argv) > 3 and sys.argv[3] == "novcam")))
"""


def _children_of_the_real_app(launcher: subprocess.Popen[Any], timeout: float) -> set[int]:
    """Inside a venv the started process is a launcher: its descendants are the real app and the stages."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = descendants(launcher.pid)
        if len(found) >= 4:  # the stages and the sender, and the real app when the launcher is above it
            return found
        assert launcher.poll() is None
        time.sleep(0.05)
    raise AssertionError(f"too few processes: {descendants(launcher.pid)}")


@pytest.mark.parametrize("how", ["ctrl_c", "close"])
def test_headless_in_a_console_ends_with_nothing_left(tmp_path: Path, face_frame_720p: np.ndarray, models_dir: Path, how: str) -> None:
    clip = _clip(tmp_path / "clip.avi", face_frame_720p)
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "AI_MAKEUP_CAM_HOME": str(tmp_path / "home"), "PYTHONUTF8": "1", parallel.SWITCH_ENV: "1"}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    app = subprocess.Popen(
        [sys.executable, "-c", _APP, "--headless", "--camera-file", str(clip), "--preset", "natural", "--no-virtual-camera"],
        cwd=ROOT, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup,
    )  # fmt: skip
    try:
        running = _children_of_the_real_app(app, 60.0)
        time.sleep(1.0)
        running = descendants(app.pid)
        sent = subprocess.run([sys.executable, "-c", _SEND, str(app.pid), how], creationflags=subprocess.CREATE_NO_WINDOW)
        if sent.returncode in (3, 4):
            pytest.skip("cannot reach the console of a child process here")
        started = time.perf_counter()
        app.wait(timeout=20.0)
        assert time.perf_counter() - started < 6.0
    finally:
        app.kill()
    assert wait_until(lambda: not any(process_alive(pid) for pid in running), 2.0), running


def test_a_killed_main_process_leaves_nothing_within_2_seconds(tmp_path: Path, face_frame_720p: np.ndarray, models_dir: Path) -> None:
    clip = _clip(tmp_path / "clip.avi", face_frame_720p)
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "AI_MAKEUP_CAM_HOME": str(tmp_path / "home"), "PYTHONUTF8": "1", parallel.SWITCH_ENV: "1"}
    app = subprocess.Popen(
        [sys.executable, "-c", _APP, "--headless", "--camera-file", str(clip), "--preset", "natural", "--no-virtual-camera"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        _children_of_the_real_app(app, 60.0)
        time.sleep(1.0)
        running = descendants(app.pid)
        real_mains = [pid for pid in running if descendants(pid)]
        assert real_mains or running
        # The real main process when there is a launcher above it; otherwise the launcher is the app.
        victim = real_mains[0] if real_mains else app.pid
        subprocess.run(["taskkill", "/F", "/PID", str(victim)], capture_output=True)
        assert wait_until(lambda: not any(process_alive(pid) for pid in running if pid != victim), 2.0), running
    finally:
        app.kill()


@pytest.mark.parametrize("vcam", ["novcam", "vcam"])
def test_closing_the_window_ends_everything_within_the_deadline(tmp_path: Path, face_frame_720p: np.ndarray, models_dir: Path, vcam: str) -> None:
    import json

    from makeup_cam.gui import RUNNER_STOP_DEADLINE_S

    if vcam == "vcam" and not obs_camera_opens():
        pytest.skip("needs OBS Virtual Camera")
    clip = _clip(tmp_path / "clip.avi", face_frame_720p)
    report = tmp_path / "report.json"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "AI_MAKEUP_CAM_HOME": str(tmp_path / "home"), "PYTHONUTF8": "1", parallel.SWITCH_ENV: "1"}
    app = subprocess.Popen([sys.executable, "-c", _GUI_CLOSES_ITSELF, str(clip), str(report), vcam], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        output, _ = app.communicate(timeout=120.0)
    finally:
        app.kill()
    assert report.is_file(), output[-2000:]
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["state"] == "running"
    assert set(data["pids"]) == set(PROCESSES)
    assert app.returncode == 0, output[-2000:]
    assert_nothing_left((data["names"][0], data["names"][1]), data["pids"])
    assert RUNNER_STOP_DEADLINE_S == 2.0
    if vcam == "vcam":
        assert wait_until(obs_camera_opens, 3.0), "the camera opens again"
