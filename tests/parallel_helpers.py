"""Shared by the tests of the pipeline of processes: a tracker and masks that give the same answer for the same frame, so
that the masks' timing is taken out and not the comparison, and ways to fail a child process from outside, with no hook in the app."""

from __future__ import annotations

import ctypes
import hashlib
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from makeup_cam.config import MakeupSettings
from makeup_cam.parallel.engine import ParallelEngine
from makeup_cam.pipeline import MakeupPipeline, ProcessResult
from makeup_cam.segmentation import SegMasks

DATA_DIR = Path(__file__).resolve().parent / "data"
FRAME_MS = 33
HEIGHT, WIDTH = 720, 1280


def base_landmarks() -> np.ndarray:
    return np.load(DATA_DIR / "face_landmarks_720p.npy").astype(np.float32)


class StableTracker:
    """The same landmarks for the same timestamp, whichever thread asks and whenever. ``lost``: timestamps without a face."""

    def __init__(self, landmarks: np.ndarray | None = None, scale: float = 1.0) -> None:
        self._landmarks = base_landmarks() * scale if landmarks is None else landmarks
        self.lost: set[int] = set()
        self.calls = 0

    def detect(self, _frame: np.ndarray, ts_ms: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        self.calls += 1
        if ts_ms in self.lost:
            return None, None
        phase = ts_ms / 400.0
        moved = self._landmarks + np.array([3.0 * np.sin(phase), 2.0 * np.cos(phase)], np.float32)
        return moved.astype(np.float32), moved[:, 0] * 0.1

    def set_smoothing(self, _smoothing: float) -> None:
        pass

    def close(self) -> None:
        pass


class StableMasks:
    """New masks every ``interval`` frames, the same ones for the same frame number. Their timestamp is that of the frame."""

    def __init__(self, shape: tuple[int, int] = (HEIGHT, WIDTH), interval: int = 2) -> None:
        self._interval = interval
        self._shape = shape
        self._current: SegMasks | None = None
        self._run = -1
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h * 0.42
        self._face = np.clip(1.6 - (((xx - cx) / (w * 0.14)) ** 2 + ((yy - cy) / (h * 0.26)) ** 2), 0.0, 1.0).astype(np.float32)
        self._hair = np.clip(1.5 - (((xx - cx) / (w * 0.2)) ** 2 + ((yy - h * 0.25) / (h * 0.26)) ** 2), 0.0, 1.0).astype(np.float32) * (1.0 - self._face)
        self._body = np.clip((yy - h * 0.72) / (h * 0.1), 0.0, 1.0).astype(np.float32)

    def update(self, _frame: np.ndarray, ts_ms: int, frame_index: int) -> SegMasks:
        run = frame_index // self._interval
        if run != self._run:
            self._run = run
            shift = run % 7
            masks = []
            for base in (self._hair, self._face, self._body):
                mask = np.roll(base, shift, axis=1)
                mask.setflags(write=False)
                masks.append(mask)
            self._current = SegMasks(masks[0], masks[1], masks[2], ts_ms)
        assert self._current is not None
        return self._current

    def set_interval(self, interval: int) -> None:
        pass

    def close(self) -> None:
        pass


def make_frames(base: np.ndarray, count: int) -> list[np.ndarray]:
    """Frames that differ a little, each with its number in the first row (see ``number_of``)."""
    frames = []
    for i in range(count):
        frame = np.roll(base, i % 5, axis=1)
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=float(i % 3))
        frame[0, :, 0] = i % 256
        frame[0, :, 1] = (i >> 8) % 256
        frames.append(frame)
    return frames


def number_of(result: ProcessResult) -> int:
    """The number of the frame this result comes from: the original keeps it, whatever the makeup does."""
    row = result.original_preview[0, 0]
    return int(row[0]) + 256 * int(row[1])


def digest(picture: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(picture).tobytes(), digest_size=12).hexdigest()


def run_in_process(
    settings: MakeupSettings,
    frames: list[np.ndarray],
    script: Callable[[int, MakeupPipeline, StableTracker], bool] | None = None,
    **tracker_options: Any,
) -> list[str]:
    tracker = StableTracker(**tracker_options)
    masks = StableMasks(frames[0].shape[:2])
    pipeline = MakeupPipeline(settings, tracker=tracker, segmenter=masks)
    hashes = []
    try:
        for i, frame in enumerate(frames):
            ts = i * FRAME_MS
            if script is not None and not script(i, pipeline, tracker):
                pipeline.prepare(frame, ts)
                continue
            pipeline.prepare(frame, ts)
            hashes.append(digest(pipeline.process(frame, ts).output))
    finally:
        pipeline.close()
    return hashes


def run_in_stages(
    settings: MakeupSettings,
    frames: list[np.ndarray],
    script: Callable[[int, MakeupPipeline, StableTracker], bool] | None = None,
    **tracker_options: Any,
) -> list[str]:
    tracker = StableTracker(**tracker_options)
    masks = StableMasks(frames[0].shape[:2])
    pipeline = MakeupPipeline(settings, tracker=tracker, segmenter=masks)
    engine = ParallelEngine()
    pipeline.attach_engine(engine)
    engine.slot_wait_s = 30.0  # offline: nothing is dropped however slow the first frames are
    hashes: list[str] = []
    expected = 0

    def on_frame(result: ProcessResult) -> None:
        hashes.append(digest(result.output))

    engine.start(frames[0].shape[:2], on_frame)
    try:
        assert engine.wait_ready(60.0)
        for i, frame in enumerate(frames):
            ts = i * FRAME_MS
            if script is not None and not script(i, pipeline, tracker):
                pipeline.prepare(frame, ts)
                continue
            pipeline.prepare(frame, ts)
            result = pipeline.process(frame, ts)
            assert result.in_flight, f"frame {i} was drawn in process: {engine.report()}"
            expected += 1
        deadline = time.monotonic() + 30.0
        while len(hashes) < expected and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(hashes) == expected
    finally:
        pipeline.close()
    return hashes


# ---- failing a child from outside ------------------------------------------------------


def suspend(pid: int) -> None:
    """A real hang: all threads of the process stop."""
    _suspend_or_resume(pid, "NtSuspendProcess")


def resume(pid: int) -> None:
    _suspend_or_resume(pid, "NtResumeProcess")


def _suspend_or_resume(pid: int, what: str) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(0x0800, 0, pid)  # PROCESS_SUSPEND_RESUME
    assert handle, ctypes.get_last_error()
    function = getattr(ctypes.WinDLL("ntdll"), what)
    function.argtypes = [ctypes.c_void_p]
    function(handle)
    kernel32.CloseHandle(handle)


def process_alive(pid: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel32.WaitForSingleObject.restype = ctypes.c_uint
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(0x00100000, 0, pid)  # SYNCHRONIZE
    if not handle:
        return False
    alive = kernel32.WaitForSingleObject(handle, 0) != 0
    kernel32.CloseHandle(handle)
    return alive


class _ProcessEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("cntUsage", ctypes.c_uint32),
        ("th32ProcessID", ctypes.c_uint32),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", ctypes.c_uint32),
        ("cntThreads", ctypes.c_uint32),
        ("th32ParentProcessID", ctypes.c_uint32),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", ctypes.c_uint32),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


def descendants(root_pid: int) -> set[int]:
    """All living Python processes below ``root_pid``, children of children too. Windows ends a child's hidden console
    host in its own time, and that is not what is being counted."""
    if sys.platform != "win32":
        return set()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    snapshot = kernel32.CreateToolhelp32Snapshot(2, 0)
    entry = _ProcessEntry()
    entry.dwSize = ctypes.sizeof(_ProcessEntry)
    parents: dict[int, int] = {}
    more = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
    while more:
        if entry.szExeFile.lower().startswith("python"):
            parents[entry.th32ProcessID] = entry.th32ParentProcessID
        more = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    kernel32.CloseHandle(snapshot)
    found: set[int] = set()
    grew = True
    while grew:
        grew = False
        for pid, parent in parents.items():
            if pid not in found and (parent == root_pid or parent in found):
                found.add(pid)
                grew = True
    return found


def wait_until(condition: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()
