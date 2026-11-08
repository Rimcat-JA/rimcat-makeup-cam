"""The main process's side of the pipeline of processes (contract sections 3, 7, 8 and 9).

``route`` and ``submit`` run on the output thread, which never waits for drawing. A collector thread takes finished slots
in order, builds the preview, frees the slots and watches the processes. A boot thread starts them, so that nothing a
frame depends on ever waits for a spawn. Until the stages are ready the output thread draws in process and hands its frames
to the sender to write at once; after a failure it does so until the stages have been started again.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import queue
import statistics
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..camera import VirtualCameraError, _start_virtual_camera, fit_frame, virtual_camera_error_message
from ..pipeline import _TO_FLOAT, PlannedFrame, _compose
from . import (
    FALLBACK_FINAL_MESSAGE,
    FALLBACK_MESSAGE,
    LOG_ENV,
    MASK_READS,
    OVERLOAD_MESSAGE,
    SLOW_MESSAGE,
    VCAM_REOPEN_MESSAGE,
    physical_cores,
    plan_stages,
    shm,
)
from .obs_queue import ObsQueue, QueueRefused

logger = logging.getLogger(__name__)

_RESTARTING_MESSAGE = "仮想カメラの準備をしています…"

HANG_LIMIT_S = 1.0
FIRST_FRAMES_HANG_LIMIT_S = 2.0  # lazily built tables: the first frames of a stage are slow, and that is no failure
FIRST_FRAMES = 30
SLOT_WAIT_S = 0.1  # the output thread never waits longer for a free slot: it drops the frame and counts it
WATCH_S = 0.025
STALL_S = 0.5  # the collector itself was not scheduled for this long: the whole machine stood still
READY_TIMEOUT_S = 60.0  # a cold disk; frames are drawn in process meanwhile, so nothing depends on it
REPLY_TIMEOUT_S = 5.0
FLUSH_TIMEOUT_S = 0.3
DRAIN_TIMEOUT_S = 0.5  # frames in flight when the frame size changes
STOP_WAIT_S = 3.0
LOCAL_CAMERA_TRIES = 3
LOCAL_CAMERA_PAUSE_S = 0.05
LOCAL_CAMERA_RETRY_S = 1.0
SLOW_MESSAGE_S = 3.0
# After a failure the stages are tried again this long after it, and after each try that fails; then no more. Long enough
# that a machine that stood still has recovered, and that a stage that keeps hanging does not freeze the picture every few seconds.
RECOVER_AFTER_S = (2.0, 20.0, 60.0)
RECOVER_FORGET_S = 60.0  # running this long, a failure counts as the first again
# Each process keeps to one thread per core. On this few cores the app's 3 to 3.5 logical CPUs leave a call app at normal
# priority too little (only with the switch forced on, as the default is off below MIN_PHYSICAL_CORES): there the steps below
# apply. On more cores, what the frames are short of goes to other programs, and giving way to them only costs frames.
FEW_CORES = 5
# Once per second: how many frames were lost or repeated, against how many came from the camera.
LOAD_SECONDS = 10
POLITE_BAD_SECONDS = 7  # of the last LOAD_SECONDS with trouble: too little CPU, not a passing stall
POLITE_CALM_S = 30  # seconds without trouble, and the priority returns
OVERLOAD_SHARE = 1.0 / 3.0  # of the frames in each of the last LOAD_SECONDS: one process does better from here on
_AGES_KEPT = 64
_FPS_SAMPLE_S = 0.5

_IDLE = "idle"
_BOOTING = "booting"
_RUNNING = "running"
_FAILING = "failing"
_FALLBACK = "fallback"
_STOPPED = "stopped"

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_instance_counter = 0
_instance_lock = threading.Lock()


class VirtualCameraNotReady(VirtualCameraError):
    """The sender is still starting: as during today's retries, the virtual camera is simply not open yet."""

    retry_soon = True


@dataclass(frozen=True)
class EngineReport:
    """What the status line needs. ``fps`` and ``processing_ms`` are None when the engine has nothing to say about them."""

    fps: float | None = None
    processing_ms: float | None = None
    processing_error: str | None = None
    virtual_camera_error: str | None = None
    lost_frames: int = 0
    slips: int = 0
    delay_ms: float | None = None  # from the camera to the virtual camera


class _JobObject:
    """Every child dies with the main process, however that ends; so do their children."""

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self._kernel32 = kernel32
        self._handle = kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise OSError(ctypes.get_last_error(), "no job object")

        class Basic(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", Basic),
                ("IoInfo", ctypes.c_ulonglong * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = Extended()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            self._handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise OSError(ctypes.get_last_error(), "the job object takes no limit")

    def add(self, process: subprocess.Popen[Any]) -> None:
        if not self._kernel32.AssignProcessToJobObject(self._handle, ctypes.c_void_p(int(process._handle))):  # type: ignore[attr-defined]
            raise OSError(ctypes.get_last_error(), "a child cannot join the job object")

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle:
            self._kernel32.CloseHandle(handle)


class _Child:
    """One process. Its stdout is JSON lines, its stderr goes to this process's log."""

    def __init__(self, name: str, module: str, args: dict[str, Any], job: _JobObject, on_status: Any = None, *, stdin: bool = False) -> None:
        self.name = name
        self._on_status = on_status
        self.started_at = time.perf_counter()
        self.ready_after: float | None = None
        command, env = shm.child_command(module, args)
        self.process: subprocess.Popen[str] = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            job.add(self.process)
        except OSError:  # outside the job nothing would end it with the main process
            self.process.kill()
            self.process.wait(1.0)
            raise
        self.replies: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._request_lock = threading.Lock()
        threading.Thread(target=self._read_stdout, name=f"{name}-stdout", daemon=True).start()
        threading.Thread(target=self._read_stderr, name=f"{name}-stderr", daemon=True).start()

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def _read_stdout(self) -> None:
        stream = self.process.stdout
        assert stream is not None
        try:
            for line in stream:
                try:
                    message = json.loads(line)
                except ValueError:
                    logger.debug("%s said: %s", self.name, line.rstrip())
                    continue
                if "neural_status" in message or "neural_wants_masks" in message:
                    if self._on_status is not None:
                        self._on_status(message)
                else:
                    self.replies.put(message)
        except (OSError, ValueError):
            pass
        self.replies.put(None)

    def _read_stderr(self) -> None:
        stream = self.process.stderr
        assert stream is not None
        try:
            for line in stream:
                text = line.rstrip()
                if text.startswith(("slip", "drop")):
                    logger.info("%s: %s", self.name, text)
                elif text:
                    logger.warning("%s: %s", self.name, text)
        except (OSError, ValueError):
            pass

    def wait_ready(self, stopping: threading.Event, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not stopping.is_set() and time.monotonic() < deadline:
            try:
                message = self.replies.get(timeout=0.05)
            except queue.Empty:
                continue
            if message is None:
                return False
            if "ready" in message:
                self.ready_after = time.perf_counter() - self.started_at
                return True
        return False

    def request(self, command: dict[str, Any], timeout: float = REPLY_TIMEOUT_S) -> dict[str, Any]:
        """Raises ``OSError`` when the process does not answer."""
        with self._request_lock:
            stream = self.process.stdin
            if stream is None:
                raise OSError("this process takes no commands")
            try:
                stream.write(json.dumps(command) + "\n")
                stream.flush()
                reply = self.replies.get(timeout=timeout)
            except (OSError, ValueError, queue.Empty) as exc:
                raise OSError(f"{self.name} did not answer: {exc!r}") from exc
            if reply is None:
                raise OSError(f"{self.name} has ended")
            return reply

    def kill(self) -> None:
        if self.alive:
            self.process.kill()

    def wait(self, timeout: float) -> bool:
        try:
            self.process.wait(max(0.0, timeout))
            return True
        except subprocess.TimeoutExpired:
            return False

    def close_pipes(self) -> None:
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass


class _Line:
    """A shared block and the stages that draw in it, for one frame size."""

    def __init__(self, index: int, shape: tuple[int, int], base: str, wake_name: str, stages: list[list[str]], job: _JobObject, log_dir: str, on_status: Any) -> None:
        self.index = index
        self.shape = shape
        height, width = shape
        self.layout = shm.Layout(height, width)
        self.block = shm.Block(self.layout)
        self.block.touch()
        self.base = f"{base}-l{index}"
        names = [f"{self.base}-e{edge}" for edge in range(len(stages))] + [wake_name]
        self._semaphores = [shm.Semaphore(name, create=True) for name in names[:-1]]
        self.done_name = f"{self.base}-done"
        self._done_semaphore = shm.Semaphore(self.done_name, create=True)
        self.into = shm.Edge(self.block.ring(0), self._semaphores[0])
        self.done = shm.Edge(self.block.ring(shm.EDGE_DONE), self._done_semaphore)
        self.free: deque[int] = deque(range(shm.N_SLOTS))
        self.free_sets: deque[int] = deque(range(shm.N_MASK_SETS))
        self.set_references: dict[int, int] = {}
        self.current_masks: tuple[Any, Any, Any] | None = None
        self.current_set: int | None = None
        self.set_generation = 0
        self.plate_references = [0, 0]
        self.current_plate: np.ndarray | None = None
        self.current_plate_slot: int | None = None
        self.plate_generation = 0
        self.frames_out = 0
        self.first_frame = True
        self.stages: list[_Child] = []
        common = {"shm": self.block.name, "layout": self.layout.args(), "parent": os.getpid()}
        try:
            for i, effects in enumerate(stages):
                later = [name for stage in stages[i + 1 :] for name in stage]
                args = dict(
                    common,
                    stage=i,
                    last=i == len(stages) - 1,
                    effects=effects,
                    reads_later={name: MASK_READS[name] for name in later if name in MASK_READS},
                    receive_semaphore=names[i],
                    send_semaphore=names[i + 1],
                )
                if log_dir:
                    args["log"] = os.path.join(log_dir, f"stage{i}.json")
                self.stages.append(_Child(f"stage {i}", "makeup_cam.parallel.stage_main", args, job, on_status))
        except Exception:
            self.kill()
            self.close()
            raise

    def stop(self, wait_s: float) -> None:
        """After the last frame. A stage that does not stop in time is killed."""
        block = self.block
        block.flags[shm.LINE_STOP] = 1.0
        for semaphore in self._semaphores:
            semaphore.post()
        deadline = time.monotonic() + wait_s
        for child in self.stages:
            if not child.wait(deadline - time.monotonic()):
                logger.warning("%s did not stop in time and is killed", child.name)
                child.kill()
                child.wait(0.5)
        self.close()

    def kill(self) -> None:
        for child in self.stages:
            child.kill()
        for child in self.stages:
            child.wait(0.5)

    def close(self) -> None:
        self.into = self.done = None  # type: ignore[assignment]
        for semaphore in [*self._semaphores, self._done_semaphore]:
            semaphore.close()
        self.block.close(unlink=True)


class _EngineCamera:
    """The virtual camera as the output thread sees it. Frames that went through the stages never pass here."""

    def __init__(self, engine: ParallelEngine, width: int, height: int, device: str) -> None:
        self._engine = engine
        self.width, self.height = width, height
        self.device = device

    def send(self, frame_bgr: np.ndarray) -> None:
        self._engine._send_at_once(fit_frame(frame_bgr, self.width, self.height))

    def close(self) -> None:
        self._engine._close_camera()


class ParallelEngine:
    """See the module docstring. One instance per pipeline; ``start`` and ``stop`` may repeat."""

    def __init__(self) -> None:
        global _instance_counter
        with _instance_lock:
            _instance_counter += 1
            self._instance = _instance_counter
        self._pipeline: Any = None
        self._state = _IDLE
        self._state_lock = threading.Lock()
        self._failed = threading.Event()
        self._stopping = threading.Event()  # a new one per session: a thread left behind by a stop must see that stop
        self._line_change = threading.Event()
        self._flight_lock = threading.Lock()
        self._flight_changed = threading.Condition(self._flight_lock)
        self._camera_lock = threading.Lock()
        self._send_lock = threading.RLock()  # taken before the camera lock; held while the local camera is written or closed
        self._way_in_lock = threading.RLock()  # a failure waits for a frame that is entering, so that it is drawn again too
        self.slot_wait_s = SLOT_WAIT_S
        self._reset_session_state()

    def _reset_session_state(self) -> None:
        self._on_frame: Any = None
        self._flight: deque[PlannedFrame] = deque()
        self._wanted_shape: tuple[int, int] | None = None
        self._line: _Line | None = None
        self._lines_made = 0
        self._job: _JobObject | None = None
        self._session: shm.SessionBlock | None = None
        self._wake: shm.Semaphore | None = None
        self._sender: _Child | None = None
        self._sender_ready = threading.Event()
        self._sender_failed = False
        self._local: Any = None
        self._camera_request: tuple[int, int, int] | None = None
        self._next_local_try = 0.0
        self._vcam_problem: str | None = None
        self._message: str | None = None
        self._next_id = 0
        self._last_out = time.perf_counter()
        self._last_watch = time.perf_counter()
        self._lost_at_way_in = 0
        self._last_loss_seen = 0
        self._last_loss_at = 0.0
        self._ages_ms: deque[float] = deque(maxlen=_AGES_KEPT)
        self._fps_sample: tuple[float, float, int] | None = None
        self._fps: float | None = None
        self._returned = 0
        self._stage_status: str | None = None
        self._stage_wants_masks = False
        self._recover_tries = 0
        self._recover_at: float | None = None
        self._running_since = 0.0
        self._senders_made = 1
        self._polite = False
        self._few_cores = physical_cores() <= FEW_CORES
        self._load_at = 0.0
        self._load_seen = (0, 0, 0)
        self._load_seconds: deque[tuple[int, int]] = deque(maxlen=LOAD_SECONDS)
        self._calm_seconds = 0
        self._delays_ms: deque[float] = deque(maxlen=_AGES_KEPT)
        self._threads: list[threading.Thread] = []
        self._logged: set[str] = set()
        self._stage_plan: list[list[str]] = []
        self._wake_name = ""
        self._grace_until = 0.0
        self._log_dir = os.environ.get(LOG_ENV, "")
        self._log: list[tuple[int, float, float]] = []
        self._events: list[Any] = []
        self._started_at = time.perf_counter()
        self._cpu_at_start = time.process_time()
        self._ready_after_ms = 0.0
        self.base = ""

    # ---- wiring -----------------------------------------------------------

    def bind(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    @property
    def running(self) -> bool:
        """Between ``start`` and ``stop``."""
        return self._state not in (_IDLE, _STOPPED)

    @property
    def stages_drawing(self) -> bool:
        """The stages draw now, or will: the main process's own effects are only a stopgap."""
        return self._state in (_BOOTING, _RUNNING)

    @property
    def stage_status(self) -> str | None:
        """What the learned finishing stage in stage 2 says."""
        return self._stage_status

    @property
    def stage_wants_masks(self) -> bool:
        """Whether the learned finishing stage in stage 2 will use the masks, as it last said."""
        return self._stage_wants_masks

    @property
    def child_pids(self) -> dict[str, int]:
        """For the drills: the processes now running."""
        pids = {} if self._line is None else {f"stage{i}": child.pid for i, child in enumerate(self._line.stages)}
        if self._sender is not None:
            pids["sender"] = self._sender.pid
        return pids

    @property
    def names_in_use(self) -> tuple[list[str], list[str]]:
        """For the tests: shared blocks and semaphores of the running session."""
        blocks: list[str] = []
        semaphores: list[str] = []
        if self._session is not None and self._wake is not None:
            blocks.append(self._session.name)
            semaphores.append(self._wake.name)
        line = self._line
        if line is not None:
            blocks.append(line.block.name)
            semaphores += [f"{line.base}-e{i}" for i in range(len(line.stages))] + [line.done_name]
        return blocks, semaphores

    def wait_ready(self, timeout: float = READY_TIMEOUT_S) -> bool:
        """For offline use and the tests: True once frames go through the stages."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._state == _RUNNING:
                return True
            if self._state in (_FAILING, _FALLBACK, _STOPPED, _IDLE):
                return False
            time.sleep(0.01)
        return False

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """For offline use and the tests: True once no frame is in flight."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._flight:
                return True
            time.sleep(0.005)
        return False

    # ---- start and stop ---------------------------------------------------

    def start(self, frame_shape: tuple[int, int], on_frame: Any = None) -> None:
        """Returns at once: the processes start in the background, and frames are drawn in process until they are ready."""
        with self._state_lock:
            if self.running:
                return
            self._reset_session_state()
            self._failed.clear()
            self._stopping = stopping = threading.Event()
            self._line_change.clear()
            self._on_frame = on_frame
            self._wanted_shape = frame_shape
            self.base = f"mkcam-{os.getpid()}-{self._instance}-{time.monotonic_ns() % 1000000}"
            self._state = _BOOTING
            collector = threading.Thread(target=self._collect, args=(stopping,), name="parallel-collector", daemon=True)
            booter = threading.Thread(target=self._boot, args=(stopping,), name="parallel-boot", daemon=True)
            self._threads = [booter, collector]
        collector.start()
        booter.start()

    def stop(self, timeout: float = STOP_WAIT_S) -> None:
        """Stop flag, all semaphores, wait up to ``timeout``, kill what remains, unlink the blocks. May be repeated."""
        with self._state_lock:
            if not self.running:
                return
            self._state = _STOPPED
            self._stopping.set()  # under the lock: what the boot thread adds under it, this stop sees
        deadline = time.monotonic() + max(0.1, min(timeout, STOP_WAIT_S))
        self._failed.set()
        self._line_change.set()
        with self._flight_changed:
            self._flight_changed.notify_all()
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(max(0.0, min(1.0, deadline - time.monotonic())))
        self._write_main_log()
        line, self._line = self._line, None
        with self._flight_lock:
            self._flight.clear()
        if line is not None:
            line.stop(max(0.05, deadline - time.monotonic()))
        self._stop_sender(deadline)
        self._close_local()
        if self._job is not None:
            self._job.close()
            self._job = None
        shm.normal_process()

    def _stop_sender(self, deadline: float) -> None:
        sender, self._sender = self._sender, None
        session, self._session = self._session, None
        wake, self._wake = self._wake, None
        if session is None:
            return
        session.flags[shm.SESSION_STOP] = 1.0
        if wake is not None:
            wake.post()
        if sender is not None:
            if not sender.wait(deadline - time.monotonic()):
                logger.warning("the sender did not stop in time and is killed")
                sender.kill()
                sender.wait(0.5)
            sender.close_pipes()
        if wake is not None:
            wake.close()
        session.close(unlink=True)

    def _boot(self, stopping: threading.Event) -> None:
        try:
            self._start_processes(stopping)
        except Exception as exc:
            if not stopping.is_set():
                self._fall_back(f"could not start: {exc!r}")
        while not stopping.is_set():
            changed = self._line_change.wait(0.25)
            if changed:
                self._line_change.clear()
            if stopping.is_set():
                return
            try:
                if changed and self._state == _BOOTING:
                    self._change_line(stopping)
                elif self._state == _FALLBACK:
                    self._recover_when_due(stopping)
            except Exception as exc:
                if not stopping.is_set():
                    self._fall_back(f"a new line: {exc!r}")

    def _start_processes(self, stopping: threading.Event) -> None:
        names = [effect.name for effect in self._pipeline._effects]
        self._stage_plan = plan_stages(names)
        job = _JobObject()
        with self._state_lock:
            if stopping.is_set():
                job.close()
                return
            self._job = job
            shm.keep_full_speed()
            # Above the other programs, as the stages are: a frame the camera thread or the tracker misses is lost before
            # any stage sees it. Next to a busy desktop: 36 frames lost in 12 runs at normal priority, 10 at this one.
            shm.above_normal_process()
        sender, session, wake = self._new_sender(job, 1)
        with self._state_lock:
            stale = stopping.is_set()
            if not stale:
                self._sender, self._session, self._wake = sender, session, wake
                self._wake_name = wake.name
        if stale:
            sender.kill()
            wake.close()
            session.close(unlink=True)
            return
        line = self._make_line(self._wanted_shape, stopping)
        if not sender.wait_ready(stopping, READY_TIMEOUT_S):
            if stopping.is_set():
                return
            self._sender_failed = True
            raise OSError("the sender did not start")
        self._sender_ready.set()
        self._bring_up(line, stopping)

    def _new_sender(self, job: _JobObject, number: int) -> tuple[_Child, shm.SessionBlock, shm.Semaphore]:
        session = shm.SessionBlock()
        wake_name = f"{self.base}-wake{number}"
        wake = shm.Semaphore(wake_name, create=True)
        args = {"session": session.name, "wake_semaphore": wake_name, "parent": os.getpid()}
        if self._log_dir:
            args["log"] = os.path.join(self._log_dir, "sender.json" if number == 1 else f"sender{number}.json")
        try:
            return _Child("sender", "makeup_cam.parallel.sender_main", args, job, stdin=True), session, wake
        except Exception:
            wake.close()
            session.close(unlink=True)
            raise

    def _make_line(self, shape: tuple[int, int] | None, stopping: threading.Event) -> _Line:
        job = self._job
        assert shape is not None and job is not None
        self._lines_made += 1
        line = _Line(self._lines_made, shape, self.base, self._wake_name, self._stage_plan, job, self._log_dir, self._note_stage_status)
        line.block.flags[shm.LINE_POLITE] = 1.0 if self._polite else 0.0
        with self._state_lock:
            stale = stopping.is_set()
            if not stale:
                self._line = line  # from here on a failure reaches its stages; frames go in only once the state says so
        if stale:
            line.kill()
            line.close()
            raise OSError("stopped meanwhile")
        return line

    def _bring_up(self, line: _Line, stopping: threading.Event) -> None:
        """The line is usable once all its stages say so and the sender has attached."""
        for child in line.stages:
            if not child.wait_ready(stopping, READY_TIMEOUT_S):
                if stopping.is_set():
                    return
                raise OSError(f"{child.name} did not start")
        sender = self._sender
        if sender is None or stopping.is_set():
            return
        sender.request({"op": "attach", "shm": line.block.name, "layout": line.layout.args(), "done_semaphore": line.done_name})
        self._last_out = time.perf_counter()
        self._ready_after_ms = (time.perf_counter() - self._started_at) * 1000.0
        if self._log_dir:
            for child in [sender, *line.stages]:
                self._events.append((child.name, "ready", child.ready_after))
        with self._state_lock:
            if self._state in (_BOOTING, _FALLBACK) and line.shape == self._wanted_shape:
                self._state = _RUNNING
                self._running_since = time.perf_counter()
                self._load_seconds.clear()
                self._load_at = 0.0
            elif self._state == _BOOTING:
                self._line_change.set()  # the size changed once more meanwhile

    def _change_line(self, stopping: threading.Event) -> None:
        """A new block and new stages for a new frame size. The old stages stop after their last frame."""
        old = self._line
        if old is not None:
            assert self._sender is not None
            self._sender.request({"op": "detach"})  # if this fails the line is still there, for the failure to end it
            self._line = None
            old.stop(1.0)
            with self._flight_lock:
                self._flight.clear()  # empty, unless a stage hung just then and took its frames along
        self._bring_up(self._make_line(self._wanted_shape, stopping), stopping)

    # ---- trying the stages again after a failure (boot thread) ------------

    def _plan_recovery(self) -> None:
        """When to try next; or, with the tries used up, say that this is how it stays."""
        if self._recover_tries < len(RECOVER_AFTER_S):
            self._recover_at = time.monotonic() + RECOVER_AFTER_S[self._recover_tries]
        else:
            self._recover_at = None
            self._message = FALLBACK_FINAL_MESSAGE

    def _recover_when_due(self, stopping: threading.Event) -> None:
        if self._recover_at is None or time.monotonic() < self._recover_at:
            return
        self._recover_tries += 1
        self._recover_at = None
        try:
            self._restart(stopping)
        except Exception as exc:
            if stopping.is_set():
                return
            logger.warning("the stages could not be started again: %r", exc)
            line, self._line = self._line, None
            if line is not None:
                line.kill()
                line.close()
            self._plan_recovery()

    def _restart(self, stopping: threading.Event) -> None:
        if not self._sender_usable():
            self._replace_sender(stopping)
        session = self._session
        assert session is not None
        session.flags[shm.SESSION_FLUSH] = 0.0
        session.flags[shm.SESSION_FLUSH_DONE] = 0.0
        self._stage_wants_masks = False
        shape = self._pipeline._frame_shape or self._wanted_shape
        self._wanted_shape = shape
        self._bring_up(self._make_line(shape, stopping), stopping)
        with self._state_lock:
            if self._state != _RUNNING:
                raise OSError(f"still {self._state} after the stages started again")
            self._message = None
        logger.warning("the stages draw again")
        self._events.append(("recovered", shm.now()))

    def _replace_sender(self, stopping: threading.Event) -> None:
        """A new sender takes over the camera. While a reader holds the dead one's queue no process can open it: then the
        new sender writes into that queue, as the rescue here did. Only if it cannot, the rescue stays where it is."""
        job = self._job
        assert job is not None
        self._senders_made += 1
        sender, session, wake = self._new_sender(job, self._senders_made)
        try:
            if not sender.wait_ready(stopping, READY_TIMEOUT_S):
                raise OSError("the new sender did not start")
            with self._send_lock, self._camera_lock:
                request = self._camera_request
                if request is not None:
                    local, self._local = self._local, None
                    if local is not None:
                        local.close()
                    command = {"op": "open", "width": request[0], "height": request[1], "fps": request[2]}
                    reply = sender.request(command)
                    for _ in range(LOCAL_CAMERA_TRIES - 1):  # a camera just closed here: its readers let go of it at their next tick
                        if reply.get("ok"):
                            break
                        time.sleep(LOCAL_CAMERA_PAUSE_S)
                        reply = sender.request(command)
                    if not reply.get("ok"):
                        logger.warning("the virtual camera goes on through the queue a reader still holds, from a new sender")
                        reply = sender.request({**command, "old_queue": True})
                    if not reply.get("ok"):
                        self._local = self._open_local_camera(*request)
                        raise OSError(f"the virtual camera stays with this process: {reply.get('error')}")
                old_session, old_wake = self._session, self._wake
                self._sender, self._session, self._wake = sender, session, wake
                self._wake_name = wake.name
                self._sender_failed = False
                self._vcam_problem = None
            if old_wake is not None:
                old_wake.close()
            if old_session is not None:
                old_session.close(unlink=True)
        except Exception:
            sender.kill()
            sender.wait(0.5)
            wake.close()
            session.close(unlink=True)
            raise

    # ---- the way in (output thread) ---------------------------------------

    def route(self, shape: tuple[int, int]) -> bool:
        """True: this frame goes through the stages. False: the caller draws it. Blocks only while a failure is being handled."""
        state = self._state
        if state == _RUNNING:
            line = self._line
            if line is not None and line.shape == shape:
                return True
            self._frame_size_changed(shape)
            return False
        if state == _FAILING:
            self._failed.wait()
            return False
        if state == _BOOTING and shape != self._wanted_shape:
            self._frame_size_changed(shape)
        return False

    def _frame_size_changed(self, shape: tuple[int, int]) -> None:
        """In-process drawing takes over at once; the frames in flight still leave first."""
        with self._state_lock:
            if self._state not in (_RUNNING, _BOOTING):
                return
            self._state = _BOOTING
            self._wanted_shape = shape
        deadline = time.monotonic() + DRAIN_TIMEOUT_S
        with self._flight_changed:
            self._flight_changed.wait_for(lambda: not self._flight or self._stopping.is_set(), max(0.0, deadline - time.monotonic()))
        self._line_change.set()

    def submit(self, planned: PlannedFrame) -> bool:
        """Into the stages. True: the engine has the frame (it may have dropped it: ``planned.dropped``).
        False: the caller draws it, as the stages have failed. Only after the frames in flight were drawn again."""
        with self._way_in_lock:
            taken = self._submit(planned)
        if not taken and self._state == _FAILING:
            self._failed.wait()
        return taken

    def submit_through(self, planned: PlannedFrame) -> bool:
        """A frame without makeup takes its turn behind the frames in flight, so the order is kept. A reset still travels."""
        with self._way_in_lock:
            taken = self._submit_through(planned)
        if not taken and self._state == _FAILING:
            self._failed.wait()
        return taken

    def _submit(self, planned: PlannedFrame) -> bool:
        line = self._take_line()
        if line is None:
            return False
        started = time.perf_counter()
        slot = self._take_slot(line)
        if slot is None:
            if self._state != _RUNNING:
                return False
            self._lost_at_way_in += 1
            planned.dropped = True
            return True
        try:
            cv2.LUT(planned.frame, _TO_FLOAT, dst=line.block.image(slot))
            mask_references, planned.mask_set = self._share_masks(line, planned.masks)
            plate_reference, planned.plate_slot = self._share_plate(line, planned.plate)
            meta = {
                "id": self._next_id + 1,
                "seq": planned.seq,
                "t_camera": planned.t_camera,
                "ts_ms": planned.ts_ms,
                "reset": planned.reset or line.first_frame,
                "through": False,
                "landmarks": planned.landmarks,
                "z": planned.z,
                "masks": mask_references,
                "masks_ts_ms": planned.masks.source_ts_ms,
                "plate": plate_reference,
                "skipped": planned.skipped,
                "mirror": planned.general.mirror_output,
                "steps": [
                    (s.effect.name, s.settings, s.intensity, s.effect.needs_hair_mask, s.effect.order) for s in planned.steps
                ],
            }
            line.block.write_meta(slot, meta)
        except Exception as exc:  # metadata too large, or a size that does not fit: fails loudly, the camera goes on
            logger.exception("a frame could not enter the stages")
            line.free.append(slot)
            self._fall_back(f"way in: {exc!r}")
            return False
        self._enter(line, planned, slot, started)
        return True

    def _submit_through(self, planned: PlannedFrame) -> bool:
        line = self._take_line()
        if line is None:
            return False
        started = time.perf_counter()
        slot = self._take_slot(line)
        if slot is None:
            if self._state != _RUNNING:
                return False
            self._lost_at_way_in += 1
            planned.dropped = True
            return True
        cv2.copyTo(planned.frame, None, line.block.out(slot))
        meta = {
            "id": self._next_id + 1,
            "seq": planned.seq,
            "t_camera": planned.t_camera,
            "reset": planned.reset or line.first_frame,
            "through": True,
            "mirror": planned.general.mirror_output,
        }
        line.block.write_meta(slot, meta)
        self._enter(line, planned, slot, started)
        return True

    def _take_line(self) -> _Line | None:
        line = self._line
        return line if self._state == _RUNNING else None

    def _take_slot(self, line: _Line) -> int | None:
        with self._flight_changed:
            if not line.free:
                self._flight_changed.wait_for(lambda: line.free or self._state != _RUNNING, self.slot_wait_s)
            return line.free.popleft() if line.free and self._state == _RUNNING else None

    def _enter(self, line: _Line, planned: PlannedFrame, slot: int, started: float) -> None:
        self._next_id += 1
        planned.identity = self._next_id
        planned.slot = slot
        planned.entered_at = time.perf_counter()
        line.first_frame = False
        with self._flight_lock:
            self._flight.append(planned)
        line.into.put(slot)
        if self._log_dir:
            self._log.append((planned.seq, planned.t_camera, round((time.perf_counter() - started) * 1000, 3)))

    def _share_masks(self, line: _Line, masks: Any) -> tuple[list[Any], int | None]:
        """Copies a new run's masks into shared memory once; frames in flight keep theirs (contract section 4.4)."""
        arrays = (masks.hair, masks.face_skin, masks.body_skin)
        if all(a is None for a in arrays):
            return [None, None, None], None
        shape = line.shape
        current = line.current_masks
        if current is None or any(a is not b for a, b in zip(arrays, current, strict=True)):
            with self._flight_lock:
                mask_set = line.free_sets.popleft()
                line.set_references[mask_set] = 1  # held while it is the newest
                previous, line.current_set = line.current_set, mask_set
                if previous is not None:
                    self._drop_set_reference(line, previous)
            for which, array in enumerate(arrays):
                if array is not None and array.shape[:2] == shape:
                    cv2.copyTo(array, None, line.block.set_mask(mask_set, which))
            line.current_masks = arrays
            line.set_generation += 1
        mask_set = line.current_set
        assert mask_set is not None
        with self._flight_lock:
            line.set_references[mask_set] += 1
        references = [
            None if array is None or array.shape[:2] != shape else ("set", mask_set, which, line.set_generation)
            for which, array in enumerate(arrays)
        ]
        return references, mask_set

    @staticmethod
    def _drop_set_reference(line: _Line, mask_set: int) -> None:
        line.set_references[mask_set] -= 1
        if line.set_references[mask_set] == 0:
            del line.set_references[mask_set]
            line.free_sets.append(mask_set)

    def _share_plate(self, line: _Line, plate: np.ndarray | None) -> tuple[Any, int | None]:
        """Two plate slots. A new plate goes into the one not in use; a slot is reused only when no frame refers to it."""
        if plate is None:
            line.current_plate = None
            if line.current_plate_slot is not None:
                with self._flight_lock:
                    line.plate_references[line.current_plate_slot] -= 1
                line.current_plate_slot = None
            return None, None
        if plate is not line.current_plate:
            spare = self._free_plate_slot(line)
            if spare is None:  # both are still in use: this frame keeps the plate it has, the next one tries again
                if line.current_plate_slot is None:
                    return None, None
            else:
                cv2.copyTo(plate, None, line.block.plate(spare))
                with self._flight_lock:
                    if line.current_plate_slot is not None:
                        line.plate_references[line.current_plate_slot] -= 1
                    line.plate_references[spare] += 1  # held while it is the newest
                line.current_plate = plate
                line.current_plate_slot = spare
                line.plate_generation += 1
        index = line.current_plate_slot
        assert index is not None
        with self._flight_lock:
            line.plate_references[index] += 1
        return (index, line.plate_generation), index

    def _free_plate_slot(self, line: _Line) -> int | None:
        with self._flight_changed:
            self._flight_changed.wait_for(lambda: 0 in line.plate_references or self._state != _RUNNING, self.slot_wait_s)
            return line.plate_references.index(0) if 0 in line.plate_references else None

    # ---- the way out (collector thread) -----------------------------------

    def _collect(self, stopping: threading.Event) -> None:
        shm.set_this_thread_priority(shm.THREAD_PRIORITY_ABOVE_NORMAL)
        milliseconds = int(WATCH_S * 1000)
        while not stopping.is_set():
            try:
                line = self._line if self._state == _RUNNING or self._flight else None
                if line is None:
                    stopping.wait(WATCH_S)
                else:
                    slot = line.done.get(milliseconds)
                    if stopping.is_set():
                        return
                    if slot is not None:
                        self._finish(line, slot)
                self._watch()
            except Exception as exc:
                if stopping.is_set():
                    return
                logger.exception("the collector failed")
                self._fall_back(f"collector: {exc!r}")

    def _finish(self, line: _Line, slot: int) -> None:
        with self._flight_lock:
            planned = self._flight.popleft() if self._flight else None
        if planned is None or planned.slot != slot:
            raise RuntimeError(f"slot {slot} came back out of order (expected {planned.slot if planned else None})")
        self._last_out = time.perf_counter()
        line.frames_out += 1
        self._returned += 1
        meta = line.block.read_meta(slot)
        age_ms = (meta["finished"] - meta["t_camera"]) * 1000.0 if meta["t_camera"] else None
        if age_ms is not None:
            self._ages_ms.append(age_ms)
        on_frame = self._on_frame
        if on_frame is not None:
            self._call_on_frame(on_frame, line.block.out(slot).copy(), planned, age_ms)
        with self._flight_changed:
            if planned.mask_set is not None:
                self._drop_set_reference(line, planned.mask_set)
            if planned.plate_slot is not None:
                line.plate_references[planned.plate_slot] -= 1
            line.free.append(slot)
            self._flight_changed.notify_all()

    def _call_on_frame(self, on_frame: Any, rendered: np.ndarray, planned: PlannedFrame, age_ms: float | None) -> None:
        timings = {} if age_ms is None else {"total": age_ms}
        try:
            on_frame(_compose(planned.frame, rendered, planned.detected, planned.general, timings))
        except Exception as exc:
            self._log_once("on_frame", exc)

    def _watch(self) -> None:
        if self._state not in (_RUNNING, _BOOTING, _FALLBACK):
            return
        moment = time.perf_counter()
        if moment - self._last_watch > STALL_S:  # slow is not hung, and neither is a machine that stood still (sleep, for one)
            self._grace_until = moment + STALL_S
        self._last_watch = moment
        if moment < self._grace_until:  # the others need a moment to run again and say so
            self._last_out = moment
            return
        sender = self._sender
        if self._state == _FALLBACK:
            if sender is not None and sender.alive is False and not self._sender_failed:
                self._lose_sender_in_fallback()
            return
        bad: str | None = None
        line = self._line
        if line is not None:
            for child in line.stages:
                if not child.alive:
                    bad = f"{child.name} ended with {child.process.returncode}"
        if bad is None and sender is not None and not sender.alive:
            bad = f"the sender ended with {sender.process.returncode}"
            self._sender_failed = True
        if bad is None and self._sender_ready.is_set() and not self._sender_beats():
            bad = "the sender does not answer"
            self._sender_failed = True
        if bad is None and line is not None and self._state == _RUNNING:
            with self._flight_lock:
                oldest = self._flight[0] if self._flight else None
            if oldest is not None:
                limit = FIRST_FRAMES_HANG_LIMIT_S if line.frames_out < FIRST_FRAMES else HANG_LIMIT_S
                since = moment - max(self._last_out, oldest.entered_at)
                if since > limit:
                    bad = f"nothing came out for {since:.3f} s"
        if bad is not None:
            self._fall_back(bad)
        elif self._state == _RUNNING:
            self._check_load(moment)

    def _check_load(self, moment: float) -> None:
        """Too little CPU shows as trouble second after second, where a stall of the machine is over in a moment. Then the
        other programs get their share again: a call app at normal priority starves behind this at above normal. And if even
        that leaves a third of the frames lost or repeated, for as long as this looks, one process does better."""
        if moment - self._load_at < 1.0:
            return
        if moment - self._running_since > RECOVER_FORGET_S:
            self._recover_tries = 0
        session = self._session
        if session is None or not self._few_cores:
            return
        first = self._load_at == 0.0
        self._load_at = moment
        seen = (int(session.flags[shm.SESSION_SLIPS]) + int(session.flags[shm.SESSION_DROPS]), self._lost_at_way_in, self._next_id)
        before, self._load_seen = self._load_seen, seen
        if first:
            return
        trouble = seen[0] - before[0] + seen[1] - before[1]
        self._load_seconds.append((trouble, seen[1] - before[1] + seen[2] - before[2]))
        self._calm_seconds = 0 if trouble else self._calm_seconds + 1
        if self._polite:
            if self._calm_seconds >= POLITE_CALM_S:
                self._be_polite(False)
        elif sum(1 for bad, _ in self._load_seconds if bad) >= POLITE_BAD_SECONDS:
            self._be_polite(True)
        if len(self._load_seconds) == LOAD_SECONDS and all(n and bad >= n * OVERLOAD_SHARE for bad, n in self._load_seconds):
            self._fall_back("too little CPU", overload=True)

    def _be_polite(self, polite: bool) -> None:
        logger.warning("%s", "too little CPU: no priority above the other programs" if polite else "enough CPU again: the priority returns")
        self._polite = polite
        self._events.append(("polite", polite, shm.now()))
        (shm.normal_process if polite else shm.above_normal_process)()
        session, line = self._session, self._line
        if session is not None:
            session.flags[shm.SESSION_POLITE] = 1.0 if polite else 0.0
        if line is not None:
            line.block.flags[shm.LINE_POLITE] = 1.0 if polite else 0.0

    def _sender_beats(self) -> bool:
        session = self._session
        if session is None:
            return True
        beat = session.flags[shm.SESSION_HEART]
        return beat == 0.0 or shm.now() - beat <= HANG_LIMIT_S

    # ---- when something goes wrong ----------------------------------------

    def _fall_back(self, why: str, *, overload: bool = False) -> None:
        """The stages are given up. Frames in flight are drawn again here, in order, with what they were planned with;
        every later frame is drawn by the output thread, in process. A failure is tried again after a while; too little CPU
        is not, as the stages would only take it from the frames again."""
        with self._state_lock:
            if self._state not in (_RUNNING, _BOOTING):
                return
            self._failed.clear()
            self._state = _FAILING
        try:
            self._give_up_the_stages(why)
            if overload:
                self._recover_at = None
                self._message = OVERLOAD_MESSAGE
                if not self._polite:
                    self._be_polite(True)
            else:
                self._plan_recovery()
        finally:
            with self._state_lock:
                if self._state == _FAILING:
                    self._state = _FALLBACK
            self._failed.set()

    def _give_up_the_stages(self, why: str) -> None:
        began = time.perf_counter()
        logger.warning("the parallel pipeline stops (%s) and one process goes on", why)
        self._events.append(("fallback", why, shm.now()))
        self._message = FALLBACK_MESSAGE
        with self._flight_changed:
            self._flight_changed.notify_all()
        with self._way_in_lock:  # a frame that was entering is now in flight, and nothing enters any more
            line, self._line = self._line, None
        if line is not None:
            line.kill()
        self._sender_ready.set()
        sender_lives = self._sender is not None and self._sender.alive and not self._sender_failed and self._flush_sender()
        if not sender_lives:
            self._kill_sender()
        pipeline = self._pipeline
        with self._flight_lock:
            frames = list(self._flight)
            self._flight.clear()
        last_written = self._last_written_id()
        pipeline._reset_effects()
        if not sender_lives:
            self._open_local_camera_after_failure()
        drawn = 0
        for planned in frames:
            if planned.identity <= last_written:  # the sender wrote it, only the preview has not had it yet
                if line is not None and self._on_frame is not None:
                    self._call_on_frame(self._on_frame, line.block.out(planned.slot).copy(), planned, None)
                continue
            drawn += 1
            try:
                self._draw_again(pipeline, planned)
            except Exception as exc:
                self._log_once("draw again", exc)
        if line is not None:
            line.close()
        self._events.append(("fallback done", round((time.perf_counter() - began) * 1000, 2), drawn))
        logger.info("%d frames drawn again in %.0f ms", drawn, (time.perf_counter() - began) * 1000)

    def _draw_again(self, pipeline: Any, planned: PlannedFrame) -> None:
        rendered = planned.frame if planned.steps is None else pipeline.draw_again(planned)
        output = cv2.flip(rendered, 1) if planned.general.mirror_output else rendered
        request = self._camera_request
        if request is not None:
            self._send_at_once(fit_frame(output, request[0], request[1]))
        if self._on_frame is not None:
            self._call_on_frame(self._on_frame, rendered, planned, None)

    def _flush_sender(self) -> bool:
        """The sender forgets the line and says which frame it dealt with last, so that no frame goes out twice."""
        session, wake = self._session, self._wake
        if session is None or wake is None or not self._sender_beats():
            return False
        session.flags[shm.SESSION_FLUSH] = 1.0
        wake.post()
        deadline = time.monotonic() + FLUSH_TIMEOUT_S
        while time.monotonic() < deadline:
            if session.flags[shm.SESSION_FLUSH_DONE]:
                return True
            if self._sender is None or not self._sender.alive:
                return False
            time.sleep(0.002)
        return False

    def _last_written_id(self) -> int:
        session = self._session
        return 0 if session is None else int(session.flags[shm.SESSION_LAST_ID])

    def _kill_sender(self) -> None:
        self._sender_failed = True
        sender = self._sender
        if sender is not None:
            sender.kill()
            sender.wait(0.5)

    def _lose_sender_in_fallback(self) -> None:
        """The sender died later, while frames were being drawn in process."""
        logger.warning("the sender ended while frames were drawn in process")
        self._sender_failed = True
        self._open_local_camera_after_failure()

    def _open_local_camera_after_failure(self) -> None:
        with self._camera_lock:
            request = self._camera_request
            if request is None:
                return
            try:
                self._local = self._open_local_camera(*request)
                self._vcam_problem = None
            except VirtualCameraError as exc:
                logger.warning("%s", exc)
                self._vcam_problem = VCAM_REOPEN_MESSAGE
                self._next_local_try = time.monotonic() + LOCAL_CAMERA_RETRY_S

    def _open_local_camera(self, width: int, height: int, fps: int) -> Any:
        """In this process, as ``pyvirtualcam.Camera``. A reader that holds the old queue keeps its name taken, and then
        that fails: the only way left is to write into that same queue, if it is what this code knows."""
        error: Exception | None = None
        for attempt in range(LOCAL_CAMERA_TRIES):
            if attempt:
                time.sleep(LOCAL_CAMERA_PAUSE_S)
            try:
                return _start_virtual_camera(width, height, fps)
            except VirtualCameraError as exc:
                error = exc
        try:
            queue_writer = ObsQueue(width, height, fps)
        except (OSError, QueueRefused) as exc:
            logger.warning("no rescue through the old queue: %s", exc)
            raise VirtualCameraError(str(error)) from exc
        logger.warning("the virtual camera goes on through the queue a reader still holds")
        return queue_writer

    def _close_local(self) -> None:
        with self._send_lock, self._camera_lock:  # not under a write: the rescue's mapping would go from under it
            local, self._local = self._local, None
        if local is not None:
            try:
                local.close()
            except Exception as exc:
                self._log_once("virtual camera", exc)

    # ---- the virtual camera (output thread, and the collector after a failure) ----

    def _sender_usable(self) -> bool:
        return self._sender is not None and not self._sender_failed and self._sender.alive

    def virtual_camera(self, *, width: int, height: int, fps: int) -> _EngineCamera:
        """What ``CameraRunner`` opens in place of ``VirtualCameraOutput``."""
        with self._send_lock, self._camera_lock:
            if self._sender_usable() or self._state == _BOOTING:
                if not self._sender_ready.is_set() or self._sender is None:
                    raise VirtualCameraNotReady(_RESTARTING_MESSAGE)
                try:
                    reply = self._sender.request({"op": "open", "width": width, "height": height, "fps": fps})
                except OSError as exc:
                    raise VirtualCameraNotReady(_RESTARTING_MESSAGE) from exc
                if not reply.get("ok"):
                    raise VirtualCameraError(virtual_camera_error_message(str(reply.get("error", ""))))
                device = str(reply["device"])
            else:
                if self._local is not None:
                    self._local.close()
                self._local = self._open_local_camera(width, height, fps)
                device = str(self._local.device)
            self._camera_request = (width, height, fps)
            self._vcam_problem = None
        return _EngineCamera(self, width, height, device)

    def _close_camera(self) -> None:
        with self._camera_lock:
            self._camera_request = None
            self._vcam_problem = None
            sender = self._sender
            usable = self._sender_usable()
        self._close_local()
        if usable and sender is not None:
            try:
                sender.request({"op": "close"}, timeout=1.0)
            except OSError as exc:
                logger.debug("closing the virtual camera: %s", exc)

    def _send_at_once(self, frame: np.ndarray) -> None:
        """Off the grid: a placeholder, or a frame drawn in the main process."""
        with self._send_lock:
            local = self._local
            if local is not None:
                try:
                    local.send(frame)
                except Exception as exc:
                    self._log_once("virtual camera", exc)
                    self._close_local()
                    self._vcam_problem = VCAM_REOPEN_MESSAGE
                return
            session, wake = self._session, self._wake
            if self._sender_usable() and session is not None and wake is not None:
                if session.can_put_now():
                    session.put_now(frame)
                    wake.post()
                return
            if self._sender_failed and time.monotonic() >= self._next_local_try:
                self._next_local_try = time.monotonic() + LOCAL_CAMERA_RETRY_S
                self._open_local_camera_after_failure()

    # ---- what the status line says ----------------------------------------

    def report(self) -> EngineReport | None:
        """None while the engine is not running."""
        if not self.running:
            return None
        session = self._session
        slips = drops = 0
        if session is not None and self._sender_usable():
            slips, drops = int(session.flags[shm.SESSION_SLIPS]), int(session.flags[shm.SESSION_DROPS])
        lost = drops + self._lost_at_way_in
        moment = time.monotonic()
        if session is not None and self._state == _RUNNING and self._camera_request is not None:
            delay = float(session.flags[shm.SESSION_DELAY_MS])
            if delay:
                self._delays_ms.append(delay)
        if lost != self._last_loss_seen:
            self._last_loss_seen = lost
            self._last_loss_at = moment
        message = self._message
        if message is None and self._last_loss_at and moment - self._last_loss_at < SLOW_MESSAGE_S:
            message = SLOW_MESSAGE
        drawing_here = self._state != _RUNNING
        ages = list(self._ages_ms)
        return EngineReport(
            fps=self._current_fps(moment),
            processing_ms=None if drawing_here or not ages else statistics.median(ages),
            processing_error=message,
            virtual_camera_error=self._vcam_problem,
            lost_frames=lost,
            slips=slips,
            delay_ms=None if drawing_here or not self._delays_ms else statistics.median(self._delays_ms),
        )

    def note_skipped(self, count: int) -> None:
        """The camera's frames that the output thread never got to, as they were overwritten while it waited. Not at the
        start of a line: the handover from drawing in process, and the stages' first frames, are slow by nature."""
        line = self._line
        if self._state == _RUNNING and line is not None and line.frames_out >= FIRST_FRAMES:
            self._lost_at_way_in += count

    def _current_fps(self, moment: float) -> float | None:
        """From the sender's writes. Without a virtual camera: what came back for the preview. Only while the stages draw."""
        session = self._session
        if session is None or not self._sender_usable():
            return None
        writes = int(session.flags[shm.SESSION_WRITES])
        count = writes if self._camera_request is not None else self._returned
        sample = self._fps_sample
        if sample is None or sample[2] > count or sample[1] != self._camera_key():
            self._fps_sample = (moment, self._camera_key(), count)
            return self._fps
        if moment - sample[0] >= _FPS_SAMPLE_S:
            self._fps = (count - sample[2]) / (moment - sample[0])
            self._fps_sample = (moment, sample[1], count)
        return self._fps

    def _camera_key(self) -> float:
        return 1.0 if self._camera_request is not None else 0.0

    # ---- small things -----------------------------------------------------

    def _note_stage_status(self, message: dict[str, Any]) -> None:
        if "neural_status" in message:
            self._stage_status = message["neural_status"]
        if "neural_wants_masks" in message:
            self._stage_wants_masks = bool(message["neural_wants_masks"])

    def _log_once(self, source: str, exc: BaseException) -> None:
        key = f"{source}: {type(exc).__name__}: {exc}"
        if key not in self._logged and len(self._logged) < 64:
            self._logged.add(key)
            logger.error("%s", key, exc_info=exc)

    def _write_main_log(self) -> None:
        if not self._log_dir:
            return
        line = self._line
        try:
            with open(os.path.join(self._log_dir, "main.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "events": self._events,
                        "start_ms": self._ready_after_ms,
                        "stages": self._stage_plan,
                        "log": self._log,
                        "cpu_s": time.process_time() - self._cpu_at_start,
                        "wall_s": time.perf_counter() - self._started_at,
                        "memory": shm.memory_mb(),
                        "layout_total_mb": 0.0 if line is None else line.layout.total / 1e6,
                        "lost_at_way_in": self._lost_at_way_in,
                        "send_log": [],
                    },
                    handle,
                )
        except OSError as exc:
            logger.warning("could not write the log: %s", exc)
