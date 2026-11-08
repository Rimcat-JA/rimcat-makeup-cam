"""What runs the model for the stage: the ``Backend`` protocol and the worker-process backend.

Threads (``docs/dev/contract-plan-b.md`` section 7.2). ``submit`` and ``latest`` are called on the camera thread and
never wait. A backend's reader thread blocks on the worker and publishes each result by swapping one reference. Results are
clamped, windowed and smoothed by ``delta_filter.DeltaFilter``: in the worker process, for ``ProcessBackend``, where that work
shares no GIL with the classic effects. ``start``, ``kill`` and ``close`` run on the stage's control thread.
"""

from __future__ import annotations

import logging
import math
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from . import worker
from .delta_filter import DeltaFilter
from .model_files import ModelSpec

logger = logging.getLogger(__name__)

BAD_RESULTS_IN_A_ROW = 3
START_TIMEOUT_S = 20.0  # the budget is 3 s; a driver's first shader compile may take far longer than a later start
_RTT_KEPT = 4096  # results kept for the statistics
_MESSAGE = struct.Struct("<q")
# The package's parent folder: "python -m makeup_cam.neural.worker" must find the package from any working folder.
_PACKAGE_PARENT = Path(__file__).resolve().parents[2]


class BackendError(Exception):
    """The backend could not start. ``code`` is a ``worker.CODE_*``, or None when the process just ended."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Result:
    """One finished job, ready to be pasted. ``delta`` is S x S x 3 float32 and must not be written to."""

    tag: int  # what ``submit`` was told: the stage's frame number
    delta: np.ndarray


class Backend(Protocol):
    """Runs the model; a fake takes its place in tests and where no ONNX package exists."""

    size: int
    busy: bool  # a job is in flight, or the backend is not serving

    def start(self) -> None:
        """Blocks until ready (control thread). Raises ``BackendError``."""

    def submit(self, crop: np.ndarray, tag: int) -> bool:
        """Send one S x S x 3 crop if no job is in flight. Never waits. False: nothing was sent."""

    def latest(self) -> Result | None:
        """The newest good result. Never waits."""

    def close(self) -> None:
        """Stop for good. May take a moment: never call it on the camera thread."""

    # What the stage's control thread and health need; ``FilteredBackend`` provides all of it.
    failure: tuple[str, int | None] | None
    good_results: int
    rejected_results: int
    slow_results: deque[float]

    def reset(self) -> None: ...
    def in_flight_since(self) -> float | None: ...
    def kill(self) -> None: ...
    def wait_for(self, tag: int, timeout: float) -> bool: ...
    def wait_for_in_flight(self, timeout: float) -> bool: ...
    def drill(self, code: int) -> None: ...


class _FromWorker:
    """Where the worker has filtered already: the copy out of the shared block, and a last check on what it holds."""

    def reset(self) -> None:
        """The worker is told by ``ProcessBackend.submit``."""

    def __call__(self, filtered: np.ndarray) -> np.ndarray | None:
        delta = cv2.copyTo(filtered, None)  # not numpy: at 512 this is 3 MB, and the classic effects wait for the GIL meanwhile
        return delta if all(math.isfinite(total) for total in cv2.sumElems(delta)[:3]) else None


class FilteredBackend:
    """Everything a backend does except reaching the model: filtering, publishing, statistics."""

    def __init__(self, size: int, max_abs_delta: float, temporal_alpha: float, clock: Callable[[], float] = time.perf_counter) -> None:
        self.size = size
        self.failure: tuple[str, int | None] | None = None  # (for the log, code): the control thread acts on it
        self.good_results = 0
        self.rejected_results = 0
        self.rtt_ms: deque[float] = deque(maxlen=_RTT_KEPT)
        self.slow_results: deque[float] = deque(maxlen=_RTT_KEPT)  # when a round trip took over a frame
        self._clock = clock
        self._filter = DeltaFilter(size, max_abs_delta, temporal_alpha)
        self._latest: Result | None = None
        self._bad_in_a_row = 0
        self._reset_from_tag = -1  # results of older jobs are dropped
        self._reset_pending = False
        self._newest_tag = threading.Condition()
        self._newest_seen = -1

    @property
    def busy(self) -> bool:
        return self.in_flight_since() is not None

    def in_flight_since(self) -> float | None:
        raise NotImplementedError

    def latest(self) -> Result | None:
        return self._latest

    def reset(self) -> None:
        """Drop the published delta and the smoothing (camera thread: only flags for the reader)."""
        self._latest = None
        self._reset_from_tag = self._last_tag_sent() + 1
        self._reset_pending = True

    def wait_for(self, tag: int, timeout: float) -> bool:
        """Offline use only (``sync``): wait for the result of the job with this tag, or anything newer."""
        deadline = time.monotonic() + timeout
        with self._newest_tag:
            while self._newest_seen < tag and self.failure is None:
                left = deadline - time.monotonic()
                if left <= 0.0:
                    return False
                self._newest_tag.wait(left)
            return self._newest_seen >= tag

    def wait_for_in_flight(self, timeout: float) -> bool:
        """Wait for the result of the last job sent. Not for the camera thread."""
        return self.wait_for(self._last_tag_sent(), timeout)

    def _last_tag_sent(self) -> int:
        raise NotImplementedError

    def _finish_job(self, tag: int, raw: np.ndarray | None, rtt_ms: float, now: float) -> None:
        """Reader thread: one job is done; ``raw`` may be overwritten after this returns. None: the worker rejected it."""
        self.rtt_ms.append(rtt_ms)
        if rtt_ms > 1000.0 / 30.0:
            self.slow_results.append(now)
        if self._reset_pending:
            self._reset_pending = False
            self._filter.reset()
        if tag >= self._reset_from_tag:
            delta = None if raw is None else self._filter(raw)
            if delta is None:
                self.rejected_results += 1
                self._bad_in_a_row += 1
                if self._bad_in_a_row >= BAD_RESULTS_IN_A_ROW:
                    self._set_failure("bad output, several in a row", None)
            else:
                self._bad_in_a_row = 0
                self.good_results += 1
                delta.setflags(write=False)
                self._latest = Result(tag, delta)
        with self._newest_tag:
            self._newest_seen = tag
            self._newest_tag.notify_all()

    def _set_failure(self, why: str, code: int | None) -> None:
        if self.failure is None:
            self.failure = (why, code)
        with self._newest_tag:
            self._newest_tag.notify_all()


class ProcessBackend(FilteredBackend):
    """Runs ``makeup_cam.neural.worker`` in its own process (contract section 7.1)."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        provider: str = "dml",
        worker_command: Sequence[str] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        super().__init__(spec.size, spec.max_abs_delta, spec.temporal_alpha, clock)
        self._filter = _FromWorker()
        self._tell_worker_to_reset = False
        self._spec = spec
        self._provider = provider
        self._command = list(worker_command) if worker_command else [sys.executable, "-m", "makeup_cam.neural.worker"]
        self._process: subprocess.Popen[bytes] | None = None
        self._block: shared_memory.SharedMemory | None = None
        self._input: np.ndarray | None = None
        self._planes: list[np.ndarray] = []
        self._output: np.ndarray | None = None
        self._ready = threading.Event()
        self._closing = False
        self._serving = False
        self._in_flight = False
        self._sent_at = 0.0
        self._job = 0
        self._tag = -1

    # ---- control thread -------------------------------------------------------------------------

    def start(self) -> None:
        size = self.size
        block_size = 2 * worker.tensor_bytes(size)
        self._block = shared_memory.SharedMemory(create=True, size=block_size)
        shape = (1, 3, size, size)
        self._input = np.ndarray(shape, np.float32, self._block.buf)
        self._planes = [self._input[0, 0], self._input[0, 1], self._input[0, 2]]
        self._output = np.ndarray((size, size, 3), np.float32, self._block.buf, offset=worker.tensor_bytes(size))
        self._input.fill(0.0)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        arguments = [
            "--shm", self._block.name, "--size", str(size), "--model", str(self._spec.file), "--provider", self._provider,
            "--max-abs-delta", repr(self._spec.max_abs_delta), "--temporal-alpha", repr(self._spec.temporal_alpha),
        ]
        if self._closing:
            self._release_block()
            raise BackendError("closed while starting")
        self._process = subprocess.Popen(
            [*self._command, *arguments],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=_PACKAGE_PARENT,
            creationflags=flags,
        )
        threading.Thread(target=self._read_worker, args=(self._process,), name="neural-reader", daemon=True).start()
        threading.Thread(target=self._log_stderr, args=(self._process,), name="neural-stderr", daemon=True).start()
        if not self._ready.wait(START_TIMEOUT_S):
            self.kill()
            raise BackendError("the worker did not get ready in time")
        if self.failure is not None:
            self.kill()
            raise BackendError(self.failure[0], self.failure[1])
        self._serving = True

    def in_flight_since(self) -> float | None:
        return self._sent_at if self._in_flight else None

    def kill(self) -> None:
        """Stop the worker at once, from any state. The block stays until ``close``."""
        self._serving = False
        process = self._process
        if process is None:
            return
        try:
            process.kill()
        except OSError:
            pass
        self._close_stdin(process)
        self._ready.set()

    def close(self) -> None:
        """Close stdin, kill after 200 ms (contract section 7.4). Never call this on the camera thread."""
        self._closing = True
        self._serving = False
        process = self._process
        if process is not None:
            self._close_stdin(process)
            try:
                process.wait(0.2)
            except subprocess.TimeoutExpired:
                self.kill()
            try:
                process.wait(1.0)
            except subprocess.TimeoutExpired:
                logger.warning("the neural worker did not end")
        self._ready.set()
        self._release_block()

    def drill(self, code: int) -> None:
        """Make the next job fail on purpose (``worker.DRILL_*``). For the evaluation's fade drill."""
        process = self._process
        if process is not None and process.stdin is not None:
            process.stdin.write(_MESSAGE.pack(code))

    @staticmethod
    def _close_stdin(process: subprocess.Popen[bytes]) -> None:
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass

    def _release_block(self) -> None:
        self._input = self._output = None
        self._planes = []
        block, self._block = self._block, None
        if block is None:
            return
        # The camera or reader thread may hold a view for a moment more; the block is small and this is rare.
        for _ in range(20):
            try:
                block.close()
                break
            except BufferError:
                time.sleep(0.01)
        else:
            _unreleased.append(block)
            return
        try:
            block.unlink()
        except OSError:
            pass

    # ---- camera thread --------------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._in_flight or not self._serving

    def submit(self, crop: np.ndarray, tag: int) -> bool:
        """At most one job is in flight, so this write cannot block and the input is never torn."""
        if self._in_flight or not self._serving:
            return False
        tensor, process = self._input, self._process
        if tensor is None or process is None or process.stdin is None:
            return False
        cv2.split(crop, self._planes)
        self._job += 1
        self._tag = tag
        self._sent_at = self._clock()
        self._in_flight = True
        try:
            if self._tell_worker_to_reset:
                self._tell_worker_to_reset = False
                process.stdin.write(_MESSAGE.pack(worker.RESET))
            process.stdin.write(_MESSAGE.pack(self._job))
        except (OSError, ValueError):  # the worker is gone; the reader thread reports it
            self._in_flight = False
            return False
        return True

    def _last_tag_sent(self) -> int:
        return self._tag

    def reset(self) -> None:
        super().reset()
        self._tell_worker_to_reset = True

    # ---- reader thread --------------------------------------------------------------------------

    def _read_worker(self, process: subprocess.Popen[bytes]) -> None:
        stream = process.stdout
        while True:
            message = _read_message(stream)
            if message is None:
                self._set_failure("the worker process ended", None)
                self._ready.set()
                return
            if message == worker.CODE_READY:
                self._ready.set()
            elif message < 0:
                self._set_failure(f"the worker reported code {message}", message)
                self._ready.set()
            else:
                self._on_job_done(message)

    def _on_job_done(self, message: int) -> None:
        output = self._output
        job = message & ~worker.REJECTED
        if job != self._job or output is None:
            self._set_failure(f"unexpected message {message} from the worker", None)
            return
        now = self._clock()
        try:
            self._finish_job(self._tag, None if message & worker.REJECTED else output, (now - self._sent_at) * 1000.0, now)
        except Exception as exc:  # a bug here must cost a restart, not the reader thread
            self._set_failure(f"result handling failed: {exc!r}", None)
        self._in_flight = False  # after the copy: only now may the camera thread send the next job

    @staticmethod
    def _log_stderr(process: subprocess.Popen[bytes]) -> None:
        for line in process.stderr:
            logger.info("neural worker: %s", line.decode("utf-8", "replace").rstrip())


_unreleased: list[shared_memory.SharedMemory] = []  # see _release_block


def _read_message(stream) -> int | None:
    data = b""
    try:
        while len(data) < _MESSAGE.size:
            part = stream.read(_MESSAGE.size - len(data))
            if not part:
                return None
            data += part
    except (OSError, ValueError):
        return None
    return _MESSAGE.unpack(data)[0]
