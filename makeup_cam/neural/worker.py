"""The inference process of the learned finishing stage (``docs/dev/contract-plan-b.md`` section 7.1).

Started by ``backend.ProcessBackend`` as ``python -m makeup_cam.neural.worker``. It runs in its own process
because after a device removal a new session in the same process never gets the GPU back, and silently lands on
the CPU; a new process gets it at once. A crash or hang here costs the look, not the call.

This module imports only numpy at the top, and onnxruntime and the result filter (cv2) inside ``main``, so the parent can
import its constants cheaply and the worker starts without Qt or MediaPipe.

Protocol. One shared-memory block holds the input tensor, ``[1, 3, S, S]`` float32 NCHW BGR, and after it the filtered result,
S x S x 3 float32 BGR (``delta_filter.DeltaFilter``: checked, clamped, windowed, smoothed; here it shares no GIL with the
camera thread). The pipes carry signed little-endian 8-byte integers:

- parent to worker: a positive job number means "run on the input tensor now". At most one job is in flight, so
  the input is never torn and the 8-byte writes cannot block. ``RESET`` drops the smoothing before the next job. ``DRILL_*`` make
  the next job fail on purpose (the evaluation's fade drill; each ``DRILL_BAD_OUTPUT`` spoils one more job). EOF means the parent
  is gone: the worker exits at once, even in the middle of a run, and also while the session is being made.
- worker to parent: the job number when the result is ready, with ``REJECTED`` added when the filter refused the output (the
  block then holds nothing to use), or a negative ``CODE_*``. Every code but ``CODE_READY`` is followed by the worker's exit.

ORT's Python wrapper prints ``EP Error`` to stdout, so the original stdout is kept for the protocol and everything
else that writes to stdout goes to stderr, which the parent logs.
"""

from __future__ import annotations

import argparse
import os
import queue
import struct
import sys
import threading
import traceback
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

CODE_READY = -1
CODE_RUN_ERROR = -2  # any exception in run (after a device removal, on a Japanese Windows: UnicodeDecodeError)
CODE_NOT_DML = -3  # the session is not on DirectML: ORT falls back to the CPU silently
CODE_LOAD_ERROR = -4  # the model file did not load
CODE_BAD_MODEL = -5  # loaded, but the input or output is not what model.json says
RESET = -103  # drop the smoothing: the next result stands on its own
REJECTED = 1 << 40  # added to a job number: this output was not acceptable
DRILL_RUN_ERROR = -101  # the next job raises
DRILL_BAD_OUTPUT = -102  # the next job writes NaN into the output tensor

WARMUP_RUNS = 5
_MESSAGE = struct.Struct("<q")


def tensor_bytes(size: int) -> int:
    """Bytes of one ``[1, 3, S, S]`` float32 tensor."""
    return 3 * size * size * 4


class Engine(Protocol):
    """Runs the model. ``OrtEngine`` in real use; tests start this module with a fake one."""

    def run(self, tensor: np.ndarray) -> np.ndarray: ...


class EngineError(Exception):
    """An engine could not be made; ``code`` says why (a ``CODE_*``)."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class OrtEngine:
    """An ONNX Runtime session on DirectML (or on the CPU, only when ``provider`` says so: offline use and tests)."""

    def __init__(self, path: str, size: int, provider: str = "dml") -> None:
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.log_severity_level = 3
            wanted = "DmlExecutionProvider" if provider == "dml" else "CPUExecutionProvider"
            session = ort.InferenceSession(path, options, providers=[wanted])
        except Exception as exc:  # ORT's types are not a stable contract, and the text may not even decode
            raise EngineError(CODE_LOAD_ERROR, f"could not load the model: {type(exc).__name__}: {exc}") from exc
        if session.get_providers()[0] != wanted:  # the only way to tell: ORT does not raise
            raise EngineError(CODE_NOT_DML, f"session is on {session.get_providers()[0]}, not {wanted}")
        _check_signature(session, size)
        self._session = session

    def run(self, tensor: np.ndarray) -> np.ndarray:
        return self._session.run(["delta"], {"input": tensor})[0]


def _check_signature(session: object, size: int) -> None:
    """The schema of ``docs/dev/contract-plan-b.md`` section 3, against what the file really has."""
    inputs, outputs = session.get_inputs(), session.get_outputs()
    wanted = [1, 3, size, size]
    problems = []
    if len(inputs) != 1 or inputs[0].name != "input" or list(inputs[0].shape) != wanted:
        problems.append(f"input {[(i.name, i.shape) for i in inputs]}")
    if len(outputs) != 1 or outputs[0].name != "delta" or list(outputs[0].shape) != wanted:
        problems.append(f"output {[(o.name, o.shape) for o in outputs]}")
    if any(item.type != "tensor(float)" for item in (*inputs, *outputs)):
        problems.append("edges must be float32")
    if problems:
        raise EngineError(CODE_BAD_MODEL, "model does not match schema 1: " + "; ".join(problems))


def _make_ort_engine(args: argparse.Namespace) -> Engine:
    return OrtEngine(args.model, args.size, args.provider)


def _send(fd: int, value: int) -> None:
    os.write(fd, _MESSAGE.pack(value))


def _read_message(fd: int) -> int | None:
    """The next integer from the pipe, or None at EOF."""
    data = b""
    try:
        while len(data) < _MESSAGE.size:
            part = os.read(fd, _MESSAGE.size - len(data))
            if not part:
                return None
            data += part
    except OSError:
        return None
    return _MESSAGE.unpack(data)[0]


def _follow_parent(fd: int, jobs: queue.SimpleQueue[int]) -> None:
    """Own thread: a run that hangs must not keep this process alive after the parent is gone."""
    while True:
        message = _read_message(fd)
        if message is None:
            os._exit(0)
        jobs.put(message)


def serve(engine: Engine, inp: np.ndarray, out: np.ndarray, jobs: queue.SimpleQueue[int], out_fd: int, result_filter: Callable[[np.ndarray], np.ndarray | None]) -> None:
    """Warm up, report ready, then run jobs until the parent is gone or a run fails. Never returns."""
    try:
        zeros = np.zeros_like(inp)
        for _ in range(WARMUP_RUNS):
            engine.run(zeros)
        _send(out_fd, CODE_READY)
        run_error = False
        bad_outputs = 0
        while True:
            message = jobs.get()
            if message == DRILL_RUN_ERROR:
                run_error = True
                continue
            if message == DRILL_BAD_OUTPUT:
                bad_outputs += 1
                continue
            if message == RESET:
                result_filter.reset()
                continue
            if run_error:
                raise RuntimeError("drill: run error")
            raw = np.array(engine.run(inp)[0])  # a copy: a fake or a session may hand out what it keeps
            if bad_outputs > 0:
                bad_outputs -= 1
                raw[0, 0, 0] = np.nan
            filtered = result_filter(raw)
            if filtered is None:
                _send(out_fd, message | REJECTED)
            else:
                out[:] = filtered
                _send(out_fd, message)
    except Exception:  # not ORT's types: see CODE_RUN_ERROR
        _report_and_exit(out_fd, CODE_RUN_ERROR)


def _report_and_exit(out_fd: int, code: int) -> None:
    traceback.print_exc(file=sys.stderr)
    try:
        _send(out_fd, code)
    except OSError:
        pass
    sys.stderr.flush()
    os._exit(1)  # no teardown: after a device loss, releasing the session may block


def main(argv: Sequence[str] | None = None, engine_factory: Callable[[argparse.Namespace], Engine] = _make_ort_engine) -> None:
    protocol_out = os.dup(1)  # the original stdout carries the protocol ...
    os.dup2(2, 1)  # ... and whatever else writes to it goes to stderr
    sys.stdout = sys.stderr
    if sys.platform == "win32":
        import msvcrt

        msvcrt.setmode(protocol_out, os.O_BINARY)  # a job number holding 0x1A or 0x0A must stay what it is
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
    parser = argparse.ArgumentParser()
    parser.add_argument("--shm", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=("dml", "cpu"), default="dml")
    parser.add_argument("--max-abs-delta", type=float, default=0.25)
    parser.add_argument("--temporal-alpha", type=float, default=0.6)
    args = parser.parse_args(argv)
    from multiprocessing import shared_memory

    block = shared_memory.SharedMemory(name=args.shm)
    if sys.platform != "win32":  # the parent owns the block: this process's tracker must not remove it
        from multiprocessing import resource_tracker

        resource_tracker.unregister(block._name, "shared_memory")  # noqa: SLF001
    shape = (1, 3, args.size, args.size)
    inp = np.ndarray(shape, np.float32, block.buf)
    out = np.ndarray((args.size, args.size, 3), np.float32, block.buf, offset=tensor_bytes(args.size))
    # Before the session is made: a driver that hangs there must not keep this process alive after the parent is gone.
    jobs: queue.SimpleQueue[int] = queue.SimpleQueue()
    threading.Thread(target=_follow_parent, args=(sys.stdin.fileno(), jobs), name="parent", daemon=True).start()
    try:
        engine = engine_factory(args)
    except EngineError as exc:
        print(exc, file=sys.stderr)
        _report_and_exit(protocol_out, exc.code)
    except Exception:
        _report_and_exit(protocol_out, CODE_LOAD_ERROR)
    import cv2

    from .delta_filter import DeltaFilter

    cv2.setNumThreads(1)  # one small array at a time: no use for a pool
    serve(engine, inp, out, jobs, protocol_out, DeltaFilter(args.size, args.max_abs_delta, args.temporal_alpha))


if __name__ == "__main__":
    main()
