"""Started by tests as the worker process: the real protocol, shared memory and pipes, with a fake engine.

``FAKE_WORKER`` says what it does. No ONNX package is needed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from makeup_cam.neural import worker  # noqa: E402

MODE = os.environ.get("FAKE_WORKER", "normal")
FAIL_AT = int(os.environ.get("FAKE_WORKER_AT", "3")) + worker.WARMUP_RUNS  # counts the warm-up runs too


class FakeEngine:
    def __init__(self) -> None:
        self.runs = 0

    def run(self, tensor: np.ndarray) -> np.ndarray:
        self.runs += 1
        if MODE == "noisy":
            print("EP Error: text on stdout, as ORT's wrapper prints it")
            sys.stdout.flush()
            os.write(1, b"more text, written straight to the descriptor\n")
        if self.runs == FAIL_AT:
            if MODE == "device_error":
                b"\xff".decode("utf-8")  # how a removed device looks on a Japanese Windows
            if MODE == "crash":
                os._exit(3)
            if MODE == "hang":
                time.sleep(3600.0)
        return tensor * 0.1


def make_engine(args: object) -> FakeEngine:
    if MODE == "not_dml":
        raise worker.EngineError(worker.CODE_NOT_DML, "session is on the CPU")
    if MODE == "load_error":
        raise worker.EngineError(worker.CODE_LOAD_ERROR, "no such model")
    if MODE == "die_at_start":
        os._exit(5)
    if MODE == "hang_at_start":
        time.sleep(3600.0)
    return FakeEngine()


if __name__ == "__main__":
    worker.main(engine_factory=make_engine)
