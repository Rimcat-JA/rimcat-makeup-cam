"""Shared by the Plan B runtime tests: a model folder without a real model, and a backend without a worker."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from makeup_cam.neural.backend import BackendError, FilteredBackend
from makeup_cam.neural.model_files import ModelSpec


def write_model_folder(folder: Path, *, size: int = 256, content: bytes = b"not really a model", **changes: Any) -> Path:
    """A folder with ``model.onnx`` and ``model.json``: a stand-in that has not been evaluated."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.onnx").write_bytes(content)
    meta: dict[str, Any] = {
        "schema": 1,
        "name": "fake",
        "version": "0.0.0",
        "file": "model.onnx",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "opset": 20,
        "stand_in": True,
        "input": {"name": "input", "size": size, "layout": "NCHW", "colors": "BGR", "range": [0.0, 1.0], "dtype": "float32"},
        "output": {
            "name": "delta",
            "kind": "delta",
            "layout": "NCHW",
            "colors": "BGR",
            "range": [-1.0, 1.0],
            "dtype": "float32",
            "max_abs_delta": 0.25,
            "temporal_alpha": 1.0,
        },
        "crop": {"definition": 1},
        "may_edit": {"eyes": False},
        "training_data": {"kind": "none", "input_preset": "natural", "note": "test"},
        "evaluation": {"verdict": "none", "report": None, "report_sha256": None, "model_sha256": None, "date": None},
    }
    meta.update(changes)
    (folder / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder


def uniform_delta(size: int, value: float = 0.04) -> np.ndarray:
    return np.full((3, size, size), value, np.float32)


class FakeBackend(FilteredBackend):
    """Answers jobs at once, or when told to (``hold = True``). Filtering and publishing are the real ones."""

    def __init__(self, spec: ModelSpec, clock: Callable[[], float], model: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        super().__init__(spec.size, spec.max_abs_delta, spec.temporal_alpha, clock)
        self.model = model or (lambda crop: uniform_delta(crop.shape[0]))
        self.hold = False
        self.start_error: BackendError | None = None
        self.started = 0
        self.killed = 0
        self.closed = 0
        self.jobs: list[int] = []
        self.pending: tuple[np.ndarray, int, float] | None = None
        self.sent_tag = -1

    def start(self) -> None:
        self.started += 1
        if self.start_error is not None:
            raise self.start_error

    def submit(self, crop: np.ndarray, tag: int) -> bool:
        if self.pending is not None:
            return False
        self.jobs.append(tag)
        self.sent_tag = tag
        self.pending = (crop.copy(), tag, self._clock())
        if not self.hold:
            self.finish()
        return True

    def finish(self, raw: np.ndarray | None = None) -> None:
        assert self.pending is not None
        crop, tag, sent_at = self.pending
        now = self._clock()
        self._finish_job(tag, self.model(crop) if raw is None else raw, (now - sent_at) * 1000.0, now)
        self.pending = None

    def in_flight_since(self) -> float | None:
        return None if self.pending is None else self.pending[2]

    def _last_tag_sent(self) -> int:
        return self.sent_tag

    def kill(self) -> None:
        self.killed += 1

    def close(self) -> None:
        self.closed += 1

    def drill(self, code: int) -> None:
        raise AssertionError("not used with this fake")
