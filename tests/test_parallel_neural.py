"""The learned finish in stage 2, with a real worker and a tiny model built here (contract sections 7 and 12.4).

Needs onnxruntime with DirectML, so it runs in ``.venv-neural`` and is skipped elsewhere. No weights on disk beyond the test's folder.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from neural_fakes import write_model_folder

from makeup_cam.neural import model_files, stage
from makeup_cam.pipeline import MakeupPipeline, ProcessResult
from makeup_cam.presets import load_preset

if sys.platform != "win32":
    pytest.skip("the pipeline of processes runs on Windows only", allow_module_level=True)

ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")
if "DmlExecutionProvider" not in ort.get_available_providers():
    pytest.skip("DirectML is not available", allow_module_level=True)

from parallel_helpers import FRAME_MS, StableMasks, StableTracker, descendants, digest, make_frames, process_alive, wait_until  # noqa: E402

from makeup_cam.parallel.engine import ParallelEngine  # noqa: E402

SIZE = 256


def _zero_delta_model(path: Path) -> None:
    """A convolution with zero weights: the delta is exactly 0, so the picture must stay exactly the classic one."""
    from onnx import TensorProto, helper, numpy_helper

    weights = numpy_helper.from_array(np.zeros((3, 3, 3, 3), np.float32), "weights")
    node = helper.make_node("Conv", ["input", "weights"], ["delta"], pads=[1, 1, 1, 1], kernel_shape=[3, 3])
    graph = helper.make_graph(
        [node],
        "zero",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("delta", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, str(path))


@pytest.fixture
def active_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A model that passes the gate of the real app: a stage takes no option to let an unevaluated one in."""
    root = tmp_path / "models"
    folder = root / "zero-1.0.0"
    folder.mkdir(parents=True)
    _zero_delta_model(folder / "model.onnx")
    content = (folder / "model.onnx").read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    report = json.dumps({"verdict": "pass", "eligible_for_calls": True, "model": {"sha256": sha}}).encode()
    (folder / model_files.EVALUATION_JSON).write_bytes(report)
    write_model_folder(
        folder,
        name="zero",
        version="1.0.0",
        size=SIZE,
        content=content,
        stand_in=False,
        evaluation={"verdict": "pass", "report": model_files.EVALUATION_JSON, "report_sha256": hashlib.sha256(report).hexdigest(), "model_sha256": sha, "date": "2026-11-07"},
    )
    monkeypatch.setenv(model_files.NEURAL_DIR_ENV, str(root))
    model_files.set_active_model("zero-1.0.0", root)
    model_files.check_gate(model_files.load_spec(folder))
    return folder


def _brighter_model(path: Path) -> None:
    """delta = 0.08 x input: a real change, whose weight comes from the masks."""
    from onnx import TensorProto, helper, numpy_helper

    weights = np.zeros((3, 3, 3, 3), np.float32)
    for channel in range(3):
        weights[channel, channel, 1, 1] = 0.08
    node = helper.make_node("Conv", ["input", "weights"], ["delta"], pads=[1, 1, 1, 1], kernel_shape=[3, 3])
    graph = helper.make_graph(
        [node],
        "brighter",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("delta", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [numpy_helper.from_array(weights, "weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, str(path))


def _settings(enabled: bool) -> object:
    settings = load_preset("natural")
    settings.neural.enabled = enabled
    settings.neural.strength = 1.0
    return settings


def test_the_finish_runs_in_stage_2_and_its_worker_goes_with_it(face_frame_720p: np.ndarray, active_model: Path) -> None:
    pipeline = MakeupPipeline(_settings(True), tracker=StableTracker(), segmenter=StableMasks())
    engine = ParallelEngine()
    pipeline.attach_engine(engine)
    engine.slot_wait_s = 30.0
    done: list[ProcessResult] = []
    engine.start((720, 1280), done.append)
    try:
        assert engine.wait_ready(60.0)
        frames = make_frames(face_frame_720p, 8)
        pids = engine.child_pids
        deadline = time.monotonic() + 45.0
        count = 0
        while pipeline.finishing_status() != stage.STATUS_RUNNING and time.monotonic() < deadline:
            pipeline.process(frames[count % 8].copy(), count * FRAME_MS)
            count += 1
            time.sleep(1 / 30)
        assert pipeline.finishing_status() == stage.STATUS_RUNNING
        here = next(e for e in pipeline._effects if e.name == "neural")
        assert here.backend is None, "the model must not run in the main process as well"
        workers = descendants(pids["stage2"])
        assert workers, "its worker is below stage 2"
        assert all(not descendants(pids[name]) for name in ("stage0", "stage1", "sender"))
    finally:
        pipeline.close()
    assert wait_until(lambda: not any(process_alive(pid) for pid in workers), 5.0)


def test_a_killed_stage_2_takes_its_worker_along(face_frame_720p: np.ndarray, active_model: Path) -> None:
    pipeline = MakeupPipeline(_settings(True), tracker=StableTracker(), segmenter=StableMasks())
    engine = ParallelEngine()
    pipeline.attach_engine(engine)
    engine.start((720, 1280))
    try:
        assert engine.wait_ready(60.0)
        frames = make_frames(face_frame_720p, 8)
        deadline = time.monotonic() + 45.0
        count = 0
        while pipeline.finishing_status() != stage.STATUS_RUNNING and time.monotonic() < deadline:
            pipeline.process(frames[count % 8].copy(), count * FRAME_MS)
            count += 1
            time.sleep(1 / 30)
        pid = engine.child_pids["stage2"]
        workers = descendants(pid)
        assert workers
        engine._line.stages[2].process.kill()
        assert wait_until(lambda: not any(process_alive(worker) for worker in workers), 5.0), workers
    finally:
        pipeline.close()


def test_with_a_delta_of_zero_the_picture_is_exactly_the_classic_one(face_frame_720p: np.ndarray, active_model: Path) -> None:
    """The same deltas in both: zero, whenever they arrive. So the finish, running in stage 2, changes no bit."""
    frames = make_frames(face_frame_720p, 120)
    classic = MakeupPipeline(_settings(False), tracker=StableTracker(), segmenter=StableMasks())
    reference = [digest(classic.process(frame, i * FRAME_MS).output) for i, frame in enumerate(frames)]
    classic.close()
    pipeline = MakeupPipeline(_settings(True), tracker=StableTracker(), segmenter=StableMasks())
    engine = ParallelEngine()
    pipeline.attach_engine(engine)
    engine.slot_wait_s = 30.0
    done: list[str] = []
    engine.start((720, 1280), lambda result: done.append(digest(result.output)))
    try:
        assert engine.wait_ready(60.0)
        for i, frame in enumerate(frames):
            assert pipeline.process(frame, i * FRAME_MS).in_flight
            time.sleep(1 / 30)
        assert wait_until(lambda: len(done) == len(frames), 20.0)
        assert pipeline.finishing_status() == stage.STATUS_RUNNING
    finally:
        pipeline.close()
    assert done == reference


class _CountingMasks(StableMasks):
    calls = 0

    def update(self, frame: np.ndarray, ts_ms: int, frame_index: int) -> object:
        type(self).calls += 1
        return super().update(frame, ts_ms, frame_index)


def _only_the_finish() -> object:
    settings = load_preset("natural")
    for name in settings.__dataclass_fields__:
        section = getattr(settings, name)
        if name not in ("general", "neural") and hasattr(section, "enabled"):
            section.enabled = False
    settings.neural.enabled = True
    settings.neural.strength = 1.0
    settings.general.mirror_output = False
    settings.general.mirror_preview = False
    return settings


def _run_only_the_finish(face: np.ndarray, with_engine: bool) -> tuple[int, int, str | None]:
    """Mask updates, pixels the last frame has changed, and the status."""
    _CountingMasks.calls = 0
    pipeline = MakeupPipeline(_only_the_finish(), tracker=StableTracker(), segmenter=_CountingMasks())
    done: list[ProcessResult] = []
    if with_engine:
        engine = ParallelEngine()
        pipeline.attach_engine(engine)
        engine.start((720, 1280), done.append)
        assert engine.wait_ready(60.0)
    frames = make_frames(face, 30)
    try:
        for i in range(330):
            result = pipeline.process(frames[i % 30], i * FRAME_MS)
            if not result.in_flight:
                done.append(result)
            time.sleep(1 / 30)
        time.sleep(0.5)
        last = done[-1]
        changed = int((np.abs(last.output.astype(int) - last.original_preview.astype(int)).max(axis=2) > 0).sum())
        return _CountingMasks.calls, changed, pipeline.finishing_status()
    finally:
        pipeline.close()


def test_a_finish_that_alone_uses_the_masks_gets_them_in_the_stage(face_frame_720p: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing else asks for masks here. Only the stage knows that its worker runs, and without masks the look changes."""
    root = tmp_path / "models"
    folder = root / "brighter-1.0.0"
    folder.mkdir(parents=True)
    _brighter_model(folder / "model.onnx")
    content = (folder / "model.onnx").read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    report = json.dumps({"verdict": "pass", "eligible_for_calls": True, "model": {"sha256": sha}}).encode()
    (folder / model_files.EVALUATION_JSON).write_bytes(report)
    write_model_folder(
        folder,
        name="brighter",
        version="1.0.0",
        size=SIZE,
        content=content,
        stand_in=False,
        evaluation={"verdict": "pass", "report": model_files.EVALUATION_JSON, "report_sha256": hashlib.sha256(report).hexdigest(), "model_sha256": sha, "date": "2026-11-07"},
    )
    monkeypatch.setenv(model_files.NEURAL_DIR_ENV, str(root))
    model_files.set_active_model("brighter-1.0.0", root)
    model_files.check_gate(model_files.load_spec(folder))
    here_calls, here_changed, here_status = _run_only_the_finish(face_frame_720p, False)
    there_calls, there_changed, there_status = _run_only_the_finish(face_frame_720p, True)
    assert here_status == there_status == stage.STATUS_RUNNING
    assert here_calls > 100
    assert there_calls > 100, "the masks are made in the main process, as the stage has asked"
    assert abs(there_changed - here_changed) <= here_changed * 0.25, (here_changed, there_changed)


def test_a_section_that_stays_the_same_gives_no_new_try_in_the_stage() -> None:
    """Each frame brings a new copy of the section to the stage. A failed worker must not be retried at every pause."""
    import pickle

    from makeup_cam.parallel.stage_main import keep_section

    effect = stage.NeuralFinishEffect(control_thread=False)
    settings = _settings(True)
    sections: dict[str, object] = {}
    effect._note_call(keep_section(sections, "neural", pickle.loads(pickle.dumps(settings.neural))), 1.0, 100.0)
    effect._rearm = False
    effect._note_call(keep_section(sections, "neural", pickle.loads(pickle.dumps(settings.neural))), 1.0, 100.3)
    assert not effect._rearm
    effect.close()
