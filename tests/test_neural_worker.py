"""The worker process: a real process, real shared memory and pipes. The engine is a fake, unless ORT is installed."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from neural_fakes import write_model_folder

from makeup_cam.config import NeuralSettings
from makeup_cam.effects.base import FrameContext
from makeup_cam.neural import stage as stage_module
from makeup_cam.neural import worker
from makeup_cam.neural.backend import BackendError, ProcessBackend
from makeup_cam.neural.model_files import ModelSpec, load_spec
from makeup_cam.neural.stage import NeuralFinishEffect

FAKE_WORKER = [sys.executable, str(Path(__file__).with_name("neural_worker_fake.py"))]
SIZE = 256


@pytest.fixture
def spec(tmp_path: Path) -> ModelSpec:
    return load_spec(write_model_folder(tmp_path / "model", size=SIZE))


@pytest.fixture
def backends() -> Any:
    made: list[ProcessBackend] = []
    yield made
    for backend in made:
        backend.kill()
        backend.close()


def fake_backend(spec: ModelSpec, made: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch, mode: str = "normal", at: int = 3) -> ProcessBackend:
    monkeypatch.setenv("FAKE_WORKER", mode)
    monkeypatch.setenv("FAKE_WORKER_AT", str(at))
    backend = ProcessBackend(spec, worker_command=FAKE_WORKER)
    made.append(backend)
    return backend


def crop_of(value: float) -> np.ndarray:
    return np.full((SIZE, SIZE, 3), value, np.float32)


def run_job(backend: ProcessBackend, tag: int, value: float = 0.5, timeout: float = 10.0) -> None:
    assert backend.submit(crop_of(value), tag)
    assert backend.wait_for(tag, timeout)


def until(condition: Any, timeout: float = 10.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


def test_jobs_go_there_and_back(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(spec, backends, monkeypatch)
    backend.start()
    for tag in range(1, 6):
        run_job(backend, tag, 0.1 * tag)
        result = backend.latest()
        assert result.tag == tag
        assert result.delta[128, 128, 0] == pytest.approx(0.01 * tag, abs=1e-5)  # the fake engine's output
    assert backend.good_results == 5
    assert len(backend.rtt_ms) == 5


def test_colors_arrive_in_their_planes(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(spec, backends, monkeypatch)
    backend.start()
    crop = crop_of(0.0)
    crop[..., 0], crop[..., 1], crop[..., 2] = 0.1, 0.2, 0.3
    assert backend.submit(crop, 1)
    assert backend.wait_for(1, 10.0)
    np.testing.assert_allclose(backend.latest().delta[128, 128], (0.01, 0.02, 0.03), atol=1e-5)


def test_at_most_one_job_is_in_flight(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(spec, backends, monkeypatch, "hang", at=1)
    backend.start()
    assert backend.submit(crop_of(0.5), 1)
    assert backend.in_flight_since() is not None and backend.busy
    assert not backend.submit(crop_of(0.5), 2)


def test_text_on_stdout_does_not_touch_the_protocol(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(spec, backends, monkeypatch, "noisy")
    backend.start()
    run_job(backend, 1)
    assert backend.failure is None and backend.good_results == 1


def test_a_device_error_is_reported_and_the_worker_exits(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    """On a Japanese Windows a removed device shows up as UnicodeDecodeError: types and text say nothing."""
    backend = fake_backend(spec, backends, monkeypatch, "device_error", at=2)
    backend.start()
    run_job(backend, 1)
    backend.submit(crop_of(0.5), 2)
    assert until(lambda: backend.failure is not None)
    assert backend.failure[1] == worker.CODE_RUN_ERROR
    assert until(lambda: backend._process.poll() is not None)


def test_a_crash_is_noticed(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(spec, backends, monkeypatch, "crash", at=1)
    backend.start()
    backend.submit(crop_of(0.5), 1)
    assert until(lambda: backend.failure is not None)
    assert backend.failure[1] is None


def test_submit_after_the_worker_is_gone_returns_false_without_raising(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(spec, backends, monkeypatch)
    backend.start()
    backend._process.kill()
    assert until(lambda: backend.failure is not None)
    for tag in range(5):
        backend.submit(crop_of(0.5), tag)  # may or may not reach the pipe; must never raise or block
    backend.kill()
    assert not backend.submit(crop_of(0.5), 9)


def test_a_worker_that_is_not_on_the_gpu_says_so(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(spec, backends, monkeypatch, "not_dml")
    with pytest.raises(BackendError) as error:
        backend.start()
    assert error.value.code == worker.CODE_NOT_DML


def test_a_model_that_does_not_load_says_so(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(spec, backends, monkeypatch, "load_error")
    with pytest.raises(BackendError) as error:
        backend.start()
    assert error.value.code == worker.CODE_LOAD_ERROR


def test_a_worker_that_dies_at_the_start_is_a_failure(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(spec, backends, monkeypatch, "die_at_start")
    with pytest.raises(BackendError):
        backend.start()


def test_a_worker_that_never_gets_ready_times_out(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("makeup_cam.neural.backend.START_TIMEOUT_S", 1.0)
    backend = fake_backend(spec, backends, monkeypatch, "hang_at_start")
    with pytest.raises(BackendError):
        backend.start()
    assert until(lambda: backend._process.poll() is not None)


def test_end_of_stdin_ends_the_worker_even_in_the_middle_of_a_run(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    """No orphans: this is what the worker sees when the app dies."""
    backend = fake_backend(spec, backends, monkeypatch, "hang", at=1)
    backend.start()
    backend.submit(crop_of(0.5), 1)
    time.sleep(0.1)
    backend._process.stdin.close()
    assert until(lambda: backend._process.poll() is not None)
    assert backend._process.returncode == 0


def test_close_never_takes_long_and_frees_the_block(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(spec, backends, monkeypatch, "hang", at=1)
    backend.start()
    backend.submit(crop_of(0.5), 1)
    started = time.perf_counter()
    backend.close()
    assert time.perf_counter() - started < 3.0
    assert backend._process.poll() is not None
    assert not backend.submit(crop_of(0.5), 2)


def test_drills_fail_the_next_job(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(spec, backends, monkeypatch)
    backend.start()
    run_job(backend, 1)
    backend.drill(worker.DRILL_BAD_OUTPUT)
    run_job(backend, 2)
    assert backend.rejected_results == 1
    run_job(backend, 3)
    assert backend.good_results == 2
    for _ in range(3):
        backend.drill(worker.DRILL_BAD_OUTPUT)
    for tag in (4, 5):
        run_job(backend, tag)
    assert backend.failure is None and backend.rejected_results == 3
    run_job(backend, 6)
    assert backend.failure is not None  # three in a row
    backend.kill()


def test_the_run_error_drill(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(spec, backends, monkeypatch)
    backend.start()
    run_job(backend, 1)
    backend.drill(worker.DRILL_RUN_ERROR)
    backend.submit(crop_of(0.5), 4)
    assert until(lambda: backend.failure is not None)
    assert backend.failure[1] == worker.CODE_RUN_ERROR


# ---- the whole stage with real processes: nothing ever freezes ----------------------------------------------


def _paced_frames(effect: NeuralFinishEffect, landmarks: np.ndarray, seconds: float, section: NeuralSettings) -> list[float]:
    """Frames as a camera thread would run them. Returns each apply's time in ms."""
    image = np.full((720, 1280, 3), 0.5, np.float32)
    times: list[float] = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        ctx = FrameContext(height=720, width=1280, landmarks=landmarks)
        started = time.perf_counter()
        effect.apply(image.copy(), ctx, section)
        times.append((time.perf_counter() - started) * 1000.0)
        time.sleep(0.01)
    return times


@pytest.fixture
def frontal(face_landmarks_720p: np.ndarray) -> np.ndarray:
    center = face_landmarks_720p.mean(axis=0)
    return center + (face_landmarks_720p - center) * 1.5


@pytest.mark.parametrize(("mode", "at"), [("device_error", 8), ("crash", 8), ("hang", 8)])
def test_a_failing_worker_costs_the_look_not_the_frames(
    spec: ModelSpec, frontal: np.ndarray, monkeypatch: pytest.MonkeyPatch, mode: str, at: int
) -> None:
    monkeypatch.setattr(stage_module, "RESTART_WAITS_S", (0.1, 0.1, 0.1))
    monkeypatch.setattr(stage_module, "WORKER_SILENT_S", 0.3)
    monkeypatch.setattr(stage_module, "GOOD_RESULTS_AFTER_RESTART", 3)
    made: list[ProcessBackend] = []

    def factory(job_spec: ModelSpec) -> ProcessBackend:
        monkeypatch.setenv("FAKE_WORKER", mode if not made else "normal")
        monkeypatch.setenv("FAKE_WORKER_AT", str(at))
        made.append(ProcessBackend(job_spec, worker_command=FAKE_WORKER))
        return made[-1]

    effect = NeuralFinishEffect(model_dir=spec.folder, allow_unevaluated=True, backend_factory=factory)
    section = NeuralSettings(enabled=True, strength=1.0)
    try:
        _paced_frames(effect, frontal, 0.05, section)
        assert until(lambda: effect.backend is not None or effect.stats()["phase"] == "off", 15.0)
        times = _paced_frames(effect, frontal, 1.5, section)
        assert until(lambda: len(made) >= 2 and effect.stats()["phase"] == "running", 20.0)
        times += _paced_frames(effect, frontal, 1.5, section)
        assert effect.restarts >= 1
        assert effect.health > 0.0  # recovered
        assert np.median(times) < 40.0
        assert max(times) < 1000.0, "a frame waited for the worker"
    finally:
        effect.close()
        for backend in made:
            assert until(lambda b=backend: b._process.poll() is not None, 5.0)


def test_closing_the_stage_ends_its_worker(spec: ModelSpec, frontal: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_WORKER", "normal")
    made: list[ProcessBackend] = []

    def factory(job_spec: ModelSpec) -> ProcessBackend:
        made.append(ProcessBackend(job_spec, worker_command=FAKE_WORKER))
        return made[-1]

    effect = NeuralFinishEffect(model_dir=spec.folder, allow_unevaluated=True, backend_factory=factory)
    _paced_frames(effect, frontal, 0.05, NeuralSettings(enabled=True, strength=1.0))
    assert until(lambda: effect.backend is not None, 15.0)
    effect.close()
    assert until(lambda: made[0]._process.poll() is not None, 5.0)
    assert until(lambda: not any(thread.name == "neural-control" for thread in threading.enumerate()), 2.0)


def test_importing_the_worker_module_needs_nothing_heavy() -> None:
    """The worker must start without Qt, MediaPipe or ONNX Runtime at import time."""
    code = (
        "import sys; import makeup_cam.neural.worker; "
        "bad = [m for m in ('PySide6', 'mediapipe', 'onnxruntime', 'cv2') if m in sys.modules]; "
        "sys.exit(str(bad) if bad else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


# ---- the real engine, where ONNX Runtime is installed --------------------------------------------------------


def _tiny_model(path: Path, size: int, *, input_name: str = "input", output_name: str = "delta", gain: float = 0.1) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    shape = [1, 3, size, size]
    graph = helper.make_graph(
        [helper.make_node("Mul", [input_name, "gain"], [output_name])],
        "tiny",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape)],
        [helper.make_tensor("gain", TensorProto.FLOAT, [], [gain])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.save(model, str(path))


def _ort_spec(folder: Path, **kwargs: Any) -> ModelSpec:
    folder.mkdir(parents=True, exist_ok=True)
    _tiny_model(folder / "model.onnx", SIZE, **kwargs)
    return load_spec(write_model_folder(folder, size=SIZE, content=(folder / "model.onnx").read_bytes()))


def test_ort_engine_on_the_cpu_when_asked(tmp_path: Path, backends: list[ProcessBackend]) -> None:
    pytest.importorskip("onnxruntime")
    backend = ProcessBackend(_ort_spec(tmp_path / "m"), provider="cpu")
    backends.append(backend)
    backend.start()
    run_job(backend, 1, 0.5, 30.0)
    assert backend.latest().delta[128, 128, 1] == pytest.approx(0.05, abs=1e-5)


def test_ort_engine_refuses_a_model_with_the_wrong_names(tmp_path: Path, backends: list[ProcessBackend]) -> None:
    pytest.importorskip("onnxruntime")
    backend = ProcessBackend(_ort_spec(tmp_path / "m", output_name="image"), provider="cpu")
    backends.append(backend)
    with pytest.raises(BackendError) as error:
        backend.start()
    assert error.value.code == worker.CODE_BAD_MODEL


def test_ort_engine_refuses_a_corrupt_file(tmp_path: Path, backends: list[ProcessBackend]) -> None:
    pytest.importorskip("onnxruntime")
    backend = ProcessBackend(load_spec(write_model_folder(tmp_path / "m", size=SIZE, content=b"not onnx")), provider="cpu")
    backends.append(backend)
    with pytest.raises(BackendError) as error:
        backend.start()
    assert error.value.code == worker.CODE_LOAD_ERROR


def _has_dml() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "DmlExecutionProvider" in ort.get_available_providers()


@pytest.mark.skipif(not _has_dml(), reason="needs the real GPU: onnxruntime-directml")
def test_real_gpu_jobs_match_the_cpu(tmp_path: Path, backends: list[ProcessBackend]) -> None:
    spec = _ort_spec(tmp_path / "m")
    gpu = ProcessBackend(spec, provider="dml")
    backends.append(gpu)
    gpu.start()
    run_job(gpu, 1, 0.5, 30.0)
    assert gpu.latest().delta[128, 128, 1] == pytest.approx(0.05, abs=1e-5)
    assert min(gpu.rtt_ms) < 1000.0


@pytest.mark.skipif(not _has_dml(), reason="needs the real GPU: onnxruntime-directml")
def test_real_gpu_stage_runs_and_recovers_from_a_killed_worker(tmp_path: Path, frontal: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage_module, "RESTART_WAITS_S", (0.1, 0.1, 0.1))
    monkeypatch.setattr(stage_module, "GOOD_RESULTS_AFTER_RESTART", 2)
    spec = _ort_spec(tmp_path / "m")
    effect = NeuralFinishEffect(model_dir=spec.folder, allow_unevaluated=True)
    section = NeuralSettings(enabled=True, strength=1.0)
    try:
        _paced_frames(effect, frontal, 0.05, section)
        assert until(lambda: effect.backend is not None, 30.0)
        _paced_frames(effect, frontal, 0.5, section)
        assert effect.health > 0.0
        effect.kill_worker()
        times = _paced_frames(effect, frontal, 0.3, section)
        assert until(lambda: effect.stats()["phase"] == "running" and effect.restarts == 1, 30.0)
        times += _paced_frames(effect, frontal, 1.0, section)
        assert effect.health > 0.0
        assert max(times) < 1000.0
    finally:
        effect.close()


# ---- no orphans, also when the worker is stuck while it starts (review finding) ---------------------


def _alive(pid: int) -> bool:
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    listed = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True, check=False)
    return f'"{pid}"'.encode() in listed.stdout


def test_end_of_stdin_ends_a_worker_stuck_in_its_start(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(spec, backends, monkeypatch, "hang_at_start")
    errors: list[BackendError] = []

    def start() -> None:
        try:
            backend.start()
        except BackendError as exc:
            errors.append(exc)

    starter = threading.Thread(target=start, daemon=True)
    starter.start()
    assert until(lambda: backend._process is not None)
    time.sleep(1.0)  # well into make_engine
    backend._process.stdin.close()
    assert until(lambda: backend._process.poll() is not None, 5.0), "the worker outlived its parent's pipe"
    assert backend._process.returncode == 0
    assert until(lambda: not starter.is_alive(), 5.0), "start kept waiting for a worker that is gone"
    assert errors


def test_closing_the_stage_reaches_a_worker_stuck_in_its_start(spec: ModelSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_WORKER", "hang_at_start")
    made: list[ProcessBackend] = []

    def factory(model_spec: ModelSpec) -> ProcessBackend:
        made.append(ProcessBackend(model_spec, worker_command=FAKE_WORKER))
        return made[-1]

    effect = NeuralFinishEffect(model_dir=tmp_path / "model", allow_unevaluated=True, backend_factory=factory)
    effect._want_stage(0.0)
    assert until(lambda: made and made[0]._process is not None)
    time.sleep(0.5)
    effect.close()
    assert until(lambda: made[0]._process.poll() is not None, 5.0), "close left the starting worker running"


def test_a_worker_stuck_in_its_start_dies_with_a_parent_that_dies(tmp_path: Path) -> None:
    script = tmp_path / "parent.py"
    fake = str(Path(__file__).with_name("neural_worker_fake.py"))
    script.write_text(
        "import os, subprocess, sys, time\n"
        f"worker = subprocess.Popen([sys.executable, {fake!r}, '--shm', sys.argv[1], '--size', '256', '--model', 'x'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)\n"
        "print(worker.pid, flush=True)\n"
        "time.sleep(1.0)\n"
        "os._exit(7)\n",
        encoding="utf-8",
    )
    block = shared_memory.SharedMemory(create=True, size=2 * worker.tensor_bytes(256))
    try:
        env = {**os.environ, "FAKE_WORKER": "hang_at_start"}
        with (tmp_path / "out.txt").open("w") as out:  # not a pipe: an orphan would hold it open, and this test would hang
            done = subprocess.run([sys.executable, str(script), block.name], stdout=out, stderr=subprocess.DEVNULL, timeout=60, env=env, check=False)
        assert done.returncode == 7
        pid = int((tmp_path / "out.txt").read_text().split()[0])
        gone = until(lambda: not _alive(pid), 5.0)
        if not gone:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False)
        assert gone, "an orphan stayed behind"
    finally:
        block.close()
        block.unlink()


# ---- the result filter runs in the worker: there it shares no GIL with the classic effects ---------------


def _spec_with(tmp_path: Path, **output: float) -> ModelSpec:
    settings = {"name": "delta", "kind": "delta", "layout": "NCHW", "colors": "BGR", "dtype": "float32", "max_abs_delta": 0.25, "temporal_alpha": 1.0, **output}
    return load_spec(write_model_folder(tmp_path / "filtered", size=SIZE, output=settings))


def test_the_worker_clamps_and_windows_what_the_model_gives(tmp_path: Path, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(_spec_with(tmp_path, max_abs_delta=0.02), backends, monkeypatch)
    backend.start()
    run_job(backend, 1, 0.5)  # the fake engine gives 0.05
    delta = backend.latest().delta
    assert delta[128, 128, 0] == pytest.approx(0.02, abs=1e-6)
    assert delta[0, 0, 0] == pytest.approx(0.0, abs=1e-4)
    assert not delta.flags.writeable


def test_the_worker_smooths_and_forgets_when_told(tmp_path: Path, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(_spec_with(tmp_path, temporal_alpha=0.5), backends, monkeypatch)
    backend.start()
    run_job(backend, 1, 0.1)
    run_job(backend, 2, 0.3)
    assert backend.latest().delta[128, 128, 0] == pytest.approx(0.5 * 0.03 + 0.5 * 0.01, abs=1e-5)
    backend.reset()
    run_job(backend, 3, 0.3)
    assert backend.latest().delta[128, 128, 0] == pytest.approx(0.03, abs=1e-5)


def test_a_result_the_worker_refuses_is_not_published(spec: ModelSpec, backends: list[ProcessBackend], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = fake_backend(spec, backends, monkeypatch)
    backend.start()
    run_job(backend, 1, 0.5)
    run_job(backend, 2, 5.0)  # the fake engine gives a mean of 0.5: over the limit of contract section 7.5
    assert backend.latest().tag == 1
    assert backend.rejected_results == 1 and backend.failure is None
