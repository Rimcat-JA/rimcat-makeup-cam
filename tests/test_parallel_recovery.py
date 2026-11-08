"""After a failure the stages draw again; with too little CPU the app makes room; a stop leaves nothing for the next start
(contract sections 8 and 9). Real processes throughout; failures come from outside."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from makeup_cam import parallel
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.presets import load_preset
from makeup_cam.runner import RunnerStatus

if sys.platform != "win32":
    pytest.skip("the pipeline of processes runs on Windows only", allow_module_level=True)

from parallel_helpers import FRAME_MS, StableMasks, StableTracker, descendants, process_alive, resume, suspend, wait_until  # noqa: E402
from test_parallel_life import Rig, RunnerRig, _report  # noqa: E402

from makeup_cam.parallel import engine as engine_module  # noqa: E402
from makeup_cam.parallel import shm  # noqa: E402
from makeup_cam.parallel.engine import FALLBACK_MESSAGE, ParallelEngine  # noqa: E402
from makeup_cam.parallel.sender_main import frame_for_camera, open_device  # noqa: E402
from makeup_cam.parallel.stage_main import keep_section  # noqa: E402

_ABOVE_NORMAL = 0x8000
_NORMAL = 0x20


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path / "home"))


@pytest.fixture
def rig(face_frame_720p: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    monkeypatch.setattr(engine_module, "RECOVER_AFTER_S", (0.3, 0.3, 0.3))
    monkeypatch.setattr(engine_module, "physical_cores", lambda: engine_module.FEW_CORES)
    made = Rig(face_frame_720p, 0)
    yield made
    for pid in made.pids.values():
        if process_alive(pid):
            resume(pid)
    made.close()


def _priority_class(pid: int) -> int:
    k32 = shm._k32
    k32.GetPriorityClass.argtypes = [shm.ctypes.c_void_p]
    k32.GetPriorityClass.restype = shm.ctypes.c_uint
    handle = k32.OpenProcess(0x1000, 0, pid)
    assert handle
    try:
        return k32.GetPriorityClass(handle)
    finally:
        k32.CloseHandle(handle)


def _keep_feeding(rig: Rig, until: Any, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if until():
            return True
        rig.feed(1)
    return until()


# ---- drawing in the stages again --------------------------------------------------------


@pytest.mark.parametrize("victim", ["stage0", "stage1", "stage2"])
def test_after_a_killed_stage_the_stages_draw_again_and_the_line_clears(rig: Rig, victim: str) -> None:
    rig.start()
    rig.feed(20)
    os.kill(rig.pids[victim], 9)
    assert _keep_feeding(rig, lambda: _report(rig.engine).processing_error == FALLBACK_MESSAGE, 3.0)
    assert _keep_feeding(rig, lambda: rig.engine._state == engine_module._RUNNING, 15.0), rig.engine._state
    assert _report(rig.engine).processing_error in (None, parallel.SLOW_MESSAGE), "frames lost a moment ago may still be mentioned"
    assert rig.engine.child_pids["sender"] == rig.pids["sender"], "a living sender stays"
    assert rig.engine.child_pids[victim] != rig.pids[victim]
    rig.feed(40)
    assert wait_until(lambda: rig.seen and rig.seen[-1] >= rig.next - 3, 3.0), rig.seen[-5:]
    assert rig.seen == sorted(set(rig.seen)), "in order, and nothing twice"


def test_a_stage_that_stands_still_for_a_second_does_not_end_the_stages_for_good(rig: Rig) -> None:
    """The stage comes back by itself after 1.2 s, too late for the limit. It is given up, and new stages take over."""
    rig.start()
    rig.feed(40)
    pid = rig.pids["stage1"]
    suspend(pid)
    assert _keep_feeding(rig, lambda: _report(rig.engine).processing_error == FALLBACK_MESSAGE, engine_module.HANG_LIMIT_S + 1.0)
    assert _keep_feeding(rig, lambda: rig.engine._state == engine_module._RUNNING, 15.0)
    assert _report(rig.engine).processing_error in (None, parallel.SLOW_MESSAGE), "frames lost a moment ago may still be mentioned"
    assert not process_alive(pid)


def test_a_killed_sender_is_replaced(rig: Rig) -> None:
    rig.start()
    rig.feed(20)
    os.kill(rig.pids["sender"], 9)
    assert _keep_feeding(rig, lambda: _report(rig.engine).processing_error == FALLBACK_MESSAGE, 3.0)
    assert _keep_feeding(rig, lambda: rig.engine._state == engine_module._RUNNING, 15.0)
    assert rig.engine.child_pids["sender"] != rig.pids["sender"]
    number = rig.next
    rig.feed(30)
    assert wait_until(lambda: rig.seen and rig.seen[-1] >= number + 20, 3.0)
    assert rig.seen == sorted(set(rig.seen))


def test_stopping_after_a_recovery_leaves_nothing(rig: Rig) -> None:
    rig.start()
    rig.feed(10)
    os.kill(rig.pids["stage2"], 9)
    assert _keep_feeding(rig, lambda: rig.engine._state == engine_module._RUNNING and rig.engine.child_pids["stage2"] != rig.pids["stage2"], 15.0)
    pids = dict(rig.engine.child_pids)
    rig.engine.stop()
    assert wait_until(lambda: not any(process_alive(pid) for pid in pids.values()), 2.0)
    assert not descendants(os.getpid()) & set(pids.values())


def test_when_the_tries_run_out_the_line_says_so_and_frames_go_on(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.start()
    rig.feed(10)
    monkeypatch.setattr(shm, "child_command", lambda module, args: (["no-such-program-at-all"], dict(os.environ)))
    os.kill(rig.pids["stage0"], 9)
    final = parallel.FALLBACK_FINAL_MESSAGE
    assert _keep_feeding(rig, lambda: _report(rig.engine).processing_error == final, 15.0)
    assert "再起動" in final
    number = rig.next
    rig.feed(5)
    assert rig.seen[-1] == number + 4
    assert rig.engine._recover_at is None


def test_a_failed_try_goes_back_to_drawing_here(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.start()
    rig.feed(10)
    original = shm.child_command
    failing = {"on": True}

    def maybe_silent(module: str, args: Any) -> tuple[list[str], dict[str, str]]:
        command, env = original(module, args)
        if failing["on"] and module.endswith("stage_main"):
            command = [command[0], "-c", "raise SystemExit(5)"]
        return command, env

    monkeypatch.setattr(shm, "child_command", maybe_silent)
    os.kill(rig.pids["stage0"], 9)
    assert _keep_feeding(rig, lambda: rig.engine._recover_tries >= 1 and rig.engine._line is None, 10.0)
    assert not rig.engine.stages_drawing or rig.engine._state == engine_module._FALLBACK
    failing["on"] = False
    assert _keep_feeding(rig, lambda: rig.engine._state == engine_module._RUNNING, 15.0)
    assert _report(rig.engine).processing_error in (None, parallel.SLOW_MESSAGE), "frames lost a moment ago may still be mentioned"


# ---- too little CPU ----------------------------------------------------------------------


_NEVER = 1e18  # the collector's own look at the load finds that the second is not over


def _begin_counting(engine: ParallelEngine) -> None:
    engine._load_at = 0.0
    engine._check_load(time.perf_counter())
    engine._load_at = _NEVER


def _seconds(engine: ParallelEngine, how_many: int, trouble: int, frames: int = 30) -> None:
    """What ``_check_load`` would see once per second, without waiting for it."""
    session = engine._session
    assert session is not None
    for _ in range(how_many):
        session.flags[shm.SESSION_SLIPS] += trouble
        engine._next_id += frames
        moment = time.perf_counter()
        engine._load_at = moment - 1.1
        engine._check_load(moment)
        engine._load_at = _NEVER


def test_one_stall_of_the_machine_changes_nothing(rig: Rig) -> None:
    rig.start()
    rig.feed(5)
    engine = rig.engine
    _begin_counting(engine)
    _seconds(engine, 1, 30)
    _seconds(engine, 9, 0)
    assert not engine._polite
    assert engine.stages_drawing
    assert _priority_class(rig.pids["stage0"]) == _ABOVE_NORMAL


def test_trouble_second_after_second_gives_the_other_programs_their_share_and_it_returns(rig: Rig) -> None:
    rig.start()
    rig.feed(5)
    engine = rig.engine
    _begin_counting(engine)
    _seconds(engine, engine_module.POLITE_BAD_SECONDS, 2)
    assert engine._polite
    rig.feed(5)  # the stages look with each frame
    pids = list(rig.pids.values())
    assert wait_until(lambda: all(_priority_class(pid) == _NORMAL for pid in pids), 2.0)
    assert _priority_class(os.getpid()) == _NORMAL
    assert engine.stages_drawing, "that alone is no failure"
    _seconds(engine, engine_module.POLITE_CALM_S, 0)
    assert not engine._polite
    rig.feed(5)
    assert wait_until(lambda: all(_priority_class(pid) == _ABOVE_NORMAL for pid in pids), 2.0)
    assert _priority_class(os.getpid()) == _ABOVE_NORMAL


def test_a_third_of_the_frames_in_trouble_for_ten_seconds_goes_to_one_process_for_good(rig: Rig) -> None:
    rig.start()
    rig.feed(5)
    engine = rig.engine
    _begin_counting(engine)
    _seconds(engine, engine_module.LOAD_SECONDS - 1, 10)
    assert engine.stages_drawing, "nine seconds are not yet enough: a slow stage for a while is no failure"
    _seconds(engine, 1, 10)
    assert _keep_feeding(rig, lambda: _report(engine).processing_error == parallel.OVERLOAD_MESSAGE, 3.0)
    assert engine._state == engine_module._FALLBACK
    assert engine._recover_at is None, "the stages would only take the CPU from the frames again"
    rig.feed(20)
    assert engine._state == engine_module._FALLBACK
    assert _priority_class(os.getpid()) == _NORMAL


def test_with_cores_to_spare_nothing_gives_way(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Short of CPU there because of other programs: the priority is what the frames still have (3 runs, 16 logical CPUs,
    x264 and 12 busy processes: 19 to 28 fps above normal, 23 to 24 at normal)."""
    monkeypatch.setattr(engine_module, "physical_cores", lambda: engine_module.FEW_CORES + 1)
    rig.start()
    rig.feed(5)
    engine = rig.engine
    _begin_counting(engine)
    _seconds(engine, engine_module.LOAD_SECONDS, 10)
    assert not engine._polite
    assert engine.stages_drawing


def test_a_core_count_follows_the_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    from makeup_cam import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "_threads_of_each_core", lambda _k: [0b11, 0b1100, 0b110000, 0b11000000, 0b1100000000, 0b110000000000])
    monkeypatch.setattr(pipeline_module, "_allowed_threads", lambda _k: 0b111111111111)
    assert parallel.physical_cores() == 6
    monkeypatch.setattr(pipeline_module, "_allowed_threads", lambda _k: 0b1010100)
    assert parallel.physical_cores() == 3
    monkeypatch.delenv(parallel.SWITCH_ENV, raising=False)
    assert not parallel.enabled()


# ---- what a stop leaves for the next start -------------------------------------------------


def test_a_start_right_after_a_stop_is_not_disturbed_by_what_was_still_starting(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """A spawn that takes 1.5 s, as behind a virus scanner: the stop does not wait for it."""
    real = subprocess.Popen
    first = threading.Event()

    def slow_once(*args: Any, **kwargs: Any) -> Any:
        if not first.is_set():
            first.set()
            time.sleep(1.5)
        return real(*args, **kwargs)

    monkeypatch.setattr(engine_module.subprocess, "Popen", slow_once)
    rig.engine.start((720, 1280), rig._collect)
    assert first.wait(2.0)
    time.sleep(0.2)
    rig.engine.stop()
    rig.start()
    time.sleep(2.0)
    assert rig.engine._state == engine_module._RUNNING
    assert _report(rig.engine).processing_error in (None, parallel.SLOW_MESSAGE), "frames lost a moment ago may still be mentioned"
    rig.feed(10)
    pids = set(rig.engine.child_pids.values())
    assert all(process_alive(pid) for pid in pids)
    assert not (descendants(os.getpid()) - pids - {child for pid in pids for child in descendants(pid)}), "nothing else was left running"


def test_a_child_that_cannot_join_the_job_does_not_stay(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[int] = []
    real = engine_module._JobObject.add

    def refused(self: Any, process: Any) -> None:
        made.append(process.pid)
        raise OSError(5, "no")

    monkeypatch.setattr(engine_module._JobObject, "add", refused)
    rig.engine.start((720, 1280), rig._collect)
    assert wait_until(lambda: _report(rig.engine).processing_error == FALLBACK_MESSAGE, 5.0)
    monkeypatch.setattr(engine_module._JobObject, "add", real)
    assert made
    assert wait_until(lambda: not any(process_alive(pid) for pid in made), 2.0), made


def test_a_hung_sender_during_a_change_of_size_still_ends_the_old_stages(rig: Rig) -> None:
    rig.start()
    rig.feed(10)
    old = {name: pid for name, pid in rig.pids.items() if name != "sender"}
    suspend(rig.pids["sender"])
    rig.engine._frame_size_changed((480, 640))
    assert wait_until(lambda: not any(process_alive(pid) for pid in old.values()), engine_module.REPLY_TIMEOUT_S + 3.0), old
    assert _report(rig.engine).processing_error == FALLBACK_MESSAGE


# ---- small things ------------------------------------------------------------------------


def test_a_camera_that_takes_nothing_is_no_tale_about_formats(capsys: pytest.CaptureFixture[str]) -> None:
    class Backend:
        class PixelFormat:
            NV12 = "nv12"
            BGR = "bgr"

        @staticmethod
        def Camera(_w: int, _h: int, _fps: float, *, fmt: str, print_fps: bool) -> str:  # noqa: N802
            raise RuntimeError("another program has it")

    with pytest.raises(RuntimeError):
        open_device(Backend, 1280, 720, 30)
    assert capsys.readouterr().err == ""


def test_a_section_that_did_not_change_stays_the_same_object() -> None:
    import pickle

    sections: dict[str, Any] = {}
    settings = load_preset("natural")
    first = keep_section(sections, "neural", pickle.loads(pickle.dumps(settings.neural)))
    assert keep_section(sections, "neural", pickle.loads(pickle.dumps(settings.neural))) is first
    settings.neural.strength = 0.25
    changed = pickle.loads(pickle.dumps(settings.neural))
    assert keep_section(sections, "neural", changed) is changed


def test_what_goes_to_the_camera_is_the_output_mirrored_and_fitted() -> None:
    picture = np.random.default_rng(4).integers(0, 256, (72, 128, 3), np.uint8)
    assert frame_for_camera(picture, False, 128, 72) is picture
    assert np.array_equal(frame_for_camera(picture, True, 128, 72), cv2.flip(picture, 1))
    fitted = frame_for_camera(picture, True, 64, 36)
    assert fitted.shape == (36, 64, 3)
    assert np.array_equal(fitted, cv2.resize(cv2.flip(picture, 1), (64, 36), interpolation=cv2.INTER_AREA))


class _Engine:
    """An engine whose stages are said to draw, for what ``_render`` asks of it."""

    stages_drawing = True
    stage_status = None
    stage_wants_masks = False
    frames: list[Any]

    def bind(self, _pipeline: Any) -> None:
        self.frames = []

    def route(self, _shape: tuple[int, int]) -> bool:
        return True

    def submit(self, planned: Any) -> bool:
        self.frames.append(planned)
        return True

    def note_skipped(self, _count: int) -> None:
        pass

    def stop(self, *_args: Any) -> None:
        pass


class _CountingMasks(StableMasks):
    calls = 0

    def update(self, frame: np.ndarray, ts_ms: int, frame_index: int) -> Any:
        type(self).calls += 1
        return super().update(frame, ts_ms, frame_index)


def _only_the_finish() -> Any:
    settings = load_preset("natural")
    for name in settings.__dataclass_fields__:
        section = getattr(settings, name)
        if name not in ("general", "neural") and hasattr(section, "enabled"):
            section.enabled = False
    settings.neural.enabled = True
    settings.neural.strength = 1.0
    return settings


def test_masks_are_made_when_the_finish_in_the_stage_will_use_them(face_frame_720p: np.ndarray) -> None:
    """The copy of the finish in this process never starts, so it cannot be the one to ask."""
    _CountingMasks.calls = 0
    pipeline = MakeupPipeline(_only_the_finish(), tracker=StableTracker(), segmenter=_CountingMasks())
    engine = _Engine()
    pipeline.attach_engine(engine)
    try:
        pipeline.process(face_frame_720p.copy(), FRAME_MS)
        assert _CountingMasks.calls == 0, "it has not said so yet"
        engine.stage_wants_masks = True
        pipeline.process(face_frame_720p.copy(), 2 * FRAME_MS)
        assert _CountingMasks.calls == 1
        assert engine.frames[-1].masks.hair is not None
        settings = _only_the_finish()
        settings.neural.strength = 0.0
        pipeline.set_settings(settings)
        pipeline.process(face_frame_720p.copy(), 3 * FRAME_MS)
        assert _CountingMasks.calls == 1, "a finish at 0 uses none, whatever the stage said last"
    finally:
        pipeline.close()


def test_the_status_line_says_the_delay() -> None:
    from makeup_cam.__main__ import format_status

    assert "遅延" not in format_status(RunnerStatus(), virtual_camera_enabled=False)
    assert "遅延 91 ms" in format_status(RunnerStatus(output_delay_ms=91.2), virtual_camera_enabled=False)


def test_a_new_frame_size_measures_the_delay_again(face_frame_720p: np.ndarray) -> None:
    """Through the real runner and sender: the first slow frames of the new stages must not pile up as lasting delay.
    The sender's clock is tested by itself in test_parallel_clock; here, that the new line reaches it."""
    pyvirtualcam = pytest.importorskip("pyvirtualcam")
    try:
        pyvirtualcam.Camera(1280, 720, 30, print_fps=False).close()
    except Exception as exc:
        pytest.skip(f"needs OBS Virtual Camera: {exc}")
    made = RunnerRig(face_frame_720p, virtual_camera=True)
    made.runner.start()
    try:
        assert made.engine.wait_ready(60.0)
        session = made.engine._session
        assert session is not None
        assert wait_until(lambda: session.flags[shm.SESSION_SETTLED_MS] > 0.0, 10.0)
        made.camera.size = (480, 640)
        assert wait_until(lambda: made.engine._line is not None and made.engine._line.shape == (480, 640) and made.engine._state == engine_module._RUNNING, 10.0)
        settled = session.flags[shm.SESSION_SETTLED_MS]
        slips = session.flags[shm.SESSION_SLIPS]
        assert wait_until(lambda: session.flags[shm.SESSION_SETTLED_MS] != settled, 10.0), "measured again for the new stages"
        assert session.flags[shm.SESSION_SLIPS] - slips <= 3, "as they come, and then a place of their own: no heap of repeats"
        report = _report(made.engine)
        assert wait_until(lambda: _report(made.engine).delay_ms is not None, 3.0), report
        assert _report(made.engine).delay_ms < 400.0
    finally:
        made.close()
