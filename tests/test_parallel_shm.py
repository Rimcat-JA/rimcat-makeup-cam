"""The boundary between the processes (contract sections 4 and 12.2)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from makeup_cam.presets import load_preset

if sys.platform != "win32":
    pytest.skip("the pipeline of processes runs on Windows only", allow_module_level=True)

from parallel_helpers import FRAME_MS, StableMasks, StableTracker, make_frames, wait_until  # noqa: E402

from makeup_cam.parallel import shm  # noqa: E402
from makeup_cam.parallel.engine import ParallelEngine  # noqa: E402
from makeup_cam.pipeline import MakeupPipeline, ProcessResult  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path / "home"))


def test_ring_keeps_the_order_and_wraps() -> None:
    cells = np.zeros(2 + shm.RING_LENGTH, np.int64)
    ring = shm.Ring(cells)
    assert ring.get() is None
    expected = 0
    written = 0
    for _ in range(5000):
        for _ in range(7):
            ring.put(written % 8)
            written += 1
        assert len(ring) == 7
        for _ in range(7):
            assert ring.get() == expected % 8
            expected += 1
        assert ring.get() is None
    assert int(cells[0]) > shm.RING_LENGTH * 100


_ECHO = r"""
import sys
from makeup_cam.parallel import shm
layout = shm.Layout(4, 4)
block = shm.Block(layout, sys.argv[1])
into = shm.Edge(block.ring(0), shm.Semaphore(sys.argv[2], create=False))
back = shm.Edge(block.ring(1), shm.Semaphore(sys.argv[3], create=False))
count = int(sys.argv[4])
for _ in range(count):
    slot = None
    while slot is None:
        slot = into.get(1000)
    back.put(slot)
"""


def test_a_ring_between_two_processes_for_more_than_65536_frames() -> None:
    layout = shm.Layout(4, 4)
    block = shm.Block(layout)
    base = f"mkcam-test-{os.getpid()}-{time.monotonic_ns()}"
    forward, backward = shm.Semaphore(f"{base}-a", create=True), shm.Semaphore(f"{base}-b", create=True)
    into = shm.Edge(block.ring(0), forward)
    back = shm.Edge(block.ring(1), backward)
    count = (1 << 16) + 500
    child = subprocess.Popen(
        [sys.executable, "-c", _ECHO, block.name, forward.name, backward.name, str(count)], cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"}
    )
    try:
        for i in range(count):
            into.put(i % 1000)
            assert back.get(5000) == i % 1000
        assert child.wait(10) == 0
    finally:
        child.kill()
        del into, back
        forward.close()
        backward.close()
        block.close(unlink=True)


def test_metadata_larger_than_the_space_fails_loudly_and_overwrites_nothing() -> None:
    block = shm.Block(shm.Layout(32, 32))
    picture = block.image(0)
    picture[:] = 0.25
    block.write_meta(0, {"small": 1})
    with pytest.raises(ValueError, match="does not fit"):
        block.write_meta(0, {"large": b"x" * shm.META_BYTES})
    assert block.read_meta(0) == {"small": 1}
    assert (picture == 0.25).all()
    del picture
    block.close(unlink=True)


@pytest.mark.parametrize("shape", [(720, 1280), (361, 641), (480, 640), (1, 3), (1081, 1921)])
def test_no_two_parts_of_a_block_overlap_at_any_size(shape: tuple[int, int]) -> None:
    layout = shm.Layout(*shape)
    block = shm.Block(layout)
    block.touch()
    arrays: list[Any] = []
    for slot in range(shm.N_SLOTS):
        arrays += [block.image(slot), block.out(slot)] + [block.slot_mask(slot, m) for m in range(shm.N_MASKS)]
    for mask_set in range(shm.N_MASK_SETS):
        arrays += [block.set_mask(mask_set, m) for m in range(shm.N_MASKS)]
    arrays += [block.plate(p) for p in range(shm.N_PLATES)]
    for i, array in enumerate(arrays):
        array[...] = i + 1
    for i, array in enumerate(arrays):
        assert (array == i + 1).all(), i
    block.write_meta(0, {"after": "all that"})
    assert block.read_meta(0)["after"] == "all that"
    assert (arrays[0] == 1).all()
    del arrays
    assert layout.total <= 800_000_000 or shape[0] > 720
    block.close(unlink=True)


def test_the_block_of_720p_has_the_size_of_the_contract() -> None:
    assert 300e6 < shm.Layout(720, 1280).total < 350e6


def test_slots_mask_sets_and_plates_all_return_after_a_run(face_frame_720p: np.ndarray) -> None:
    masks = StableMasks(interval=2)
    pipeline = MakeupPipeline(load_preset("natural"), tracker=StableTracker(), segmenter=masks)
    engine = ParallelEngine()
    pipeline.attach_engine(engine)
    engine.slot_wait_s = 30.0
    done: list[ProcessResult] = []
    engine.start((720, 1280), done.append)
    try:
        assert engine.wait_ready(60.0)
        frames = make_frames(face_frame_720p, 40)
        plate_a = np.full((720, 1280, 3), 0.25, np.float32)
        plate_b = np.full((720, 1280, 3), 0.5, np.float32)
        line = engine._line
        assert line is not None
        for i, frame in enumerate(frames):
            if i == 10:
                pipeline._plate = plate_a
            elif i == 20:
                pipeline._plate = plate_b
            elif i == 30:
                pipeline._plate = None
            assert pipeline.process(frame, i * FRAME_MS).in_flight
            with engine._flight_lock:
                in_flight = list(engine._flight)
                slots = [p.slot for p in in_flight]
                free = list(line.free)
                assert len(set(slots)) == len(slots), "a slot in two frames at once"
                assert not set(slots) & set(free), "a slot both free and in flight"
                for planned in in_flight:
                    if planned.mask_set is not None:
                        assert planned.mask_set in line.set_references, "a mask set reused while a frame refers to it"
                    if planned.plate_slot is not None:
                        assert line.plate_references[planned.plate_slot] > 0, "a plate reused while a frame refers to it"
        assert wait_until(lambda: len(done) == len(frames), 30.0)
        assert wait_until(lambda: len(line.free) == shm.N_SLOTS, 5.0)
        assert sorted(line.free) == list(range(shm.N_SLOTS))
        assert line.set_references == {line.current_set: 1}, "only the newest set is held"
        assert len(line.free_sets) == shm.N_MASK_SETS - 1
        assert line.plate_references == [0, 0]
    finally:
        pipeline.close()


def test_two_plates_never_share_a_slot_while_in_use(face_frame_720p: np.ndarray) -> None:
    pipeline = MakeupPipeline(load_preset("natural"), tracker=StableTracker(), segmenter=StableMasks())
    engine = ParallelEngine()
    pipeline.attach_engine(engine)
    engine.slot_wait_s = 30.0
    done: list[ProcessResult] = []
    engine.start((720, 1280), done.append)
    try:
        assert engine.wait_ready(60.0)
        frames = make_frames(face_frame_720p, 16)
        line = engine._line
        assert line is not None
        seen: list[tuple[int | None, int]] = []
        for i, frame in enumerate(frames):
            pipeline._plate = np.full((720, 1280, 3), (i + 1) / 32.0, np.float32)  # a new plate on every frame
            pipeline.process(frame, i * FRAME_MS)
            seen.append((line.current_plate_slot, line.plate_generation))
        generations = [generation for _, generation in seen]
        assert generations == sorted(generations)
        assert wait_until(lambda: len(done) == len(frames), 30.0)
    finally:
        pipeline.close()


def test_children_start_the_real_interpreter_so_that_a_kill_hits_them() -> None:
    command, env = shm.child_command("makeup_cam.parallel.stage_main", {"a": 1})
    assert command[1:3] == ["-m", "makeup_cam.parallel.stage_main"]
    assert str(ROOT) in env["PYTHONPATH"].split(os.pathsep)
    if os.path.normcase(sys.executable) != os.path.normcase(getattr(sys, "_base_executable", sys.executable)):
        assert env["__PYVENV_LAUNCHER__"] == sys.executable
        assert command[0] == sys._base_executable  # type: ignore[attr-defined]
