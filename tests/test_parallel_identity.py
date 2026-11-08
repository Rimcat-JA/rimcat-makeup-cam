"""The picture of the stages is bit-identical to the in-process chain (contract sections 5 and 12.1). Nothing is rounded."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from makeup_cam import parallel
from makeup_cam.config import MakeupSettings
from makeup_cam.effects import load_builtin_effects
from makeup_cam.effects.base import FrameContext
from makeup_cam.models import FACE_LANDMARKER, SELFIE_MULTICLASS, ensure_model
from makeup_cam.pipeline import MakeupPipeline, PlannedFrame
from makeup_cam.presets import load_preset
from makeup_cam.segmentation import SegMasks, Segmenter
from makeup_cam.tracking import FaceTracker

if sys.platform != "win32":
    pytest.skip("the pipeline of processes runs on Windows only", allow_module_level=True)

from parallel_helpers import (  # noqa: E402
    FRAME_MS,
    HEIGHT,
    WIDTH,
    StableMasks,
    StableTracker,
    base_landmarks,
    digest,
    make_frames,
    run_in_process,
    run_in_stages,
)

from makeup_cam.parallel.engine import ParallelEngine  # noqa: E402
from makeup_cam.parallel.shm import ArrayViews, Block, Layout  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLIP_1072 = ROOT / ".test-media" / "qa" / "fps" / "clips" / "1072_IEO_HAP_HI.avi"
MASK_FIELDS = ("hair_mask", "face_skin_mask", "body_skin_mask")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path / "home"))


def assert_same(in_process: list[str], in_stages: list[str]) -> None:
    assert len(in_process) == len(in_stages)
    different = [i for i, (a, b) in enumerate(zip(in_process, in_stages, strict=True)) if a != b]
    assert not different, f"{len(different)} of {len(in_process)} frames differ, first {different[:10]}"


@pytest.mark.parametrize("preset", ["natural", "polished", "strong"])
def test_stages_equal_the_chain_on_every_frame(face_frame_720p: np.ndarray, preset: str) -> None:
    settings = load_preset(preset)
    frames = make_frames(face_frame_720p, 40)
    assert_same(run_in_process(settings, frames), run_in_stages(settings, frames))


def _script_settings(preset: str) -> dict[str, MakeupSettings]:
    settings = load_preset(preset)
    slider = settings.copy()
    slider.lips.opacity = min(1.0, slider.lips.opacity + 0.2)
    slider.skin.smoothing = max(0.0, slider.skin.smoothing - 0.2)
    slider.general.intensity = 0.8
    off = settings.copy()
    off.hairshape.enabled = False
    off.skin.enabled = False
    zero = settings.copy()
    zero.general.intensity = 0.0
    mirrored = settings.copy()
    mirrored.general.mirror_output = True
    return {"other": load_preset("natural" if preset != "natural" else "strong"), "base": settings, "slider": slider, "off": off, "zero": zero, "mirrored": mirrored}


def _plate(frame_like: np.ndarray, amount: float) -> np.ndarray:
    return cv2.GaussianBlur(frame_like, (0, 0), amount).astype(np.float32) / 255.0


def _script(looks: dict[str, MakeupSettings], first_frame: np.ndarray) -> Any:
    """Preset glide, slider during a glide, sections off and on, intensity 0, bypass, face lost and found, lost frames,
    plate set, replaced and cleared, mirror on and off."""
    plates = {"a": _plate(first_frame, 25), "b": _plate(first_frame, 9)}

    def run(i: int, pipeline: MakeupPipeline, tracker: StableTracker) -> bool:
        if i == 12:
            pipeline.glide_to_settings(looks["other"])
        elif i == 15:
            pipeline.set_settings(looks["slider"])  # a slider, during the glide
        elif i == 28:
            pipeline.glide_to_settings(looks["base"])
        elif i == 40:
            pipeline.set_settings(looks["off"])
        elif i == 48:
            pipeline.set_settings(looks["base"])
        elif i == 54:
            pipeline.set_settings(looks["zero"])
        elif i == 58:
            pipeline.set_settings(looks["base"])
        elif i == 62:
            pipeline.bypass = True
        elif i == 68:
            pipeline.bypass = False
        elif 76 <= i < 90:
            tracker.lost.add(i * FRAME_MS)  # the face is lost, fades out, is forgotten, and comes back
        elif i == 100:
            pipeline._plate = plates["a"]
        elif i == 106:
            pipeline._plate = plates["b"]
        elif i == 112:
            pipeline._plate = None
        elif i == 116:
            pipeline.set_settings(looks["mirrored"])
        elif i == 122:
            pipeline.set_settings(looks["base"])
        return i not in (96, 97, 98)  # the camera lost frames: announced, never processed

    return run


@pytest.mark.parametrize("preset", ["strong", "natural"])
def test_stages_equal_the_chain_through_a_script(face_frame_720p: np.ndarray, preset: str) -> None:
    looks = _script_settings(preset)
    frames = make_frames(face_frame_720p, 130)
    in_process = run_in_process(looks["base"], frames, _script(looks, face_frame_720p))
    in_stages = run_in_stages(looks["base"], frames, _script(looks, face_frame_720p))
    assert_same(in_process, in_stages)


def test_the_comparison_is_able_to_see_a_difference(face_frame_720p: np.ndarray) -> None:
    settings = load_preset("natural")
    frames = make_frames(face_frame_720p, 12)
    other = settings.copy()
    other.lips.opacity = min(1.0, settings.lips.opacity + 0.3)
    assert run_in_process(settings, frames) != run_in_process(other, frames)


def test_odd_sizes_go_through_the_stages(face_frame_720p: np.ndarray) -> None:
    small = cv2.resize(face_frame_720p, (641, 361), interpolation=cv2.INTER_AREA)
    settings = load_preset("natural")
    frames = make_frames(small, 15)
    landmarks = base_landmarks() * 0.5
    assert_same(run_in_process(settings, frames, landmarks=landmarks), run_in_stages(settings, frames, landmarks=landmarks))


@pytest.mark.parametrize("preset", ["natural", "polished", "strong"])
def test_clip_300_frames(preset: str, models_dir: Path) -> None:
    """The real tracker on every frame and the real segmentation model inline, so that both runs see the same masks."""
    if not CLIP_1072.is_file():
        pytest.skip("needs the clips in .test-media")

    def frames() -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(CLIP_1072))
        taken = []
        for _ in range(300):
            ok, frame = capture.read()
            if not ok:
                break
            taken.append(frame)
        capture.release()
        return taken

    def build() -> MakeupPipeline:
        settings = load_preset(preset)
        tracker = FaceTracker(ensure_model(FACE_LANDMARKER, directory=models_dir), settings.general.landmark_smoothing)
        segmenter = Segmenter(ensure_model(SELFIE_MULTICLASS, directory=models_dir), settings.general.segmentation_interval, background=False)
        return MakeupPipeline(settings, tracker=tracker, segmenter=segmenter)

    clip = frames()
    assert len(clip) == 300
    first = build()
    in_process = []
    try:
        for i, frame in enumerate(clip):
            in_process.append(digest(first.process(frame, i * FRAME_MS).output))
    finally:
        first.close()
    second = build()
    engine = ParallelEngine()
    second.attach_engine(engine)
    engine.slot_wait_s = 30.0
    in_stages: list[str] = []
    engine.start(clip[0].shape[:2], lambda result: in_stages.append(digest(result.output)))
    try:
        assert engine.wait_ready(60.0)
        for i, frame in enumerate(clip):
            assert second.process(frame, i * FRAME_MS).in_flight
        deadline = time.monotonic() + 30.0
        while len(in_stages) < len(clip) and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        second.close()
    assert_same(in_process, in_stages)


# ---- what crosses a cut -----------------------------------------------------------------


class _Recording(FrameContext):
    """A context that remembers what is read and written after it was made."""

    def start_recording(self) -> None:
        object.__setattr__(self, "reads", set())
        object.__setattr__(self, "writes", set())

    def __getattribute__(self, name: str) -> Any:
        if name in (*MASK_FIELDS, "cache", "landmarks", "background"):
            reads = object.__getattribute__(self, "__dict__").get("reads")
            if reads is not None:
                reads.add(name)
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        writes = object.__getattribute__(self, "__dict__").get("writes")
        if writes is not None:
            writes.add(name)
        object.__setattr__(self, name, value)


def _record_effects(face_frame_720p: np.ndarray) -> dict[str, tuple[set[str], set[str]]]:
    tracker, masks = StableTracker(), StableMasks()
    settings = load_preset("strong")
    recorded: dict[str, tuple[set[str], set[str]]] = {}
    for cls in load_builtin_effects():
        if cls.name == "neural":  # needs a model: the source is checked below
            continue
        section = getattr(settings, cls.name)
        section.enabled = True
        effect = cls()
        reads: set[str] = set()
        writes: set[str] = set()
        for i in range(8):
            ts = (i + 1) * FRAME_MS
            landmarks, z = tracker.detect(face_frame_720p, ts)
            current: SegMasks = masks.update(face_frame_720p, ts, i)
            # Skin looks at the hair mask only without a face-skin mask, so that case is part of what it reads.
            ctx = _Recording(
                height=HEIGHT, width=WIDTH, landmarks=landmarks, landmarks_z=z,
                hair_mask=current.hair, face_skin_mask=current.face_skin if i < 6 else None, body_skin_mask=current.body_skin,
                timestamp_ms=ts, masks_timestamp_ms=current.source_ts_ms,
            )  # fmt: skip
            ctx.start_recording()
            effect.apply(face_frame_720p.astype(np.float32) / 255.0, ctx, section)
            reads |= ctx.reads
            writes |= ctx.writes
        recorded[cls.name] = (reads, writes)
    return recorded


def test_the_table_of_masks_is_what_the_effects_read(face_frame_720p: np.ndarray) -> None:
    table = parallel.MASK_READS
    for name, (reads, _) in _record_effects(face_frame_720p).items():
        assert reads & set(MASK_FIELDS) == set(table.get(name, ())), name


def test_the_table_knows_every_effect_that_mentions_a_mask_in_its_source() -> None:
    """The stage of the learned finish cannot run here without a model, so its source is read; the others too, as a second check."""
    root = ROOT / "makeup_cam"
    for cls in load_builtin_effects():
        module = sys.modules[cls.__module__]
        source = Path(module.__file__ or "").read_text(encoding="utf-8")
        if cls.name == "neural":
            assert set(re.findall(r"ctx\.(hair_mask|face_skin_mask|body_skin_mask)", source)) == set(parallel.MASK_READS["neural"])
        elif re.search(r"ctx\.(hair_mask|face_skin_mask|body_skin_mask)", source) and cls.name not in parallel.MASK_READS:
            assert cls.name == "tearbag", f"{cls.name} in {module.__file__}"  # shares contour.py and reads no mask
    assert root.is_dir()


def test_effects_write_only_what_crosses_the_cut_and_use_no_cache(face_frame_720p: np.ndarray) -> None:
    for name, (reads, writes) in _record_effects(face_frame_720p).items():
        assert "cache" not in reads, f"{name} uses ctx.cache: it MUST then cross explicitly"
        assert writes <= set(parallel.CONTEXT_WRITES.get(name, ())), name


def test_nothing_in_the_effects_uses_the_cache() -> None:
    for path in [*(ROOT / "makeup_cam" / "effects").glob("*.py"), ROOT / "makeup_cam" / "neural" / "stage.py"]:
        if path.name != "base.py":
            assert "ctx.cache" not in path.read_text(encoding="utf-8"), path


def test_stages_follow_the_order_of_the_effects() -> None:
    names = [cls.name for cls in load_builtin_effects()]
    stages = parallel.plan_stages(names)
    assert [name for stage in stages for name in stage] == names
    assert len(stages) == 3
    assert stages[0] == ["reshape", "hairshape"]
    assert stages[1] == ["hair", "skin", "contour"]
    assert stages[2][:3] == ["brows", "eyeshadow", "tearbag"]
    assert stages[2][-1] == "neural"


# ---- identity of arrays -----------------------------------------------------------------


@pytest.fixture
def block() -> Any:
    made = Block(Layout(48, 64))
    yield made
    made.close(unlink=True)


def test_one_array_per_mask_set_and_generation(block: Block) -> None:
    views = ArrayViews(block)
    first = views.mask(("set", 2, 0, 7))
    assert views.mask(("set", 2, 0, 7)) is first
    assert views.mask(("set", 2, 1, 7)) is not first
    refilled = views.mask(("set", 2, 0, 8))
    assert refilled is not first
    assert views.mask(("set", 2, 0, 8)) is refilled
    assert not first.flags.writeable


def test_a_mask_changed_by_a_stage_is_a_new_array_for_each_frame(block: Block) -> None:
    views = ArrayViews(block)
    assert views.mask(("slot", 1, 0)) is not views.mask(("slot", 1, 0))
    assert views.mask(None) is None


def test_a_new_plate_is_seen_as_new(block: Block) -> None:
    views = ArrayViews(block)
    first = views.plate((0, 1))
    assert views.plate((0, 1)) is first
    assert views.plate((1, 2)) is not first
    assert views.plate((0, 3)) is not first
    assert views.plate(None) is None
    assert not first.flags.writeable


def test_the_hair_search_runs_once_per_mask_set(block: Block, face_frame_720p: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    """In a stage as in process: the same view for the frames of one set, a new one when the set is filled again."""
    from makeup_cam.effects import hair as hair_module

    calls = {"count": 0}
    original = hair_module._hair_blobs

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(hair_module, "_hair_blobs", counted)
    big = Block(Layout(HEIGHT, WIDTH))
    try:
        views = ArrayViews(big)
        masks = StableMasks()
        current = masks.update(face_frame_720p, 0, 0)
        cv2.copyTo(current.hair, None, big.set_mask(0, 0))
        cv2.copyTo(current.face_skin, None, big.set_mask(0, 1))
        settings = load_preset("strong")
        settings.hair.enabled = True
        effect = hair_module.HairEffect()
        landmarks, z = StableTracker().detect(face_frame_720p, 0)

        def frame(generation: int) -> None:
            ctx = FrameContext(
                height=HEIGHT, width=WIDTH, landmarks=landmarks, landmarks_z=z,
                hair_mask=views.mask(("set", 0, 0, generation)), face_skin_mask=views.mask(("set", 0, 1, generation)),
            )  # fmt: skip
            effect.apply(face_frame_720p.astype(np.float32) / 255.0, ctx, settings.hair)

        frame(1)
        frame(1)
        frame(1)
        once = calls["count"]
        assert once >= 1
        frame(2)
        assert calls["count"] > once
        del views
    finally:
        big.close(unlink=True)


# ---- resets -----------------------------------------------------------------------------


class _RecordingEngine:
    """Stands in for the engine: says what each frame carries."""

    stages_drawing = True
    stage_status = None

    def __init__(self) -> None:
        self.planned: list[PlannedFrame] = []
        self.drop_next = False

    def bind(self, _pipeline: Any) -> None:
        pass

    def route(self, _shape: tuple[int, int]) -> bool:
        return True

    def submit(self, planned: PlannedFrame) -> bool:
        self.planned.append(planned)
        planned.dropped, self.drop_next = self.drop_next, False
        return True

    submit_through = submit

    def note_skipped(self, _count: int) -> None:
        pass

    def stop(self) -> None:
        pass


def test_a_reset_reaches_the_stages_with_its_own_frame_and_stays_if_that_frame_is_dropped(face_frame_720p: np.ndarray) -> None:
    tracker = StableTracker()
    engine = _RecordingEngine()
    pipeline = MakeupPipeline(load_preset("natural"), tracker=tracker, segmenter=StableMasks())
    pipeline.attach_engine(engine)
    frames = make_frames(face_frame_720p, 8)
    pipeline.process(frames[0], 0)
    assert engine.planned[0].reset  # the first size is a change of size
    pipeline.process(frames[1], FRAME_MS)
    assert not engine.planned[1].reset
    pipeline.bypass = True  # bypass forgets the face: the effects reset now
    engine.drop_next = True
    pipeline.process(frames[2], 2 * FRAME_MS)
    assert engine.planned[2].reset and engine.planned[2].steps is None and engine.planned[2].dropped
    pipeline.process(frames[3], 3 * FRAME_MS)
    assert engine.planned[3].reset  # the dropped frame's reset is not lost
    pipeline.process(frames[4], 4 * FRAME_MS)
    assert not engine.planned[4].reset
    pipeline.close()


def test_a_frame_dropped_on_the_way_in_counts_as_skipped_for_the_next(face_frame_720p: np.ndarray) -> None:
    engine = _RecordingEngine()
    pipeline = MakeupPipeline(load_preset("natural"), tracker=StableTracker(), segmenter=StableMasks())
    pipeline.attach_engine(engine)
    frames = make_frames(face_frame_720p, 4)
    for i, frame in enumerate(frames):
        pipeline.prepare(frame, i * FRAME_MS)
        engine.drop_next = i == 1
        pipeline.process(frame, i * FRAME_MS)
    assert [p.skipped for p in engine.planned] == [0, 0, 1, 0]
    pipeline.close()


def test_a_stage_does_not_load_the_models() -> None:
    source = (ROOT / "makeup_cam" / "parallel" / "stage_main.py").read_text(encoding="utf-8")
    assert "FaceTracker" not in source and "Segmenter" not in source


def test_the_sender_imports_no_effect() -> None:
    import subprocess

    code = (
        "import sys, json\n"
        "sys.argv = ['x', '{}']\n"
        "import makeup_cam.parallel.sender_main\n"
        "import makeup_cam.camera, makeup_cam.parallel.shm, makeup_cam.parallel.clock, makeup_cam.parallel.nv12\n"
        "print(json.dumps([m for m in sys.modules if m.startswith('makeup_cam.effects') or m.startswith('mediapipe')]))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"})
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
