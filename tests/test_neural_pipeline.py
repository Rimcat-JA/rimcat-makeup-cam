"""The stage inside the real pipeline, with fake tracker and segmenter: order, settings, presets, status, closing."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from neural_fakes import FakeBackend, uniform_delta, write_model_folder

import makeup_cam.effects
from makeup_cam.config import SECTION_LABELS, SECTIONS, MakeupSettings, section_fields
from makeup_cam.effects import BUILTIN_EFFECT_MODULES, load_builtin_effects
from makeup_cam.effects.base import Effect, FrameContext
from makeup_cam.neural.stage import NeuralFinishEffect
from makeup_cam.pipeline import LAST_GEOMETRY_ORDER, OPTIONAL_MASK_USERS, MakeupPipeline
from makeup_cam.presets import list_presets, load_preset
from makeup_cam.runner import RunnerStatus
from makeup_cam.segmentation import SegMasks

H, W = 720, 1280


class ScriptedTracker:
    def __init__(self, landmarks: np.ndarray) -> None:
        self.landmarks = landmarks

    def detect(self, frame: np.ndarray, ts_ms: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.landmarks.copy(), np.zeros(len(self.landmarks), np.float32)

    def set_smoothing(self, smoothing: float) -> None: ...
    def close(self) -> None: ...


class NoMasks:
    def __init__(self) -> None:
        self.updates = 0

    def update(self, frame: np.ndarray, ts_ms: int, frame_index: int) -> SegMasks:
        self.updates += 1
        return SegMasks()

    def set_interval(self, interval: int) -> None: ...
    def close(self) -> None: ...


@pytest.fixture
def landmarks(face_landmarks_720p: np.ndarray) -> np.ndarray:
    center = face_landmarks_720p.mean(axis=0)
    return center + (face_landmarks_720p - center) * 1.5


def test_the_stage_is_registered_after_every_classic_effect() -> None:
    assert "neural" in BUILTIN_EFFECT_MODULES
    effects = load_builtin_effects()
    assert effects[-1] is NeuralFinishEffect
    assert NeuralFinishEffect.order == 90 and NeuralFinishEffect.name == "neural" and NeuralFinishEffect.needs_face
    assert all(effect.order < 90 for effect in effects[:-1])
    assert "neural" in OPTIONAL_MASK_USERS


def test_the_section_is_off_by_default_and_has_japanese_labels() -> None:
    settings = MakeupSettings()
    assert settings.neural.enabled is False
    assert SECTION_LABELS["neural"] == "学習モデルの仕上げ"
    fields = {name: meta for name, meta, _ in section_fields(SECTIONS["neural"])}
    assert fields["enabled"]["label"] == "学習モデルの仕上げを使う"
    assert fields["strength"]["label"] == "仕上げの強さ"
    assert (fields["strength"]["min"], fields["strength"]["max"]) == (0.0, 1.0)


def test_older_settings_without_the_section_get_it_off() -> None:
    assert MakeupSettings.from_dict({"lips": {"opacity": 0.2}}).neural.enabled is False


def test_values_out_of_range_are_clamped() -> None:
    settings = MakeupSettings.from_dict({"neural": {"strength": 7.0, "enabled": "yes"}})
    assert settings.neural.strength == 1.0 and settings.neural.enabled is False


@pytest.mark.parametrize("info", [info for info in list_presets() if info.builtin], ids=lambda info: info.name)
def test_no_builtin_preset_uses_the_stage(info: Any) -> None:
    settings = load_preset(info.name)
    assert settings.neural.enabled is False
    # Switching the section on must do something, and the strength must equal the default: settings saved before the
    # section existed get the default, and would otherwise show as changed against the preset.
    assert settings.neural.strength == MakeupSettings().neural.strength > 0.0


@pytest.mark.parametrize("info", [info for info in list_presets() if info.builtin], ids=lambda info: info.name)
def test_settings_saved_before_the_section_existed_equal_the_preset(info: Any) -> None:
    fresh = load_preset(info.name).to_dict()
    older = {key: value for key, value in fresh.items() if key != "neural"}
    assert MakeupSettings.from_dict(older).to_dict() == fresh


def _frames(pipeline: MakeupPipeline, frame: np.ndarray, count: int) -> np.ndarray:
    result = None
    for index in range(count):
        result = pipeline.process(frame, index * 33)
    return result.output


def _pipeline(landmarks: np.ndarray, settings: MakeupSettings, effects: list[Any] | None = None) -> MakeupPipeline:
    return MakeupPipeline(settings, tracker=ScriptedTracker(landmarks), segmenter=NoMasks(), effects=effects)


def _still_frame() -> np.ndarray:
    return np.random.default_rng(3).integers(40, 200, (H, W, 3), np.uint8)


def test_section_off_no_model_and_strength_zero_all_give_the_classic_output(
    landmarks: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AI_MAKEUP_CAM_NEURAL_DIR", str(tmp_path / "nothing"))
    frame = _still_frame()
    natural = load_preset("natural")
    classic_effects = [cls() for cls in load_builtin_effects() if cls is not NeuralFinishEffect]
    classic = _pipeline(landmarks, natural.copy(), classic_effects)

    zero = natural.copy()
    zero.neural.enabled = True
    zero.neural.strength = 0.0
    no_model = natural.copy()
    no_model.neural.enabled = True
    no_model.neural.strength = 1.0
    others = [_pipeline(landmarks, natural.copy()), _pipeline(landmarks, zero), _pipeline(landmarks, no_model)]
    for index in range(16):
        expected = classic.process(frame, index * 33).output
        for pipeline in others:
            assert np.array_equal(pipeline.process(frame, index * 33).output, expected)
    for pipeline in (classic, *others):
        pipeline.close()


def test_the_finish_comes_on_top_of_the_classic_frame_and_fades_with_the_face(landmarks: np.ndarray, tmp_path: Path) -> None:
    frame = np.full((H, W, 3), 128, np.uint8)
    settings = MakeupSettings.from_dict(load_preset("off").to_dict())
    settings.neural.enabled = True
    settings.neural.strength = 1.0
    settings.general.mirror_preview = False
    effect = NeuralFinishEffect(
        model_dir=write_model_folder(tmp_path / "m", size=256),
        allow_unevaluated=True,
        backend_factory=lambda spec: FakeBackend(spec, time.perf_counter, lambda crop: uniform_delta(crop.shape[0], 0.1)),
    )
    pipeline = _pipeline(landmarks, settings, [effect])
    deadline = time.perf_counter() + 10.0
    index = 0
    while effect.health < 1.0 and time.perf_counter() < deadline:
        pipeline.process(frame, index * 33)
        index += 1
    result = pipeline.process(frame, index * 33)
    x, y = landmarks[50].astype(int)  # a cheek point
    assert int(result.output[y, x, 0]) > 145
    assert np.array_equal(result.output[:20, :20], frame[:20, :20])
    assert "neural" in result.timings_ms
    assert pipeline.finishing_status() == "動作中"
    pipeline.close()
    assert effect.stats()["phase"] == "closed"


def test_a_frame_size_change_drops_the_delta(landmarks: np.ndarray, tmp_path: Path) -> None:
    settings = MakeupSettings.from_dict(load_preset("off").to_dict())
    settings.neural.enabled = True
    settings.neural.strength = 1.0
    effect = NeuralFinishEffect(
        model_dir=write_model_folder(tmp_path / "m", size=256),
        allow_unevaluated=True,
        backend_factory=lambda spec: FakeBackend(spec, time.perf_counter),
    )
    pipeline = _pipeline(landmarks, settings, [effect])
    frame = np.full((H, W, 3), 128, np.uint8)
    deadline = time.perf_counter() + 10.0
    index = 0
    while effect.health <= 0.0 and time.perf_counter() < deadline:
        pipeline.process(frame, index * 33)
        index += 1
    assert effect.health > 0.0
    pipeline.process(np.full((H // 2, W // 2, 3), 128, np.uint8), index * 33)
    assert effect.health < 0.2  # started over: at most one step of the rise
    pipeline.close()


def test_a_broken_stage_is_isolated_by_the_pipeline(landmarks: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = MakeupSettings.from_dict(load_preset("off").to_dict())
    settings.neural.enabled = True
    settings.neural.strength = 1.0
    effect = NeuralFinishEffect(control_thread=False)
    monkeypatch.setattr(effect, "apply", lambda *args: (_ for _ in ()).throw(RuntimeError("a bug")))
    pipeline = _pipeline(landmarks, settings, [effect])
    frame = np.full((H, W, 3), 128, np.uint8)
    assert np.array_equal(_frames(pipeline, frame, 3), frame)
    pipeline.close()


def test_an_overshoot_is_clipped_by_the_pipeline_and_does_not_wrap(landmarks: np.ndarray, tmp_path: Path) -> None:
    for value, level, expected in ((0.14, 250, 255), (-0.14, 5, 0)):
        folder = write_model_folder(tmp_path / str(value), size=256)
        stage = NeuralFinishEffect(
            model_dir=folder,
            allow_unevaluated=True,
            backend_factory=lambda spec, v=value: FakeBackend(spec, time.perf_counter, lambda crop: uniform_delta(crop.shape[0], v)),
        )
        settings = _only_blush(1.0)
        settings.general.mirror_output = False
        pipeline = _pipeline(landmarks, settings, [stage])
        frame = np.full((H, W, 3), level, np.uint8)
        output = None
        for index in range(200):
            output = pipeline.process(frame, index * 33).output
            if stage.health >= 1.0 and index > 40:
                break
            time.sleep(0.005)
        center = landmarks[1].astype(int)
        assert stage.health >= 1.0
        assert output[center[1], center[0]].tolist() == [expected] * 3
        pipeline.close()


def test_the_pipeline_starts_the_paste_ahead_once_the_geometry_is_final(landmarks: np.ndarray) -> None:
    seen: list[Any] = []
    moved = landmarks + 1.0

    class Geometry(Effect):
        name = "reshape"
        order = LAST_GEOMETRY_ORDER

        def apply(self, frame: np.ndarray, ctx: FrameContext, settings: Any) -> np.ndarray:
            ctx.landmarks = moved
            return frame

    class Later(Effect):
        name = "skin"
        order = LAST_GEOMETRY_ORDER + 1

        def apply(self, frame: np.ndarray, ctx: FrameContext, settings: Any) -> np.ndarray:
            seen.append("later")
            return frame

    class Stage(Effect):
        name = "neural"
        order = 90

        def start_ahead(self, ctx: FrameContext, settings: Any) -> None:
            seen.append(ctx.landmarks)

        def apply(self, frame: np.ndarray, ctx: FrameContext, settings: Any) -> np.ndarray:
            seen.append("stage")
            return frame

    settings = load_preset("natural")
    settings.neural.enabled = True
    pipeline = _pipeline(landmarks, settings, [Geometry(), Later(), Stage()])
    pipeline.process(_still_frame(), 0)
    pipeline.close()
    assert seen[0] is moved and seen[1:] == ["later", "stage"]


def test_only_effects_up_to_the_last_geometry_order_change_the_geometry() -> None:
    """What ``start_ahead`` relies on. Fails when a later effect starts writing the landmarks or a mask into the context."""
    pattern = re.compile(r"ctx\.(landmarks|landmarks_z|hair_mask|face_skin_mask|body_skin_mask)(, ctx\.\w+)*\s*=[^=]")
    writers = {}
    root = Path(makeup_cam.effects.__file__).parent
    for path in root.glob("*.py"):
        if pattern.search(path.read_text(encoding="utf-8")):
            writers[path.stem] = next(cls.order for cls in load_builtin_effects() if cls.__module__.endswith("." + path.stem))
    assert writers == {"reshape": 5, "hairshape": 8}
    assert max(writers.values()) <= LAST_GEOMETRY_ORDER


def test_a_failing_start_ahead_is_isolated(landmarks: np.ndarray) -> None:
    class Stage(Effect):
        name = "neural"
        order = 90

        def start_ahead(self, ctx: FrameContext, settings: Any) -> None:
            raise RuntimeError("drill")

        def apply(self, frame: np.ndarray, ctx: FrameContext, settings: Any) -> np.ndarray:
            return frame

    settings = load_preset("natural")
    settings.neural.enabled = True
    pipeline = _pipeline(landmarks, settings, [Stage()])
    result = pipeline.process(_still_frame(), 0)
    pipeline.close()
    assert result.output.shape == (H, W, 3)


class CountingMasks(NoMasks):
    def update(self, frame: np.ndarray, ts_ms: int, frame_index: int) -> SegMasks:
        self.updates += 1
        full = np.ones((H, W), np.float32)
        return SegMasks(hair=full * 0.0, face_skin=full * 0.5, body_skin=full * 0.0)


def _only_blush(strength: float) -> MakeupSettings:
    """A look with no effect that asks for masks, and one that uses them when they are there."""
    settings = load_preset("natural")
    for name in SECTIONS:
        section = getattr(settings, name)
        if name != "general" and hasattr(section, "enabled"):
            section.enabled = False
    settings.blush.enabled = True
    settings.neural.enabled = True
    settings.neural.strength = strength
    return settings


@pytest.mark.parametrize("strength", [0.0, 1.0])
def test_a_stage_that_does_nothing_starts_no_segmentation_and_changes_no_other_effect(
    landmarks: np.ndarray, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strength: float
) -> None:
    monkeypatch.setenv("AI_MAKEUP_CAM_NEURAL_DIR", str(tmp_path / "nothing"))
    frame = _still_frame()
    classic_settings = _only_blush(strength)
    classic_settings.neural.enabled = False
    classic = MakeupPipeline(classic_settings, tracker=ScriptedTracker(landmarks), segmenter=CountingMasks())
    masks = CountingMasks()
    pipeline = MakeupPipeline(_only_blush(strength), tracker=ScriptedTracker(landmarks), segmenter=masks)
    for index in range(30):
        assert np.array_equal(pipeline.process(frame, index * 33).output, classic.process(frame, index * 33).output)
        time.sleep(0.001)  # time for the stage's control thread to find that there is no model
    pipeline.close()
    classic.close()
    assert masks.updates == 0


def test_status_is_none_while_the_section_is_off(landmarks: np.ndarray) -> None:
    pipeline = _pipeline(landmarks, MakeupSettings())
    assert pipeline.finishing_status() is None
    pipeline.close()


def test_runner_status_has_the_message_field() -> None:
    assert RunnerStatus().neural_message is None


def test_the_command_line_only_offers_the_override_to_the_offline_modes() -> None:
    from makeup_cam.__main__ import parse_args

    args = parse_args(["--image", "a.png", "--out", "b.png", "--neural-model", "m", "--neural-strength", "0.5"])
    assert args.neural_model == Path("m") and args.neural_strength == 0.5
    for argv in (["--headless", "--neural-model", "m"], ["--neural-model", "m"]):
        with pytest.raises(SystemExit):
            parse_args(argv)
    with pytest.raises(SystemExit):
        parse_args(["--image", "a.png", "--out", "b.png", "--neural-model", "m", "--neural-strength", "2"])


# ---- what the stage learns from the pipeline ------------------------------------------------------------------------------


class _Listening(Effect):
    name = "skin"
    order = 30
    needs_face = False

    def __init__(self) -> None:
        self.seen: list[FrameContext] = []

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: Any) -> np.ndarray:
        self.seen.append(ctx)
        return frame


class _DatedMasks(NoMasks):
    def update(self, frame: np.ndarray, ts_ms: int, frame_index: int) -> SegMasks:
        return SegMasks(source_ts_ms=ts_ms - 66)


def _listening_pipeline(landmarks: np.ndarray) -> tuple[MakeupPipeline, _Listening]:
    listening = _Listening()
    return MakeupPipeline(MakeupSettings(), effects=[listening], tracker=ScriptedTracker(landmarks), segmenter=_DatedMasks()), listening


class _Ahead(Effect):
    name = "neural"
    order = 90
    needs_face = False

    def __init__(self) -> None:
        self.ahead = 0

    def start_ahead(self, ctx: FrameContext, settings: Any) -> None:
        self.ahead += 1

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: Any) -> np.ndarray:
        return frame


def test_nothing_is_prepared_ahead_when_nothing_runs_in_the_meantime(landmarks: np.ndarray) -> None:
    """With every classic effect off the helper only stood in the way: 4.5 ms on the camera thread instead of 3.4."""
    settings = MakeupSettings()
    settings.neural.enabled = True
    frame = np.zeros((H, W, 3), np.uint8)
    alone = _Ahead()
    pipeline = MakeupPipeline(settings, effects=[alone], tracker=ScriptedTracker(landmarks), segmenter=_DatedMasks())
    pipeline.process(frame, 0)
    pipeline.close()
    assert alone.ahead == 0
    company = _Ahead()
    pipeline = MakeupPipeline(settings, effects=[_Listening(), company], tracker=ScriptedTracker(landmarks), segmenter=_DatedMasks())
    pipeline.process(frame, 0)
    pipeline.close()
    assert company.ahead == 1


def test_effects_are_told_which_frames_were_never_rendered(landmarks: np.ndarray) -> None:
    pipeline, listening = _listening_pipeline(landmarks)
    frames = [np.full((H, W, 3), value, np.uint8) for value in range(8)]
    try:
        pipeline.prepare(frames[0], 0)
        pipeline.process(frames[0], 0)  # the first: nothing before it to compare with
        pipeline.prepare(frames[1], 33)
        pipeline.process(frames[1], 33)
        pipeline.prepare(frames[2], 66)  # never rendered
        pipeline.prepare(frames[3], 99)
        pipeline.process(frames[3], 99)
        pipeline.prepare(frames[4], 132)
        pipeline.prepare(frames[5], 165)  # announced, but it is still its turn to come
        pipeline.process(frames[4], 132)
        pipeline.process(frames[5], 165)
        for index in (6, 7):  # two lost at once
            pipeline.prepare(frames[index], index * 33)
        pipeline.prepare(frames[0], 8 * 33)
        pipeline.process(frames[0], 8 * 33)
    finally:
        pipeline.close()
    assert [ctx.skipped_frames for ctx in listening.seen] == [0, 0, 1, 0, 0, 2]


def test_without_announcements_nothing_counts_as_skipped(landmarks: np.ndarray) -> None:
    pipeline, listening = _listening_pipeline(landmarks)
    frame = np.zeros((H, W, 3), np.uint8)
    try:
        for index in range(3):
            pipeline.process(frame, index * 1000)
    finally:
        pipeline.close()
    assert [ctx.skipped_frames for ctx in listening.seen] == [0, 0, 0]


def test_effects_are_told_when_the_frame_and_its_masks_are_from(landmarks: np.ndarray) -> None:
    pipeline, listening = _listening_pipeline(landmarks)
    try:
        pipeline.process(np.zeros((H, W, 3), np.uint8), 500)
    finally:
        pipeline.close()
    assert listening.seen[0].timestamp_ms == 500 and listening.seen[0].masks_timestamp_ms == 434
