"""Training code that needs no torch: runs in the app's own environment, on small synthetic frames.

Everything is written to pytest's temp folder. The landmarks are the stored ones of the test portrait; no face picture is
involved, so that these tests need neither scikit-image nor MediaPipe.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from makeup_cam import regions
from makeup_cam.neural import geometry
from training import augment, config, heldout, layout, masks, metadata, pairs, samples

ROOT = Path(__file__).resolve().parent.parent
LANDMARKS = np.load(Path(__file__).parent / "data" / "face_landmarks_720p.npy").astype(np.float32)
SCALE = 0.5  # small frames: these tests must stay fast
FRAME_LANDMARKS = LANDMARKS * SCALE
FRAME_SHAPE = (360, 640, 3)


def synthetic_frame(index: int) -> np.ndarray:
    """A textured uint8 BGR frame that moves slowly with the index."""
    y, x = np.mgrid[0 : FRAME_SHAPE[0], 0 : FRAME_SHAPE[1]].astype(np.float32)
    x = x + index
    pattern = np.stack(
        [90 + 40 * np.sin(x / 9.0), 120 + 40 * np.sin((x + y) / 13.0), 150 + 40 * np.sin(y / 7.0)], axis=2
    )
    return pattern.astype(np.uint8)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def make_sequence(
    root: Path,
    name: str,
    frames: int = 6,
    kind: str = "classic_pipeline",
    change: tuple[int, int, int] = (-6, -2, 12),
    consecutive: bool = True,
    with_masks: bool = False,
) -> Path:
    """A sequence in the contract's layout: the target is the input with a fixed colour change everywhere."""
    folder = root / name
    for index in range(frames):
        frame = synthetic_frame(index)
        write_png(folder / "input" / f"{index:06d}.png", frame)
        write_png(folder / "target" / f"{index:06d}.png", np.clip(frame.astype(np.int16) + change, 0, 255).astype(np.uint8))
        if with_masks:
            write_png(folder / "masks" / f"{index:06d}.png", np.full(FRAME_SHAPE[:2], 255, np.uint8))
    shift = np.arange(frames, dtype=np.float32)[:, None, None] * np.array([0.2, 0.0], dtype=np.float32)
    np.save(folder / "landmarks.npy", FRAME_LANDMARKS[None] + shift)
    layout.write_manifest(folder, kind, "off", 30.0, frames, consecutive, "test")
    return folder


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def test_default_config_is_valid_and_survives_a_round_trip():
    original = config.RunConfig()
    config.validate(original)
    assert config.from_dict(json.loads(config.dumps(original))) == original


def test_shipped_configs_load_and_hold_no_personal_data():
    files = sorted((ROOT / "training" / "configs").glob("*.json"))
    assert len(files) >= 4
    for path in files:
        loaded = config.load(path)
        assert config.find_personal_data(config.to_dict(loaded)) == []


@pytest.mark.parametrize(
    "values",
    [
        {"nonsense": 1},
        {"train": {"stepz": 1}},
        {"data": {"size": 300}},
        {"export": {"opset": 23}},  # DirectML would run Conv on the CPU
        {"export": {"max_abs_delta": 0.4}},
        {"export": {"name": "a/b"}},
        {"model": {"norm": "group"}},
        {"model": {"activation": "gelu"}},  # Gelu falls to the CPU under DirectML
        {"model": {"attention": True}},  # a student is exported
        {"stage": "teacher", "model": {"float16": True}},
        {"data": {"size": 256}, "model": {"depth": 9}},
    ],
)
def test_config_refuses(values):
    with pytest.raises(config.ConfigError):
        config.from_dict(values)


@pytest.mark.parametrize(
    "text",
    ["C:\\data\\faces", "c:/data", "/home/someone/data", "D:\\Users\\someone\\x", "folder/Users/someone", "name@example.com"],
)
def test_config_refuses_paths_and_addresses(text):
    with pytest.raises(config.ConfigError):
        config.from_dict({"description": text})
    with pytest.raises(config.ConfigError):
        config.from_dict({"export": {"note": text}})
    with pytest.raises(config.ConfigError):
        config.from_dict({"data": {"train_sequences": [text]}})


def test_config_allows_sequence_patterns_and_plain_notes():
    loaded = config.from_dict({"data": {"train_sequences": ["cremad-*", "2026-11-10_main-*"]}, "export": {"note": "stand-in, 2 people"}})
    assert loaded.data.train_sequences == ["cremad-*", "2026-11-10_main-*"]


def test_fingerprint_ignores_only_what_does_not_change_training():
    a = config.RunConfig()
    b = config.from_dict({"train": {"steps": 99, "log_every": 7, "num_workers": 0}, "description": "x", "export": {"note": "n"}})
    assert config.fingerprint(a) == config.fingerprint(b)
    assert config.fingerprint(a) != config.fingerprint(config.from_dict({"train": {"learning_rate": 0.5}}))
    assert config.fingerprint(a) != config.fingerprint(config.from_dict({"model": {"base": 8}}))


# --------------------------------------------------------------------------
# Dataset layout and session folders
# --------------------------------------------------------------------------


def test_sequence_is_read_and_checked(tmp_path):
    folder = make_sequence(tmp_path, "cremad-1002-A", 3)
    sequence = layout.read_sequence(folder)
    assert (sequence.kind, sequence.frames, sequence.consecutive, sequence.group) == ("classic_pipeline", 3, True, "cremad-1002")
    assert sequence.landmarks().shape == (3, 478, 2)
    assert sequence.has_masks is False


@pytest.mark.parametrize(
    "breakage",
    [
        lambda f: (f / "manifest.json").write_text("{", encoding="utf-8"),
        lambda f: (f / "manifest.json").write_text(json.dumps({"kind": "video", "frames": 3}), encoding="utf-8"),
        lambda f: (f / "manifest.json").write_text(json.dumps({"kind": "recording", "frames": 4}), encoding="utf-8"),
        lambda f: (f / "landmarks.npy").unlink(),
        lambda f: (f / "target" / "000002.png").unlink(),
    ],
)
def test_broken_sequences_are_refused(tmp_path, breakage):
    folder = make_sequence(tmp_path, "s-a", 3)
    breakage(folder)
    with pytest.raises(layout.DatasetError):
        layout.read_sequence(folder)


def test_held_out_person_never_reaches_training(tmp_path):
    for name in ("cremad-1002-A", "cremad-1002-B", "cremad-1012-A", "cremad-1013-A"):
        make_sequence(tmp_path, name, 2)
    sequences = layout.find_sequences(tmp_path)
    # A pattern that names only one sequence of an actor still takes all of that actor's sequences out of training.
    train, kept = layout.split_sequences(sequences, ["*"], ["cremad-1002-A"])
    assert {s.name for s in kept} == {"cremad-1002-A", "cremad-1002-B"}
    assert {s.name for s in train} == {"cremad-1012-A", "cremad-1013-A"}
    train, kept = layout.split_sequences(sequences, ["cremad-1012-*"], [])
    assert [s.name for s in train] == ["cremad-1012-A"] and kept == []


def test_data_summary_says_no_paths(tmp_path):
    make_sequence(tmp_path, "cremad-1002-A", 2)
    summary = layout.data_summary(layout.find_sequences(tmp_path))
    assert summary == {"kinds": ["classic_pipeline"], "input_presets": ["off"], "sequences": 1, "groups": 1, "frames": 2}


def _take(folder: Path, step: int, key: str, take: int, status: str, video: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"step{step:02d}_{key}_take{take:02d}"
    if video:
        (folder / f"{stem}.mkv").write_bytes(b"x")
    (folder / f"{stem}.json").write_text(
        json.dumps({"status": status, "video": {"file": f"{stem}.mkv"}, "fps": {"container": 30.0}, "frames": {"written": 90}}),
        encoding="utf-8",
    )


def test_session_folders_follow_the_capture_guide(tmp_path):
    sessions = tmp_path / "sessions"
    bare, makeup = sessions / "2026-11-10_main_bare", sessions / "2026-11-10_main_makeup"
    _take(bare, 3, "head", 1, "completed")
    _take(bare, 3, "head", 2, "completed")
    _take(bare, 3, "head", 3, "aborted")  # newest, but not completed
    _take(bare, 5, "talk", 1, "completed", video=False)  # video gone
    _take(makeup, 3, "head", 1, "completed")
    _take(makeup, 5, "talk", 1, "completed")
    _take(bare, 7, "gaze", 1, "failed")
    _take(sessions / "2026-11-11_window_bare", 3, "head", 1, "completed")  # no made-up side
    _take(sessions / "2026-11-20_mock_bare", 1, "talk", 1, "completed")  # evaluation: never for training
    _take(sessions / "2026-11-20_mock_makeup", 1, "talk", 1, "completed")
    pairs_found = layout.find_session_pairs(tmp_path)
    assert [p.name for p in pairs_found] == ["2026-11-10_main"]
    pair = pairs_found[0]
    assert pair.steps == [3]
    assert pair.bare[3].take == 2 and pair.makeup[3].take == 1
    assert pair.bare[3].fps == 30.0 and pair.bare[3].frames == 90


# --------------------------------------------------------------------------
# Crop space
# --------------------------------------------------------------------------


def test_landmarks_map_to_where_the_crop_shows_them():
    """A dot drawn at a landmark in the frame must show up in the crop where the landmark is said to be."""
    size = 256
    for index in (1, 33, 263, 61, 152):
        frame = np.zeros((720, 1280, 3), np.float32)
        x, y = LANDMARKS[index]
        cv2.circle(frame, (int(x), int(y)), 5, (1.0, 1.0, 1.0), -1, cv2.LINE_AA)
        crop = geometry.extract_crop(frame, LANDMARKS, size)
        ys, xs = np.nonzero(crop[..., 0] > 0.5)
        found = np.array([xs.mean() + 0.5, ys.mean() + 0.5])  # pixel i covers [i, i + 1)
        expected = masks.landmarks_to_crop(np.array([[int(x) + 0.5, int(y) + 0.5]]), geometry.crop_matrix(LANDMARKS, size))[0]
        assert np.linalg.norm(found - expected) < 0.6


def test_crop_weight_is_exactly_zero_where_nothing_may_change():
    size = 256
    matrix = geometry.crop_matrix(LANDMARKS, size)
    weight = masks.crop_weight(LANDMARKS, matrix, size)
    crop_points = masks.landmarks_to_crop(LANDMARKS, matrix)
    roi = regions.Roi(0, 0, size, size)
    grown = regions.expand_polygon(crop_points[regions.LIPS_INNER], 0.01 * regions.face_scale(crop_points))
    mouth = regions.polygon_mask(grown, roi) >= 1.0
    eyes = np.zeros_like(mouth)
    for side in regions.SIDES:
        eyes |= regions.eye_opening_mask(crop_points, side, roi) >= 1.0
    assert eyes.sum() > 20
    assert (weight[eyes] == 0.0).all()
    assert (weight[mouth] == 0.0).all()
    assert weight[0, 0] == 0.0 and weight[size - 1, size - 1] == 0.0  # outside the face
    cheek = crop_points[regions.side_indices("left").eye].mean(axis=0) + np.array([0.0, 40.0])
    assert weight[int(cheek[1]), int(cheek[0])] == 1.0
    with_eyes = masks.crop_weight(LANDMARKS, matrix, size, may_edit_eyes=True)
    assert with_eyes[eyes].mean() > 0.9


def test_crop_weight_agrees_with_the_runtime_in_the_frame():
    """Rasterized in the crop here and in the frame at runtime: the two must be the same region."""
    size = 256
    matrix = geometry.crop_matrix(LANDMARKS, size)
    in_crop = masks.crop_weight(LANDMARKS, matrix, size)
    roi = regions.Roi(0, 0, 1280, 720)
    in_frame = geometry.edit_weight(LANDMARKS, roi)
    warped = cv2.warpAffine(in_frame, matrix, (size, size), flags=cv2.INTER_LINEAR)
    assert np.abs(in_crop - warped).mean() < 0.01


def test_crop_weight_takes_the_frame_mask_as_extra_allowed_area():
    size = 256
    matrix = geometry.crop_matrix(LANDMARKS, size)
    assert masks.crop_weight(LANDMARKS, matrix, size, np.ones((720, 1280), np.float32))[0:20, :].min() == 1.0
    assert masks.crop_weight(LANDMARKS, matrix, size, np.zeros((720, 1280), np.float32))[0, 0] == 0.0


def test_apply_delta_follows_the_contract():
    size = 64
    crop = np.full((size, size, 3), 0.5, np.float32)
    delta = np.full((size, size, 3), 0.5, np.float32)  # over the clamp
    weight = np.ones((size, size), np.float32)
    assert (masks.apply_delta(crop, delta, weight, 0.0) == crop).all()
    window = np.ones((size, size), np.float32)
    assert np.allclose(masks.apply_delta(crop, delta, weight, 1.0, 0.25, window), 0.75)
    assert np.allclose(masks.apply_delta(crop, delta, weight, 0.5, 0.25, window), 0.625)
    out = masks.apply_delta(crop, delta, np.zeros((size, size), np.float32), 1.0)
    assert (out == crop).all()
    assert (masks.apply_delta(crop, delta, weight, 1.0)[0, 0] - 0.5 < 1e-3).all()  # the edge window


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------


def test_identity_photometric_changes_nothing():
    crop = np.random.default_rng(0).random((32, 32, 3), dtype=np.float32)
    assert np.allclose(augment.Photometric.identity().apply(crop), crop)


def test_photometric_is_the_room_and_is_reproducible():
    settings = config.AugmentConfig()
    a = augment.Photometric.sample(np.random.default_rng(3), settings)
    b = augment.Photometric.sample(np.random.default_rng(3), settings)
    assert a == b
    crop = np.full((32, 32, 3), 0.4, np.float32)
    lit = a.apply(crop)
    assert lit.dtype == np.float32 and lit.min() >= 0.0 and lit.max() <= 1.0
    assert not np.allclose(lit, crop)


def test_degrade_stays_in_range_and_draws_fresh_noise():
    settings = config.AugmentConfig()
    crop = np.random.default_rng(0).random((64, 64, 3), dtype=np.float32)
    a = augment.degrade(crop, np.random.default_rng(1), settings)
    b = augment.degrade(crop, np.random.default_rng(2), settings)
    assert a.dtype == np.float32 and a.shape == crop.shape
    assert 0.0 <= a.min() and a.max() <= 1.0
    assert not np.allclose(a, b)
    assert np.allclose(a, augment.degrade(crop, np.random.default_rng(1), settings))


# --------------------------------------------------------------------------
# Samples
# --------------------------------------------------------------------------


def _builder(root: Path, enabled: bool = False, **data: object) -> samples.SampleBuilder:
    sequences = layout.find_sequences(root)
    return samples.SampleBuilder(
        sequences, config.DataConfig(**data), config.AugmentConfig(enabled=enabled), max_abs_delta=0.25, seed=5
    )


def test_a_sample_is_determined_by_its_step_and_position(tmp_path):
    make_sequence(tmp_path, "s-a", 4)
    builder = _builder(tmp_path, enabled=True)
    a = builder.build(7, 1)
    b = builder.build(7, 1)
    c = builder.build(7, 2)
    assert set(a) == {"input", "delta", "weight", "flow", "flow_confidence", "has_next"}
    assert a["input"].shape == (2, 3, 256, 256) and a["weight"].shape == (2, 1, 256, 256)
    assert a["flow"].shape == (2, 256, 256) and a["flow_confidence"].shape == (1, 256, 256)
    for key in a:
        assert a[key].dtype == np.float32
        assert np.array_equal(a[key], b[key])
    assert not np.array_equal(a["input"], c["input"])


def test_delta_is_target_minus_input_and_noise_stays_out_of_it(tmp_path):
    make_sequence(tmp_path, "s-a", 3, change=(-6, -2, 12))
    plain = _builder(tmp_path, enabled=False, jitter_shift=0.0, jitter_degrees=0.0, jitter_scale=0.0)
    sample = plain.build(0, 0)
    middle = sample["delta"][0][:, 100:156, 100:156].mean(axis=(1, 2))
    assert np.allclose(middle, np.array([-6, -2, 12]) / 255.0, atol=0.004)
    noisy = _builder(tmp_path, enabled=True, jitter_shift=0.0, jitter_degrees=0.0, jitter_scale=0.0)
    # Fixed light: any noise in the delta would show as spread in this flat change.
    noisy.augment = config.AugmentConfig(exposure_stops=0.0, white_balance=0.0, gamma=0.0, lighting_gradient=0.0, jpeg_quality_min=70, read_noise=0.03)
    sample = noisy.build(0, 0)
    assert sample["delta"][0].std(axis=(1, 2)).max() < 0.01
    assert not np.allclose(sample["input"], plain.build(0, 0)["input"], atol=0.001)


def test_delta_never_exceeds_the_clamp(tmp_path):
    make_sequence(tmp_path, "s-a", 3, change=(120, -120, 120))
    sample = _builder(tmp_path).build(0, 0)
    assert np.abs(sample["delta"]).max() <= 0.25 + 1e-6


def test_no_next_frame_means_no_temporal_pair(tmp_path):
    make_sequence(tmp_path, "s-a", 3)
    make_sequence(tmp_path, "t-b", 3, consecutive=False)
    builder = _builder(tmp_path)
    sequences = {s.name: s for s in builder.sequences}
    rng = np.random.default_rng(0)
    assert builder.build_at(sequences["s-a"], 0, rng)["has_next"] == 1.0
    assert builder.build_at(sequences["s-a"], 2, rng)["has_next"] == 0.0  # the last frame
    assert builder.build_at(sequences["t-b"], 0, rng)["has_next"] == 0.0


def test_masks_from_the_sequence_are_used(tmp_path):
    make_sequence(tmp_path, "s-a", 2, with_masks=True)
    builder = _builder(tmp_path, jitter_shift=0.0, jitter_degrees=0.0, jitter_scale=0.0)
    assert builder.sequences[0].has_masks
    assert builder.build(0, 0)["weight"][0, 0, 10, 128] == 1.0  # top of the crop: outside the oval, allowed by the mask


def test_each_person_is_picked_equally_often(tmp_path):
    for name in ("p-1", "p-2", "p-3", "p-4"):
        make_sequence(tmp_path, name, 2)  # one person with four sequences
    make_sequence(tmp_path, "q-1", 2)  # another with one
    builder = _builder(tmp_path)
    rng = np.random.default_rng(0)
    picks = [builder.pick(rng)[0].group for _ in range(400)]
    assert 150 < picks.count("q") < 250


def test_backward_flow_finds_a_shift_and_warping_undoes_it():
    noise = np.random.default_rng(0).random((256, 256, 3), dtype=np.float32)
    image = np.clip(0.5 + 3.0 * (cv2.GaussianBlur(noise, (0, 0), 3.0) - 0.5), 0.0, 1.0)
    moved = np.roll(image, 3, axis=1)  # the next frame: moved 3 px to the right
    flow = samples.backward_flow(moved, image)
    assert abs(float(np.median(flow[..., 0])) + 3.0) < 0.3 and abs(float(np.median(flow[..., 1]))) < 0.3
    warped = samples.warp_by_flow(image, flow)
    assert np.abs(warped - moved)[32:-32, 32:-32].mean() < 0.03
    assert samples.flow_confidence(moved, warped)[32:-32, 32:-32].mean() > 0.5
    assert samples.flow_confidence(moved, np.zeros_like(moved)).mean() < 0.01


# --------------------------------------------------------------------------
# Metadata and the operator rules
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kinds", "kind", "stand_in"),
    [
        ([], "none", True),
        (["classic_pipeline"], "classic_pipeline", True),
        (["3d_render"], "3d_render", False),
        (["recording"], "recording", False),
        (["3d_render", "recording"], "mixed", False),
        (["3d_render", "classic_pipeline"], "mixed", True),  # stand-in pairs in the mix: never a real model
    ],
)
def test_stand_in_rule(kinds, kind, stand_in):
    assert metadata.training_data_kind(kinds) == kind
    assert metadata.is_stand_in(kinds) is stand_in


def test_model_json_follows_the_contract(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model bytes")
    export = config.ExportConfig(name="finish-student", version="0.1.0", note="n")
    data = {"kinds": ["classic_pipeline"], "input_presets": ["off"]}
    meta = metadata.build_model_json(export, 512, path, data, {"step": 3})
    assert meta["schema"] == 1 and meta["opset"] == 20 and meta["stand_in"] is True
    assert meta["sha256"] == metadata.sha256_of(path) and meta["size_bytes"] == 11 and meta["file"] == "model.onnx"
    assert meta["input"] == {"name": "input", "size": 512, "layout": "NCHW", "colors": "BGR", "range": [0.0, 1.0], "dtype": "float32", "stage": "after_classic"}
    assert meta["output"]["kind"] == "delta" and meta["output"]["colors"] == "BGR" and meta["output"]["max_abs_delta"] == 0.25
    assert meta["crop"] == {"definition": 1} and meta["may_edit"] == {"eyes": False}
    assert meta["training_data"] == {"kind": "classic_pipeline", "input_preset": "off", "note": "n"}
    assert meta["evaluation"] == {"verdict": "none", "report": None, "report_sha256": None, "model_sha256": None, "date": None}
    assert str(tmp_path) not in json.dumps(meta)


def test_model_json_is_readable_by_the_runtime(tmp_path):
    """The runtime's gate must accept the file's shape, and must refuse a stand-in for a call."""
    model_files = pytest.importorskip("makeup_cam.neural.model_files")
    folder = tmp_path / "finish-student-0.1.0"
    folder.mkdir()
    path = folder / "model.onnx"
    path.write_bytes(b"model bytes")
    meta = metadata.build_model_json(config.ExportConfig(name="finish-student", version="0.1.0"), 256, path, {"kinds": ["classic_pipeline"], "input_presets": ["off"]})
    (folder / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    spec = model_files.load_spec(folder)
    assert spec.stand_in is True
    with pytest.raises(model_files.ModelRefused):
        model_files.check_gate(spec)
    model_files.check_gate(spec, allow_unevaluated=True)  # offline modes and tests only


@pytest.mark.parametrize(
    ("nodes", "fragment"),
    [
        ([("", "Gelu", {})], "Gelu"),
        ([("", "GroupNormalization", {})], "GroupNormalization"),
        ([("", "Resize", {"mode": "cubic"})], "cubic"),
        ([("", "Resize", {"mode": "linear", "coordinate_transformation_mode": "asymmetric"})], "座標"),
        ([("", "LayerNormalization", {})], "LayerNormalization"),
        ([("com.microsoft", "FusedConv", {})], "FusedConv"),
    ],
)
def test_forbidden_nodes_are_named(nodes, fragment):
    problems = metadata.check_nodes(nodes)
    assert len(problems) == 1 and fragment in problems[0]


def test_allowed_nodes_pass():
    nodes = [(domain, op, {}) for domain, op in (("", "Conv"), ("", "Relu"), ("ai.onnx", "Concat"), ("", "Tanh"), ("", "Cast"), ("", "Constant"), ("", "InstanceNormalization"))]
    nodes.append(("", "Resize", {"mode": "linear", "coordinate_transformation_mode": "half_pixel"}))
    nodes.append(("", "Resize", {"mode": "nearest", "coordinate_transformation_mode": "asymmetric"}))
    assert metadata.check_nodes(nodes) == []


# --------------------------------------------------------------------------
# Held-out numbers
# --------------------------------------------------------------------------


def test_heldout_tells_a_perfect_model_from_doing_nothing(tmp_path):
    change = (-6, -2, 12)
    make_sequence(tmp_path, "p-a", 3, change=change)
    make_sequence(tmp_path, "q-a", 3, change=(0, 0, 25))
    sequences = layout.find_sequences(tmp_path)

    def nothing(batch: np.ndarray) -> np.ndarray:
        return np.zeros_like(batch)

    report = heldout.evaluate(nothing, sequences, 256, 2)
    assert report["summary"]["ratio_mean"] == pytest.approx(1.0)
    assert report["strength_monotonic"] is False
    assert report["summary"]["identity_lab"] > 0.5

    def half_of_p(batch: np.ndarray) -> np.ndarray:
        return np.full_like(batch, 0.0) + (np.array(change, np.float32) / 255.0).reshape(1, 3, 1, 1)

    report = heldout.evaluate(half_of_p, sequences, 256, 2)
    assert report["groups"]["p"]["ratio"] < 0.1
    assert report["groups"]["q"]["ratio"] > 0.3
    assert report["summary"]["worst_group"] == "q"
    assert report["strength_monotonic"] is True
    assert report["strength"]["1.0"]["lab_error"] < report["strength"]["1.0"]["lab_error_identity"]
    assert report["raw_delta_where_weight_is_zero"]["mean"] == pytest.approx(np.abs(np.array(change) / 255.0).mean(), abs=1e-3)
    assert report["frame_step"]["model_mean"] == pytest.approx(0.0, abs=1e-6)


def test_contact_sheet_says_what_it_is(tmp_path):
    make_sequence(tmp_path, "p-a", 3)
    sheet = heldout.contact_sheet(lambda b: np.zeros_like(b), layout.find_sequences(tmp_path), 256)
    assert sheet.shape[1] == 5 * 256 and sheet.shape[0] == 48 + 256


# --------------------------------------------------------------------------
# Making pairs
# --------------------------------------------------------------------------


def test_only_women_are_used(tmp_path):
    table = tmp_path / "demographics.csv"
    table.write_text('"ActorID","Age","Sex"\n1001,51,"Male"\n1002,21,"Female"\n1012,23,"Female"\n', encoding="utf-8")
    women = pairs.women_actors(table)
    assert women == {"1002", "1012"}
    for name in ("1001_A.flv", "1002_A.flv", "1012_A.flv", "other.flv"):
        (tmp_path / name).write_bytes(b"x")
    pattern = str(tmp_path / "*.flv")
    assert [p.name for p in pairs.select_clips(pattern, women)] == ["1002_A.flv", "1012_A.flv"]
    assert [p.name for p in pairs.select_clips(pattern, women, ["1012"])] == ["1012_A.flv"]
    with pytest.raises(layout.DatasetError):
        pairs.select_clips(pattern, women, ["1001"])


def test_test_media_lists_all_clips_as_women():
    """The 11 actors of the test clips, from the table that comes with them. Skipped where the clips are absent."""
    table = ROOT / ".test-media" / "testvids" / "demographics.csv"
    if not table.is_file():
        pytest.skip("no test media")
    clips = pairs.select_clips(str(ROOT / ".test-media" / "testvids" / "*.flv"), pairs.women_actors(table))
    assert len(clips) == 44


def test_data_and_models_cannot_land_in_tracked_folders(tmp_path):
    pairs.refuse_tracked_folder(tmp_path / "data")
    pairs.refuse_tracked_folder(ROOT / ".test-media" / "planb" / "pairs")
    for folder in (ROOT / "training" / "data", ROOT / "docs", ROOT):
        with pytest.raises(layout.DatasetError):
            pairs.refuse_tracked_folder(folder)


def test_faces_and_reports_of_a_comparison_cannot_land_in_tracked_folders(tmp_path, capsys, monkeypatch):
    def no_model(*args):
        raise AssertionError("refused before anything is loaded or written")

    monkeypatch.setattr(heldout, "onnx_predictor", no_model)
    base = ["--model", str(tmp_path), "--data", str(tmp_path), "--sequences", "x-*"]
    monkeypatch.chdir(ROOT)
    for flags in (["--out", "heldout.json"], ["--sheet", "heldout.png"], ["--sheet", str(ROOT / "docs" / "sheet.png")], ["--out", str(tmp_path / "ok.json"), "--sheet", "heldout.png"]):
        assert heldout.main(base + flags) == 2
        assert "Git" in capsys.readouterr().err
    assert not (ROOT / "heldout.json").exists() and not (ROOT / "heldout.png").exists()


def test_a_folder_elsewhere_in_a_repository_counts_as_tracked_unless_ignored(tmp_path):
    repo = tmp_path / "other"
    repo.mkdir()
    done = subprocess.run(["git", "init", "-q", str(repo)], capture_output=True, check=False)
    if done.returncode != 0:
        pytest.skip("no git")
    (repo / ".gitignore").write_text("kept-out/\n", encoding="utf-8")
    assert layout.in_tracked_folder(repo / "results" / "deep") is True
    assert layout.in_tracked_folder(repo / "kept-out") is False
    assert layout.in_tracked_folder(tmp_path / "elsewhere") is False
    # Above this folder, in the repository it belongs to: the path alone would miss it.
    parent = ROOT.parent
    if subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=parent, capture_output=True, check=False).returncode == 0:
        assert layout.in_tracked_folder(parent / "faces") is True


def test_the_guide_writes_nothing_with_faces_next_to_the_code():
    guide = (ROOT / "docs" / "plan-b-training.md").read_text(encoding="utf-8")
    for line in guide.splitlines():
        if "training.heldout" in line and "--out" in line:
            for flag in ("--out", "--sheet"):
                value = line.split(flag + " ")[1].split()[0]
                assert "/" in value and not value.startswith(("docs", "tests", "training", "makeup_cam")), line


def test_weights_are_ignored_by_git_wherever_a_tool_was_told_to_write_them():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    for pattern in ("*.onnx", "*.pt"):
        assert pattern in ignored


def test_sequence_writer_is_all_or_nothing(tmp_path):
    frame = synthetic_frame(0)
    writer = pairs.SequenceWriter(tmp_path, "s-a", "classic_pipeline", "off", 30.0)
    for _ in range(pairs._MIN_SEQUENCE):
        writer.add(frame, frame, FRAME_LANDMARKS, np.ones(FRAME_SHAPE[:2], np.float32))
        assert not (tmp_path / "s-a").exists()
    finished = writer.finish()
    assert finished == tmp_path / "s-a"
    sequence = layout.read_sequence(finished)
    assert sequence.frames == pairs._MIN_SEQUENCE and sequence.has_masks
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())
    short = pairs.SequenceWriter(tmp_path, "s-b", "classic_pipeline", "off", 30.0)
    short.add(frame, frame, FRAME_LANDMARKS)
    assert short.finish() is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s-a"]
    with pytest.raises(layout.DatasetError):
        pairs.SequenceWriter(tmp_path, "s-a", "classic_pipeline", "off", 30.0)


class _FakeResult:
    def __init__(self, frame: np.ndarray, found: bool, tint: int) -> None:
        self.output = frame + tint
        self.landmarks = FRAME_LANDMARKS if found else None
        self.face_found = found


def test_a_standin_with_the_same_preset_on_both_sides_is_refused(tmp_path):
    with pytest.raises(layout.DatasetError):
        pairs.check_standin_presets("natural", "natural")
    pairs.check_standin_presets("off", "natural")
    pairs.check_standin_presets("natural", "polished")


def test_the_standin_input_preset_can_be_chosen(tmp_path, monkeypatch):
    seen = {}

    def fake(args):
        seen.update(vars(args))
        return 0

    monkeypatch.setattr(pairs, "command_standin", fake)
    assert pairs.main(["standin", "--input-preset", "natural", "--target-preset", "polished", "--out", str(tmp_path)]) == 0
    assert seen["input_preset"] == "natural" and seen["target_preset"] == "polished"
    assert pairs.main(["standin", "--out", str(tmp_path)]) == 0
    assert seen["input_preset"] == "off"


def test_pair_frames_splits_runs_and_skips_the_fade():
    frames = [(np.full(FRAME_SHAPE, i, np.uint8), None) for i in range(12)]
    lost = {5}
    stream = list(pairs.pair_frames(frames, None, lambda f, t: _FakeResult(f, int(f[0, 0, 0]) not in lost, 100), None, 2))
    runs, current = [], []
    for item in stream:
        if item is None:
            runs.append(current)
            current = []
        else:
            current.append(item)
    runs.append(current)
    assert [[int(f.input[0, 0, 0]) for f in run] for run in runs] == [[2, 3, 4], [8, 9, 10, 11]]
    assert int(runs[0][0].target[0, 0, 0]) == 102  # target side through the (fake) pipeline
    assert runs[0][0].landmarks.shape == (478, 2)


def test_pair_frames_with_an_aligned_target_source():
    frames = [(np.full(FRAME_SHAPE, i, np.uint8), np.full(FRAME_SHAPE, 200 - i, np.uint8)) for i in range(5)]
    stream = list(pairs.pair_frames(frames, lambda f, t: _FakeResult(f, True, 1), None, None, 1))
    assert [int(f.input[0, 0, 0]) for f in stream] == [2, 3, 4, 5]
    assert [int(f.target[0, 0, 0]) for f in stream] == [199, 198, 197, 196]


def test_presets_that_move_the_face_are_refused():
    from makeup_cam.presets import load_preset

    assert pairs.reshape_is_neutral(load_preset("natural"))
    assert not pairs.reshape_is_neutral(load_preset("strong"))


def test_test_clips_are_prepared_like_the_other_video_checks():
    frame = pairs.prepare_test_clip_frame(np.zeros((360, 480, 3), np.uint8))
    assert frame.shape == (720, 1280, 3)


def test_matching_finds_the_same_pose_and_expression():
    rng = np.random.default_rng(0)
    base = pairs.normalized_landmarks(LANDMARKS)
    moves = rng.normal(0.0, 0.01, (5, 478, 2)).astype(np.float32)
    candidates = base[None] + moves
    order = np.array([3, 0, 4])
    found, distance = pairs.match_frames(candidates[order] + 1e-4, candidates)
    assert found.tolist() == order.tolist()
    assert (distance < 0.002).all()


def test_normalized_landmarks_ignore_position_and_size():
    moved = LANDMARKS * 0.7 + np.array([40.0, -30.0], np.float32)
    assert np.abs(pairs.normalized_landmarks(moved) - pairs.normalized_landmarks(LANDMARKS)).max() < 1e-3


def test_alignment_recovers_a_similarity_and_rejects_another_expression():
    angle = np.deg2rad(3.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32) * 1.02
    makeup = LANDMARKS @ rotation.T + np.array([7.0, -4.0], np.float32)
    alignment = pairs.align_landmarks(LANDMARKS, makeup)
    assert alignment is not None and alignment.fit_error < 0.002
    changed = makeup.copy()
    changed[regions.LIPS_INNER] += np.array([0.0, 25.0], np.float32)
    changed[regions.FACE_OVAL] += np.array([0.0, 25.0], np.float32)
    assert pairs.align_landmarks(LANDMARKS, changed).fit_error > pairs._MAX_FIT_ERROR


def test_target_is_put_in_place_and_colour_matched():
    frame = np.full((720, 1280, 3), (90, 120, 150), np.uint8)
    y, x = np.mgrid[0:720, 0:1280]
    frame[..., 1] = (120 + 30 * np.sin(x / 40.0)).astype(np.uint8)
    matrix = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0]])  # made-up frame to bare frame
    makeup = cv2.warpAffine(frame, cv2.invertAffineTransform(matrix), (1280, 720), borderMode=cv2.BORDER_REPLICATE)
    darker = np.clip(makeup.astype(np.float32) * np.array([0.9, 0.9, 0.95], np.float32), 0, 255).astype(np.uint8)
    target = pairs.align_target(frame, darker, pairs.Alignment(matrix, 0.0), LANDMARKS)
    assert np.abs(target.astype(np.float32) - frame.astype(np.float32))[100:600, 100:1100].mean() < 2.0


# --------------------------------------------------------------------------
# Separation from the app
# --------------------------------------------------------------------------


def test_the_app_never_imports_training():
    for path in (ROOT / "makeup_cam").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] == "training" for alias in node.names), path.name
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or "").split(".")[0] != "training", path.name


def test_training_imports_only_the_light_parts_of_the_app_at_module_level():
    """Contract section 10. ``.venv-train`` has no MediaPipe, so anything more must wait inside a function."""
    allowed = {"makeup_cam", "makeup_cam.regions", "makeup_cam.neural", "makeup_cam.neural.geometry"}
    names = {"regions", "geometry"}
    for path in (ROOT / "training").glob("*.py"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").startswith("makeup_cam"):
                assert node.module in allowed, path.name
                assert {alias.name for alias in node.names} <= names, path.name
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith("makeup_cam") for alias in node.names), path.name


def test_torch_free_modules_really_are():
    code = (
        "import sys\n"
        "import training.config, training.layout, training.masks, training.augment, training.samples\n"
        "import training.metadata, training.heldout, training.latency, training.pairs\n"
        "bad = [m for m in ('torch', 'onnx', 'onnxruntime', 'mediapipe', 'PySide6') if m in sys.modules]\n"
        "sys.exit(repr(bad) if bad else 0)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def test_no_weights_or_media_in_the_training_folder():
    suffixes = {".pt", ".pth", ".onnx", ".ckpt", ".safetensors", ".png", ".jpg", ".mkv", ".mp4", ".flv", ".npy"}
    assert not [p.name for p in (ROOT / "training").rglob("*") if p.suffix.lower() in suffixes]
