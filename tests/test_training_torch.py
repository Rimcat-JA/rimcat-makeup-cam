"""Training code that needs torch, onnx and onnxruntime: skipped in the app's environment, run in ``.venv-train``.

Small synthetic frames in pytest's temp folder. No test here says anything about any look.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_training_core import make_sequence  # noqa: E402

from training import checkpoint, config, layout, losses, models, samples, train  # noqa: E402
from training.models import DeltaUNet  # noqa: E402

TINY = {"base": 4, "depth": 2}


def tiny_config(steps: int = 6, **sections: dict) -> config.RunConfig:
    values = {
        "stage": "student",
        "data": {"size": 256, "heldout_sequences": ["held-*"]},
        "model": dict(TINY),
        "train": {
            "steps": steps,
            "batch_size": 2,
            "num_workers": 0,
            "warmup_steps": 1,
            "checkpoint_every": 2,
            "validate_every": 1000,
            "validate_frames": 1,
            "log_every": 2,
            "ema_decay": 0.9,
            "threads": 2,
        },
        "export": {"name": "test-student", "version": "0.0.1"},
    }
    for name, section in sections.items():
        values[name] = {**values.get(name, {}), **section} if isinstance(section, dict) else section
    return config.from_dict(values)


@pytest.fixture(scope="module")
def data_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("pairs")
    for name in ("p-a", "q-a", "held-a"):
        make_sequence(root, name, 4, change=(-10, -3, 20))
    return root


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [256, 384, 512])
def test_student_shape_and_exact_zero_at_the_start(size):
    model = DeltaUNet(config.ModelConfig(**TINY)).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, size, size))
    assert out.shape == (1, 3, size, size)
    assert (out == 0.0).all()  # an untrained model changes nothing


def test_teacher_options_build_and_run():
    model = DeltaUNet(config.ModelConfig(base=8, depth=3, blocks=2, residual=True, norm="instance", attention=True)).eval()
    with torch.no_grad():
        assert model(torch.rand(1, 3, 64, 64)).shape == (1, 3, 64, 64)


def test_no_feature_goes_silent_on_negative_input():
    """A ReLU student went dead on the stand-in pairs and stayed there. Leaky units keep passing gradients."""
    model = DeltaUNet(config.ModelConfig(**TINY))
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, -0.1)  # every unit negative: a ReLU network would give no gradient at all
    torch.nn.init.constant_(model.last.weight, 0.001)
    torch.nn.init.zeros_(model.last.bias)
    model(torch.rand(1, 3, 32, 32)).sum().backward()
    assert model.first[0].conv1.weight.grad.abs().sum() > 0.0


def test_output_stays_in_range_with_large_weights():
    model = DeltaUNet(config.ModelConfig(**TINY)).eval()
    torch.nn.init.constant_(model.last.bias, 50.0)
    with torch.no_grad():
        assert model(torch.rand(1, 3, 64, 64)).abs().max() <= 1.0


def test_shipped_sizes_are_what_their_configs_say():
    root = Path(__file__).resolve().parent.parent / "training" / "configs"
    student = models.weights_megabytes(DeltaUNet(config.load(root / "student-512.json").model))
    assert 8.0 < student < 9.0  # contract 7.3: the size the runtime has room for


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------


def test_lab_agrees_with_opencv():
    image = np.random.default_rng(0).random((16, 16, 3), dtype=np.float32)
    expected = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    ours = losses.bgr_to_lab(torch.from_numpy(image.transpose(2, 0, 1))[None])[0].numpy().transpose(1, 2, 0)
    assert np.abs(ours[..., 0] * 100.0 - expected[..., 0]).max() < 0.2
    assert np.abs(ours[..., 1:] * 100.0 - expected[..., 1:]).max() < 0.5


def test_lab_gradient_is_finite_at_black_and_white():
    image = torch.tensor([0.0, 1.0, 0.5]).view(1, 3, 1, 1).requires_grad_()
    losses.bgr_to_lab(image).sum().backward()
    assert torch.isfinite(image.grad).all()


def test_masked_mean_of_an_empty_mask_is_zero_not_nan():
    assert losses.masked_mean(torch.ones(1, 3, 4, 4), torch.zeros(1, 1, 4, 4)) == 0.0


def test_warp_matches_the_data_side():
    image = np.random.default_rng(0).random((32, 32, 3), dtype=np.float32)
    flow = np.stack([np.full((32, 32), 2.5, np.float32), np.full((32, 32), -1.0, np.float32)], axis=2)
    expected = samples.warp_by_flow(image, flow)
    ours = losses.warp(torch.from_numpy(image.transpose(2, 0, 1))[None], torch.from_numpy(flow.transpose(2, 0, 1))[None])
    assert np.abs(ours[0].numpy().transpose(1, 2, 0) - expected)[4:-4, 4:-4].max() < 1e-4


def test_temporal_loss_counts_only_change_beyond_the_targets():
    size = 16
    flow = torch.zeros(1, 2, size, size)
    mask = torch.ones(1, 1, size, size)
    still = torch.full((1, 3, size, size), 0.1)
    moved = still + 0.05
    assert losses.temporal_loss(still, still, still, still, flow, mask, mask) == 0.0
    flickers = losses.temporal_loss(still, moved, still, still, flow, mask, mask)
    follows = losses.temporal_loss(still, moved, still, moved, flow, mask, mask)
    assert flickers > follows > 0.0
    assert flickers == pytest.approx(0.05 * 1.1, rel=1e-3)
    assert losses.temporal_loss(still, moved, still, still, flow, mask * 0.0, mask) == 0.0  # a failed flow does not count


def test_low_frequency_guard_ignores_what_the_target_also_has():
    size = 32
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    fine = ((x + y) % 2).float().view(1, 1, size, size).repeat(1, 3, 1, 1) * 0.1
    mask = torch.ones(1, 1, size, size)
    smooth = torch.full_like(fine, 0.1)
    assert losses.low_frequency_loss(fine, smooth, mask) > 0.01
    assert losses.low_frequency_loss(fine, fine, mask) == 0.0
    assert losses.low_frequency_loss(smooth, smooth, mask) == 0.0


def test_region_loss_sees_only_the_forbidden_area():
    delta = torch.zeros(1, 3, 8, 8)
    weight = torch.ones(1, 1, 8, 8)
    weight[..., :4] = 0.0
    assert losses.region_loss(delta, weight) == 0.0
    delta[..., 4:] = 0.5  # all allowed
    assert losses.region_loss(delta, weight) == pytest.approx(0.0, abs=1e-6)
    delta[..., :4] = 0.5
    assert losses.region_loss(delta, weight) == pytest.approx(0.25, abs=1e-4)  # half the crop, per pixel of the crop


def test_magnitude_loss_bites_hard_beyond_the_clamp():
    mask = torch.ones(1, 1, 4, 4)
    small = losses.magnitude_loss(torch.full((1, 3, 4, 4), 0.1), 0.25, mask)
    large = losses.magnitude_loss(torch.full((1, 3, 4, 4), 0.35), 0.25, mask)
    assert small == pytest.approx(0.1, abs=1e-5)
    assert large == pytest.approx(0.35 + 1.0, abs=1e-5)


def _batch(has_next: float = 1.0) -> losses.Batch:
    size = 32
    return losses.Batch(
        input=torch.rand(2, 2, 3, size, size),
        delta=torch.zeros(2, 2, 3, size, size) + 0.05,
        weight=torch.ones(2, 2, 1, size, size),
        flow=torch.zeros(2, 2, size, size),
        flow_confidence=torch.ones(2, 1, size, size),
        has_next=torch.full((2,), has_next),
    )


def _losses(prediction: torch.Tensor, batch: losses.Batch, loss: config.LossConfig, **kwargs) -> dict:
    window = torch.ones(1, 1, 32, 32)
    return losses.compute_losses(prediction, batch, loss, window, 0.25, torch.full((2, 1, 1, 1), 0.5), **kwargs)


def test_all_terms_are_finite_and_reach_the_model():
    model = DeltaUNet(config.ModelConfig(base=4, depth=2))
    batch = _batch()
    prediction = model(batch.input.reshape(4, 3, 32, 32))
    terms = _losses(prediction, batch, config.LossConfig())
    assert {"lab", "lab_partial", "detail", "region", "magnitude", "low_frequency", "temporal", "lag", "total"} <= set(terms)
    assert all(torch.isfinite(value) for value in terms.values())
    terms["total"].backward()
    assert model.last.weight.grad.abs().sum() > 0.0


def test_no_temporal_terms_without_neighbours():
    batch = _batch(has_next=0.0)
    terms = _losses(torch.zeros(4, 3, 32, 32), batch, config.LossConfig())
    assert "temporal" not in terms and "lag" not in terms


def test_a_loss_weight_of_zero_removes_the_term():
    terms = _losses(torch.zeros(4, 3, 32, 32), _batch(), config.LossConfig(region=0.0, detail=0.0))
    assert "region" not in terms and "detail" not in terms


def test_distillation_pulls_towards_the_teacher():
    batch = _batch()
    prediction = torch.zeros(4, 3, 32, 32)
    away = _losses(prediction, batch, config.LossConfig(), teacher_delta=torch.full_like(prediction, 0.1))
    same = _losses(prediction, batch, config.LossConfig(), teacher_delta=prediction)
    assert away["distill"] == pytest.approx(0.1, abs=1e-4)
    assert same["distill"] == 0.0


def test_identity_guard_with_a_fake_embedder():
    class Fake(torch.nn.Module):
        def forward(self, rgb: torch.Tensor) -> torch.Tensor:
            return rgb.mean(dim=(2, 3))

    batch = _batch()
    prediction = torch.zeros(4, 3, 32, 32)
    terms = _losses(prediction, batch, config.LossConfig(identity=1.0), embedder=Fake())
    assert terms["identity"] == pytest.approx(0.0, abs=1e-5)
    changed = _losses(prediction + torch.tensor([0.3, 0.0, -0.3]).view(1, 3, 1, 1), batch, config.LossConfig(identity=1.0), embedder=Fake())
    assert changed["identity"] > 0.001


def test_perceptual_weights_are_local_only(tmp_path, data_root):
    """A missing file is an error at the start; nothing downloads and nothing silently goes without."""
    tv = pytest.importorskip("torchvision")
    weights = tmp_path / "vgg.pt"
    torch.save(tv.models.vgg16(weights=None).state_dict(), weights)
    loss = losses.PerceptualLoss(str(weights))
    image = torch.rand(1, 3, 64, 64)
    mask = torch.ones(1, 1, 64, 64)
    assert loss(image, image, mask) == 0.0
    assert loss(image * 0.5, image, mask) > 0.0
    assert loss(image * 0.5, image, mask * 0.0) == 0.0
    with pytest.raises(SystemExit):
        train.run(tiny_config(loss={"perceptual": 1.0}), data_root, tmp_path / "run")


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


def _state(run_dir: Path) -> dict:
    return checkpoint.load(run_dir / "ckpt" / checkpoint.LAST)


def test_a_run_writes_numbers_only_and_keeps_people_apart(tmp_path, data_root):
    run_dir = tmp_path / "run"
    result = train.run(tiny_config(), data_root, run_dir)
    assert result.steps == 6
    assert result.summary["data"]["groups"] == 2  # "held" did not train
    assert result.summary["heldout_sequences"] == ["held-a"]
    assert result.summary["stand_in"] is True
    assert result.summary["device"] == "cpu"
    assert (run_dir / "metrics.jsonl").read_text(encoding="utf-8").count("\n") >= 3
    for name in ("run_config.json", "summary.json", "metrics.jsonl"):
        assert str(data_root) not in (run_dir / name).read_text(encoding="utf-8")
        assert str(tmp_path) not in (run_dir / name).read_text(encoding="utf-8")
    assert str(data_root) not in json.dumps({k: v for k, v in _state(run_dir).items() if k in ("config", "data", "teacher")})
    assert config.from_dict(json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))).stage == "student"


def test_validation_compares_with_doing_nothing(tmp_path, data_root):
    run_dir = tmp_path / "run"
    result = train.run(tiny_config(train={"steps": 2, "validate_every": 2}), data_root, run_dir)
    assert result.summary["heldout"]["identity_lab"] > 0.0
    assert result.summary["heldout"]["worst_group"] == "held"


def test_interrupted_and_resumed_run_equals_one_that_never_stopped(tmp_path, data_root):
    straight = train.run(tiny_config(), data_root, tmp_path / "straight")
    assert straight.steps == 6
    train.run(tiny_config(), data_root, tmp_path / "stopped", stop_after=3)
    assert _state(tmp_path / "stopped")["step"] == 3
    resumed = train.run(tiny_config(), data_root, tmp_path / "stopped", resume=True)
    assert resumed.steps == 6
    a, b = _state(tmp_path / "straight"), _state(tmp_path / "stopped")
    assert a["step"] == b["step"] == 6
    for key in a["model"]:
        assert torch.allclose(a["model"][key], b["model"][key], atol=1e-5), key
        assert torch.allclose(a["ema"][key], b["ema"][key], atol=1e-5), key


def test_an_existing_run_is_not_overwritten_and_a_changed_one_not_resumed(tmp_path, data_root):
    run_dir = tmp_path / "run"
    train.run(tiny_config(steps=2), data_root, run_dir)
    with pytest.raises(SystemExit):
        train.run(tiny_config(steps=2), data_root, run_dir)
    with pytest.raises(SystemExit):
        train.run(tiny_config(steps=4, train={"steps": 4, "learning_rate": 0.1}), data_root, run_dir, resume=True)
    assert train.run(tiny_config(steps=4, train={"steps": 4}), data_root, run_dir, resume=True).steps == 4  # may run longer


def test_old_checkpoints_are_dropped_and_writes_are_atomic(tmp_path):
    folder = tmp_path / "ckpt"
    for step in range(1, 5):
        checkpoint.save(folder, step, {"step": step}, keep=2)
    assert sorted(p.name for p in folder.iterdir()) == ["last.pt", "step-00000003.pt", "step-00000004.pt"]
    assert checkpoint.load(folder / checkpoint.LAST)["step"] == 4


def test_loss_falls_on_a_mapping_that_can_be_learned(tmp_path):
    # Masks that allow everything: there is no face in these frames, so a model could not see where the oval is.
    root = tmp_path / "pairs"
    for name in ("p-a", "held-a"):
        make_sequence(root, name, 4, change=(-10, -3, 20), with_masks=True)
    result = train.run(
        tiny_config(
            steps=40,
            model={"base": 8, "depth": 2},
            train={"steps": 40, "learning_rate": 0.003, "warmup_steps": 2, "checkpoint_every": 40, "validate_every": 40, "log_every": 10},
            augment={"enabled": False},
            loss={"lab_partial": 0.0, "temporal": 0.0, "lag": 0.0},
        ),
        root,
        tmp_path / "run",
    )
    lines = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    totals = [line["losses"]["total"] for line in lines if "losses" in line]
    assert totals[-1] < 0.7 * totals[0]
    assert result.last_loss < result.first_loss
    assert result.summary["heldout"]["ratio_mean"] < 0.8  # the same change on another sequence


def test_distillation_run_and_who_may_teach(tmp_path, data_root):
    teacher_config = tiny_config(steps=2, model={"base": 6, "depth": 2, "residual": True, "norm": "instance"}, stage="teacher")
    train.run(teacher_config, data_root, tmp_path / "teacher")
    student = train.run(tiny_config(steps=2), data_root, tmp_path / "student", teacher_path=tmp_path / "teacher" / "ckpt" / "last.pt")
    assert student.summary["teacher"]["step"] == 2
    lines = (tmp_path / "student" / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"distill"' in lines
    with pytest.raises(SystemExit):  # a student cannot teach
        train.run(tiny_config(steps=1), data_root, tmp_path / "other", teacher_path=tmp_path / "student" / "ckpt" / "last.pt")
    with pytest.raises(SystemExit):  # a teacher has no teacher
        train.run(teacher_config, data_root, tmp_path / "t2", teacher_path=tmp_path / "teacher" / "ckpt" / "last.pt")


def test_non_finite_loss_skips_the_step_then_stops(tmp_path, data_root, monkeypatch):
    def broken(*args, **kwargs):
        return {"total": torch.tensor(float("nan"))}

    monkeypatch.setattr(train, "compute_losses", broken)
    with pytest.raises(RuntimeError):
        train.run(tiny_config(steps=30, train={"steps": 30, "checkpoint_every": 100}), data_root, tmp_path / "run")


def test_dataset_errors_reach_the_command_line_as_messages(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text('{"stage": "student"}', encoding="utf-8")
    assert train.main(["--config", str(path), "--data", str(tmp_path / "missing"), "--run-dir", str(tmp_path / "run")]) == 2
    assert "データのフォルダ" in capsys.readouterr().err


def test_learning_rate_schedule_depends_on_the_step_alone():
    cfg = tiny_config(steps=100, train={"steps": 100, "warmup_steps": 10, "learning_rate": 0.01})
    assert train.learning_rate(cfg, 0) == pytest.approx(0.001)
    assert train.learning_rate(cfg, 10) == pytest.approx(0.01)
    assert train.learning_rate(cfg, 99) < 0.001
    assert train.ema_decay(cfg, 0) < 0.2 and train.ema_decay(cfg, 10_000) == cfg.train.ema_decay


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from training import export, heldout  # noqa: E402


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, data_root) -> Path:
    run_dir = tmp_path_factory.mktemp("run")
    train.run(tiny_config(steps=4, train={"steps": 4}), data_root, run_dir)
    return run_dir / "ckpt" / "last.pt"


def test_export_writes_what_the_contract_asks(tmp_path, trained_run, data_root):
    out = tmp_path / "test-student-0.0.1"
    result = export.export_checkpoint(trained_run, data_root, out)
    assert sorted(p.name for p in out.iterdir()) == ["export_check.json", "model.json", "model.onnx"]
    check, meta = result["check"], result["model_json"]
    assert check["passed"] and check["max_abs_difference_against_torch"] <= 1e-3
    assert "LeakyRelu" in check["ops"] and "Relu" not in check["ops"]
    assert meta["stand_in"] is True  # trained on classic-pipeline pairs: can never reach a call
    assert meta["training_data"]["kind"] == "classic_pipeline"
    assert meta["sha256"] == __import__("hashlib").sha256((out / "model.onnx").read_bytes()).hexdigest()
    assert str(tmp_path) not in (out / "model.json").read_text(encoding="utf-8")

    proto = onnx.load(str(out / "model.onnx"))
    assert [o.version for o in proto.opset_import if o.domain in ("", "ai.onnx")] == [20]
    assert not metadata_problems(proto)
    session = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    assert (session.get_inputs()[0].name, session.get_inputs()[0].shape) == ("input", [1, 3, 256, 256])
    assert (session.get_outputs()[0].name, session.get_outputs()[0].shape) == ("delta", [1, 3, 256, 256])
    assert session.get_inputs()[0].type == "tensor(float)" and session.get_outputs()[0].type == "tensor(float)"


def metadata_problems(proto) -> list[str]:
    return export.check_graph(proto, 256, 20)


@pytest.mark.parametrize("size", [384, 512])
def test_export_at_another_allowed_size(tmp_path, trained_run, data_root, size):
    result = export.export_checkpoint(trained_run, data_root, tmp_path / "m", size=size)
    assert result["model_json"]["input"]["size"] == size
    assert result["model_json"]["training_run"]["trained_size"] == 256


def test_export_refuses_sizes_teachers_and_tracked_folders(tmp_path, trained_run, data_root):
    with pytest.raises(export.ExportError):
        export.export_checkpoint(trained_run, data_root, tmp_path / "m", size=300)
    with pytest.raises(export.ExportError):
        export.export_checkpoint(trained_run, data_root, Path(export.__file__).parent / "model-here")
    teacher = tiny_config(steps=1, stage="teacher", model={"base": 4, "depth": 2})
    train.run(teacher, data_root, tmp_path / "teacher")
    with pytest.raises(export.ExportError):
        export.export_checkpoint(tmp_path / "teacher" / "ckpt" / "last.pt", data_root, tmp_path / "t")


def test_a_run_cannot_keep_its_checkpoints_in_a_tracked_folder(data_root):
    run_dir = Path(train.__file__).parent / "run-here"
    with pytest.raises(layout.DatasetError):
        train.run(tiny_config(steps=1), data_root, run_dir)
    assert not run_dir.exists()


def test_float16_graph_has_float32_edges_and_matches(tmp_path, data_root):
    run_dir = tmp_path / "run"
    train.run(tiny_config(steps=2, model={**TINY, "float16": True}), data_root, run_dir)
    out = tmp_path / "m"
    result = export.export_checkpoint(run_dir / "ckpt" / "last.pt", data_root, out)
    assert result["check"]["float16"] is True and "Cast" in result["check"]["ops"]
    assert result["check"]["max_abs_difference_against_torch"] <= 5e-3
    assert not export.check_graph(onnx.load(str(out / "model.onnx")), 256, 20)


def test_a_wrong_graph_is_caught_and_leaves_no_model_behind(tmp_path, monkeypatch, trained_run, data_root):
    def wrong(*args, **kwargs):
        raise export.ExportError("broken")

    monkeypatch.setattr(export, "check_export", wrong)
    out = tmp_path / "m"
    with pytest.raises(export.ExportError):
        export.export_checkpoint(trained_run, data_root, out)
    assert not (out / "model.onnx").exists() and not (out / "model.json").exists()


def test_exporting_over_a_model_needs_saying_so(tmp_path, trained_run, data_root):
    out = tmp_path / "m"
    export.export_checkpoint(trained_run, data_root, out)
    before = (out / "model.onnx").read_bytes()
    with pytest.raises(export.ExportError):
        export.export_checkpoint(trained_run, data_root, out)
    assert (out / "model.onnx").read_bytes() == before
    export.export_checkpoint(trained_run, data_root, out, overwrite=True)


def test_check_graph_names_each_rule():
    from onnx import TensorProto, helper

    def graph(op: str, shape: list[int], opset: int, **attrs: object):
        node = helper.make_node(op, ["input"], ["delta"], **attrs)
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
        out = helper.make_tensor_value_info("delta", TensorProto.FLOAT, shape)
        return helper.make_model(helper.make_graph([node], "g", [inp], [out]), opset_imports=[helper.make_opsetid("", opset)])

    assert not export.check_graph(graph("Tanh", [1, 3, 256, 256], 20), 256, 20)
    assert any("Gelu" in p for p in export.check_graph(graph("Gelu", [1, 3, 256, 256], 20), 256, 20))
    assert any("opset" in p for p in export.check_graph(graph("Tanh", [1, 3, 256, 256], 23), 256, 23))
    assert any("形" in p for p in export.check_graph(graph("Tanh", [1, 3, 256, 128], 20), 256, 20))
    dynamic = graph("Tanh", [1, 3, 256, 256], 20)
    dynamic.graph.input[0].type.tensor_type.shape.dim[2].dim_param = "height"
    assert any("形" in p for p in export.check_graph(dynamic, 256, 20))


def test_heldout_runs_on_an_exported_model(tmp_path, trained_run, data_root):
    out = tmp_path / "m"
    export.export_checkpoint(trained_run, data_root, out)
    report_path = tmp_path / "report.json"
    sheet_path = tmp_path / "sheet.png"
    code = heldout.main(
        ["--model", str(out), "--data", str(data_root), "--sequences", "held-*", "--frames", "1", "--out", str(report_path), "--sheet", str(sheet_path)]
    )
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["conditions"]["stand_in"] is True and report["conditions"]["plumbing_test_only"] is True
    assert report["summary"]["identity_lab"] > 0.0
    assert str(tmp_path) not in report_path.read_text(encoding="utf-8")
    assert sheet_path.stat().st_size > 1000


def test_layout_module_is_reused_for_real_kinds_without_stand_in(tmp_path):
    """Real pairs give a model that is not marked a stand-in, and that alone: the gate's other conditions stay."""
    root = tmp_path / "pairs"
    make_sequence(root, "2026-11-10_main-step03_head", 3, kind="3d_render")
    summary = layout.data_summary(layout.find_sequences(root))
    cfg = tiny_config(steps=1, train={"steps": 1}, data={"heldout_sequences": []})
    train.run(cfg, root, tmp_path / "run")
    result = export.export_checkpoint(tmp_path / "run" / "ckpt" / "last.pt", root, tmp_path / "m")
    assert summary["kinds"] == ["3d_render"]
    assert result["model_json"]["stand_in"] is False
    assert result["model_json"]["evaluation"]["verdict"] == "none"  # still not allowed in a call
