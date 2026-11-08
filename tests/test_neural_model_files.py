"""Where models live, ``model.json`` and the gate (contract section 6)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from neural_fakes import write_model_folder

from makeup_cam.neural import model_files
from makeup_cam.neural.model_files import ModelRefused, check_gate, load_spec


def passed_model(folder: Path, **changes: Any) -> Path:
    """A model folder that goes through the gate."""
    write_model_folder(folder, stand_in=False)
    meta = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    report = json.dumps({"verdict": "pass", "eligible_for_calls": True, "model": {"sha256": meta["sha256"]}}).encode("utf-8")
    (folder / "evaluation.json").write_bytes(report)
    meta["evaluation"] = {
        "verdict": "pass",
        "report": "evaluation.json",
        "report_sha256": hashlib.sha256(report).hexdigest(),
        "model_sha256": meta["sha256"],
        "date": "2026-11-06T12:00:00",
    }
    meta.update(changes)
    (folder / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder


def test_a_passed_model_goes_through(tmp_path: Path) -> None:
    check_gate(load_spec(passed_model(tmp_path / "m")))


def test_a_stand_in_never_reaches_a_camera_even_with_a_pass(tmp_path: Path) -> None:
    with pytest.raises(ModelRefused) as refused:
        check_gate(load_spec(passed_model(tmp_path / "m", stand_in=True)))
    assert refused.value.status == model_files.NOT_PASSED_MESSAGE


def test_a_missing_stand_in_flag_counts_as_a_stand_in(tmp_path: Path) -> None:
    folder = passed_model(tmp_path / "m")
    meta = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    del meta["stand_in"]
    (folder / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    assert load_spec(folder).stand_in


@pytest.mark.parametrize("verdict", ["none", "fail", "incomplete"])
def test_only_a_pass_passes(tmp_path: Path, verdict: str) -> None:
    folder = passed_model(tmp_path / "m")
    meta = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    meta["evaluation"]["verdict"] = verdict
    (folder / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ModelRefused):
        check_gate(load_spec(folder))


def test_a_changed_model_file_is_refused_with_and_without_the_offline_override(tmp_path: Path) -> None:
    folder = passed_model(tmp_path / "m")
    (folder / "model.onnx").write_bytes(b"something else")
    spec = load_spec(folder)
    with pytest.raises(ModelRefused):
        check_gate(spec)
    with pytest.raises(ModelRefused):
        check_gate(spec, allow_unevaluated=True)


def test_a_changed_report_is_refused(tmp_path: Path) -> None:
    folder = passed_model(tmp_path / "m")
    (folder / "evaluation.json").write_bytes(b'{"verdict": "pass", "eligible_for_calls": true, "edited": true}')
    with pytest.raises(ModelRefused):
        check_gate(load_spec(folder))


def test_an_evaluation_of_another_file_is_refused(tmp_path: Path) -> None:
    folder = passed_model(tmp_path / "m")
    meta = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    meta["evaluation"]["model_sha256"] = "0" * 64
    (folder / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ModelRefused):
        check_gate(load_spec(folder))


def test_the_offline_override_lets_an_unevaluated_stand_in_run(tmp_path: Path) -> None:
    spec = load_spec(write_model_folder(tmp_path / "m"))
    with pytest.raises(ModelRefused):
        check_gate(spec)
    check_gate(spec, allow_unevaluated=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": 2},
        {"crop": {"definition": 2}},
        {"input": {"name": "input", "size": 300, "layout": "NCHW", "colors": "BGR", "dtype": "float32"}},
        {"input": {"name": "input", "size": 256, "layout": "NCHW", "colors": "RGB", "dtype": "float32"}},
        {"output": {"name": "image", "kind": "image", "layout": "NCHW", "colors": "BGR", "dtype": "float32"}},
        {"output": {"name": "delta", "kind": "delta", "layout": "NCHW", "colors": "BGR", "dtype": "float32", "max_abs_delta": 0.5}},
        {"output": {"name": "delta", "kind": "delta", "layout": "NCHW", "colors": "BGR", "dtype": "float32", "temporal_alpha": 0.0}},
        {"file": "../elsewhere/model.onnx"},
        {"file": "C:model.onnx"},
    ],
)
def test_what_the_runtime_does_not_know_is_refused(tmp_path: Path, changes: dict[str, Any]) -> None:
    with pytest.raises(ModelRefused):
        load_spec(write_model_folder(tmp_path / "m", **changes))


def test_broken_or_missing_metadata_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ModelRefused) as missing:
        load_spec(tmp_path)
    assert missing.value.status == model_files.NO_MODEL_MESSAGE
    (tmp_path / "model.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelRefused):
        load_spec(tmp_path)
    (tmp_path / "model.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ModelRefused):
        load_spec(tmp_path)


def test_unknown_extra_keys_are_ignored(tmp_path: Path) -> None:
    assert load_spec(write_model_folder(tmp_path / "m", something_new={"a": 1})).name == "fake"


def test_the_folder_comes_from_the_environment_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(model_files.NEURAL_DIR_ENV, str(tmp_path))
    assert model_files.neural_dir() == tmp_path
    monkeypatch.delenv(model_files.NEURAL_DIR_ENV)
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path))
    assert model_files.neural_dir() == tmp_path / "neural_models"


def test_active_model_is_chosen_by_its_file(tmp_path: Path) -> None:
    with pytest.raises(ModelRefused) as refused:
        model_files.active_model_dir(tmp_path)
    assert refused.value.status == model_files.NO_MODEL_MESSAGE
    write_model_folder(tmp_path / "fake-0.0.0")
    assert model_files.list_models(tmp_path) == ["fake-0.0.0"]
    model_files.set_active_model("fake-0.0.0", tmp_path)
    assert model_files.active_model_name(tmp_path) == "fake-0.0.0"
    assert model_files.active_model_dir(tmp_path) == tmp_path / "fake-0.0.0"
    with pytest.raises(ValueError):
        model_files.set_active_model("no-such", tmp_path)


@pytest.mark.parametrize("text", ['{"model": "../outside"}', '{"model": 3}', "[]", "{broken"])
def test_a_bad_active_file_means_no_model(tmp_path: Path, text: str) -> None:
    (tmp_path / "active.json").write_text(text, encoding="utf-8")
    assert model_files.active_model_name(tmp_path) is None


# ---- what a hostile or broken file must not do (review findings) ---------------------------------


def _pad_json(path: Path, megabytes: float) -> None:
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["padding"] = [12345678] * int(megabytes * 1024 * 1024 / 10)
    path.write_text(json.dumps(meta), encoding="utf-8")


def _never_parse(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("an oversized metadata file reached the JSON parser, which holds the GIL while it runs")


def test_an_oversized_model_json_is_refused_without_being_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = write_model_folder(tmp_path / "m")
    _pad_json(folder / "model.json", 3.0)
    monkeypatch.setattr(model_files.json, "loads", _never_parse)
    with pytest.raises(ModelRefused) as refused:
        load_spec(folder)
    assert refused.value.status == model_files.NOT_PASSED_MESSAGE


def test_an_oversized_active_file_means_no_model_without_being_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "active.json").write_text(json.dumps({"model": "a", "padding": "x" * (1 << 21)}), encoding="utf-8")
    monkeypatch.setattr(model_files.json, "loads", _never_parse)
    assert model_files.active_model_name(tmp_path) is None


def test_a_real_sized_model_json_still_loads(tmp_path: Path) -> None:
    folder = write_model_folder(tmp_path / "m", note="x" * 100_000)
    assert load_spec(folder).name == "fake"


def test_deeply_nested_metadata_is_refused_not_raised(tmp_path: Path) -> None:
    folder = write_model_folder(tmp_path / "m")
    (folder / "model.json").write_text("[" * 200_000, encoding="utf-8")
    with pytest.raises(ModelRefused):
        load_spec(folder)
    (tmp_path / "active.json").write_text("[" * 200_000, encoding="utf-8")
    assert model_files.active_model_name(tmp_path) is None


@pytest.mark.parametrize("name", ["NUL", "nul.onnx", "CON", "COM1.onnx", "LPT9", "model\u0000.onnx", "model.onnx.", "model.onnx "])
def test_names_windows_treats_as_devices_or_cuts_short_are_refused(tmp_path: Path, name: str) -> None:
    with pytest.raises(ModelRefused):
        load_spec(write_model_folder(tmp_path / "m", file=name))


def test_a_file_name_a_path_cannot_hold_is_refused_at_the_gate_too(tmp_path: Path) -> None:
    spec = load_spec(write_model_folder(tmp_path / "m"))
    broken = model_files.ModelSpec(**{**spec.__dict__, "file": spec.folder / "model\u0000.onnx"})
    with pytest.raises(ModelRefused):
        check_gate(broken, allow_unevaluated=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"opset": 23},
        {"opset": "20"},
        {"schema": True},
        {"schema": 1.0},
        {"crop": {"definition": True}},
        {"input": {"name": "input", "size": 256.0, "layout": "NCHW", "colors": "BGR", "dtype": "float32"}},
        {"output": {"name": "delta", "kind": "delta", "layout": "NCHW", "colors": "BGR", "dtype": "float32", "max_abs_delta": "0.2"}},
        {"output": {"name": "delta", "kind": "delta", "layout": "NCHW", "colors": "BGR", "dtype": "float32", "temporal_alpha": True}},
        {"output": {"name": "delta", "kind": "delta", "layout": "NCHW", "colors": "BGR", "dtype": "float32", "max_abs_delta": float("nan")}},
    ],
)
def test_numbers_of_the_wrong_type_are_refused(tmp_path: Path, changes: dict[str, Any]) -> None:
    with pytest.raises(ModelRefused):
        load_spec(write_model_folder(tmp_path / "m", **changes))


@pytest.mark.parametrize(
    "report",
    [
        b'{"verdict": "fail", "eligible_for_calls": false}',
        b'{"verdict": "pass", "eligible_for_calls": false}',
        b'{"verdict": "pass", "eligible_for_calls": true, "model": {"sha256": "0000"}}',
        b"[1, 2]",
        b"{broken",
    ],
)
def test_the_gate_reads_the_report_itself_not_only_its_hash(tmp_path: Path, report: bytes) -> None:
    folder = passed_model(tmp_path / "m")
    (folder / "evaluation.json").write_bytes(report)
    meta = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    meta["evaluation"]["report_sha256"] = hashlib.sha256(report).hexdigest()
    (folder / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ModelRefused):
        check_gate(load_spec(folder))


def test_the_command_line_lists_a_broken_folder_in_japanese_and_goes_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from makeup_cam.neural.__main__ import main

    monkeypatch.setenv(model_files.NEURAL_DIR_ENV, str(tmp_path))
    write_model_folder(tmp_path / "standin-1")
    write_model_folder(tmp_path / "deep-1")
    (tmp_path / "deep-1" / "model.json").write_text("[" * 200_000, encoding="utf-8")
    write_model_folder(tmp_path / "null-1")
    meta = json.loads((tmp_path / "null-1" / "model.json").read_text(encoding="utf-8"))
    meta["file"] = "model\u0000.onnx"
    (tmp_path / "null-1" / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    assert main([]) == 0
    out = capsys.readouterr().out
    for name in ("standin-1", "deep-1", "null-1"):
        assert f"{name}: 使えません（{model_files.NOT_PASSED_MESSAGE}）" in out
    assert "stand-in" not in out and "Error" not in out
