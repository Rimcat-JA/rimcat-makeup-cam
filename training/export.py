"""Exports a student to the contract's ONNX model and checks it (sections 3, 6 and 8). Needs torch, onnx and onnxruntime.

    python -m training.export --checkpoint RUN/ckpt/last.pt --data DATA --out .test-media/planb/models/NAME-VERSION

The check runs on the CPU here. It covers everything but DirectML itself: that is what ``training.latency`` (in
``.venv-neural``) and the evaluation tool are for. A model that fails any check is deleted, so that no folder ever holds
a model.onnx that did not pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from . import checkpoint, config as config_module, metadata
from .config import ExportConfig, ModelConfig
from .layout import DatasetError, find_sequences, in_tracked_folder
from .models import DeltaUNet, Float32Edges
from .samples import crop_view

_REPOSITORY = Path(__file__).resolve().parent.parent


class ExportError(RuntimeError):
    """The model breaks a rule of the contract. The message is for the person running the tool."""


def refuse_tracked_folder(folder: Path) -> None:
    """Weights never go to a tracked path (contract 6.1)."""
    if in_tracked_folder(folder):
        raise ExportError("モデルは Git で管理する場所に置けません。.test-media/ の下か、リポジトリの外にしてください")


def graph_nodes(model: onnx.ModelProto) -> list[tuple[str, str, dict[str, Any]]]:
    nodes = []
    for node in model.graph.node:
        attributes = {}
        for attribute in node.attribute:
            value = onnx.helper.get_attribute_value(attribute)
            attributes[attribute.name] = value.decode() if isinstance(value, bytes) else value
        nodes.append((node.domain, node.op_type, attributes))
    return nodes


def check_graph(model: onnx.ModelProto, size: int, opset: int) -> list[str]:
    """Everything that can be told from the file alone."""
    problems = metadata.check_nodes(graph_nodes(model))
    versions = {entry.domain: entry.version for entry in model.opset_import}
    version = versions.get("", versions.get("ai.onnx"))
    if version != opset or not 17 <= version <= 21:
        problems.append(f"opset が {opset} ではありません: {version}（17〜21 のみ。23 は Conv が CPU に回ります）")
    if model.functions:
        problems.append("グラフの中に関数が残っています")
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            problems.append("外部データのファイルを使っています（1 ファイルにしてください）")
            break
    for values, name in ((model.graph.input, "input"), (model.graph.output, "delta")):
        if len(values) != 1 or values[0].name != name:
            problems.append(f"グラフの入出力の名前が {name} ではありません")
            continue
        tensor_type = values[0].type.tensor_type
        if tensor_type.elem_type != onnx.TensorProto.FLOAT:
            problems.append(f"{name} が float32 ではありません")
        shape = [d.dim_value if d.HasField("dim_value") else None for d in tensor_type.shape.dim]
        if shape != [1, 3, size, size]:
            problems.append(f"{name} の形が [1, 3, {size}, {size}] ではありません: {shape}")
    return problems


def real_crop(data_root: Path, size: int) -> tuple[np.ndarray, str]:
    """A real crop to compare on: random numbers would not exercise what the model does on faces."""
    sequences = find_sequences(data_root)
    sequence = sequences[0]
    landmarks = sequence.landmarks()
    index = sequence.frames // 2
    view = crop_view(sequence, landmarks[index], landmarks[index], index, size)
    return view.input.transpose(2, 0, 1)[None].astype(np.float32), sequence.name


def export_model(
    model: torch.nn.Module,
    folder: Path,
    export: ExportConfig,
    size: int,
    crop: np.ndarray,
    data: dict[str, Any],
    float16: bool = False,
    training_run: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Writes model.onnx, model.json and export_check.json into ``folder``. Returns what the check measured."""
    refuse_tracked_folder(folder)
    if (folder / "model.onnx").exists() and not overwrite:
        raise ExportError(f"{folder.name} にはすでにモデルがあります。バージョンを変えるか、--overwrite を付けてください（評価の結果も無効になります）")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model.onnx"
    model = model.cpu().eval()
    # The comparison is on the model as exported: a float16 graph is compared with the float16 weights.
    exported: torch.nn.Module = (Float32Edges(model) if float16 else model).eval()
    example = torch.zeros(1, 3, size, size, dtype=torch.float32)
    # In a temporary folder first: the exporter leaves a half-written file behind when it fails.
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch) / "model.onnx"
        torch.onnx.export(
            exported,
            (example,),
            str(scratch_path),
            dynamo=True,
            opset_version=export.opset,
            external_data=False,
            verbose=False,
            input_names=["input"],
            output_names=["delta"],
        )
        leftovers = [p.name for p in Path(scratch).iterdir() if p.name != "model.onnx"]
        if leftovers:
            raise ExportError(f"1 ファイルになりませんでした: {leftovers}")
        path.write_bytes(scratch_path.read_bytes())

    try:
        result = check_export(path, exported, crop, export, size, float16)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    meta = metadata.build_model_json(export, size, path, data, training_run)
    (folder / "model.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (folder / "export_check.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"check": result, "model_json": meta}


def check_export(
    path: Path, model: torch.nn.Module, crop: np.ndarray, export: ExportConfig, size: int, float16: bool
) -> dict[str, Any]:
    """Contract section 8's export check. Raises ExportError listing every problem."""
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)
    problems = check_graph(proto, size, export.opset)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if [i.name for i in inputs] != ["input"] or [o.name for o in outputs] != ["delta"]:
        problems.append("ONNX Runtime から見た入出力の名前が違います")
    output = session.run(None, {"input": crop})[0]
    with torch.no_grad():
        expected = model(torch.from_numpy(crop)).numpy()
    if output.shape != (1, 3, size, size):
        problems.append(f"出力の形が違います: {output.shape}")
    tolerance = export.tolerance_float16 if float16 else export.tolerance
    difference = float(np.abs(output - expected).max())
    if difference > tolerance:
        problems.append(f"PyTorch との差が大きすぎます: {difference:.2e} > {tolerance:.2e}")
    if float(np.abs(output).max()) > 1.0 + 1e-6:
        problems.append("出力が [-1, 1] に収まっていません")
    if problems:
        raise ExportError("書き出したモデルが契約を満たしません:\n- " + "\n- ".join(problems))
    return {
        "passed": True,
        "onnxruntime_cpu": ort.__version__,
        "max_abs_difference_against_torch": difference,
        "tolerance": tolerance,
        "ops": sorted({node.op_type for node in proto.graph.node}),
        "nodes": len(proto.graph.node),
        "float16": float16,
        "output_abs_max": float(np.abs(output).max()),
    }


def export_checkpoint(checkpoint_path: Path, data_root: Path, out: Path | None, size: int | None = None, weights: str = "ema", overwrite: bool = False) -> dict[str, Any]:
    state = checkpoint.load(checkpoint_path)
    config = config_module.from_dict(state["config"])
    if config.stage != "student":
        raise ExportError("書き出せるのは生徒モデルだけです")
    size = size or config.data.size
    if size not in config_module.ALLOWED_SIZES:
        raise ExportError(f"S は {config_module.ALLOWED_SIZES} のどれかです")
    model = DeltaUNet(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state[weights])
    if out is None:
        out = _REPOSITORY / ".test-media" / "planb" / "models" / f"{config.export.name}-{config.export.version}"
    crop, _ = real_crop(data_root, size)
    run_info = {"step": int(state["step"]), "fingerprint": state["fingerprint"], "trained_size": config.data.size, "data": state["data"], "distilled": state["teacher"] is not None}
    return export_model(model, out, config.export, size, crop, state["data"], config.model.float16, run_info, overwrite)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生徒モデルを ONNX に書き出して確認します")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="確認に使う実際のクロップを取るデータのフォルダ")
    parser.add_argument("--out", type=Path, help="出力先（既定は .test-media/planb/models/<name>-<version>）")
    parser.add_argument("--size", type=int, help="学習と別の S で書き出すとき（畳み込みだけなのでできますが、確かめてください）")
    parser.add_argument("--overwrite", action="store_true", help="同じフォルダにあるモデルを置き換えます")
    args = parser.parse_args(argv)
    try:
        result = export_checkpoint(args.checkpoint, args.data, args.out, args.size, overwrite=args.overwrite)
    except (ExportError, DatasetError, config_module.ConfigError) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(json.dumps(result["check"], indent=2, ensure_ascii=False))
    meta = result["model_json"]
    print(f"stand_in={meta['stand_in']} sha256={meta['sha256'][:12]}… {meta['size_bytes'] / 1e6:.2f} MB")
    if meta["stand_in"]:
        print("このモデルは代用品です。配線の確認にしかなりません。見た目については何も示しません", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
