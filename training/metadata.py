"""``model.json`` and the export rules of the contract (sections 6 and 8). Needs only the standard library.

The stand-in rule lives here so that no caller can get it wrong: a model is a stand-in unless every sequence it trained
on was real paired data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import ExportConfig

SCHEMA = 1
CROP_DEFINITION = 1
REAL_KINDS = frozenset({"3d_render", "recording"})
# Contract section 8, verified on the GPU.
ALLOWED_OPS = frozenset(
    "Conv ConvTranspose Resize Concat Add Sub Mul Div Pow Relu LeakyRelu PRelu Sigmoid Tanh HardSwish HardSigmoid Elu Selu "
    "Softplus Softsign Mish Clip BatchNormalization InstanceNormalization AveragePool MaxPool GlobalAveragePool ReduceMean "
    "DepthToSpace SpaceToDepth Pad Slice Split Transpose Reshape Softmax MatMul GridSample Einsum Cast".split()
)
# No kernel runs for these: they only name a value. The exporter leaves them in a few graphs.
ALLOWED_STRUCTURAL = frozenset({"Constant", "Identity"})
FORBIDDEN_RESIZE_MODES = frozenset({"cubic"})
ALLOWED_RESIZE_TRANSFORMS = frozenset({"half_pixel", "pytorch_half_pixel", "align_corners", "asymmetric"})


def training_data_kind(kinds: Iterable[str]) -> str:
    found = set(kinds)
    if not found:
        return "none"
    if len(found) == 1:
        return next(iter(found))
    return "mixed"


def is_stand_in(kinds: Iterable[str]) -> bool:
    """True for anything but purely real paired data (contract 6.2). Classic-pipeline pairs in the mix make it a stand-in."""
    found = set(kinds)
    return not found or not found <= REAL_KINDS


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build_model_json(
    export: ExportConfig,
    size: int,
    model_path: Path,
    data: dict[str, Any],
    training_run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``model.json`` for a model file (contract 6.2). The evaluation block starts empty: only the evaluation tool writes it."""
    kinds = data.get("kinds", [])
    presets = data.get("input_presets", [])
    return {
        "schema": SCHEMA,
        "name": export.name,
        "version": export.version,
        "file": model_path.name,
        "sha256": sha256_of(model_path),
        "size_bytes": model_path.stat().st_size,
        "opset": export.opset,
        "stand_in": is_stand_in(kinds),
        "input": {"name": "input", "size": size, "layout": "NCHW", "colors": "BGR", "range": [0.0, 1.0], "dtype": "float32", "stage": "after_classic"},
        "output": {
            "name": "delta",
            "kind": "delta",
            "layout": "NCHW",
            "colors": "BGR",
            "range": [-1.0, 1.0],
            "dtype": "float32",
            "max_abs_delta": export.max_abs_delta,
            "temporal_alpha": export.temporal_alpha,
        },
        "crop": {"definition": CROP_DEFINITION},
        "may_edit": {"eyes": export.may_edit_eyes},
        "training_data": {"kind": training_data_kind(kinds), "input_preset": presets[0] if len(presets) == 1 else "mixed", "note": export.note},
        "evaluation": {"verdict": "none", "report": None, "report_sha256": None, "model_sha256": None, "date": None},
        "training_run": training_run or {},
    }


def check_nodes(nodes: Iterable[tuple[str, str, dict[str, Any]]]) -> list[str]:
    """Problems in a graph's nodes, each (domain, op type, attributes). Japanese messages; empty means all is allowed."""
    problems = []
    for domain, op, attributes in nodes:
        if domain not in ("", "ai.onnx"):
            problems.append(f"標準でない演算があります: {domain}.{op}")
        elif op == "Gelu":
            problems.append("Gelu は CPU に回ります（契約 §8）")
        elif op == "GroupNormalization":
            problems.append("GroupNormalization は一部が CPU に回ります（契約 §8）")
        elif op not in ALLOWED_OPS and op not in ALLOWED_STRUCTURAL:
            problems.append(f"許可されていない演算です: {op}")
        elif op == "Resize":
            mode = attributes.get("mode", "nearest")
            transform = attributes.get("coordinate_transformation_mode", "half_pixel")
            if mode in FORBIDDEN_RESIZE_MODES:
                problems.append("Resize の cubic は CPU に回ります（契約 §8）")
            elif mode == "linear" and transform not in ("half_pixel", "pytorch_half_pixel", "align_corners"):
                problems.append(f"Resize の座標の取り方が許可されていません: {transform}")
    return problems
