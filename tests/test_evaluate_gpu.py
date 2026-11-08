"""Parts of the evaluation that need onnxruntime with DirectML. Skipped without them. Tiny models built here, no weights on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from makeup_cam import evaluate as ev

ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

if "DmlExecutionProvider" not in ort.get_available_providers():
    pytest.skip("DirectML is not available", allow_module_level=True)

SIZE = 64


def _conv_model(path: Path, opset: int) -> Path:
    """One convolution from ``input`` to ``delta``, with the fixed shapes of the contract."""
    from onnx import TensorProto, helper, numpy_helper

    weights = numpy_helper.from_array(np.zeros((3, 3, 3, 3), np.float32), "weights")
    node = helper.make_node("Conv", ["input", "weights"], ["delta"], pads=[1, 1, 1, 1], kernel_shape=[3, 3])
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("delta", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.save(model, path)
    return path


def test_a_graph_on_the_gpu_passes(tmp_path: Path) -> None:
    result = ev.check_kernels(_conv_model(tmp_path / "ok.onnx", 20), SIZE)
    assert result["available"] is True
    assert result["not_on_gpu"] == []


def test_a_node_on_the_cpu_is_found(tmp_path: Path) -> None:
    """The contract measured this: at opset 23 DirectML leaves Conv to the CPU, without a warning."""
    result = ev.check_kernels(_conv_model(tmp_path / "cpu.onnx", 23), SIZE)
    assert result["available"] is True
    assert "Conv" in result["not_on_gpu"]
    metrics, _ = ev.build_metrics(ev.MODE_CLASSIC, [ev.ClipResult("c", 100, 100, 30.0, {}, [], {})], kernels=result)
    assert metrics["all_kernels_dml"]["passed"] is False
