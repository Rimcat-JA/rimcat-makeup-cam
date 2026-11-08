"""Latency of an exported model under ONNX Runtime, settings as in the runtime's worker. Needs numpy and onnxruntime.

    .venv-neural/Scripts/python -m training.latency --model DIR --runs 300 --hz 30

Paced by default, since a GPU clocks down between frames and a flat-out loop looks faster than a call will ever be. This
is the model alone, without the app: the evaluation tool measures it next to the app. Nothing is sent anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def percentiles_ms(seconds: list[float]) -> dict[str, float]:
    values = np.asarray(seconds) * 1000.0
    return {
        "median": round(float(np.median(values)), 2),
        "p95": round(float(np.percentile(values, 95)), 2),
        "p99": round(float(np.percentile(values, 99)), 2),
        "max": round(float(values.max()), 2),
    }


def process_vram_mb() -> float | None:
    """This process's dedicated GPU memory, from Windows' performance counters. None when not available."""
    if sys.platform != "win32":
        return None
    query = rf"(Get-Counter '\GPU Process Memory(pid_{os.getpid()}_*)\Dedicated Usage' -ErrorAction Stop).CounterSamples | Measure-Object -Property CookedValue -Sum | ForEach-Object {{ $_.Sum }}"
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True, timeout=30).stdout.strip()
        return round(float(out) / 1e6, 1) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def kernels_off_dml(profile_path: str) -> dict[str, int]:
    """{op: count} of kernels that did not run on DirectML, from ORT's profile file."""
    events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    found: dict[str, int] = {}
    for event in events:
        if event.get("cat") != "Node":
            continue
        args = event.get("args", {})
        provider = args.get("provider")
        if provider and provider != "DmlExecutionProvider" and event.get("name", "").endswith("kernel_time"):
            op = args.get("op_name", "?")
            found[op] = found.get(op, 0) + 1
    return found


def measure(model_path: Path, provider: str, runs: int, hz: float) -> dict[str, Any]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    if provider == "dml":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = ["DmlExecutionProvider"]
    start = time.perf_counter()
    session = ort.InferenceSession(str(model_path), options, providers=providers)
    create = time.perf_counter() - start
    if provider == "dml" and session.get_providers()[0] != "DmlExecutionProvider":
        raise RuntimeError("DirectML が使えません（CPU に切り替わりました）")
    shape = session.get_inputs()[0].shape
    rng = np.random.default_rng(0)
    inputs = [rng.random(shape, dtype=np.float32) for _ in range(8)]
    start = time.perf_counter()
    output = session.run(None, {"input": inputs[0]})[0]
    first = time.perf_counter() - start
    for _ in range(4):  # the worker warms up the same way before it reports ready
        session.run(None, {"input": inputs[0]})
    period = 1.0 / hz if hz > 0 else 0.0
    paced = []
    next_time = time.perf_counter()
    for i in range(runs):
        next_time += period
        start = time.perf_counter()
        output = session.run(None, {"input": inputs[i % 8]})[0]
        paced.append(time.perf_counter() - start)
        remaining = next_time - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
    result = {
        "file_mb": round(model_path.stat().st_size / 1e6, 2),
        "size": shape[2],
        "providers": session.get_providers(),
        "onnxruntime": ort.__version__,
        "create_ms": round(create * 1000.0, 1),
        "first_run_ms": round(first * 1000.0, 1),
        "hz": hz,
        "runs": runs,
        "latency_ms": percentiles_ms(paced),
        "finite": bool(np.isfinite(output).all()),
        "process_vram_mb": process_vram_mb() if provider == "dml" else None,
        "input": "random numbers: latency only",
    }
    if provider == "dml":
        with tempfile.TemporaryDirectory() as scratch:
            profiled = ort.SessionOptions()
            profiled.enable_mem_pattern = False
            profiled.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            profiled.log_severity_level = 3
            profiled.enable_profiling = True
            profiled.profile_file_prefix = str(Path(scratch) / "profile")
            profiling = ort.InferenceSession(str(model_path), profiled, providers=["DmlExecutionProvider"])
            profiling.run(None, {"input": inputs[0]})
            result["kernels_off_dml"] = kernels_off_dml(profiling.end_profiling())
            del profiling
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="書き出したモデルの遅延を測ります（モデル単独）")
    parser.add_argument("--model", type=Path, required=True, help="model.onnx のフォルダ")
    parser.add_argument("--provider", choices=("dml", "cpu"), default="dml")
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--hz", type=float, default=30.0, help="0 で全力")
    args = parser.parse_args(argv)
    try:
        result = measure(args.model / "model.onnx", args.provider, args.runs, args.hz)
    except Exception as exc:  # ORT's errors have no common base class, and on a Japanese Windows some are not ORT's at all
        print(f"測れませんでした: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
