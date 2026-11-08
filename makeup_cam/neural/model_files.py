"""Where models live, ``model.json``, and the gate (``docs/dev/contract-plan-b.md`` section 6).

All of this reads files and computes hashes, so it runs on the stage's control thread, never on a frame's time.
Refusals carry the Japanese status text the GUI shows.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import user_data_dir

NEURAL_DIR_ENV = "AI_MAKEUP_CAM_NEURAL_DIR"
ACTIVE_FILE = "active.json"
MODEL_JSON = "model.json"
EVALUATION_JSON = "evaluation.json"

KNOWN_SCHEMAS = (1,)
KNOWN_CROP_DEFINITIONS = (1,)  # geometry.CROP_DEFINITION; this module does not import cv2
KNOWN_OPSETS = (17, 18, 19, 20, 21)
ALLOWED_SIZES = (256, 384, 512)  # geometry.ALLOWED_SIZES
DEFAULT_MAX_ABS_DELTA = 0.25
MAX_MAX_ABS_DELTA = 0.35
DEFAULT_TEMPORAL_ALPHA = 0.6
# json.loads holds the GIL for all of its work, and the camera thread needs the GIL: 21 MB stopped other Python threads for
# 276 ms, 419 MB for 4.5 s. Real files: model.json about 2 KB, evaluation.json about 90 KB.
MAX_METADATA_BYTES = 1 << 20
MAX_REPORT_BYTES = 1 << 21
# What Windows opens as a device whatever the folder or the extension.
_DEVICE_NAMES = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))})

NO_MODEL_MESSAGE = "モデルがありません"
NOT_PASSED_MESSAGE = "評価に合格したモデルがありません"


class ModelRefused(Exception):
    """This model must not run. ``status`` is the status text; the exception's text is for the log."""

    def __init__(self, status: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status


@dataclass(frozen=True)
class ModelSpec:
    """What the runtime needs from one ``model.json``."""

    folder: Path
    name: str
    version: str
    file: Path
    sha256: str
    size: int
    max_abs_delta: float
    temporal_alpha: float
    may_edit_eyes: bool
    stand_in: bool
    verdict: str
    report_sha256: str | None
    evaluated_model_sha256: str | None


def neural_dir() -> Path:
    """The folder that holds the model folders and ``active.json``."""
    override = os.environ.get(NEURAL_DIR_ENV)
    return Path(override) if override else user_data_dir() / "neural_models"


def list_models(root: Path | None = None) -> list[str]:
    """Names (``<name>-<version>``) of the folders under ``root`` that have a ``model.json``."""
    root = neural_dir() if root is None else root
    try:
        return sorted(p.name for p in root.iterdir() if (p / MODEL_JSON).is_file())
    except OSError:
        return []


def active_model_name(root: Path | None = None) -> str | None:
    """What ``active.json`` names, or None when there is no usable file."""
    root = neural_dir() if root is None else root
    try:
        data = _read_json(root / ACTIVE_FILE, MAX_METADATA_BYTES)
    except (OSError, ValueError):
        return None
    name = data.get("model") if isinstance(data, dict) else None
    return name if isinstance(name, str) and _is_plain_name(name) else None


def set_active_model(name: str, root: Path | None = None) -> None:
    """Point ``active.json`` at a model folder, atomically. ``ValueError`` when there is no such model."""
    root = neural_dir() if root is None else root
    if not _is_plain_name(name) or not (root / name / MODEL_JSON).is_file():
        raise ValueError(name)
    temporary = root / (ACTIVE_FILE + ".tmp")
    temporary.write_text(json.dumps({"model": name}), encoding="utf-8")
    os.replace(temporary, root / ACTIVE_FILE)


def active_model_dir(root: Path | None = None) -> Path:
    """The active model's folder. Raises ``ModelRefused`` when none is chosen or it is gone."""
    root = neural_dir() if root is None else root
    name = active_model_name(root)
    if name is None or not (root / name).is_dir():
        raise ModelRefused(NO_MODEL_MESSAGE, f"no active model under {ACTIVE_FILE}")
    return root / name


def load_spec(folder: Path) -> ModelSpec:
    """Read and check ``model.json`` (contract section 6.2). Raises ``ModelRefused``."""
    try:
        data = _read_json(folder / MODEL_JSON, MAX_METADATA_BYTES)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except OSError as exc:
        raise ModelRefused(NO_MODEL_MESSAGE, f"cannot read {MODEL_JSON}: {exc}") from exc
    except ValueError as exc:
        raise ModelRefused(NOT_PASSED_MESSAGE, f"{MODEL_JSON} does not parse: {exc}") from exc
    try:
        return _spec_from(folder, data)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ModelRefused(NOT_PASSED_MESSAGE, f"{MODEL_JSON} is not usable: {exc!r}") from exc


def _read_json(path: Path, limit: int) -> Any:
    """The parsed file. ``ValueError`` when it is larger than ``limit`` (it is then not parsed) or does not parse."""
    return _parse_json(_read_limited(path, limit))


def _read_limited(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{path.name} is over {limit} bytes")
    return data


def _parse_json(data: bytes) -> Any:
    try:
        return json.loads(data.decode("utf-8-sig"))
    except RecursionError as exc:  # a deeply nested file; UnicodeDecodeError and JSONDecodeError are ValueErrors already
        raise ValueError("nested too deeply") from exc


def _is_int(value: Any) -> bool:
    return type(value) is int  # True and 1.0 compare equal to 1


def _number(value: Any) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"not a number: {value!r}")
    return float(value)


def _spec_from(folder: Path, data: dict[str, Any]) -> ModelSpec:
    if not _is_int(data["schema"]) or data["schema"] not in KNOWN_SCHEMAS:
        raise ValueError(f"unknown schema {data['schema']!r}")
    if not _is_int(data["crop"]["definition"]) or data["crop"]["definition"] not in KNOWN_CROP_DEFINITIONS:
        raise ValueError(f"unknown crop definition {data['crop']['definition']!r}")
    inp, out = data["input"], data["output"]
    size = inp["size"]
    if not _is_int(size) or size not in ALLOWED_SIZES:
        raise ValueError(f"size {size!r}")
    expected_input = ("input", "NCHW", "BGR", "float32")
    if (inp["name"], inp["layout"], inp["colors"], inp["dtype"]) != expected_input:
        raise ValueError(f"input {inp!r}")
    if (out["name"], out["kind"], out["layout"], out["colors"], out["dtype"]) != ("delta", "delta", "NCHW", "BGR", "float32"):
        raise ValueError(f"output {out!r}")
    max_abs = _number(out.get("max_abs_delta", DEFAULT_MAX_ABS_DELTA))
    alpha = _number(out.get("temporal_alpha", DEFAULT_TEMPORAL_ALPHA))
    if not 0.0 < max_abs <= MAX_MAX_ABS_DELTA:
        raise ValueError(f"max_abs_delta {max_abs!r}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"temporal_alpha {alpha!r}")
    opset = data.get("opset", KNOWN_OPSETS[0])  # contract section 8: at 23, DirectML runs Conv on the CPU, 40 x slower, with no warning
    if not _is_int(opset) or opset not in KNOWN_OPSETS:
        raise ValueError(f"opset {opset!r}")
    file_name = data["file"]
    if not isinstance(file_name, str) or not _is_plain_name(file_name):
        raise ValueError(f"file {file_name!r}")
    evaluation = data.get("evaluation") or {}
    return ModelSpec(
        folder=folder,
        name=str(data["name"]),
        version=str(data["version"]),
        file=folder / file_name,
        sha256=str(data["sha256"]).lower(),
        size=int(size),
        max_abs_delta=max_abs,
        temporal_alpha=alpha,
        may_edit_eyes=data.get("may_edit", {}).get("eyes", False) is True,
        stand_in=data.get("stand_in", True) is not False,  # anything but an explicit false counts as a stand-in
        verdict=str(evaluation.get("verdict", "none")),
        report_sha256=evaluation.get("report_sha256"),
        evaluated_model_sha256=evaluation.get("model_sha256"),
    )


def check_gate(spec: ModelSpec, allow_unevaluated: bool = False) -> None:
    """Contract section 6.3. Raises ``ModelRefused`` unless this model may run.

    ``allow_unevaluated`` is for the offline modes and tests only: it skips the evaluation and the stand-in rule,
    not the check that the file is the one ``model.json`` describes.
    """
    try:
        actual = sha256_of(spec.file)
    except OSError as exc:
        raise ModelRefused(NO_MODEL_MESSAGE, f"cannot read {spec.file.name}: {exc}") from exc
    except ValueError as exc:  # a path Windows cannot hold, such as an embedded NUL
        raise ModelRefused(NOT_PASSED_MESSAGE, f"cannot read {spec.file.name!r}: {exc}") from exc
    if actual != spec.sha256:
        raise ModelRefused(NOT_PASSED_MESSAGE, f"{spec.file.name} does not match the sha256 of {MODEL_JSON}")
    if allow_unevaluated:
        return
    if spec.stand_in:
        raise ModelRefused(NOT_PASSED_MESSAGE, "a stand-in may not reach a virtual camera")
    if spec.verdict != "pass":
        raise ModelRefused(NOT_PASSED_MESSAGE, f"evaluation verdict is {spec.verdict!r}")
    if spec.evaluated_model_sha256 != actual:
        raise ModelRefused(NOT_PASSED_MESSAGE, "the evaluation is of another file")
    try:
        report = _read_limited(spec.folder / EVALUATION_JSON, MAX_REPORT_BYTES)
    except (OSError, ValueError) as exc:
        raise ModelRefused(NOT_PASSED_MESSAGE, f"cannot read {EVALUATION_JSON}: {exc}") from exc
    if hashlib.sha256(report).hexdigest() != spec.report_sha256:
        raise ModelRefused(NOT_PASSED_MESSAGE, f"{EVALUATION_JSON} does not match {MODEL_JSON}")
    _check_report(report, actual)


def _check_report(report: bytes, model_sha256: str) -> None:
    """The report must say what ``model.json`` says it says. This catches mistakes; anyone can edit both files."""
    try:
        data = _parse_json(report)
    except ValueError as exc:
        raise ModelRefused(NOT_PASSED_MESSAGE, f"{EVALUATION_JSON} does not parse: {exc}") from exc
    model = data.get("model") if isinstance(data, dict) else None
    if not isinstance(model, dict) or data.get("verdict") != "pass" or data.get("eligible_for_calls") is not True or model.get("sha256") != model_sha256:
        raise ModelRefused(NOT_PASSED_MESSAGE, f"{EVALUATION_JSON} does not report a pass of this file")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_plain_name(name: str) -> bool:
    """A single path component: nothing in a model's files may point outside its folder."""
    if not name or name in (".", "..") or name[-1] in ". " or any(c in '/\\:' or ord(c) < 32 for c in name):
        return False
    return name.split(".")[0].upper() not in _DEVICE_NAMES
