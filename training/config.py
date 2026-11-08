"""The configuration of one training run: a JSON file, strict, with no paths and no personal data.

Data, run and checkpoint folders are command-line arguments and never part of a config, so a config (and the copy
every run keeps next to its checkpoints) can be read, shared and archived without saying anything about anyone.
Unknown keys are errors: a typo must not silently train something else.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_SIZES = (256, 384, 512)  # contract section 3; repeated so that this module needs nothing else
ALLOWED_OPSETS = range(17, 22)  # contract section 8
MAX_ABS_DELTA_LIMIT = 0.35  # contract section 5

# Things that have no place in a config. Sequence patterns like "cremad-1002-*" must stay possible.
_PERSONAL_PATTERNS = (
    (re.compile(r"[A-Za-z]:[\\/]"), "絶対パス"),
    (re.compile(r"(^|[\\/])(Users|home)[\\/]", re.IGNORECASE), "ユーザーのフォルダ"),
    (re.compile(r"^[\\/]"), "絶対パス"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "メールアドレス"),
)


class ConfigError(ValueError):
    """The config cannot be used. The message is for the person running the tool."""


@dataclass
class DataConfig:
    size: int = 256  # S of the crop: 256, 384 or 512
    train_sequences: list[str] = field(default_factory=lambda: ["*"])  # glob patterns over sequence folder names
    heldout_sequences: list[str] = field(default_factory=list)  # never trained on, whatever train_sequences says
    jitter_shift: float = 0.005  # landmark jitter: shift, as a share of face_scale
    jitter_degrees: float = 0.3
    jitter_scale: float = 0.005
    temporal_warp: str = "flow"  # "flow": align neighbours by optical flow on top of the crop; "crop": the crop alone


@dataclass
class AugmentConfig:
    enabled: bool = True
    # Same for input and target, and for both frames of a pair: the room changes, the makeup does not.
    exposure_stops: float = 0.5
    white_balance: float = 0.08
    gamma: float = 0.15
    lighting_gradient: float = 0.2
    # On the input only: the model must not learn to remove noise or to copy it into the delta.
    read_noise: float = 0.012
    shot_noise: float = 0.012
    blur_sigma: float = 0.6
    jpeg_quality_min: int = 70  # 100 turns it off


@dataclass
class ModelConfig:
    base: int = 16  # channels at the top level
    depth: int = 3  # number of downsampling steps
    blocks: int = 1  # blocks per level
    residual: bool = False
    norm: str = "none"  # "none" or "instance": the contract allows no GroupNormalization in a student
    activation: str = "leaky"  # "leaky" or "relu"; see models.py
    attention: bool = False  # teacher only: self-attention at the bottleneck
    float16: bool = False  # inside the exported graph (student only)


@dataclass
class LossConfig:
    lab: float = 1.0  # L1 in Lab at full strength
    lab_partial: float = 0.5  # L1 in Lab at a random strength: the slider's middle must also be right
    strength_min: float = 0.25
    detail: float = 0.2  # luminance gradients at two scales; needs no pretrained weights
    temporal: float = 0.5  # stability across neighbouring frames
    lag: float = 0.1  # a delta is drawn one frame late at runtime: it must still fit the next frame
    region: float = 5.0  # the delta must already be 0 where the runtime's weight would make it 0
    magnitude: float = 0.05  # identity guard: limit the size of the delta
    low_frequency: float = 0.2  # identity guard: no more fine detail in the delta than in the target's
    perceptual: float = 0.0  # needs --perceptual-weights, a local file; off unless asked for
    identity: float = 0.0  # needs --identity-weights, a local file; off unless asked for
    distill: float = 1.0  # student against teacher; used only when a teacher checkpoint is given


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 4
    learning_rate: float = 1e-3
    warmup_steps: int = 50
    grad_clip: float = 1.0
    ema_decay: float = 0.995
    amp: bool = False  # mixed precision on CUDA only
    num_workers: int = 2
    threads: int = 0  # torch CPU threads; 0 keeps torch's default
    seed: int = 1
    checkpoint_every: int = 500
    keep_checkpoints: int = 2
    validate_every: int = 500
    validate_frames: int = 12  # per held-out sequence
    log_every: int = 50


@dataclass
class ExportConfig:
    name: str = "finish-student"
    version: str = "0.0.1"
    opset: int = 20
    max_abs_delta: float = 0.25
    temporal_alpha: float = 0.6
    may_edit_eyes: bool = False
    tolerance: float = 1e-3  # largest ONNX Runtime against torch difference on a real crop (contract section 8)
    tolerance_float16: float = 5e-3  # a float16 graph rounds differently from the float32 weights
    note: str = ""  # goes into model.json's training_data.note; technical, no personal details


@dataclass
class RunConfig:
    stage: str = "student"  # "teacher" or "student"
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def find_personal_data(value: Any, where: str = "") -> list[str]:
    """Where a config holds a path or an address, as readable messages. Empty when it holds none."""
    if isinstance(value, dict):
        return [problem for key, item in value.items() for problem in find_personal_data(item, f"{where}.{key}".strip("."))]
    if isinstance(value, (list, tuple)):
        return [problem for i, item in enumerate(value) for problem in find_personal_data(item, f"{where}[{i}]")]
    if isinstance(value, str):
        return [f"{where}: {what}は書けません" for pattern, what in _PERSONAL_PATTERNS if pattern.search(value)]
    return []


def _from_dict(cls: type, values: Any, where: str) -> Any:
    if not isinstance(values, dict):
        raise ConfigError(f"{where or '設定'} はオブジェクトで書いてください")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - set(fields))
    if unknown:
        raise ConfigError(f"{where or '設定'} に知らない項目があります: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        spec = fields[name]
        default = spec.default_factory() if spec.default_factory is not dataclasses.MISSING else spec.default
        kwargs[name] = _from_dict(type(default), value, f"{where}.{name}".strip(".")) if dataclasses.is_dataclass(default) else value
    return cls(**kwargs)


def validate(config: RunConfig) -> None:
    """Raises ConfigError for anything that could not be trained or exported under the contract."""
    problems = find_personal_data(to_dict(config))
    if config.stage not in ("teacher", "student"):
        problems.append("stage は teacher か student です")
    if config.data.size not in ALLOWED_SIZES:
        problems.append(f"data.size は {ALLOWED_SIZES} のどれかです")
    if config.data.temporal_warp not in ("flow", "crop"):
        problems.append("data.temporal_warp は flow か crop です")
    if config.model.norm not in ("none", "instance"):
        problems.append("model.norm は none か instance です")
    if config.model.activation not in ("leaky", "relu"):
        problems.append("model.activation は leaky か relu です")
    if config.stage == "student" and config.model.attention:
        problems.append("生徒モデルに attention は使えません（書き出しの規則）")
    if config.stage == "teacher" and config.model.float16:
        problems.append("教師モデルは書き出さないので float16 は使えません")
    if config.model.base < 1 or config.model.depth < 1 or config.model.blocks < 1:
        problems.append("model.base, depth, blocks は 1 以上です")
    if config.model.depth >= 1 and config.data.size % 2**config.model.depth != 0:
        problems.append("data.size は 2^model.depth で割り切れる必要があります")
    if config.export.opset not in ALLOWED_OPSETS:
        problems.append("export.opset は 17〜21 です（契約 §8。23 だと Conv が CPU に回ります）")
    if not 0.0 < config.export.max_abs_delta <= MAX_ABS_DELTA_LIMIT:
        problems.append(f"export.max_abs_delta は 0 より大きく {MAX_ABS_DELTA_LIMIT} 以下です")
    if not 0.0 < config.export.temporal_alpha <= 1.0:
        problems.append("export.temporal_alpha は 0 より大きく 1 以下です")
    if not 0.0 < config.loss.strength_min <= 1.0:
        problems.append("loss.strength_min は 0 より大きく 1 以下です")
    if config.train.steps < 1 or config.train.batch_size < 1:
        problems.append("train.steps と train.batch_size は 1 以上です")
    if not 1 <= config.augment.jpeg_quality_min <= 100:
        problems.append("augment.jpeg_quality_min は 1〜100 です")
    for label, value in (("export.name", config.export.name), ("export.version", config.export.version)):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            problems.append(f"{label} は英数字と . _ - だけです（フォルダ名になります）")
    if problems:
        raise ConfigError("設定を直してください:\n- " + "\n- ".join(problems))


def from_dict(values: Any) -> RunConfig:
    config = _from_dict(RunConfig, values, "")
    validate(config)
    return config


def load(path: str | Path) -> RunConfig:
    try:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"設定ファイルを読めません: {Path(path).name}") from exc
    except ValueError as exc:
        raise ConfigError(f"設定ファイルが JSON として読めません: {Path(path).name}: {exc}") from exc
    return from_dict(values)


def to_dict(config: RunConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)


def dumps(config: RunConfig) -> str:
    return json.dumps(to_dict(config), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def fingerprint(config: RunConfig) -> str:
    """Identifies what a run trains. A resumed run must have the same one; it may only run for longer."""
    values = to_dict(config)
    for key in ("steps", "checkpoint_every", "keep_checkpoints", "validate_every", "validate_frames", "log_every", "num_workers", "threads"):
        values["train"].pop(key, None)
    values.pop("description", None)
    values.pop("export", None)  # export settings do not change the training
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
