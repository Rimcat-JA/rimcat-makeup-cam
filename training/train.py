"""Trains a teacher or a student. Needs torch. Same code on the CPU, on CUDA and on torch-directml.

    python -m training.train --config training/configs/smoke-student.json --data DATA --run-dir RUN --device cpu
    (again with --resume after an interruption: it continues exactly; --teacher RUN/ckpt/last.pt to distill)

Everything random comes from (seed, step), so a resumed run is the same as one that never stopped, and a run keeps no
state but its weights, its optimizer and its step. A run folder holds only numbers and the config: no paths, no pictures.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from makeup_cam.neural import geometry

from . import checkpoint, config as config_module, heldout, metadata
from .config import ModelConfig, RunConfig
from .layout import DatasetError, data_summary, find_sequences, in_tracked_folder, split_sequences
from .losses import Batch, PerceptualLoss, compute_losses
from .models import DeltaUNet, weights_megabytes
from .samples import SampleBuilder

_MAX_BAD_STEPS = 20
_STRENGTH_STREAM = 1_000_000  # keeps the slider's random numbers apart from those of the samples


class _StepDataset(Dataset):
    """A sample is named by (step, position): see ``samples``."""

    def __init__(self, builder: SampleBuilder) -> None:
        self.builder = builder

    def __getitem__(self, key: tuple[int, int]) -> dict[str, np.ndarray]:
        return self.builder.build(*key)


class _StepBatches:
    """The batches from one step to the end of the run."""

    def __init__(self, start: int, end: int, batch_size: int) -> None:
        self.start, self.end, self.batch_size = start, end, batch_size

    def __iter__(self):
        for step in range(self.start, self.end):
            yield [(step, position) for position in range(self.batch_size)]

    def __len__(self) -> int:
        return self.end - self.start


def _init_worker(_: int) -> None:
    import cv2

    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA が使えません")
        return torch.device("cuda")
    if name == "dml":
        try:
            import torch_directml
        except ImportError as exc:
            raise SystemExit("torch-directml が入っていません（docs/plan-b-training.md）") from exc
        return torch_directml.device()
    return torch.device("cpu")


def learning_rate(config: RunConfig, step: int) -> float:
    """Linear warm-up, then a cosine down to 5 %. A function of the step alone, so that resuming needs no scheduler state."""
    base = config.train.learning_rate
    warmup = config.train.warmup_steps
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, config.train.steps - warmup)
    return base * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def ema_decay(config: RunConfig, step: int) -> float:
    """Short memory at the start, so that the zero-initialized first weights do not linger."""
    return min(config.train.ema_decay, (1.0 + step) / (10.0 + step))


def build_teacher(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    state = checkpoint.load(path)
    if state["config"]["stage"] != "teacher":
        raise SystemExit("--teacher に指定したチェックポイントは教師モデルのものではありません")
    teacher = DeltaUNet(ModelConfig(**state["config"]["model"]))
    teacher.load_state_dict(state["ema"])
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, {"step": state["step"], "fingerprint": state["fingerprint"]}


def predictor(model: nn.Module, device: torch.device) -> heldout.Predict:
    def predict(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(batch).to(device)).cpu().numpy()

    return predict


@dataclass
class RunResult:
    steps: int
    first_loss: float
    last_loss: float
    summary: dict[str, Any]


def run(
    config: RunConfig,
    data_root: Path,
    run_dir: Path,
    device_name: str = "cpu",
    resume: bool = False,
    teacher_path: Path | None = None,
    perceptual_weights: Path | None = None,
    identity_weights: Path | None = None,
    stop_after: int | None = None,
) -> RunResult:
    """``stop_after`` ends the process of training early at that step, as an interruption would; for tests."""
    if config.loss.perceptual > 0.0 and perceptual_weights is None:
        raise SystemExit("loss.perceptual を使うには --perceptual-weights が必要です（ダウンロードはしません）")
    if config.loss.identity > 0.0 and identity_weights is None:
        raise SystemExit("loss.identity を使うには --identity-weights が必要です（ダウンロードはしません）")
    if config.stage == "teacher" and teacher_path is not None:
        raise SystemExit("教師モデルの学習に --teacher は使えません")

    train_set, heldout_set = split_sequences(find_sequences(data_root), config.data.train_sequences, config.data.heldout_sequences)
    if not train_set:
        raise DatasetError("学習に使うシーケンスがありません")
    if in_tracked_folder(run_dir):  # checkpoints are weights, and the run holds what the data looks like
        raise DatasetError("学習の記録とチェックポイントは Git で管理する場所に置けません。.test-media/ の下か、リポジトリの外にしてください")
    device = resolve_device(device_name)
    if config.train.threads:
        torch.set_num_threads(config.train.threads)
    use_amp = config.train.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ckpt_dir = run_dir / "ckpt"
    last = ckpt_dir / checkpoint.LAST
    if last.exists() and not resume:
        raise SystemExit("このフォルダには途中までの学習があります。続けるなら --resume を付けてください")
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.train.seed)
    model = DeltaUNet(config.model).to(device)
    ema = copy.deepcopy(model).eval()
    # foreach=False: the default is not implemented on every device, and the result is the same.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    fingerprint = config_module.fingerprint(config)
    start = 0
    if resume and last.exists():
        state = checkpoint.load(last)
        if state["fingerprint"] != fingerprint:
            raise SystemExit("設定が途中までの学習と違うので続けられません（変えてよいのは steps と記録の間隔だけです）")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        if use_amp and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start = int(state["step"])
        print(f"{start} ステップ目から続けます")

    teacher, teacher_info = (None, None)
    if teacher_path is not None:
        teacher, teacher_info = build_teacher(teacher_path, device)
    perceptual = PerceptualLoss(str(perceptual_weights)).to(device) if perceptual_weights is not None else None
    embedder = None
    if identity_weights is not None:
        embedder = torch.jit.load(str(identity_weights), map_location=device).eval()
        for parameter in embedder.parameters():
            parameter.requires_grad_(False)

    (run_dir / "run_config.json").write_text(config_module.dumps(config), encoding="utf-8")
    summary_of_data = data_summary(train_set)
    builder = SampleBuilder(
        train_set, config.data, config.augment, config.export.max_abs_delta, config.export.may_edit_eyes, config.train.seed
    )
    workers = config.train.num_workers
    loader = DataLoader(
        _StepDataset(builder),
        batch_sampler=_StepBatches(start, config.train.steps, config.train.batch_size),
        num_workers=workers,
        worker_init_fn=_init_worker if workers else None,
        pin_memory=device.type == "cuda",
    )
    window = torch.from_numpy(geometry.edge_window(config.data.size)).to(device)[None, None]
    device_label = str(device) if device.type != "cuda" else torch.cuda.get_device_name(0)
    print(
        f"学習: stage={config.stage} S={config.data.size} 重み {weights_megabytes(model):.2f} MB デバイス {device_label} "
        f"スレッド {torch.get_num_threads()} シーケンス {len(train_set)}（{summary_of_data['groups']} 人）"
        f" 未学習 {len(heldout_set)} 教師 {'あり' if teacher is not None else 'なし'}",
        flush=True,
    )

    log_path = run_dir / "metrics.jsonl"
    step_times: list[float] = []
    data_times: list[float] = []
    window_losses: list[float] = []
    terms_sum: dict[str, float] = {}
    first_loss = last_loss = float("nan")
    bad_steps = 0
    val_report: dict[str, Any] | None = None
    completed = start

    def save(at_step: int) -> None:
        state = {
            "step": at_step,
            "config": config_module.to_dict(config),
            "fingerprint": fingerprint,
            "data": summary_of_data,
            "teacher": teacher_info,
            "model": checkpoint.to_cpu(model.state_dict()),
            "ema": checkpoint.to_cpu(ema.state_dict()),
            "optimizer": checkpoint.to_cpu(optimizer.state_dict()),
        }
        if use_amp:
            state["scaler"] = checkpoint.to_cpu(scaler.state_dict())
        checkpoint.save(ckpt_dir, at_step, state, config.train.keep_checkpoints)

    def log(record: dict[str, Any]) -> None:
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def validate(at_step: int) -> dict[str, Any] | None:
        if not heldout_set:
            return None
        report = heldout.evaluate(
            predictor(ema, device),
            heldout_set,
            config.data.size,
            config.train.validate_frames,
            config.export.max_abs_delta,
            config.export.may_edit_eyes,
        )
        s = report["summary"]
        print(
            f"  未学習の人 step {at_step}: Lab 誤差 モデル {s['model_lab']:.3f} / 何も変えない {s['identity_lab']:.3f}"
            f"（比 平均 {s['ratio_mean']:.3f}、最悪 {s['ratio_worst']:.3f}）",
            flush=True,
        )
        log({"step": at_step, "heldout": report})
        return report

    model.train()
    try:
        data_start = time.perf_counter()
        for step, numpy_batch in enumerate(loader, start=start):
            data_times.append(time.perf_counter() - data_start)
            step_start = time.perf_counter()
            batch = Batch(**{k: v.to(device, non_blocking=True) for k, v in numpy_batch.items()})
            n, _, c, s, _ = batch.input.shape
            x = batch.input.reshape(2 * n, c, s, s)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(config, step)
            rng = np.random.default_rng([config.train.seed, step, _STRENGTH_STREAM])
            strengths = torch.from_numpy(rng.uniform(config.loss.strength_min, 1.0, n).astype(np.float32)).to(device).view(n, 1, 1, 1)
            with torch.autocast("cuda", enabled=use_amp) if device.type == "cuda" else _nothing():
                prediction = model(x)
                teacher_delta = None
                if teacher is not None:
                    with torch.no_grad():
                        teacher_delta = teacher(x)
                terms = compute_losses(
                    prediction, batch, config.loss, window, config.export.max_abs_delta, strengths, teacher_delta, perceptual, embedder
                )
            total = terms["total"]
            value = float(total.detach().cpu())
            if not math.isfinite(value):
                bad_steps += 1
                if bad_steps >= _MAX_BAD_STEPS:
                    raise RuntimeError("損失が続けて数値でなくなったので止めました")
                optimizer.zero_grad(set_to_none=True)
                data_start = time.perf_counter()
                continue
            bad_steps = 0
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            decay = ema_decay(config, step)
            with torch.no_grad():
                for tracked, current in zip(ema.parameters(), model.parameters()):
                    tracked.mul_(decay).add_(current.detach(), alpha=1.0 - decay)
            completed = step + 1
            step_times.append(time.perf_counter() - step_start)

            if math.isnan(first_loss):
                first_loss = value
            last_loss = value
            window_losses.append(value)
            for name, term in terms.items():
                terms_sum[name] = terms_sum.get(name, 0.0) + float(term.detach().cpu())

            done = completed
            if done % config.train.log_every == 0 or done == config.train.steps:
                count = len(window_losses)
                recent_step = float(np.median(step_times[-count:])) * 1000.0
                recent_data = float(np.median(data_times[-count:])) * 1000.0
                record = {
                    "step": done,
                    "lr": learning_rate(config, step),
                    "losses": {k: v / count for k, v in terms_sum.items()},
                    "step_ms": recent_step,
                    "wait_for_data_ms": recent_data,
                }
                log(record)
                print(f"step {done}/{config.train.steps} 損失 {record['losses']['total']:.4f} 計算 {recent_step:.0f} ms データ待ち {recent_data:.0f} ms", flush=True)
                window_losses.clear()
                terms_sum.clear()
            if done % config.train.validate_every == 0 or done == config.train.steps:
                val_report = validate(done)
            if done % config.train.checkpoint_every == 0 or done == config.train.steps:
                save(done)
            if stop_after is not None and done >= stop_after:
                save(done)
                break
            data_start = time.perf_counter()
    except KeyboardInterrupt:
        print("止めました。チェックポイントを保存します", file=sys.stderr)
        save(completed)
        raise

    summary = {
        "stage": config.stage,
        "device": device_label,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "steps_this_process": len(step_times),
        "final_step": completed,
        "weights_mb": weights_megabytes(model),
        "step_ms_median": float(np.median(step_times)) * 1000.0 if step_times else None,
        "wait_for_data_ms_median": float(np.median(data_times)) * 1000.0 if data_times else None,
        "data": summary_of_data,
        "heldout_sequences": [s.name for s in heldout_set],
        "heldout": val_report["summary"] if val_report else None,
        "teacher": teacher_info,
        "stand_in": metadata.is_stand_in(summary_of_data["kinds"]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return RunResult(completed, first_loss, last_loss, summary)


class _nothing:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="教師モデルまたは生徒モデルを学習します")
    parser.add_argument("--config", type=Path, required=True, help="設定ファイル（JSON）")
    parser.add_argument("--data", type=Path, required=True, help="データのフォルダ")
    parser.add_argument("--run-dir", type=Path, required=True, help="この学習の記録とチェックポイントを置くフォルダ")
    parser.add_argument("--device", choices=("cpu", "cuda", "dml"), default="cpu")
    parser.add_argument("--resume", action="store_true", help="途中までの学習があれば続けます")
    parser.add_argument("--teacher", type=Path, help="蒸留に使う教師モデルのチェックポイント")
    parser.add_argument("--perceptual-weights", type=Path, help="VGG16 の重みファイル（手元のもの）")
    parser.add_argument("--identity-weights", type=Path, help="顔認識モデル（TorchScript、手元のもの）")
    args = parser.parse_args(argv)
    try:
        config = config_module.load(args.config)
        result = run(config, args.data, args.run_dir, args.device, args.resume, args.teacher, args.perceptual_weights, args.identity_weights)
    except (config_module.ConfigError, DatasetError) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
