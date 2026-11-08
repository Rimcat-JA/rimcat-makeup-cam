"""How close a model comes to the targets on people it never saw, against leaving the crop unchanged. Needs numpy and cv2.

This is a training tool, not the evaluation of contract section 9. It answers one question: did the training learn the
mapping in its pairs? On stand-in pairs a good number proves the plumbing and nothing about any look.

Also works from the command line on an exported model (ONNX Runtime, CPU or DirectML):
    python -m training.heldout --model DIR --data ROOT --sequences "cremad-1060-*" --out OUT/report.json --sheet OUT/sheet.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from makeup_cam.neural import geometry

from .layout import Sequence, find_sequences, in_tracked_folder, split_sequences
from .masks import apply_delta
from .samples import crop_view

Predict = Callable[[np.ndarray], np.ndarray]  # [1, 3, S, S] float32 to the same
STRENGTHS = (0.25, 0.5, 1.0)
_FULL_WEIGHT = 0.99


def _lab(crop: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(crop, 0.0, 1.0).astype(np.float32), cv2.COLOR_BGR2Lab)


def _lab_l1(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    if not region.any():
        return float("nan")
    return float(np.abs(_lab(a) - _lab(b))[region].mean())


def _delta_e(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    if not region.any():
        return float("nan")
    return float(np.linalg.norm(_lab(a) - _lab(b), axis=2)[region].mean())


def frame_indices(sequence: Sequence, count: int) -> list[int]:
    """Evenly spaced frames, each with a following frame when the sequence has one."""
    last = sequence.frames - 2 if sequence.consecutive and sequence.frames > 1 else sequence.frames - 1
    return sorted({int(round(i)) for i in np.linspace(0, last, min(count, last + 1))})


def evaluate(
    predict: Predict,
    sequences: list[Sequence],
    size: int,
    frames_per_sequence: int = 12,
    max_abs_delta: float = 0.25,
    may_edit_eyes: bool = False,
) -> dict[str, Any]:
    """Numbers per group (person) and overall. ``lab`` values are mean absolute Lab differences (cv2's float scale: L
    0..100, a and b about -127..127) over the pixels a model may fully change, between the result and the target."""
    window = geometry.edge_window(size)
    per_group: dict[str, list[dict[str, float]]] = {}
    strength_totals = {s: {"applied": [], "error": [], "baseline": []} for s in STRENGTHS}
    steps_model, steps_target, forbidden = [], [], []
    for sequence in sequences:
        landmarks = sequence.landmarks()
        for index in frame_indices(sequence, frames_per_sequence):
            view = crop_view(sequence, landmarks[index], landmarks[index], index, size, may_edit_eyes)
            delta = predict(view.input.transpose(2, 0, 1)[None])[0].transpose(1, 2, 0)
            region = view.weight * window >= _FULL_WEIGHT
            output = apply_delta(view.input, delta, view.weight, 1.0, max_abs_delta, window)
            per_group.setdefault(sequence.group, []).append(
                {
                    "identity_lab": _lab_l1(view.input, view.target, region),
                    "model_lab": _lab_l1(output, view.target, region),
                    "identity_delta_e": _delta_e(view.input, view.target, region),
                    "model_delta_e": _delta_e(output, view.target, region),
                }
            )
            target_delta = np.clip(view.target - view.input, -max_abs_delta, max_abs_delta)
            for strength in STRENGTHS:
                partial = apply_delta(view.input, delta, view.weight, strength, max_abs_delta, window)
                wanted = apply_delta(view.input, target_delta, view.weight, strength, max_abs_delta, window)
                strength_totals[strength]["applied"].append(float(np.abs(partial - view.input)[region].mean()) if region.any() else 0.0)
                strength_totals[strength]["error"].append(_lab_l1(partial, wanted, region))
                strength_totals[strength]["baseline"].append(_lab_l1(view.input, wanted, region))
            outside = view.weight <= 0.0
            if outside.any():
                forbidden.append(np.abs(delta)[outside])
            if sequence.consecutive and index + 1 < sequence.frames:
                following = crop_view(sequence, landmarks[index + 1], landmarks[index + 1], index + 1, size, may_edit_eyes)
                next_delta = predict(following.input.transpose(2, 0, 1)[None])[0].transpose(1, 2, 0)
                next_target = np.clip(following.target - following.input, -max_abs_delta, max_abs_delta)
                both = region & (following.weight * window >= _FULL_WEIGHT)
                if both.any():
                    steps_model.append(float(np.abs(next_delta - delta)[both].mean()))
                    steps_target.append(float(np.abs(next_target - target_delta)[both].mean()))

    groups = {}
    for group, rows in per_group.items():
        keys = rows[0].keys()
        mean = {key: float(np.nanmean([row[key] for row in rows])) for key in keys}
        mean["ratio"] = mean["model_lab"] / max(mean["identity_lab"], 1e-9)
        mean["frames"] = len(rows)
        groups[group] = mean
    ratios = {group: values["ratio"] for group, values in groups.items()}
    worst = max(ratios, key=ratios.get) if ratios else None
    applied = [float(np.mean(strength_totals[s]["applied"])) if strength_totals[s]["applied"] else 0.0 for s in STRENGTHS]
    all_forbidden = np.concatenate(forbidden) if forbidden else np.zeros(1)
    return {
        "groups": groups,
        "summary": {
            "identity_lab": float(np.mean([v["identity_lab"] for v in groups.values()])) if groups else None,
            "model_lab": float(np.mean([v["model_lab"] for v in groups.values()])) if groups else None,
            "ratio_mean": float(np.mean(list(ratios.values()))) if ratios else None,
            "ratio_worst": ratios[worst] if worst else None,
            "worst_group": worst,
        },
        "strength": {
            str(s): {
                "applied_mean": applied[i],
                "lab_error": float(np.nanmean(strength_totals[s]["error"])) if strength_totals[s]["error"] else None,
                "lab_error_identity": float(np.nanmean(strength_totals[s]["baseline"])) if strength_totals[s]["baseline"] else None,
            }
            for i, s in enumerate(STRENGTHS)
        },
        "strength_monotonic": all(a < b for a, b in zip(applied, applied[1:])),
        "frame_step": {
            "model_mean": float(np.mean(steps_model)) if steps_model else None,
            "target_mean": float(np.mean(steps_target)) if steps_target else None,
        },
        "raw_delta_where_weight_is_zero": {
            "mean": float(all_forbidden.mean()),
            "p99": float(np.percentile(all_forbidden, 99)),
            "note": "before the runtime's mask, which sets these pixels to exactly 0 anyway",
        },
    }


def contact_sheet(
    predict: Predict,
    sequences: list[Sequence],
    size: int,
    per_group: int = 2,
    max_abs_delta: float = 0.25,
    may_edit_eyes: bool = False,
) -> np.ndarray:
    """uint8 BGR rows: input, target, model, target change x4, model change x4. For looking at, not for measuring."""
    window = geometry.edge_window(size)
    rows = []
    seen: dict[str, int] = {}
    for sequence in sequences:
        if seen.get(sequence.group, 0) >= per_group:
            continue
        seen[sequence.group] = seen.get(sequence.group, 0) + 1
        landmarks = sequence.landmarks()
        index = sequence.frames // 2
        view = crop_view(sequence, landmarks[index], landmarks[index], index, size, may_edit_eyes)
        delta = predict(view.input.transpose(2, 0, 1)[None])[0].transpose(1, 2, 0)
        output = apply_delta(view.input, delta, view.weight, 1.0, max_abs_delta, window)
        wanted = apply_delta(view.input, view.target - view.input, view.weight, 1.0, max_abs_delta, window)
        cells = [view.input, view.target, output, 0.5 + 4.0 * (wanted - view.input), 0.5 + 4.0 * (output - view.input)]
        row = np.hstack([np.clip(cell, 0.0, 1.0) for cell in cells])
        row = (row * 255.0 + 0.5).astype(np.uint8)
        cv2.putText(row, sequence.name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    if not rows:
        return np.zeros((32, 32, 3), np.uint8)
    sheet = np.vstack(rows)
    banner = np.zeros((48, sheet.shape[1], 3), np.uint8)
    cv2.putText(banner, "PLUMBING TEST ONLY: stand-in pairs. Says nothing about any look.", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, "input | target | model | target change x4 | model change x4", (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, sheet])


def onnx_predictor(model_dir: Path, provider: str) -> tuple[Predict, dict[str, Any]]:
    """A predict function for an exported model. Reports the provider really in use: ORT can fall back silently."""
    import onnxruntime as ort

    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers = ["DmlExecutionProvider"] if provider == "dml" else ["CPUExecutionProvider"]
    if provider == "dml":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model_dir / meta["file"]), options, providers=providers)
    if provider == "dml" and session.get_providers()[0] != "DmlExecutionProvider":
        raise RuntimeError("DirectML が使えません（CPU に切り替わりました）")

    def predict(batch: np.ndarray) -> np.ndarray:
        return session.run(None, {"input": batch.astype(np.float32)})[0]

    return predict, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="書き出したモデルと、何も変えない場合を、未学習の人のデータで比べます")
    parser.add_argument("--model", type=Path, required=True, help="model.onnx と model.json のフォルダ")
    parser.add_argument("--data", type=Path, required=True, help="データのフォルダ")
    parser.add_argument("--sequences", nargs="+", required=True, help="シーケンス名のパターン（人単位で全部使います）")
    parser.add_argument("--provider", choices=("cpu", "dml"), default="cpu")
    parser.add_argument("--frames", type=int, default=12, help="シーケンスごとのフレーム数")
    parser.add_argument("--out", type=Path, help="結果の JSON")
    parser.add_argument("--sheet", type=Path, help="見比べ用の画像（PNG）")
    args = parser.parse_args(argv)
    for path in (args.out, args.sheet):  # the sheet shows faces, and the report names people's sequences
        if path is not None and in_tracked_folder(path.resolve().parent):
            print(f"{path.name} は Git で管理する場所に置けません。.test-media/ の下か、リポジトリの外にしてください", file=sys.stderr)
            return 2
    predict, meta = onnx_predictor(args.model, args.provider)
    _, sequences = split_sequences(find_sequences(args.data), [], args.sequences)
    if not sequences:
        print("パターンに合うシーケンスがありません", file=sys.stderr)
        return 1
    size = int(meta["input"]["size"])
    max_abs = float(meta["output"]["max_abs_delta"])
    eyes = bool(meta["may_edit"]["eyes"])
    report = evaluate(predict, sequences, size, args.frames, max_abs, eyes)
    report["conditions"] = {
        "model": f"{meta['name']}-{meta['version']}",
        "stand_in": meta["stand_in"],
        "provider": args.provider,
        "sequences": [s.name for s in sequences],
        "plumbing_test_only": meta["stand_in"],
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.sheet:
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", contact_sheet(predict, sequences, size, 2, max_abs, eyes))
        if ok:
            encoded.tofile(str(args.sheet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
