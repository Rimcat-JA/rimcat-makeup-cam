"""Makes datasets in the contract's layout (section 10). Run in ``.venv-neural``: it needs the classic pipeline and MediaPipe.

Three sources:

- ``standin``: CREMA-D women through the classic pipeline. Bare to made-up by the classic effects. **A plumbing test only:**
  a model trained on this can at best imitate what the app already does, and it says nothing about any learned look.
- ``render``: two aligned sources, for instance a bare and a made-up 3D render (docs/3d-model-commission-spec.md). The
  input side goes through the classic pipeline with ``--input-preset``.
- ``recording``: the capture guide's sessions (docs/capture-guide.md, section 5). Bare and made-up takes are not in step, so
  each bare frame gets the made-up frame nearest in pose and expression, aligned by a similarity transform, and is
  dropped when they do not fit. No real recording exists yet: this path has only been run on synthetic and test data.

This is the only module under ``training/`` that imports more of ``makeup_cam`` than ``regions`` and ``neural.geometry``, and
only inside the commands, since ``.venv-train`` has no MediaPipe. The functions on arrays need numpy and cv2 alone.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import re
import shutil
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from makeup_cam import regions
from makeup_cam.neural import geometry

from .layout import DatasetError, Take, find_session_pairs, in_tracked_folder, write_manifest

_MIN_SEQUENCE = 8
_FRAME_SIZE = (1280, 720)
_CLIP_NAME = re.compile(r"^(\d{4})_(.+)$")

PLUMBING_NOTE = "plumbing test only: classic pipeline as target; says nothing about any learned look"

# Recording pairs: landmarks that do not move with speech or gaze align the two frames (as the crop does) ...
STABLE_POINTS = [33, 133, 362, 263, 2, 168, 6, 197, 10, 152, 234, 454]
# ... and these must fit after that, or the pair is dropped: the frame is not the same pose and expression.
_MAX_FIT_ERROR = 0.015  # p95 of all landmarks, x face scale
_MAX_MATCH_DISTANCE = 0.012  # root mean square landmark distance in the crop, as a share of its side


def refuse_tracked_folder(folder: Path) -> None:
    if in_tracked_folder(folder):
        raise DatasetError("データは Git で管理する場所に置けません。.test-media/ の下か、リポジトリの外にしてください")


def prepare_test_clip_frame(frame: np.ndarray) -> np.ndarray:
    """The small test clips as in the other video checks: 960x720, padded to 1280x720 (contract section 9.1)."""
    resized = cv2.resize(frame, (960, 720), interpolation=cv2.INTER_CUBIC)
    pad = (_FRAME_SIZE[0] - 960) // 2
    return cv2.copyMakeBorder(resized, 0, 0, pad, pad, cv2.BORDER_REPLICATE)


def women_actors(demographics_csv: Path) -> set[str]:
    with demographics_csv.open(encoding="utf-8", newline="") as file:
        return {row["ActorID"] for row in csv.DictReader(file) if row["Sex"] == "Female"}


def select_clips(pattern: str, women: set[str], actors: Iterable[str] | None = None) -> list[Path]:
    """Clips of the listed women. An actor who is not on the list of women is an error, not a silent skip."""
    wanted = set(actors) if actors is not None else None
    if wanted is not None and not wanted <= women:
        raise DatasetError(f"女性の一覧にない番号です: {', '.join(sorted(wanted - women))}")
    clips = []
    for path in sorted(Path(p) for p in glob.glob(pattern)):
        match = _CLIP_NAME.match(path.stem)
        if match and match.group(1) in women and (wanted is None or match.group(1) in wanted):
            clips.append(path)
    return clips


def read_frames(source: Path, transform: Callable[[np.ndarray], np.ndarray] | None = None) -> Iterator[np.ndarray]:
    """uint8 BGR frames of a video, or of a folder of images by name."""
    if source.is_dir():
        for path in sorted(p for p in source.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")):
            frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise DatasetError(f"画像を読めません: {path.name}")
            yield transform(frame) if transform else frame
        return
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise DatasetError(f"動画を開けません: {source.name}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield transform(frame) if transform else frame
    finally:
        capture.release()


class SequenceWriter:
    """Writes one sequence to a temporary folder and renames it when complete: half a sequence never looks like one."""

    def __init__(self, root: Path, name: str, kind: str, input_preset: str, fps: float, consecutive: bool = True, note: str = "") -> None:
        self.final = root / name
        self.folder = root / f".{name}.tmp"
        if self.final.exists():
            raise DatasetError(f"すでにあります: {name}")
        shutil.rmtree(self.folder, ignore_errors=True)
        for part in ("input", "target"):
            (self.folder / part).mkdir(parents=True)
        self.kind, self.input_preset, self.fps, self.consecutive, self.note = kind, input_preset, fps, consecutive, note
        self.count = 0
        self.landmarks: list[np.ndarray] = []
        self.has_masks = False

    def add(self, input_frame: np.ndarray, target_frame: np.ndarray, landmarks: np.ndarray, mask: np.ndarray | None = None) -> None:
        name = f"{self.count:06d}.png"
        # Low compression: these are working files, and reading them back is what a training run spends its time on.
        options = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        for part, image in (("input", input_frame), ("target", target_frame)):
            ok, encoded = cv2.imencode(".png", image, options)
            if not ok:
                raise DatasetError("PNG を書けません")
            encoded.tofile(str(self.folder / part / name))
        if mask is not None:
            (self.folder / "masks").mkdir(exist_ok=True)
            ok, encoded = cv2.imencode(".png", (np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), options)
            if not ok:
                raise DatasetError("PNG を書けません")
            encoded.tofile(str(self.folder / "masks" / name))
            self.has_masks = True
        self.landmarks.append(np.asarray(landmarks, dtype=np.float32))
        self.count += 1

    def finish(self) -> Path | None:
        """The finished folder, or None when too short to be of use (then nothing stays behind)."""
        if self.count < _MIN_SEQUENCE:
            self.discard()
            return None
        np.save(self.folder / "landmarks.npy", np.stack(self.landmarks))
        write_manifest(self.folder, self.kind, self.input_preset, self.fps, self.count, self.consecutive, self.note)
        self.folder.rename(self.final)
        return self.final

    def discard(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)


def reshape_is_neutral(settings: Any) -> bool:
    """A preset that moves the face's geometry cannot give pixel-aligned pairs."""
    section = settings.reshape
    if not section.enabled:
        return True
    names = ("face_slim", "jaw", "chin_length", "eye_size", "nose_width", "nose_length", "mouth_width", "mouth_corner", "lip_fullness", "brow_lift")
    return all(getattr(section, name) == 0 for name in names)


@dataclass
class _Classic:
    """The classic pipeline for one preset, with reproducible masks."""

    settings: Any
    pipeline: Any


def _classic_side(preset: str, sync_masks: bool = True) -> _Classic:
    from makeup_cam.models import SELFIE_MULTICLASS, ensure_model
    from makeup_cam.pipeline import MakeupPipeline
    from makeup_cam.presets import load_preset
    from makeup_cam.segmentation import Segmenter

    settings = load_preset(preset)
    segmenter = Segmenter(ensure_model(SELFIE_MULTICLASS), settings.general.segmentation_interval, background=not sync_masks)
    return _Classic(settings, MakeupPipeline(settings, segmenter=segmenter))


def _mask_source() -> Any:
    from makeup_cam.models import SELFIE_MULTICLASS, ensure_model
    from makeup_cam.segmentation import Segmenter

    return Segmenter(ensure_model(SELFIE_MULTICLASS), 1, background=False)


def _settle_frames() -> int:
    from makeup_cam.pipeline import FADE_FRAMES, STEADY_FRAMES

    return STEADY_FRAMES + FADE_FRAMES + 1


@dataclass
class _Frame:
    input: np.ndarray
    target: np.ndarray
    landmarks: np.ndarray
    mask: np.ndarray | None


def pair_frames(
    frames: Iterable[tuple[np.ndarray, np.ndarray | None]],
    input_process: Callable[[np.ndarray, int], Any] | None,
    target_process: Callable[[np.ndarray, int], Any] | None,
    masks: Callable[[np.ndarray, int, int], np.ndarray | None] | None,
    settle: int,
    fps: float = 30.0,
) -> Iterator[_Frame | None]:
    """Yields each pair with a settled face, and None where the run of consecutive frames breaks. ``frames`` gives
    (input source, target source or None). One frame at a time: a long take must not have to fit in memory.

    A process function returns a ``ProcessResult``. The landmarks are those of the input side, or of the target side when
    the input side is the raw frame (preset ``off`` needs no second tracker).
    """
    settled = 0
    for index, (source, target_source) in enumerate(frames):
        timestamp = int(index * 1000.0 / fps)
        side_input = input_process(source, timestamp) if input_process else None
        side_target = target_process(source, timestamp) if target_process else None
        tracked = side_input if side_input is not None else side_target
        if tracked is None or tracked.landmarks is None or not tracked.face_found:
            if settled > settle:
                yield None
            settled = 0
            continue
        settled += 1
        if settled <= settle:
            continue
        input_frame = side_input.output if side_input is not None else source
        target_frame = side_target.output if side_target is not None else target_source
        mask = masks(source, timestamp, index) if masks else None
        yield _Frame(input_frame, target_frame, np.array(tracked.landmarks), mask)


def _union_masks(source: Any) -> Callable[[np.ndarray, int, int], np.ndarray | None]:
    def compute(frame: np.ndarray, timestamp: int, index: int) -> np.ndarray | None:
        result = source.update(frame, timestamp, index)
        if result.hair is None and result.face_skin is None:
            return None
        if result.hair is None or result.face_skin is None:
            return result.hair if result.hair is not None else result.face_skin
        return np.maximum(result.hair, result.face_skin)

    return compute


def _write_runs(stream: Iterable[_Frame | None], root: Path, name: str, kind: str, input_preset: str, fps: float, note: str) -> list[Path]:
    """One sequence per run of consecutive frames: ``name``, then ``name_2`` and so on."""
    written: list[Path] = []
    writer: SequenceWriter | None = None
    number = 0
    try:
        for frame in stream:
            if frame is None:
                if writer is not None and (finished := writer.finish()) is not None:
                    written.append(finished)
                writer = None
                continue
            if writer is None:
                number += 1
                writer = SequenceWriter(root, name if number == 1 else f"{name}_{number}", kind, input_preset, fps, True, note)
            writer.add(frame.input, frame.target, frame.landmarks, frame.mask)
        if writer is not None and (finished := writer.finish()) is not None:
            written.append(finished)
            writer = None
    finally:
        if writer is not None:
            writer.discard()
    return written


def _result_ok(preset: str, settings: Any) -> None:
    if not reshape_is_neutral(settings):
        raise DatasetError(f"プリセット {preset} は顔の形を変えるので、位置のそろったペアを作れません")


def check_standin_presets(input_preset: str, target_preset: str) -> None:
    """The same preset on both sides gives a target delta of 0 everywhere: nothing to learn."""
    if input_preset == target_preset:
        raise DatasetError(f"入力と目標が同じプリセットです（{input_preset}）。差がないので学習できません")


def command_standin(args: argparse.Namespace) -> int:
    from makeup_cam.presets import load_preset

    refuse_tracked_folder(args.out)
    clips_pattern = str(args.clips)
    found = glob.glob(clips_pattern)
    if not found:
        raise DatasetError("クリップがありません")
    demographics = Path(args.demographics) if args.demographics else Path(found[0]).parent / "demographics.csv"
    if not demographics.is_file():
        raise DatasetError("demographics.csv がないので、女性のクリップを選べません")
    actors = args.actors.split(",") if args.actors else None
    clips = select_clips(clips_pattern, women_actors(demographics), actors)
    if not clips:
        raise DatasetError("使えるクリップがありません")
    check_standin_presets(args.input_preset, args.target_preset)
    _result_ok(args.target_preset, load_preset(args.target_preset))
    _result_ok(args.input_preset, load_preset(args.input_preset))
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(clips)} 本を古典パイプラインに通します（{args.input_preset} → {args.target_preset}）。これは配線の確認専用のデータです", flush=True)
    settle = _settle_frames()
    total = 0
    for clip in clips:
        match = _CLIP_NAME.match(clip.stem)
        name = f"cremad-{match.group(1)}-{match.group(2)}"
        if (args.out / name).exists():
            continue
        target = _classic_side(args.target_preset)
        # "off" needs no second pipeline: the input side is then the raw frame.
        side = None if args.input_preset == "off" else _classic_side(args.input_preset)
        source = _mask_source()
        try:
            frames = ((frame, None) for frame in read_frames(clip, prepare_test_clip_frame))
            if args.max_frames:
                frames = (item for i, item in enumerate(frames) if i < args.max_frames)
            input_process = None if side is None else (lambda f, t: side.pipeline.process(f, t))
            runs = pair_frames(frames, input_process, lambda f, t: target.pipeline.process(f, t), _union_masks(source), settle)
            for path in _write_runs(runs, args.out, name, "classic_pipeline", args.input_preset, 30.0, PLUMBING_NOTE):
                count = len(list((path / "input").iterdir()))
                total += count
                print(f"  {path.name}: {count} フレーム", flush=True)
        finally:
            target.pipeline.close()
            if side is not None:
                side.pipeline.close()
            source.close()
    print(f"合計 {total} フレーム")
    return 0


def command_render(args: argparse.Namespace) -> int:
    from makeup_cam.presets import load_preset

    refuse_tracked_folder(args.out)
    settings = load_preset(args.input_preset)
    _result_ok(args.input_preset, settings)
    args.out.mkdir(parents=True, exist_ok=True)
    side = _classic_side(args.input_preset)
    source = _mask_source()
    try:
        frames = zip(read_frames(args.input), read_frames(args.target))
        runs = pair_frames(frames, lambda f, t: side.pipeline.process(f, t), None, _union_masks(source), _settle_frames(), args.fps)
        written = _write_runs(runs, args.out, args.name, args.kind, args.input_preset, args.fps, args.note)
    finally:
        side.pipeline.close()
        source.close()
    print(f"{len(written)} シーケンスを書きました")
    return 0


# --------------------------------------------------------------------------
# Recordings: frames are matched, not in step.
# --------------------------------------------------------------------------


def normalized_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """(N, 2) in the crop's own frame, as shares of its side: the head's position and size no longer count."""
    matrix = geometry.crop_matrix(landmarks, 256)
    return ((np.asarray(landmarks, dtype=np.float64) @ matrix[:, :2].T + matrix[:, 2]) / 256.0).astype(np.float32)


def match_frames(bare: np.ndarray, makeup: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """For each bare frame, the index of the made-up frame nearest in pose and expression, and the root mean square landmark
    distance. Both inputs are (N, 478, 2) from ``normalized_landmarks``."""
    a = bare.reshape(len(bare), -1)
    b = makeup.reshape(len(makeup), -1)
    indices = np.zeros(len(a), dtype=np.int64)
    distances = np.zeros(len(a), dtype=np.float32)
    points = bare.shape[1]
    for start in range(0, len(a), chunk):
        part = a[start : start + chunk]
        squared = ((part[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        best = squared.argmin(axis=1)
        indices[start : start + chunk] = best
        distances[start : start + chunk] = np.sqrt(squared[np.arange(len(part)), best] / points)
    return indices, distances


@dataclass(frozen=True)
class Alignment:
    matrix: np.ndarray  # 2x3, made-up frame to bare frame
    fit_error: float  # p95 of all landmarks after the alignment, x face scale


def align_landmarks(bare: np.ndarray, makeup: np.ndarray) -> Alignment | None:
    """The similarity taking the made-up frame's stable landmarks onto the bare frame's."""
    matrix, _ = cv2.estimateAffinePartial2D(makeup[STABLE_POINTS].astype(np.float32), bare[STABLE_POINTS].astype(np.float32), method=cv2.LMEDS)
    if matrix is None:
        return None
    moved = makeup @ matrix[:, :2].T + matrix[:, 2]
    errors = np.linalg.norm(moved - bare, axis=1) / float(regions.face_scale(bare))
    return Alignment(matrix, float(np.percentile(errors, 95)))


def colour_gains(bare: np.ndarray, warped: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Exposure and white balance between the takes, from what does not wear makeup: everything outside the face."""
    height, width = bare.shape[:2]
    roi = regions.Roi(0, 0, width, height)
    outside = regions.face_oval_mask(landmarks, roi, expand_px=0.1 * float(regions.face_scale(landmarks))) <= 0.0
    usable = outside & (warped.min(axis=2) > 8) & (bare.max(axis=2) < 247) & (bare.min(axis=2) > 8)
    if np.count_nonzero(usable) < 1000:
        return np.ones(3, dtype=np.float32)
    gains = np.median(bare[usable].astype(np.float32) / warped[usable].astype(np.float32), axis=0)
    return np.clip(gains, 0.75, 1.33).astype(np.float32)


def align_target(bare: np.ndarray, makeup: np.ndarray, alignment: Alignment, bare_landmarks: np.ndarray) -> np.ndarray:
    """The made-up frame, in the bare frame's place and colour balance."""
    height, width = bare.shape[:2]
    warped = cv2.warpAffine(makeup, alignment.matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gains = colour_gains(bare, warped, bare_landmarks)
    return np.clip(warped.astype(np.float32) * gains + 0.5, 0, 255).astype(np.uint8)


def _take_landmarks(take: Take, smoothing: float) -> np.ndarray | None:
    """(N, 478, 2) with NaN for frames without a face."""
    from makeup_cam.models import FACE_LANDMARKER, ensure_model
    from makeup_cam.tracking import FaceTracker

    tracker = FaceTracker(ensure_model(FACE_LANDMARKER), smoothing)
    rows = []
    try:
        for index, frame in enumerate(read_frames(take.video)):
            landmarks, _ = tracker.detect(frame, int(index * 1000.0 / take.fps))
            rows.append(np.full((478, 2), np.nan, np.float32) if landmarks is None else np.asarray(landmarks, np.float32))
    finally:
        tracker.close()
    return np.stack(rows) if rows else None


def command_recording(args: argparse.Namespace) -> int:
    """One sequence per step of a session. Marked as not consecutive: the matched targets jump in time."""
    from makeup_cam.presets import load_preset

    refuse_tracked_folder(args.out)
    _result_ok(args.input_preset, load_preset(args.input_preset))
    pairs = find_session_pairs(args.capture)
    if args.sessions:
        pairs = [pair for pair in pairs if any(fnmatch.fnmatchcase(pair.name, pattern) for pattern in args.sessions)]
    if not pairs:
        raise DatasetError("すっぴんとメイクがそろったセッションがありません")
    args.out.mkdir(parents=True, exist_ok=True)
    source = _mask_source()
    try:
        for pair in pairs:
            for step in pair.steps:
                bare_take, makeup_take = pair.bare[step], pair.makeup[step]
                name = f"{pair.name}-step{step:02d}_{bare_take.key}"
                if (args.out / name).exists():
                    continue
                written, offered = _write_recording_step(bare_take, makeup_take, args, name, source)
                print(f"  {name}: {offered} 枚のうち {written} 枚が合いました", flush=True)
    finally:
        source.close()
    return 0


def _write_recording_step(bare_take: Take, makeup_take: Take, args: argparse.Namespace, name: str, source: Any) -> tuple[int, int]:
    settings_side = _classic_side(args.input_preset)
    makeup_landmarks = _take_landmarks(makeup_take, settings_side.settings.general.landmark_smoothing)
    if makeup_landmarks is None:
        settings_side.pipeline.close()
        return 0, 0
    usable = np.flatnonzero(np.isfinite(makeup_landmarks).all(axis=(1, 2)))
    normalized = np.stack([normalized_landmarks(makeup_landmarks[i]) for i in usable]) if len(usable) else np.zeros((0, 478, 2), np.float32)
    capture = cv2.VideoCapture(str(makeup_take.video))
    writer = SequenceWriter(args.out, name, "recording", args.input_preset, bare_take.fps / args.stride, False, args.note)
    offered = settled = 0
    masks = _union_masks(source)
    try:
        for index, frame in enumerate(read_frames(bare_take.video)):
            result = settings_side.pipeline.process(frame, int(index * 1000.0 / bare_take.fps))
            settled = settled + 1 if result.face_found and result.landmarks is not None else 0
            if settled <= _settle_frames() or index % args.stride or not len(usable):
                continue
            offered += 1
            landmarks = np.asarray(result.landmarks, np.float32)
            chosen, distance = match_frames(normalized_landmarks(landmarks)[None], normalized)
            if float(distance[0]) > _MAX_MATCH_DISTANCE:
                continue
            makeup_index = int(usable[int(chosen[0])])
            alignment = align_landmarks(landmarks, makeup_landmarks[makeup_index])
            if alignment is None or alignment.fit_error > _MAX_FIT_ERROR:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, makeup_index)
            ok, makeup = capture.read()
            if not ok:
                continue
            # The target is aligned to the raw bare frame; the input side keeps the classic look.
            target = align_target(frame, makeup, alignment, landmarks)
            writer.add(result.output, target, landmarks, masks(frame, int(index * 1000.0 / bare_take.fps), index))
        finished = writer.finish()
        return (writer.count if finished else 0), offered
    finally:
        capture.release()
        settings_side.pipeline.close()
        if not writer.final.exists():
            writer.discard()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="学習用のデータを作ります（.venv-neural で実行します）")
    commands = parser.add_subparsers(dest="command", required=True)

    standin = commands.add_parser("standin", help="CREMA-D の女性を古典パイプラインに通す。配線の確認専用")
    standin.add_argument("--clips", default=".test-media/testvids/*.flv", help="クリップのパターン")
    standin.add_argument("--demographics", help="demographics.csv（既定はクリップと同じフォルダ）")
    standin.add_argument("--actors", help="番号をカンマで区切って。既定は全員")
    standin.add_argument("--input-preset", default="off", help="入力側のプリセット。モデルを natural の上で動かすなら natural にします（off のままだとメイクが二重になります）")
    standin.add_argument("--target-preset", default="natural")
    standin.add_argument("--max-frames", type=int, default=0)
    standin.add_argument("--out", type=Path, required=True)
    standin.set_defaults(run=command_standin)

    render = commands.add_parser("render", help="位置のそろった 2 つの素材（3D の描画など）")
    render.add_argument("--input", type=Path, required=True, help="入力側の動画か画像フォルダ")
    render.add_argument("--target", type=Path, required=True, help="目標側の動画か画像フォルダ")
    render.add_argument("--name", required=True, help="シーケンス名（個人名を入れないでください）")
    render.add_argument("--kind", choices=("3d_render", "recording"), default="3d_render")
    render.add_argument("--input-preset", default="natural")
    render.add_argument("--fps", type=float, default=30.0)
    render.add_argument("--note", default="")
    render.add_argument("--out", type=Path, required=True)
    render.set_defaults(run=command_render)

    recording = commands.add_parser("recording", help="撮影ガイドのセッション（すっぴんとメイク）")
    recording.add_argument("--capture", type=Path, required=True, help="sessions/ のあるフォルダ")
    recording.add_argument("--sessions", nargs="*", help="使うセッション（例: 2026-11-10_main）")
    recording.add_argument("--input-preset", default="natural")
    recording.add_argument("--stride", type=int, default=3, help="何フレームごとに使うか")
    recording.add_argument("--note", default="")
    recording.add_argument("--out", type=Path, required=True)
    recording.set_defaults(run=command_recording)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except DatasetError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
