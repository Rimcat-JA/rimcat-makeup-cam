"""Evaluation of a finishing model, or of a preset alone: is the result stable and still the same person?

``python -m makeup_cam.evaluate`` runs clips through the real pipeline as a library and measures the
result against thresholds. Output: ``evaluation.json`` (machine-readable), ``report_ja.md`` and a Japanese
summary on stdout, frame numbers and saved crops for every breakdown. See ``docs/dev/evaluation.md``
and section 9 of ``docs/dev/contract-plan-b.md``.

What is compared with what
    With ``--model`` the reference is the classic output of the same run: a read-only probe sits just
    before the ``neural`` step, so both come from the same landmarks and masks, and "weight exactly 0"
    can be checked bit for bit. Without a model the output is the classic pipeline, and the reference is the
    input. With ``--rendered`` a finished video is measured, against ``--reference`` if given, else the input.

    Every number here depends on the conditions in the report. No threshold says a look is good: they
    only find what breaks. All thresholds are first proposals, as no real makeup model exists yet.

Identity
    Landmark geometry and edge structure need nothing extra. An embedding model is optional
    (``--identity-model``), never downloaded by this tool, and its licence is for the caller to state.

Detector report
    ``--deepfake-report`` only prints what a detector the user ran themselves said. It is a quality signal
    for the user's own review and never takes part in the verdict. Passing is no guarantee of anything, and this tool
    is not for hiding manipulation: it does not call, bundle, download or train against a detector.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import glob as globlib
import hashlib
import json
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import regions
from .config import MakeupSettings
from .effects.base import Effect, FrameContext
from .neural import geometry

logger = logging.getLogger(__name__)

SCHEMA = 1
FRAME_SIZE = (1280, 720)  # (width, height): the size every clip is brought to
ANALYSIS_CROP = 256  # the crop the temporal measures work in
LAG_CROP = 128  # the crop the paint-lag estimate works in
SETTLE_FRAMES = 20  # the landmark model's first answers are stretched, and the fades run in (the stage's takes 15): not measured
MIN_PASS_FRAMES = 1000  # measured frames below which a verdict is at best "incomplete"
MIN_SOAK_MINUTES = 30.0
FULL_STRENGTH = 1.0  # the top of the strength slider
SWAY_CLIPS = 8  # clips also measured with a sway added (of all that were given, evenly spread)
NOD_CLIPS = 4  # the first of those also measured with a nod and a bow added
BUDGET_PRESET = "natural"  # the contract's latency budget is stated at this preset
# What the app's output thread does after each frame and this tool, which opens no virtual camera, does not: the frame into the
# camera's format and shared memory, and the preview's hand-over. An estimate: the conversion and copy alone took 0.46 ms here,
# with OpenCV's thread pool; the virtual camera itself has not been measured. It matters because the frame rate does not fall
# with the time a frame takes, but all at once where the frame time is used up: without this the same model on the same clip read
# anything from 0.0 to 1.1 fps, with a 2 ms cost in the output thread a steady 1.1. See REALTIME_ROUNDS.
DELIVERY_MS = 1.5
# Model runs per case, each between classic runs. The worst counts: near the limit of the frame time a single run may pass by luck.
REALTIME_ROUNDS = 3
RENDER_COST_FRAMES = 30  # measured per clip, to find the clip that is heaviest at a preset that was not evaluated
OPEN_MOUTH = 0.06  # inner lip gap x face scale from which the mouth counts as open
MIN_EYE_AREA = 0.06  # x face scale squared: smaller eye openings (a blink) have too few edges to compare
MAX_CROPS_PER_KIND = 12  # saved crops per clip and kind of event
EVENT_MERGE_GAP = 5  # frames: the same kind again within this counts as the same event

EXIT_PASS, EXIT_FAIL, EXIT_INCOMPLETE, EXIT_ERROR = 0, 1, 2, 3

# Modes: what the reference is.
MODE_CLASSIC = "classic"  # the model's output against the classic output of the same run
MODE_INPUT = "input"  # the output against the bare input
MODE_PAIRED = "paired"  # a rendered output against a paired reference video
ALL_MODES = frozenset({MODE_CLASSIC, MODE_INPUT, MODE_PAIRED})
CLASSIC_ONLY = frozenset({MODE_CLASSIC})
LIVE_MODEL = CLASSIC_ONLY  # needs the running model


@dataclass(frozen=True)
class Threshold:
    """One line of the table below."""

    key: str
    label: str  # Japanese, for the report
    unit: str
    op: str  # "le", "ge" or "true"
    value: float | None  # limit of the median over clips (or of the single value)
    worst: float | None = None  # limit of the worst clip
    provisional: bool = True
    required: bool = True  # False: shown, but never takes part in the verdict
    modes: frozenset[str] = ALL_MODES
    basis: str = ""
    # Limits when the reference is the bare input (or a paired video): makeup itself changes much more there.
    input_value: float | None = None
    input_worst: float | None = None

    def limits(self, mode: str) -> tuple[float | None, float | None]:
        if mode != MODE_CLASSIC and self.input_value is not None:
            return self.input_value, self.input_worst
        return self.value, self.worst


# The spatial checks (a review found the temporal measures blind to a stable seam, cast or warp). Numbers: see docs/dev/evaluation.md.
WARP_FLOW_LIMITS = (0.45, 0.7)
WARP_FLOW_BASIS = "44 clips, clean imitations at most 0.145 (0.23 with a nod); stand-in students through the real worker 0.15 (p32 at 512) to 0.54 (p16 at 256, a small face). Median over clips: nose 15 % wider 0.69, eyes 10 % larger 0.83, nose 30 % wider 1.42. Single clips over the worst limit: 18 of 44, 38 of 44, 40 of 44; jaw 20 % slimmer 44 of 44, at least 0.92. A nose 6 % wider gives 0.30: not seen. Crop pixels on calm skin"
SEAM_EDGE_LIMITS = (1.2, 2.0)
SEAM_EDGE_BASIS = "44 clips, clean imitations at most 1.61 (a smoothing one; median 0.98), stand-ins through the real worker at most 0.61; a grid of 5 levels 2.52 at best, a step of 20 levels 3.03, a step of 10 levels: median over clips 1.30, so seen across clips only; a step of 4 levels 0.42 at most: not seen"
SIDE_CAST_LIMIT = 3.0
SIDE_CAST_BASIS = "44 clips, clean imitations at most 0.29, stand-ins through the real worker at most 1.06; one side redder by 8 levels: 5.28 at best"
SKIN_CAST_LIMIT = 12.0
SKIN_CAST_BASIS = "44 clips, clean imitations at most 2.41, made 3 x stronger 6.8, stand-ins through the real worker at most 2.6; the whole face redder by 25 levels: 15.55 at best"
RIM_CHANGE_LIMIT = 0.25
RIM_CHANGE_BASIS = "44 clips, clean imitations at most 0.21, stand-ins through the real worker at most 0.005; 20 levels painted on the soft rim, through the weight: 0.28 at best"
FRAME_LOCKED_LIMITS = (1.0, 1.6)
FRAME_LOCKED_BASIS = "44 clips with a sway or a nod, clean imitations: median 0, at most 1.42; stand-ins through the real worker 0.5 to 0.76. A soft patch of 8 levels fixed to the frame: median over clips 1.49, at least 0.94, so seen across clips, not on each. Needs motion: only clips with 25 frames that move get a value"
PASTE_DRIFT_LIMIT = 0.2
PASTE_DRIFT_BASIS = "x face scale, of the frames not weakened to under half. 4 clips: talking at most 0.16, with a nod 0.11. Without the stage's motion gate, from recorded tracks: a nod at age 1 0.12, at age 3 0.24, a fast bow at age 1 0.59"

# Every threshold of the tool, in one table. All are first proposals: no real model exists. ``basis`` says where each comes from.
# "clean" is the worst clip of the clean runs (classic presets, and two imitations of a well-behaved model, also with an added
# sway), "defect" the injected defect closest to the limit. Over all 44 CREMA-D clips; the tables are in docs/dev/evaluation.md.
_CONTRACT = "contract, section 7.3"
THRESHOLDS: tuple[Threshold, ...] = (
    Threshold("flicker_ratio", "顔の中のちらつき（基準との比）", "倍", "le", 1.08, 1.15, basis="clean 1.066; noise of 2 levels: median 1.40, 1 level: 1.14. Contract's first 1.10 / 1.25 missed more of the 2-level noise"),
    Threshold("flicker_block_ratio", "顔の中のちらつき（広い範囲の明滅）", "倍", "le", 1.05, 1.10, basis="clean 1.009; a face pumping by 1 level: lowest clip 2.9"),
    Threshold("flicker_outside_ratio", "顔の外のちらつき（基準との比）", "倍", "le", 1.05, 1.10, basis="clean 1.000; noise of 1 level on the wall: lowest clip 2.2"),
    Threshold("delta_flicker", "加えた変化そのもののちらつき", "階調", "le", 0.5, 0.7, input_value=1.0, input_worst=2.2, basis="clean 0.55 (input mode: 2.05, strong preset); noise of 1 level: lowest clip 0.88"),
    Threshold("mouth_interior_max", "口の内側の変化（最大）", "階調", "le", 0.0, provisional=False, modes=CLASSIC_ONLY, basis="contract: bit for bit. Clean runs: 0 on all clips"),
    Threshold("eye_interior_max", "目の開口部の変化（最大）", "階調", "le", 0.0, provisional=False, modes=CLASSIC_ONLY, basis="contract: bit for bit. Clean runs: 0 on all clips"),
    Threshold("outside_max", "変更禁止の領域の変化（最大）", "階調", "le", 0.0, provisional=False, modes=CLASSIC_ONLY, basis="contract: bit for bit. Clean runs: 0 on all clips"),
    Threshold("mouth_band_edge_corr", "口の縁の形（エッジの相関）", "", "ge", 0.90, input_value=0.65, basis="clean 0.923 (a smoothing imitation; the contract's first 0.98 would fail it); teeth moved by 1 px: 0.91 at best, median 0.87"),
    Threshold("mouth_inner_edge_corr", "歯の形（エッジの相関）", "", "ge", 0.95, input_value=0.75, basis="clean 0.999 (input mode 0.85); teeth moved by 1 px: 0.84 at best (input mode, 2 px: 0.59)"),
    Threshold("eye_edge_corr", "目の形（エッジの相関）", "", "ge", 0.95, input_value=0.55, basis="clean 0.989 (input mode 0.56); eyes moved by 1 px: 0.87 at best"),
    Threshold("background_change", "背景の変化（遠い背景の平均）", "階調", "le", 0.1, 0.2, basis="clean 0.001; a soft patch of 4 levels on the wall: lowest clip 0.34"),
    Threshold("landmark_shift_p99", "検出点のずれ（p99、3 フレームの中央値）", "×顔の大きさ", "le", 0.06, basis="clean 0.045; a real worker with a random-weights stand-in 0.054; another face, aligned: lowest clip 0.043, median 0.09. The contract's first 0.015 fails clean clips: the landmark model itself differs by more on nearly equal frames"),
    Threshold("identity_geometry_p99", "顔の形の変化（相似変換を除いたずれ、p99）", "×顔の大きさ", "le", 0.02, input_value=0.025, basis="clean 0.015 (input mode 0.021); another face, warped onto the same landmarks: median 0.021, so this alone cannot see that case"),
    Threshold("identity_structure_p01", "顔の構造の相関（素顔との比較、p01）", "", "ge", 0.75, basis="clean 0.848 (strong preset), 0.92 for the imitations; another face: 0.62 at best"),
    Threshold("identity_cosine_p01", "顔認識モデルの類似度（p01、素顔との類似度）", "", "ge", 0.70, modes=ALL_MODES, basis="output against the bare frame. SFace: natural 0.890 at worst, strong 0.764 at worst, another face 0.519 at best. The contract's first rule, at most 0.02 below classic, rejects any added makeup: a stand-in that imitates natural from off lost 0.028 to 0.059, as much as a 30 % wider nose; shape now has warp_flow_px. Only with an optional model; see docs/dev/evaluation.md for that trial"),
    Threshold("warp_flow_px", "顔の形の動き（色ではなく位置の変化）", "px", "le", WARP_FLOW_LIMITS[0], WARP_FLOW_LIMITS[1], modes=CLASSIC_ONLY, basis=WARP_FLOW_BASIS),
    Threshold("seam_edge", "変化の中の段差や格子", "階調/px", "le", SEAM_EDGE_LIMITS[0], SEAM_EDGE_LIMITS[1], modes=CLASSIC_ONLY, basis=SEAM_EDGE_BASIS),
    Threshold("side_cast", "顔の左右で違う色かぶり", "階調", "le", SIDE_CAST_LIMIT, modes=CLASSIC_ONLY, basis=SIDE_CAST_BASIS),
    Threshold("skin_cast", "肌全体の色かぶり", "階調", "le", SKIN_CAST_LIMIT, modes=CLASSIC_ONLY, basis=SKIN_CAST_BASIS),
    Threshold("rim_change", "髪や肩のまわりの塗り（マスクの弱い帯）", "階調", "le", RIM_CHANGE_LIMIT, modes=CLASSIC_ONLY, basis=RIM_CHANGE_BASIS),
    Threshold("frame_locked_ratio", "顔についてこない変化（フレームに固定された貼り付きなど）", "倍", "le", FRAME_LOCKED_LIMITS[0], FRAME_LOCKED_LIMITS[1], modes=CLASSIC_ONLY, basis=FRAME_LOCKED_BASIS),
    Threshold("delta_abs_p99", "肌の上の変化の大きさ（p99）", "0〜1", "le", 0.15, modes=CLASSIC_ONLY, basis="contract. Clean 0.055; an imitation made 20 x stronger: lowest clip 0.235"),
    Threshold("mouth_latency_frames", "口の動きの遅れ（口元の同期の目安）", "フレーム", "le", 0.3, 0.6, basis="clean 0.32 (the landmark model's own difference on made-up lips); lips 1 frame late: lowest clip 0.93"),
    Threshold("paint_lag_frames", "メイクが顔に遅れる量", "フレーム", "le", 0.6, 0.9, basis="clean 0.70 to 0.78 in two runs (a fixed tint through the real worker: the landmark smoothing's own lag; the rounding of the imitations hides part of it: 0.41); the whole change 1 frame late: lowest clip 0.96 with the imitation, 1.33 on the real path. Needs motion: only clips with 25 frames that move get a value"),
    Threshold("paint_lag_local_frames", "一部のメイクが顔に遅れる量（局所）", "フレーム", "le", 0.8, 1.35, modes=CLASSIC_ONLY, basis="clean 1.32 (an imitation that depends on the picture; 1.15 before masks counted only where sure; a fixed tint 0 to 0.47), median over clips 0.42; a patch 2 frames late: median 1.63, 3 frames: 2.18, 5 frames: 3.04, 1 frame: at most 1.04, so that one is not seen. Needs motion. Without a model the makeup itself moves relative to the rigid crop (clean up to 1.51), so it only applies with a model"),
    Threshold("paint_offset", "メイクの位置ずれ（唇の外に出た色の割合）", "", "le", 0.15, 0.3, basis="clean 0.25; lip color 0.12 x face scale off: lowest clip 0.34. 0.06 x face scale is mostly not seen"),
    Threshold("breaks_per_30min", "崩れの回数（30 分あたり）", "回", "le", 1.0, provisional=False, basis="contract"),
    Threshold("paired_abs", "対の参照動画との差（参考。判定には使いません）", "階調", "le", None, required=False, basis="review only: a pair filmed separately is never aligned pixel for pixel"),
    Threshold("deepfake_margin", "検出器のスコアの差（参考。判定には使いません）", "", "le", None, required=False, basis="review only, never part of the verdict: see the module docstring"),
    Threshold("all_kernels_dml", "全ノードが GPU 上で動く", "", "true", None, provisional=False, modes=LIVE_MODEL, basis="contract, section 8"),
    Threshold("rtt_p95_ms", "モデルの往復時間 p95", "ms", "le", 12.0, provisional=False, modes=LIVE_MODEL, basis=_CONTRACT),
    Threshold("rtt_p99_ms", "モデルの往復時間 p99", "ms", "le", 20.0, provisional=False, modes=LIVE_MODEL, basis=_CONTRACT),
    Threshold("stage_p95_ms", "カメラスレッド上の処理時間 p95", "ms", "le", 5.0, provisional=False, modes=LIVE_MODEL, basis=_CONTRACT),
    Threshold("fps_loss", "フレームレートの低下（最悪の組み合わせ）", "fps", "le", 0.5, provisional=False, modes=LIVE_MODEL, basis=_CONTRACT),
    Threshold("miss_pct", "2 フレーム以上古い差分で描いた割合", "%", "le", 0.5, provisional=False, modes=LIVE_MODEL, basis="contract, section 9.2"),
    Threshold("paste_drift_p99", "貼った差分のずれ（p99、弱めずに貼ったフレーム）", "×顔の大きさ", "le", PASTE_DRIFT_LIMIT, modes=LIVE_MODEL, basis=PASTE_DRIFT_BASIS),
    Threshold("vram_mb", "ワーカーの GPU メモリ", "MB", "le", 1500.0, provisional=False, modes=LIVE_MODEL, basis=_CONTRACT),
    Threshold("strength_zero_identical", "強さ 0 で従来の出力と完全に一致", "", "true", None, provisional=False, modes=LIVE_MODEL, basis="contract, section 9.2"),
    Threshold("strength_monotonic", "強さとともに変化が増える", "", "true", None, provisional=False, modes=LIVE_MODEL, basis="contract, section 9.2"),
    Threshold("fade_drill", "障害時に従来の出力へ滑らかに戻る", "", "true", None, provisional=False, modes=LIVE_MODEL, basis="contract, section 9.2"),
)
THRESHOLD_BY_KEY = {t.key: t for t in THRESHOLDS}

# What a single frame must show to count as a breakdown. The same status as the table above. "held": for HELD_FRAMES frames.
# The numbers in the comments: worst clean clip, and the defect closest to the limit, per frame, over all 44 clips.
EVENT_RULES: dict[str, float] = {
    "spike_factor": 8.0,  # a local one-frame change over this x the running median (clean 7.8, input mode 11.5 once; 6 levels on a cheek 7.6) ...
    "spike_floor": 3.0,  # ... and over this many levels, as a block mean
    "background_block": 1.0,  # levels, block mean in the far background (clean 0.57; noise of 1 level 1.05)
    "background_mean": 0.1,  # levels, mean of the far background (clean 0.001; a soft patch of 4 levels 0.34)
    "teeth_edge_corr": 0.90,  # open mouth, edges inside it (clean 0.996; teeth moved by 1 px 0.83)
    "eye_edge_corr": 0.90,  # edges inside the eye opening (clean 0.96; eyes moved by 1 px 0.86)
    "input_teeth_edge_corr": 0.70,  # the same against the bare input (clean 0.79, strong preset; teeth moved by 2 px 0.56)
    "input_eye_edge_corr": 0.30,  # the same (clean 0.35, strong preset; eyes moved by 2 px: 0.62, so this one finds little)
    "stage_jump": 0.07,  # x face scale: the landmarks drawn with jump against the raw ones (clean 0.033, with a sway 0.082 on few clips; a jump of 0.08 finds 0.079)
    "output_jump": 0.13,  # x face scale: the output's landmarks jump against the reference's (clean 0.125, the landmark model's own glitches)
    "stage_offset": 0.06,  # x face scale, held: the landmarks drawn with are off the face (clean 0.033; injected 0.1)
    "lip_offset": 0.37,  # share, held: lip color beside the lips (clean 0.36; 0.12 x face scale off 0.37)
    "lag_local_frames": 3.0,  # frames, one box, over a window of LAG_WINDOW frames, for gross cases only (clean 2.84; a patch 5 frames late: median 4.09, 3 frames: 2.77)
    "lag_frames": 3.0,  # frames, over a window of LAG_WINDOW frames, for gross cases only (clean 2.8 through the real worker; the whole change 3 frames late: median 3.2)
    "lip_min_strength": 1.0,  # levels of color change on the lips below which the lips have no paint to find
    "frozen_output": 0.02,  # levels, mean frame step of the output, held ...
    "frozen_input": 0.2,  # ... while the input's is over this
    "identity_landmarks": 0.05,  # x face scale, held (clean 0.043; another face, aligned 0.037)
    "identity_structure": 0.70,  # held (clean 0.86; another face 0.66)
}
SPIKE_REFERENCE_SHARE = 0.25  # of the reference's own second difference at that place: see ``ClipAnalyzer._measure_face``
HELD_FRAMES = 3  # frames a "held" rule must hold
SPIKE_MEDIAN_FRAMES = 30
LAG_WINDOW = 15
LAG_LOCAL_BOX = 32  # crop pixels of LAG_CROP (about 1.1 x face scale): the size of what can lag on its own
LAG_LOCAL_FLOOR = 0.2  # squared levels per moving frame, mean over a box: below this a box has too little pattern for a fit
LAG_LOCAL_COVER = 0.4  # share of a box inside the measured area: boxes on its rim hold a few noisy pixels (they gave all the false alarms)
_FRAME_LOCK_FLOOR = 0.4  # levels, root mean square: a box with less change than this has nothing to hold still
EVENT_LABELS: dict[str, str] = {
    "spike": "一瞬だけの局所的な変化",
    "off_target": "メイクの位置ずれ",
    "landmark_jump": "検出点の飛び",
    "teeth_altered": "歯・口の内側の変化",
    "eyes_altered": "目の開口部の変化",
    "background_changed": "背景の変化",
    "paint_behind": "顔が動いたあとにメイクが残る",
    "face_lost": "出力で顔が見つからない",
    "frozen": "出力が固まっている",
    "identity_shift": "顔の形・構造の変化",
    "result_rejected": "モデルの結果を破棄",
    "health_fall": "モデルの強さが自動で下がった",
    "worker_restart": "ワーカーの再起動",
}

# Landmarks that identity geometry and drawing checks use.
_RIGID = [33, 133, 362, 263, 168, 6, 197, 2]
_IDENTITY = sorted(set(regions.FACE_OVAL) | set(regions.LIPS_OUTER) | {33, 133, 362, 263, 168, 6, 197, 2, 1, 98, 327, 70, 105, 107, 300, 334, 336})
_INNER_FACE = [10, 338, 297, 332, 284, 251, 389, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 162, 21, 54, 103, 67, 109]


class EvaluationError(Exception):
    """The evaluation could not run. The message is in Japanese."""


# --------------------------------------------------------------------------
# Frames and masks
# --------------------------------------------------------------------------


def bring_to_size(frame: np.ndarray, size: tuple[int, int] = FRAME_SIZE) -> np.ndarray:
    """A clip's frame at the evaluation's size: scaled to fit, centered on black."""
    width, height = size
    h, w = frame.shape[:2]
    if (w, h) == size:
        return frame
    k = min(width / w, height / h)
    new_w, new_h = round(w * k), round(h * k)
    scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC if k > 1.0 else cv2.INTER_AREA)
    left, top = (width - new_w) // 2, (height - new_h) // 2
    return cv2.copyMakeBorder(scaled, top, height - new_h - top, left, width - new_w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def read_frames(path: Path, limit: int | None = None) -> Iterator[np.ndarray]:
    """Decoded frames of a file, at the evaluation's size."""
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            raise EvaluationError(f"動画を開けませんでした: {path.name}")
        count = 0
        while limit is None or count < limit:
            ok, frame = capture.read()
            if not ok or frame is None:
                return
            yield bring_to_size(frame)
            count += 1
    finally:
        capture.release()


def file_fps(path: Path) -> float:
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    return fps if 5.0 <= fps <= 240.0 else 30.0


def crop_roi(landmarks: np.ndarray, width: int, height: int) -> regions.Roi:
    """The frame rectangle the aligned crop covers (the whole frame if it covers none of it)."""
    roi = geometry.crop_roi(geometry.crop_matrix(landmarks, ANALYSIS_CROP), ANALYSIS_CROP, (height, width))
    return regions.Roi(0, 0, width, height) if roi is None else roi


@dataclass
class Zones:
    """The regions of section 5 of the contract, for one frame, from the landmarks the stage used."""

    roi: regions.Roi
    weight: np.ndarray  # float32, in ``roi``
    mouth_exact: np.ndarray  # bool, in ``roi``: the mouth opening, where the weight is exactly 0
    eyes_exact: np.ndarray  # bool, in ``roi``: the eye openings, where the weight is exactly 0
    allowed_soft: np.ndarray  # float32, in ``roi``: where the stage could draw at all
    skin: np.ndarray  # bool, in ``roi``
    mouth_band: np.ndarray | None = None  # bool, in ``roi``: the ring around the inner lip line; None: mouth closed
    mouth_inner: np.ndarray | None = None  # bool, in ``roi``: inside the inner lip line
    eyes_open: bool = False  # enough of the eye openings shows to compare edges
    rim: np.ndarray | None = None  # bool, in ``roi``: a mask's soft edge, away from the face: wall, collar, shoulders


def build_zones(
    landmarks: np.ndarray,
    hair_mask: np.ndarray | None,
    skin_mask: np.ndarray | None,
    width: int,
    height: int,
    may_edit_eyes: bool = False,
) -> Zones:
    """The weight of the contract's section 5, plus what the checks need."""
    roi = crop_roi(landmarks, width, height)
    s = regions.face_scale(landmarks)
    weight = geometry.edit_weight(landmarks, roi, hair_mask, skin_mask, may_edit_eyes)
    oval = regions.face_oval_mask(landmarks, roi, expand_px=0.03 * s, feather_px=0.05 * s)
    allowed = geometry.allowed_area(landmarks, roi, hair_mask, skin_mask)
    mouth = regions.polygon_mask(regions.expand_polygon(landmarks[regions.LIPS_INNER], 0.01 * s), roi)
    eyes = np.zeros(mouth.shape, np.float32)
    for side in regions.SIDES:
        eyes = np.maximum(eyes, regions.eye_opening_mask(landmarks, side, roi, expand_px=0.005 * s))
    keep_out = mouth if may_edit_eyes else np.maximum(mouth, eyes)
    skin = (skin_mask[roi.slices] > 0.5) if skin_mask is not None else (oval > 0.5)
    skin = skin & (keep_out == 0.0) & (oval > 0.5)
    # Where the weight is exactly 0 inside what the stage may draw on: the openings, and the rim the runtime adds to them.
    kept = (weight == 0.0) & (allowed > 0.0)
    near_mouth = cv2.dilate((mouth > 0.0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    zones = Zones(roi, weight, kept & near_mouth, kept & ~near_mouth, allowed, skin)
    zones.rim = _mask_rim(hair_mask, skin_mask, roi, oval)
    zones.eyes_open = int(np.count_nonzero(zones.eyes_exact)) >= MIN_EYE_AREA * s * s
    if mouth_gap(landmarks) > OPEN_MOUTH:
        lip_line = landmarks[regions.LIPS_INNER]
        grown = regions.polygon_mask(regions.expand_polygon(lip_line, 0.04 * s), roi)
        shrunk = regions.polygon_mask(regions.expand_polygon(lip_line, -0.04 * s), roi)
        zones.mouth_band = (grown > 0.5) & (shrunk < 0.5)
        zones.mouth_inner = regions.polygon_mask(regions.expand_polygon(lip_line, -0.005 * s), roi) > 0.5
    return zones


def _mask_rim(hair_mask: np.ndarray | None, skin_mask: np.ndarray | None, roi: regions.Roi, oval: np.ndarray) -> np.ndarray | None:
    """Where a mask is unsure and the face oval is not there: what the stage draws here lands on the wall or the collar."""
    if hair_mask is None and skin_mask is None:
        return None
    strongest = np.zeros((roi.h, roi.w), np.float32)
    for mask in (hair_mask, skin_mask):
        if mask is not None:
            cv2.max(strongest, mask[roi.slices], dst=strongest)
    low, high = geometry.MASK_SURE
    return (strongest > 0.02) & (strongest < 0.5 * (low + high)) & (oval == 0.0)


def _edge_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def edge_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Correlation of the gradient magnitudes of two gray images inside a bool mask. NaN if too little."""
    if int(mask.sum()) < 16:
        return float("nan")
    x = _edge_magnitude(a)[mask].astype(np.float64)
    y = _edge_magnitude(b)[mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denominator < 1e-6:
        return 1.0 if float(np.abs(x - y).max()) < 1e-3 else 0.0
    return float((x * y).sum() / denominator)


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


_THIRDS = np.float32([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])


def _channel_mean(image: np.ndarray) -> np.ndarray:
    """The mean over the color channels of a float32 image. NumPy's ``mean(axis=2)`` takes ten times as long."""
    return cv2.transform(image, _THIRDS)


def _channel_max(image: np.ndarray) -> np.ndarray:
    """The largest color channel of a uint8 image."""
    a, b, c = cv2.split(image)
    return cv2.max(cv2.max(a, b), c)


def _edge_correlation_in(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """``edge_correlation`` of two BGR images, working only on the box around the mask."""
    if int(np.count_nonzero(mask)) < 16:
        return float("nan")
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    x0, y0 = max(x - 2, 0), max(y - 2, 0)
    box = slice(y0, y + h + 2), slice(x0, x + w + 2)
    return edge_correlation(_gray(a[box]), _gray(b[box]), mask[box])


def similarity_residual(a: np.ndarray, b: np.ndarray) -> float:
    """Mean distance between two point sets after the best similarity transform, in pixels of ``b``."""
    za = (a[:, 0] + 1j * a[:, 1]).astype(np.complex128)
    zb = (b[:, 0] + 1j * b[:, 1]).astype(np.complex128)
    za = za - za.mean()
    zb = zb - zb.mean()
    denominator = float((np.abs(za) ** 2).sum())
    if denominator < 1e-9:
        return float("nan")
    k = (np.conj(za) * zb).sum() / denominator
    return float(np.abs(k * za - zb).mean())


def mouth_gap(landmarks: np.ndarray) -> float:
    return float(np.linalg.norm(landmarks[regions.LIPS_INNER_TOP] - landmarks[regions.LIPS_INNER_BOTTOM])) / max(regions.face_scale(landmarks), 1.0)


def estimate_latency(reference: np.ndarray, output: np.ndarray, max_lag: int = 8) -> tuple[float, float]:
    """How many frames ``output`` lags behind ``reference``: (lag, correlation). NaN if there is no motion.

    Both are mouth-opening series. Correlates the changes, with a parabola through the peak for the fraction.
    """
    good = np.isfinite(reference) & np.isfinite(output)
    if good.sum() < 2 * max_lag + 8:
        return float("nan"), float("nan")
    a = np.interp(np.arange(len(reference)), np.flatnonzero(good), reference[good])
    b = np.interp(np.arange(len(output)), np.flatnonzero(good), output[good])
    a, b = np.gradient(a), np.gradient(b)
    a = a - a.mean()
    b = b - b.mean()
    if a.std() < 5e-4 or b.std() < 5e-4:
        return float("nan"), float("nan")
    scores = []
    for lag in range(-max_lag, max_lag + 1):
        x = a[max(0, -lag) : len(a) - max(0, lag)]
        y = b[max(0, lag) : len(b) - max(0, -lag)]
        scores.append(float((x * y).sum() / max(np.sqrt((x * x).sum() * (y * y).sum()), 1e-12)))
    scores_array = np.array(scores)
    index = int(np.argmax(scores_array))
    lag = float(index - max_lag)
    if 0 < index < len(scores) - 1:
        left, mid, right = scores_array[index - 1 : index + 2]
        curve = left - 2 * mid + right
        if curve < -1e-9:
            lag += 0.5 * (left - right) / curve
    return lag, float(scores_array[index])


# --------------------------------------------------------------------------
# Per-frame measures
# --------------------------------------------------------------------------


@dataclass
class FrameRef:
    """What is known about a frame before looking at the output under test."""

    index: int
    stage: np.ndarray | None  # landmarks drawn with (as reshape left them)
    raw_bare: np.ndarray | None  # unsmoothed landmarks of the input
    raw_ref: np.ndarray | None  # unsmoothed landmarks of the reference
    zones: Zones | None
    matrix: np.ndarray | None  # frame -> analysis crop
    crop_ref: np.ndarray | None  # float32 levels (0..255)
    crop_bare: np.ndarray | None
    inside: np.ndarray | None  # float32 mask in crop space: where the stage may draw
    structure_mask: np.ndarray | None  # bool in crop space: eyes, nose and mouth
    calm: np.ndarray | None  # float32 in crop space: ``inside``, without what moves by itself (eyes, lids, mouth)
    half_ref: np.ndarray  # uint8 half size
    half_bare: np.ndarray
    far: np.ndarray | None  # bool half size: the far background
    mouth_open: bool = False


def _crop_levels(frame: np.ndarray, landmarks: np.ndarray, size: int = ANALYSIS_CROP) -> np.ndarray:
    """``geometry.extract_crop`` on an even-aligned part of the frame (a third of the time), in levels (0..255)."""
    matrix = geometry.crop_matrix(landmarks, size)
    roi = geometry.crop_roi(matrix, size, frame.shape)
    if roi is None:
        return np.zeros((size, size, 3), np.float32)
    x0, y0 = (max(roi.x0 - 16, 0) // 16) * 16, (max(roi.y0 - 16, 0) // 16) * 16
    image = frame[y0 : roi.y1 + 16, x0 : roi.x1 + 16].astype(np.float32)
    matrix[:, 2] += matrix[:, :2] @ np.array([x0, y0], np.float64)
    reduction = geometry.SIDE_PER_FACE_SCALE * float(regions.face_scale(landmarks)) / size
    level = 0
    while reduction / 2**level > 1.5:
        image, level = cv2.pyrDown(image), level + 1
    matrix[:, :2] *= 2**level
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _half(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA)


def make_frame_ref(
    index: int,
    bare: np.ndarray,
    reference: np.ndarray,
    stage: np.ndarray | None,
    raw_bare: np.ndarray | None,
    raw_ref: np.ndarray | None,
    hair_mask: np.ndarray | None,
    skin_mask: np.ndarray | None,
    may_edit_eyes: bool = False,
) -> FrameRef:
    """Measure the reference side of one frame. Costs about 15 ms at 720p."""
    height, width = reference.shape[:2]
    half_ref, half_bare = _half(reference), _half(bare)
    if stage is None:
        return FrameRef(index, None, raw_bare, raw_ref, None, None, None, None, None, None, None, half_ref, half_bare, None)
    zones = build_zones(stage, hair_mask, skin_mask, width, height, may_edit_eyes)
    matrix = geometry.crop_matrix(stage, ANALYSIS_CROP)
    crop_ref = _crop_levels(reference, stage)
    crop_bare = _crop_levels(bare, stage)
    roi_matrix = matrix.copy()
    roi_matrix[:, 2] += matrix[:, :2] @ np.array([zones.roi.x0, zones.roi.y0], np.float64)
    inside = cv2.warpAffine(zones.weight, roi_matrix, (ANALYSIS_CROP, ANALYSIS_CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    inside = (inside > 0.5).astype(np.float32)
    structure = np.zeros((ANALYSIS_CROP, ANALYSIS_CROP), np.uint8)
    s = regions.face_scale(stage)
    hull = regions.expand_polygon(stage[_INNER_FACE], -0.2 * s).astype(np.float64)
    crop_points = hull @ matrix[:, :2].T + matrix[:, 2]
    cv2.fillPoly(structure, [np.round(crop_points).astype(np.int32)], 1)
    moving = np.zeros((ANALYSIS_CROP, ANALYSIS_CROP), np.uint8)
    for polygon in (
        regions.expand_polygon(stage[regions.side_indices("left").eye], 0.35 * s),
        regions.expand_polygon(stage[regions.side_indices("right").eye], 0.35 * s),
        regions.expand_polygon(stage[regions.LIPS_OUTER], 0.2 * s),
    ):
        cv2.fillPoly(moving, [np.round(polygon.astype(np.float64) @ matrix[:, :2].T + matrix[:, 2]).astype(np.int32)], 1)
    calm = inside * (1.0 - moving.astype(np.float32))
    # Far background: nothing the stage or a makeup effect has a reason to touch.
    near = np.zeros(half_ref.shape[:2], np.uint8)
    pad = max(int(0.35 * s / 2), 1)
    grown = cv2.dilate((zones.allowed_soft > 0.02).astype(np.uint8), np.ones((3, 3), np.uint8))
    small = cv2.resize(grown, (max(zones.roi.w // 2, 1), max(zones.roi.h // 2, 1)), interpolation=cv2.INTER_NEAREST)
    y0, x0 = zones.roi.y0 // 2, zones.roi.x0 // 2
    near[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small[: near.shape[0] - y0, : near.shape[1] - x0]
    if hair_mask is not None or skin_mask is not None:
        for mask in (hair_mask, skin_mask):
            if mask is not None:
                near = np.maximum(near, (cv2.resize(mask, (near.shape[1], near.shape[0]), interpolation=cv2.INTER_AREA) > 0.02).astype(np.uint8))
    near = cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)))
    far = (near == 0) & (_channel_max(half_bare) > 4)  # the black bars around a narrow clip do not count
    gap = mouth_gap(stage)
    return FrameRef(index, stage, raw_bare, raw_ref, zones, matrix, crop_ref, crop_bare, inside, structure > 0, calm, half_ref, half_bare, far, gap > OPEN_MOUTH)


def _block_max(values: np.ndarray, mask: np.ndarray | None, block: int) -> float:
    """The largest mean of ``values`` over a square block, with blocks mostly inside ``mask``."""
    if mask is None:
        return float(cv2.blur(values, (block, block)).max()) if values.size else 0.0
    weights = cv2.blur(mask, (block, block))
    means = cv2.blur(values * mask, (block, block))
    good = weights > 0.5
    return float((means[good] / weights[good]).max()) if good.any() else 0.0


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    total = float(mask.sum())
    return float((values * mask).sum() / total) if total > 0.0 else float("nan")


def _max_diff(diff: np.ndarray, mask: np.ndarray) -> float:
    return float(diff[mask].max()) if mask.any() else 0.0


def _lip_measures(bare: np.ndarray, output: np.ndarray, landmarks: np.ndarray) -> tuple[float, float]:
    """Lip color that landed beside the lips: its share (0..1), and its strength in levels.

    Color change just outside the lip line, against the same on the lips. Only what stands out from the surroundings
    counts, as a change all over the skin says nothing about the lips. Lip color on target has little beside it.
    """
    s = regions.face_scale(landmarks)
    roi = regions.roi_around(landmarks[regions.LIPS_OUTER], output.shape, pad=0.0, margin_px=0.25 * s)
    if roi is None or roi.w < 8 or roi.h < 8:
        return float("nan"), 0.0
    lab_o = cv2.cvtColor(output[roi.slices], cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(bare[roi.slices], cv2.COLOR_BGR2LAB).astype(np.float32)
    change = cv2.magnitude(lab_o[..., 1] - lab_b[..., 1], lab_o[..., 2] - lab_b[..., 2])
    outer = regions.polygon_mask(landmarks[regions.LIPS_OUTER], roi)
    inner = regions.polygon_mask(regions.expand_polygon(landmarks[regions.LIPS_INNER], 0.01 * s), roi)
    lips = outer * (1.0 - inner)
    if float(lips.sum()) < 16.0:
        return float("nan"), 0.0
    reach = max(int(round(0.08 * s)), 1)
    grown = cv2.dilate((outer > 0.0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1)))
    wider = cv2.dilate(grown, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1)))
    beside = (grown > 0) & (outer == 0.0)
    around = wider == 0
    if not beside.any() or not around.any():
        return float("nan"), 0.0
    change = np.maximum(change - float(np.median(change[around])), 0.0)
    on_lips = float((change * lips).sum() / lips.sum())
    off_lips = float(change[beside].mean())
    strength = max(on_lips, off_lips)
    if strength <= 0.0:
        return float("nan"), 0.0
    return off_lips / (on_lips + off_lips), strength


class LagEstimator:
    """How many frames the applied change lags behind the face, from image change and face motion.

    The change, seen in face-aligned coordinates, should hold still. If it sits where the face was ``k`` frames
    ago, it differs from its average by about ``k`` times (motion per frame) dot (its gradient). A least-squares fit
    of that gives ``k``, for a whole clip and for a window. Fast, no search, and meaningless without motion:
    the fit's denominator says how much motion and pattern there was.

    A small part that lags (a pasted patch) hardly moves the fit over the whole face, so the same fit also runs per box of
    ``LAG_LOCAL_BOX`` pixels, away from the eyes and the mouth, which move by themselves. The largest box counts.
    """

    _WARMUP = 15
    _SPAN = 2  # motion is measured over this many frames each way
    MIN_SPEED = 0.3  # crop pixels per frame (about 1.2 frame pixels at the median face): slower frames say nothing
    MIN_FRAMES = 25  # frames with motion a clip needs to get a value (with 8 to 15, clean clips reach 1.15)
    MIN_WINDOW_FRAMES = 8  # the same for one window of LAG_WINDOW frames

    def __init__(self) -> None:
        self._matrices: deque[np.ndarray | None] = deque(maxlen=2 * self._SPAN + 1)
        self._crops: deque[tuple[np.ndarray, ...] | None] = deque(maxlen=2 * self._SPAN + 1)
        self._frame_template: np.ndarray | None = None
        self._template: np.ndarray | None = None
        self._count = 0
        ys, xs = np.mgrid[0 : LAG_CROP, 0 : LAG_CROP].astype(np.float32)
        self._xs, self._ys = xs, ys
        self.numerator = 0.0
        self.denominator = 0.0
        self.moving_frames = 0
        self._window: deque[tuple[float, float]] = deque(maxlen=LAG_WINDOW)
        self.local_numerator = np.zeros((LAG_CROP, LAG_CROP), np.float64)
        self.local_denominator = np.zeros((LAG_CROP, LAG_CROP), np.float64)
        self.local_cover = np.zeros((LAG_CROP, LAG_CROP), np.float64)
        self.local_residual = np.zeros((LAG_CROP, LAG_CROP), np.float64)
        self.local_residual_frame = np.zeros((LAG_CROP, LAG_CROP), np.float64)
        self._local_window: deque[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = deque(maxlen=LAG_WINDOW)
        self.window_local = float("nan")  # the last window's largest box

    def update(self, delta_quarter: np.ndarray | None, raw: np.ndarray | None, inside: np.ndarray | None, calm: np.ndarray | None = None) -> tuple[float, float]:
        """Feed one frame; returns the window's (lag, denominator); the frame measured is ``_SPAN`` old."""
        self.window_local = float("nan")
        if delta_quarter is None or raw is None or inside is None:
            self._matrices.append(None)
            self._crops.append(None)
            return float("nan"), 0.0
        matrix = geometry.crop_matrix(raw, LAG_CROP)
        quarter = matrix.copy()
        quarter[:, 2] += 1.5 * quarter[:, :2].sum(axis=1)  # quarter pixel i is centered on frame pixel 4i + 1.5
        quarter[:, :2] *= 4.0
        crop = cv2.warpAffine(delta_quarter, quarter, (LAG_CROP, LAG_CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        small = cv2.resize(inside, (LAG_CROP, LAG_CROP), interpolation=cv2.INTER_AREA)
        small_calm = small if calm is None else cv2.resize(calm, (LAG_CROP, LAG_CROP), interpolation=cv2.INTER_AREA)
        self._matrices.append(matrix)
        self._crops.append((crop, small, small_calm, delta_quarter, quarter))
        return self._measure()

    def _measure(self) -> tuple[float, float]:
        if len(self._matrices) < self._matrices.maxlen or any(m is None for m in self._matrices):
            return float("nan"), 0.0
        center = self._crops[self._SPAN]
        if center is None:
            return float("nan"), 0.0
        crop, mask, calm, frame_delta, frame_matrix = center
        self._count += 1
        if self._template is None:
            self._template = crop.copy()
            self._frame_template = frame_delta.astype(np.float32).copy()
            return float("nan"), 0.0
        self._template += (crop - self._template) / self._count
        self._frame_template += (frame_delta - self._frame_template) / self._count
        if self._count < self._WARMUP:
            return float("nan"), 0.0
        before, after = self._matrices[0], self._matrices[-1]
        # Crop coordinates now -> crop coordinates of the same frame point a frame earlier (averaged over the span).
        back = _compose_affine(before, cv2.invertAffineTransform(after))
        ux = (back[0, 0] - 1.0) * self._xs + back[0, 1] * self._ys + back[0, 2]
        uy = back[1, 0] * self._xs + (back[1, 1] - 1.0) * self._ys + back[1, 2]
        ux, uy = ux / (2 * self._SPAN), uy / (2 * self._SPAN)
        if float(np.sqrt(np.mean(ux * ux + uy * uy))) < self.MIN_SPEED:
            self._window.append((0.0, 0.0))
            self._local_window.append(None)
            return float("nan"), 0.0
        self.moving_frames += 1
        gx = cv2.Sobel(self._template, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(self._template, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        predicted = ux[..., None] * gx + uy[..., None] * gy
        residual = crop - self._template
        weight = mask[..., None]
        product = residual * predicted
        power = predicted * predicted
        numerator = float((product * weight).sum())
        denominator = float((power * weight).sum())
        self.numerator += numerator
        self.denominator += denominator
        self._window.append((numerator, denominator))
        calm_weight = calm[..., None]
        local_numerator, local_denominator = (product * calm_weight).sum(axis=2), (power * calm_weight).sum(axis=2)
        self.local_numerator += local_numerator
        self.local_denominator += local_denominator
        self.local_cover += calm
        # Blurred first: fine detail in a change that follows the picture moves by itself. A patch is a broad thing.
        broad = cv2.blur(residual, (9, 9))
        self.local_residual += _channel_mean(broad * broad) * calm
        in_frame = cv2.blur(frame_delta - self._frame_template, (5, 5))
        self.local_residual_frame += cv2.warpAffine(_channel_mean(in_frame * in_frame), frame_matrix, (LAG_CROP, LAG_CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * calm
        self._local_window.append((local_numerator, local_denominator, calm))
        window_den = sum(d for _, d in self._window)
        moving = sum(1 for _, d in self._window if d > 0.0)
        if moving < self.MIN_WINDOW_FRAMES or window_den <= 1e-9:
            return float("nan"), 0.0
        kept = [item for item in self._local_window if item is not None]
        self.window_local = _largest_box_lag(sum(item[0] for item in kept), sum(item[1] for item in kept), sum(item[2] for item in kept), len(kept))
        return sum(n for n, _ in self._window) / window_den, window_den

    def clip_lag(self) -> tuple[float, float]:
        if self.denominator <= 1e-9 or self.moving_frames < self.MIN_FRAMES:
            return float("nan"), float(self.moving_frames)
        return self.numerator / self.denominator, float(self.moving_frames)

    def clip_frame_locked(self) -> float:
        """The largest box's change over the moving face, in face coordinates against frame coordinates: over 1 means it holds
        stiller on the frame than on the face. Paint holds still on the face. Something fixed to the frame sweeps through the face
        as it moves, and a soft patch has too little edge for the lag fit to see. A model's change may vary with the picture in
        both, so only the ratio tells. Boxes with next to no change do not count. NaN without motion.
        """
        if self.moving_frames < self.MIN_FRAMES:
            return float("nan")
        box = (LAG_LOCAL_BOX, LAG_LOCAL_BOX)
        cover = cv2.blur(self.local_cover, box, borderType=cv2.BORDER_CONSTANT)
        good = cover / self.moving_frames >= LAG_LOCAL_COVER
        if not good.any():
            return 0.0
        on_face = np.sqrt(cv2.blur(self.local_residual, box, borderType=cv2.BORDER_CONSTANT)[good] / cover[good])
        on_frame = np.sqrt(cv2.blur(self.local_residual_frame, box, borderType=cv2.BORDER_CONSTANT)[good] / cover[good])
        moving = on_face >= _FRAME_LOCK_FLOOR
        return float((on_face[moving] / np.maximum(on_frame[moving], 1e-3)).max()) if moving.any() else 0.0

    def clip_lag_local(self) -> float:
        """The largest box over the clip. 0 when no box has enough pattern to lag; NaN without motion."""
        if self.moving_frames < self.MIN_FRAMES:
            return float("nan")
        return _largest_box_lag(self.local_numerator, self.local_denominator, self.local_cover, self.moving_frames)


def _largest_box_lag(numerator: np.ndarray, denominator: np.ndarray, cover: np.ndarray, frames: int) -> float:
    box = (LAG_LOCAL_BOX, LAG_LOCAL_BOX)
    top = cv2.blur(numerator, box, borderType=cv2.BORDER_CONSTANT) / frames
    bottom = cv2.blur(denominator, box, borderType=cv2.BORDER_CONSTANT) / frames
    good = (bottom >= LAG_LOCAL_FLOOR) & (cv2.blur(cover, box, borderType=cv2.BORDER_CONSTANT) / frames >= LAG_LOCAL_COVER)
    return float((top[good] / bottom[good]).max()) if good.any() else 0.0


def _compose_affine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """``first`` after ``second``, both 2x3."""
    a = np.vstack([first, [0.0, 0.0, 1.0]])
    b = np.vstack([second, [0.0, 0.0, 1.0]])
    return (a @ b)[:2]


SERIES_KEYS = (
    "flick_in_out", "flick_in_ref", "flick_block_out", "flick_block_ref", "flick_bg_out", "flick_bg_ref", "delta_flick",
    "step_local", "impulse", "impulse_bg", "delta_p99", "delta_mean",
    "mouth_max", "eye_max", "outside_max", "band_corr", "inner_corr", "eye_corr",
    "far_mean", "far_block", "lm_shift", "geom", "structure", "structure_ref",
    "stage_offset", "stage_jump", "output_jump", "lip_offset", "lip_strength", "lag", "lag_den", "lag_local",
    "out_step", "bare_step", "lost", "gap_bare", "gap_out", "cosine", "cosine_ref", "paired_abs",
    "seam", "side_cast", "skin_cast", "warp_flow", "rim_change", "face_scale",
)


@dataclass
class ClipEvent:
    clip: str
    kind: str
    start: int
    end: int
    peak: float
    crops: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"clip": self.clip, "kind": self.kind, "label": EVENT_LABELS.get(self.kind, self.kind), "start_frame": self.start, "end_frame": self.end, "peak": _rounded(self.peak), "crops": self.crops}


@dataclass
class ClipResult:
    name: str
    frames: int
    measured_frames: int
    fps: float
    metrics: dict[str, float]
    events: list[ClipEvent]
    series: dict[str, np.ndarray]


class EventRecorder:
    """Turns per-frame findings into merged events, and saves what they looked like."""

    def __init__(self, clip: str, crops_dir: Path | None) -> None:
        self.clip = clip
        self.crops_dir = crops_dir
        self.events: list[ClipEvent] = []
        self._open: dict[str, ClipEvent] = {}
        self._crops: dict[str, int] = {}

    def add(self, kind: str, frame: int, peak: float, images: Callable[[], tuple[np.ndarray, ...]] | None = None) -> None:
        event = self._open.get(kind)
        if event is not None and frame - event.end <= EVENT_MERGE_GAP:
            event.end = frame
            event.peak = max(event.peak, peak) if np.isfinite(peak) else event.peak
            return
        event = ClipEvent(self.clip, kind, frame, frame, peak)
        self.events.append(event)
        self._open[kind] = event
        if self.crops_dir is not None and images is not None and self._crops.get(kind, 0) < MAX_CROPS_PER_KIND:
            self._crops[kind] = self._crops.get(kind, 0) + 1
            path = self.crops_dir / f"{self.clip}" / f"{kind}_{frame:06d}.png"
            if _save_triptych(path, images()):
                event.crops.append(str(Path("events") / self.clip / path.name).replace(os.sep, "/"))


def _save_triptych(path: Path, images: tuple[np.ndarray, ...]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, data = cv2.imencode(".png", np.hstack(images))  # imwrite cannot take every path on Windows
        if ok:
            path.write_bytes(data.tobytes())
        return bool(ok)
    except (OSError, cv2.error, ValueError) as exc:
        logger.warning("could not save a crop: %s", exc)
        return False


def event_images(bare: np.ndarray, reference: np.ndarray, output: np.ndarray, landmarks: np.ndarray | None, side: int = 320) -> tuple[np.ndarray, ...]:
    """Input, reference, output, and the change x 8, around the face."""
    height, width = output.shape[:2]
    if landmarks is None:
        roi = regions.Roi(0, 0, width, height)
    else:
        roi = crop_roi(landmarks, width, height)
    diff = np.clip(cv2.absdiff(output[roi.slices], reference[roi.slices]).astype(np.float32) * 8.0, 0, 255).astype(np.uint8)
    parts = (bare[roi.slices], reference[roi.slices], output[roi.slices], diff)
    k = side / max(roi.h, 1)
    size = (max(round(roi.w * k), 1), side)
    return tuple(cv2.resize(part, size, interpolation=cv2.INTER_AREA) for part in parts)


class FindingState:
    """What ``frame_findings`` remembers from frame to frame."""

    def __init__(self) -> None:
        self.impulses: deque[float] = deque(maxlen=SPIKE_MEDIAN_FRAMES)
        self.impulses_bg: deque[float] = deque(maxlen=SPIKE_MEDIAN_FRAMES)
        self.held: dict[str, int] = {}
        self.lost_run = 0
        self.frozen_run = 0

    def held_for(self, key: str, active: bool) -> bool:
        self.held[key] = self.held.get(key, 0) + 1 if active else 0
        return self.held[key] >= HELD_FRAMES


def frame_findings(sample: dict[str, float], state: FindingState, rules: dict[str, float], mode: str) -> list[tuple[str, float]]:
    """The breakdowns one frame shows: (kind, peak). A pure function of the frame's measures and the state, so
    that stored series can be replayed with other rules. A comparison with NaN (not measured) finds nothing."""
    found: list[tuple[str, float]] = []
    for key, history in (("impulse", state.impulses), ("impulse_bg", state.impulses_bg)):
        step = sample[key]
        if np.isfinite(step):
            if len(history) >= 8 and step > rules["spike_floor"] and step > rules["spike_factor"] * statistics.median(history):
                found.append(("spike", step))
            history.append(step)
    far_block, far_mean = sample["far_block"], sample["far_mean"]
    if far_block > rules["background_block"] or far_mean > rules["background_mean"]:
        found.append(("background_changed", far_block))
    exact = mode == MODE_CLASSIC
    if exact and sample["outside_max"] > 0.0:
        found.append(("background_changed", sample["outside_max"]))
    if (exact and sample["mouth_max"] > 0.0) or sample["inner_corr"] < rules["teeth_edge_corr" if exact else "input_teeth_edge_corr"]:
        found.append(("teeth_altered", sample["mouth_max"] if exact else sample["inner_corr"]))
    if (exact and sample["eye_max"] > 0.0) or sample["eye_corr"] < rules["eye_edge_corr" if exact else "input_eye_edge_corr"]:
        found.append(("eyes_altered", sample["eye_max"] if exact else sample["eye_corr"]))
    if sample["stage_jump"] > rules["stage_jump"]:
        found.append(("landmark_jump", sample["stage_jump"]))
    if sample["output_jump"] > rules["output_jump"]:
        found.append(("landmark_jump", sample["output_jump"]))
    if state.held_for("stage_offset", sample["stage_offset"] > rules["stage_offset"]):
        found.append(("off_target", sample["stage_offset"]))
    painted = sample["lip_strength"] >= rules["lip_min_strength"]
    if state.held_for("lip_offset", painted and sample["lip_offset"] > rules["lip_offset"]):
        found.append(("off_target", sample["lip_offset"]))
    if sample["lag"] > rules["lag_frames"]:
        found.append(("paint_behind", sample["lag"]))
    # Series stored before this measure existed have no such key.
    if mode == MODE_CLASSIC and sample.get("lag_local", float("nan")) > rules["lag_local_frames"]:
        found.append(("paint_behind", sample["lag_local"]))
    state.lost_run = state.lost_run + 1 if sample["lost"] == 1.0 else 0
    if state.lost_run >= 2:
        found.append(("face_lost", float(state.lost_run)))
    frozen = sample["out_step"] < rules["frozen_output"] and sample["bare_step"] > rules["frozen_input"]
    state.frozen_run = state.frozen_run + 1 if frozen else 0
    if state.frozen_run >= HELD_FRAMES:
        found.append(("frozen", float(state.frozen_run)))
    changed = sample["lm_shift"] > rules["identity_landmarks"] or sample["geom"] > rules["identity_landmarks"] or sample["structure"] < rules["identity_structure"]
    if state.held_for("identity", changed):
        found.append(("identity_shift", sample["lm_shift"]))
    return found


class ClipAnalyzer:
    """Measures one output video, frame by frame, against its ``FrameRef`` s. Holds about 3 frames in memory."""

    def __init__(
        self,
        name: str,
        fps: float,
        mode: str,
        track_output: Callable[[np.ndarray, int], np.ndarray | None] | None = None,
        crops_dir: Path | None = None,
        rules: dict[str, float] | None = None,
        embedder: Callable[..., tuple[float, float]] | None = None,
        settle_frames: int = SETTLE_FRAMES,
    ) -> None:
        self.name = name
        self.fps = fps
        self.mode = mode
        self._track = track_output
        self._rules = EVENT_RULES if rules is None else rules
        self._embedder = embedder
        self._settle = settle_frames
        self._recorder = EventRecorder(name, crops_dir)
        self._series: dict[str, list[float]] = {key: [] for key in SERIES_KEYS}
        self._frames = 0
        self._crops: deque[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = deque(maxlen=3)  # output, reference, inside, change, calm
        self._halves: deque[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = deque(maxlen=3)
        self._last_delta_crop: np.ndarray | None = None
        self._last_half_out: np.ndarray | None = None
        self._last_half_bare: np.ndarray | None = None
        self._stage_vs_raw: np.ndarray | None = None
        self._out_vs_ref: np.ndarray | None = None
        self._findings = FindingState()
        self._lag = LagEstimator()
        self._flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    # ---- feeding -----------------------------------------------------------------------------

    def feed(self, ref: FrameRef, bare: np.ndarray, reference: np.ndarray, output: np.ndarray, paired: np.ndarray | None = None) -> dict[str, float]:
        sample: dict[str, float] = dict.fromkeys(SERIES_KEYS, float("nan"))
        index = ref.index
        self._frames += 1
        half_out = _half(output)
        raw_out = self._track(output, index) if self._track is not None else None
        self._measure_outside_flicker(ref, half_out, sample)
        self._measure_steps(ref, half_out, sample)
        if ref.stage is not None:
            self._measure_face(ref, bare, reference, output, raw_out, sample)
        else:
            self._crops.append(None)
            self._last_delta_crop = None
            self._stage_vs_raw = None
            self._out_vs_ref = None
            self._lag.update(None, None, None)
        self._measure_tracks(ref, bare, reference, output, raw_out, sample)
        if paired is not None:
            sample["paired_abs"] = float(cv2.absdiff(half_out, _half(paired)).mean())
        for key, value in sample.items():
            self._series[key].append(value)
        if index >= self._settle:
            self._find_events(ref, bare, reference, output, sample)
        return sample

    def _measure_outside_flicker(self, ref: FrameRef, half_out: np.ndarray, sample: dict[str, float]) -> None:
        self._halves.append((half_out, ref.half_ref, ref.far))
        if len(self._halves) == 3 and self._halves[1][2] is not None:
            far = self._halves[1][2]
            if far.any():
                out = [h[0].astype(np.float32) for h in self._halves]
                rf = [h[1].astype(np.float32) for h in self._halves]
                second_out = _channel_mean(cv2.absdiff(out[2] + out[0], 2.0 * out[1]))
                second_ref = _channel_mean(cv2.absdiff(rf[2] + rf[0], 2.0 * rf[1]))
                sample["flick_bg_out"] = float(cv2.mean(second_out, far.astype(np.uint8))[0])
                sample["flick_bg_ref"] = float(cv2.mean(second_ref, far.astype(np.uint8))[0])
                impulse = _channel_mean(cv2.absdiff(out[2] + out[0] - 2.0 * out[1], rf[2] + rf[0] - 2.0 * rf[1]))
                sample["impulse_bg"] = _block_max(impulse, far.astype(np.float32), 16)

    def _measure_steps(self, ref: FrameRef, half_out: np.ndarray, sample: dict[str, float]) -> None:
        if ref.far is not None and ref.far.any():
            mask = ref.far.astype(np.float32)
            delta = _channel_mean(cv2.absdiff(half_out, ref.half_ref).astype(np.float32))
            sample["far_mean"] = _masked_mean(delta, mask)
            sample["far_block"] = _block_max(delta, mask, 16)
        if self._last_half_out is not None:
            sample["out_step"] = float(np.mean(cv2.absdiff(half_out, self._last_half_out)))
            sample["bare_step"] = float(np.mean(cv2.absdiff(ref.half_bare, self._last_half_bare)))
        self._last_half_out = half_out
        self._last_half_bare = ref.half_bare

    def _measure_face(self, ref: FrameRef, bare: np.ndarray, reference: np.ndarray, output: np.ndarray, raw_out: np.ndarray | None, sample: dict[str, float]) -> None:
        zones = ref.zones
        crop_out = _crop_levels(output, ref.stage)
        delta = crop_out - ref.crop_ref
        self._crops.append((crop_out, ref.crop_ref, ref.inside, delta, ref.calm))
        if len(self._crops) == 3 and all(c is not None for c in self._crops):
            mask = self._crops[1][2]
            sample["flick_in_out"], sample["flick_block_out"] = _second_difference_crop([c[0] for c in self._crops], mask)
            sample["flick_in_ref"], sample["flick_block_ref"] = _second_difference_crop([c[1] for c in self._crops], mask)
            second = _channel_mean(cv2.absdiff(self._crops[2][3] + self._crops[0][3], 2.0 * self._crops[1][3]))
            sample["delta_flick"] = _masked_mean(second, mask)
            # A model's change follows the picture: at a blink the picture jumps by 26 levels over a block, and a change
            # that follows it by 3 is no break (the contract: "while classic shows none"). So what the reference itself did in
            # that place takes its share off first. Against the bare input makeup moves with every blink and word in any case.
            reference_second = _channel_mean(cv2.absdiff(self._crops[2][1] + self._crops[0][1], 2.0 * self._crops[1][1]))
            if self.mode == MODE_CLASSIC:
                sample["impulse"] = _block_max(cv2.subtract(second, SPIKE_REFERENCE_SHARE * reference_second), mask, 16)
            else:
                sample["impulse"] = _block_max(second, self._crops[1][4], 16)
        sample["delta_mean"] = _masked_mean(_channel_mean(np.abs(delta)), ref.inside)
        sample["face_scale"] = float(regions.face_scale(ref.stage))
        if self.mode == MODE_CLASSIC:
            self._measure_spatial(ref, crop_out, delta, sample)
            if zones.rim is not None and zones.rim.any():
                sample["rim_change"] = float(np.mean(_channel_max(cv2.absdiff(output[zones.roi.slices], reference[zones.roi.slices]))[zones.rim]))
        if self._last_delta_crop is not None:
            sample["step_local"] = _block_max(_channel_mean(cv2.absdiff(delta, self._last_delta_crop)), ref.inside, 16)
        self._last_delta_crop = delta
        # Exact checks, and what they miss: edges.
        roi = zones.roi
        o_roi, r_roi = output[roi.slices], reference[roi.slices]
        diff = _channel_max(cv2.absdiff(o_roi, r_roi))
        sample["mouth_max"] = _max_diff(diff, zones.mouth_exact)
        sample["eye_max"] = _max_diff(diff, zones.eyes_exact)
        outside_max = _max_diff(diff, zones.allowed_soft == 0.0)
        if self.mode == MODE_CLASSIC:  # outside the rectangle the stage cannot draw at all
            whole = _channel_max(cv2.absdiff(output, reference))
            whole[roi.slices] = 0
            outside_max = max(outside_max, float(whole.max()))
        sample["outside_max"] = outside_max
        if zones.skin.any():
            sample["delta_p99"] = float(np.percentile(cv2.absdiff(o_roi, r_roi)[zones.skin], 99)) / 255.0
        if zones.mouth_band is not None:
            sample["band_corr"] = _edge_correlation_in(o_roi, r_roi, zones.mouth_band)
            sample["inner_corr"] = _edge_correlation_in(o_roi, r_roi, zones.mouth_inner)
        if zones.eyes_open:
            sample["eye_corr"] = _edge_correlation_in(o_roi, r_roi, zones.eyes_exact)
        # Structure against the bare frame: same pose, so what differs in the edges was changed.
        gray_bare = _gray_float(ref.crop_bare)
        sample["structure"] = edge_correlation(_gray_float(crop_out), gray_bare, ref.structure_mask)
        sample["structure_ref"] = edge_correlation(_gray_float(ref.crop_ref), gray_bare, ref.structure_mask)
        if raw_out is not None:
            sample["lip_offset"], sample["lip_strength"] = _lip_measures(bare, output, raw_out)
        quarter = None
        if output.shape[:2] == (FRAME_SIZE[1], FRAME_SIZE[0]):
            half_change = _half(output).astype(np.float32) - ref.half_ref.astype(np.float32)
            quarter = cv2.resize(half_change, (FRAME_SIZE[0] // 4, FRAME_SIZE[1] // 4), interpolation=cv2.INTER_AREA)
        sample["lag"], sample["lag_den"] = self._lag.update(quarter, ref.raw_bare, ref.inside, ref.calm)
        sample["lag_local"] = self._lag.window_local

    def _measure_spatial(self, ref: FrameRef, crop_out: np.ndarray, delta: np.ndarray, sample: dict[str, float]) -> None:
        """What holds still and so never shows in the temporal measures: a seam, a grid, a cast, content that was moved."""
        calm = cv2.erode((ref.calm > 0.5).astype(np.uint8), _SPATIAL_ERODE)
        if int(np.count_nonzero(calm)) < 64:
            return
        gray_ref = _gray_float(ref.crop_ref)
        sample["seam"] = seam_edge(delta, gray_ref, calm)
        left, right = cv2.mean(delta[:, : ANALYSIS_CROP // 2], calm[:, : ANALYSIS_CROP // 2]), cv2.mean(delta[:, ANALYSIS_CROP // 2 :], calm[:, ANALYSIS_CROP // 2 :])
        whole = cv2.mean(delta, calm)
        sample["side_cast"] = max(abs(left[k] - right[k]) for k in range(3))
        sample["skin_cast"] = max(abs(whole[k]) for k in range(3))
        if ref.index % WARP_FLOW_EVERY == 0:
            sample["warp_flow"] = warp_flow(self._flow, gray_ref, _gray_float(crop_out), calm)

    def _measure_tracks(self, ref: FrameRef, bare: np.ndarray, reference: np.ndarray, output: np.ndarray, raw_out: np.ndarray | None, sample: dict[str, float]) -> None:
        raw_bare, raw_ref, stage = ref.raw_bare, ref.raw_ref, ref.stage
        if raw_bare is not None:
            sample["gap_bare"] = mouth_gap(raw_bare)
        if raw_out is not None:
            sample["gap_out"] = mouth_gap(raw_out)
        if self._track is not None:
            sample["lost"] = 1.0 if raw_ref is not None and raw_out is None else 0.0
        if stage is not None and raw_bare is not None:
            s = max(regions.face_scale(raw_bare), 1.0)
            offset = np.median(stage[_RIGID] - raw_bare[_RIGID], axis=0)
            sample["stage_offset"] = float(np.linalg.norm(offset)) / s
            if self._stage_vs_raw is not None:
                sample["stage_jump"] = float(np.linalg.norm(offset - self._stage_vs_raw)) / s
            self._stage_vs_raw = offset
        else:
            self._stage_vs_raw = None
        if raw_out is not None and raw_ref is not None:
            s = max(regions.face_scale(raw_ref), 1.0)
            sample["lm_shift"] = float(np.median(np.linalg.norm(raw_out - raw_ref, axis=1))) / s
            sample["geom"] = similarity_residual(raw_out[_IDENTITY], raw_ref[_IDENTITY]) / s
            offset = np.median(raw_out[_RIGID] - raw_ref[_RIGID], axis=0)
            if self._out_vs_ref is not None:
                sample["output_jump"] = float(np.linalg.norm(offset - self._out_vs_ref)) / s
            self._out_vs_ref = offset
        else:
            self._out_vs_ref = None
        if self._embedder is not None and raw_out is not None and raw_bare is not None:
            try:
                sample["cosine"], sample["cosine_ref"] = self._embedder(ref.index, bare, raw_bare, reference, raw_ref, output, raw_out)
            except Exception as exc:  # an optional model the user supplied must not stop the evaluation
                logger.warning("the identity model failed: %s", exc)

    def break_continuity(self) -> None:
        """The next frame does not follow the last one (a dropped frame, a cut): nothing is compared across it."""
        self._crops.clear()
        self._halves.clear()
        self._last_delta_crop = None
        self._last_half_out = None
        self._last_half_bare = None
        self._stage_vs_raw = None
        self._out_vs_ref = None

    # ---- events ------------------------------------------------------------------------------

    def _find_events(self, ref: FrameRef, bare: np.ndarray, reference: np.ndarray, output: np.ndarray, sample: dict[str, float]) -> None:
        for kind, peak in frame_findings(sample, self._findings, self._rules, self.mode):
            self._recorder.add(kind, ref.index, peak, lambda: event_images(bare, reference, output, ref.stage))

    # ---- the end of the clip -----------------------------------------------------------------

    def finish(self) -> ClipResult:
        series = {key: np.array(values, np.float64) for key, values in self._series.items()}
        metrics = clip_metrics(series, self._settle, self._lag.clip_lag(), self.mode, self._lag.clip_lag_local())
        metrics["frame_locked_ratio"] = self._lag.clip_frame_locked()
        measured = int(np.isfinite(series["flick_in_out"][self._settle :]).sum()) if self._frames > self._settle else 0
        return ClipResult(self.name, self._frames, measured, self.fps, metrics, self._recorder.events, series)


def _gray_float(crop: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(crop, 0.0, 255.0), cv2.COLOR_BGR2GRAY)


_SPATIAL_ERODE = np.ones((7, 7), np.uint8)  # the weight's own fall at its rim is not a seam
WARP_FLOW_EVERY = 3  # frames: the flow takes a few ms
_SEAM_REFERENCE_SHARE = 0.6  # a change may follow the picture's own edges by this much
_SEAM_PERCENTILE = 99.5  # one line across a face is about 1 % of the area
_FLOW_MIN_EDGE = 1.5  # levels per pixel: flat skin has nothing to tell a move from a change of color
_FLOW_PERCENTILE = 95.0


def seam_edge(delta: np.ndarray, gray_reference: np.ndarray, mask: np.ndarray) -> float:
    """Levels per pixel: the sharpest edges of the change, in any color channel, where the picture itself has no edge."""
    smooth = cv2.blur(delta, (3, 3))
    edges = cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    a, b, c = cv2.split(edges)
    strongest = cv2.max(cv2.max(a, b), c)
    excess = strongest - _SEAM_REFERENCE_SHARE * _edge_magnitude(cv2.blur(gray_reference, (3, 3))) / 8.0
    return float(np.percentile(excess[mask > 0], _SEAM_PERCENTILE))


def warp_flow(flow: Any, gray_reference: np.ndarray, gray_output: np.ndarray, mask: np.ndarray) -> float:
    """Crop pixels: how far content moved between the reference and the output, where the picture has edges to follow.

    A delta that only changes color moves nothing. Flow of 1 crop pixel is about 2 frame pixels at the median face.
    """
    a = np.clip(gray_reference, 0.0, 255.0).astype(np.uint8)
    b = np.clip(gray_output, 0.0, 255.0).astype(np.uint8)
    field = flow.calc(a, b, None)
    magnitude = cv2.magnitude(field[..., 0], field[..., 1])
    textured = (_edge_magnitude(gray_reference) / 8.0 > _FLOW_MIN_EDGE) & (mask > 0)
    return float(np.percentile(magnitude[textured], _FLOW_PERCENTILE)) if int(np.count_nonzero(textured)) >= 32 else float("nan")


def _second_difference_crop(crops: Sequence[np.ndarray], mask: np.ndarray) -> tuple[float, float]:
    """The second difference at pixel level, and over blocks of 16 crop pixels (about 30 frame pixels)."""
    second = crops[2] - 2.0 * crops[1] + crops[0]
    fine = _masked_mean(_channel_mean(np.abs(second)), mask)
    weights = cv2.blur(mask, (16, 16))
    blocks = cv2.blur(second * mask[..., None], (16, 16))
    good = weights > 0.5
    coarse = float((np.abs(blocks[good]) / weights[good][..., None]).mean()) if good.any() else float("nan")
    return fine, coarse


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    good = np.isfinite(numerator) & np.isfinite(denominator)
    if not good.any():
        return float("nan")
    bottom = float(denominator[good].mean())
    top = float(numerator[good].mean())
    if bottom < 1e-6:
        return 1.0 if top < 1e-6 else float("inf")
    return top / bottom


def clip_metrics(series: dict[str, np.ndarray], settle: int, lag: tuple[float, float], mode: str, lag_local: float = float("nan")) -> dict[str, float]:
    """One clip's metrics from its per-frame series."""
    s = {key: values[settle:] for key, values in series.items()}
    out: dict[str, float] = {}
    out["flicker_ratio"] = _ratio(s["flick_in_out"], s["flick_in_ref"])
    out["flicker_block_ratio"] = _ratio(s["flick_block_out"], s["flick_block_ref"])
    out["flicker_outside_ratio"] = _ratio(s["flick_bg_out"], s["flick_bg_ref"])
    out["delta_flicker"] = _mean(s["delta_flick"])
    for key, name in (("mouth_interior_max", "mouth_max"), ("eye_interior_max", "eye_max"), ("outside_max", "outside_max")):
        out[key] = _max(s[name])
    out["mouth_band_edge_corr"] = _percentile(s["band_corr"], 5)
    out["mouth_inner_edge_corr"] = _percentile(s["inner_corr"], 5)
    out["eye_edge_corr"] = _percentile(s["eye_corr"], 5)
    out["background_change"] = _percentile(s["far_mean"], 99)
    out["background_block_max"] = _max(s["far_block"])
    # A 3-frame median first: the landmark model has single-frame glitches that the picture does not.
    out["landmark_shift_p99"] = _percentile(_median_of_three(s["lm_shift"]), 99)
    out["identity_geometry_p99"] = _percentile(_median_of_three(s["geom"]), 99)
    out["identity_structure_p01"] = _percentile(s["structure"], 1)
    out["identity_structure_ref_p01"] = _percentile(s["structure_ref"], 1)
    out["delta_abs_p99"] = _percentile(s["delta_p99"], 95)
    for key, name in (("seam_edge", "seam"), ("side_cast", "side_cast"), ("skin_cast", "skin_cast"), ("warp_flow_px", "warp_flow"), ("rim_change", "rim_change")):
        out[key] = _percentile(s[name], 50)  # what holds for most of the clip: a stable defect shows at the median
    out["face_scale_median"] = _percentile(s["face_scale"], 50)
    latency, correlation = estimate_latency(series["gap_bare"], series["gap_out"])
    out["mouth_latency_frames"] = abs(latency) if np.isfinite(latency) else float("nan")
    out["mouth_latency_signed"] = latency
    out["mouth_latency_corr"] = correlation
    out["paint_lag_frames"] = lag[0]
    out["paint_lag_motion"] = lag[1]
    out["paint_lag_local_frames"] = lag_local
    painted = s["lip_strength"] >= EVENT_RULES["lip_min_strength"]
    out["paint_offset"] = _percentile(s["lip_offset"][painted], 50) if painted.any() else float("nan")
    out["identity_cosine_p01"] = _percentile(s["cosine"], 1)
    out["identity_cosine_ref_p01"] = _percentile(s["cosine_ref"], 1)
    out["paired_abs"] = _mean(s["paired_abs"])
    out["lost_frames"] = float(np.nansum(s["lost"])) if len(s["lost"]) else 0.0
    return out


def _median_of_three(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values
    stacked = np.vstack([values[:-2], values[1:-1], values[2:]])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all NaN: not measured
        return np.nanmedian(stacked, axis=0)


def _mean(values: np.ndarray) -> float:
    good = _finite(values)
    return float(good.mean()) if good.size else float("nan")


def _max(values: np.ndarray) -> float:
    good = _finite(values)
    return float(good.max()) if good.size else float("nan")


def _percentile(values: np.ndarray, q: float) -> float:
    good = _finite(values)
    return float(np.percentile(good, q)) if good.size else float("nan")


def _rounded(value: float, digits: int = 5) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


# --------------------------------------------------------------------------
# The pipeline as a library
# --------------------------------------------------------------------------


class StageProbe(Effect):
    """Sits just before the ``neural`` step and only looks: the classic output and what the stage is given.

    Named after the ``general`` section because a section must exist and that one can never be switched off.
    It writes nothing into the frame, so a run with the probe draws the same pixels as one without it.
    """

    name = "general"
    order = 89
    needs_face = False

    def __init__(self) -> None:
        self.classic: np.ndarray | None = None
        self.landmarks: np.ndarray | None = None
        self.hair_mask: np.ndarray | None = None
        self.face_skin_mask: np.ndarray | None = None

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: Any) -> np.ndarray:
        # The same conversion as the pipeline's last two steps.
        self.classic = cv2.convertScaleAbs(cv2.max(frame, 0.0), alpha=255.0)
        self.landmarks = ctx.landmarks
        self.hair_mask = ctx.hair_mask
        self.face_skin_mask = ctx.face_skin_mask
        return frame

    def reset(self) -> None:
        self.classic = None
        self.landmarks = None


def builtin_effects(extra: Sequence[Effect] = (), without_neural: bool = False, neural: Effect | None = None) -> list[Effect]:
    """The registered effects, with the ones given. ``neural`` replaces the registered finishing stage."""
    from .effects import load_builtin_effects

    effects: list[Effect] = []
    for effect_cls in load_builtin_effects():
        if effect_cls.name == "neural":
            if neural is None and not without_neural:
                effects.append(effect_cls())
            continue
        effects.append(effect_cls())
    if neural is not None:
        effects.append(neural)
    effects.extend(extra)
    return effects


def default_tracker_factory(smoothing: float = 0.0, model_dir: Path | None = None) -> Any:
    from .models import FACE_LANDMARKER, ensure_model
    from .tracking import FaceTracker

    return FaceTracker(ensure_model(FACE_LANDMARKER, directory=model_dir), smoothing)


def default_segmenter(interval: int, model_dir: Path | None = None, background: bool = False) -> Any:
    from .models import SELFIE_MULTICLASS, ensure_model
    from .segmentation import Segmenter

    return Segmenter(ensure_model(SELFIE_MULTICLASS, directory=model_dir), interval, background=background)


def file_settings(settings: MakeupSettings) -> MakeupSettings:
    """The frames are compared unmirrored; mirroring happens after everything measured here."""
    copy = settings.copy()
    copy.general.mirror_output = False
    copy.general.mirror_preview = False
    copy.general.show_landmarks = False
    return copy


@dataclass
class Tools:
    """What the evaluation builds on. Tests replace these with fakes; the defaults load the real models."""

    tracker_factory: Callable[[float], Any] = default_tracker_factory
    segmenter_factory: Callable[..., Any] = default_segmenter
    backend_factory: Callable[[Any], Any] | None = None  # None: the real worker process
    embedder: Any | None = None
    # On Windows ``time.monotonic`` moves in steps of about 16 ms, which round-trip times cannot be read from.
    now: Callable[[], float] = time.perf_counter


class SFaceEmbedder:
    """An optional face-recognition model in OpenCV's ``FaceRecognizerSF`` format, supplied by the user. Not bundled."""

    def __init__(self, path: Path) -> None:
        self._model = cv2.FaceRecognizerSF.create(str(path), "")

    def embed(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
        # Eyes, nose tip and mouth corners from the landmarks: the five points this model aligns with.
        right, left = regions.eye_center(landmarks, "right"), regions.eye_center(landmarks, "left")
        points = [right, left, landmarks[regions.NOSE_TIP], landmarks[regions.MOUTH_RIGHT_CORNER], landmarks[regions.MOUTH_LEFT_CORNER]]
        x, y = landmarks.min(axis=0)
        w, h = landmarks.max(axis=0) - landmarks.min(axis=0)
        row = np.array([[x, y, w, h, *np.array(points, np.float32).ravel(), 1.0]], np.float32)
        aligned = self._model.alignCrop(frame, row)
        feature = self._model.feature(aligned).ravel().astype(np.float64)
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm > 0.0 else None


def wait_for_stage(stage: Any, timeout_s: float = 2.0) -> None:
    """Between frames only, never inside one: the next frame is then drawn with the delta of this one (age 1)."""
    backend = stage.backend
    if backend is None:
        return
    deadline = time.perf_counter() + timeout_s
    while backend.busy and time.perf_counter() < deadline:
        time.sleep(0.0005)


def wait_until_serving(stage: Any, timeout_s: float = 30.0) -> None:
    """After the first frame with a face. Raises ``EvaluationError`` if the stage stayed off."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        info = stage.stats()
        if info["phase"] == "running":
            return
        if info["phase"] in ("off", "failed", "closed"):
            logger.warning("the stage stayed off: %s", info["refused_because"])
            raise EvaluationError(f"学習モデルの仕上げが動きません: {info['status']}")
        time.sleep(0.01)
    raise EvaluationError("学習モデルの仕上げの準備が 30 秒で終わりませんでした")


def build_stage(model_dir: Path, tools: Tools, clock: Callable[[], float] | None = None, keep_frame_rate: bool = True) -> Any:
    from .neural.stage import NeuralFinishEffect

    return NeuralFinishEffect(model_dir=model_dir, allow_unevaluated=True, backend_factory=tools.backend_factory, clock=clock or tools.now, keep_frame_rate=keep_frame_rate)


def _build_pipeline(settings: MakeupSettings, tools: Tools, probe: Effect | None, stage: Any | None, background_masks: bool = False) -> Any:
    from .pipeline import MakeupPipeline

    tracker = tools.tracker_factory(settings.general.landmark_smoothing)
    try:
        segmenter = tools.segmenter_factory(settings.general.segmentation_interval, None, background_masks)
    except BaseException:
        tracker.close()
        raise
    extra = [] if probe is None else [probe]
    return MakeupPipeline(settings, tracker=tracker, segmenter=segmenter, effects=builtin_effects(extra, without_neural=stage is None, neural=stage))


# --------------------------------------------------------------------------
# Quality pass: every frame, in order, measured
# --------------------------------------------------------------------------


@dataclass
class RunOptions:
    settings: MakeupSettings
    model_dir: Path | None = None
    may_edit_eyes: bool = False
    out_dir: Path | None = None
    frames_per_clip: int | None = None
    strength: float = 1.0
    realtime_rounds: int = REALTIME_ROUNDS


def _with_model(settings: MakeupSettings, strength: float) -> MakeupSettings:
    copy = file_settings(settings)
    copy.neural.enabled = True
    copy.neural.strength = strength
    return copy


def _without_model(settings: MakeupSettings) -> MakeupSettings:
    copy = file_settings(settings)
    copy.neural.enabled = False
    return copy


def sway_frames(frames: Iterable[np.ndarray], amplitude: float = 30.0, period: float = 40.0) -> Iterator[np.ndarray]:
    """The same clip with the head (and the room) swaying: about 4.7 px per frame at the fastest. Paint that lags
    behind the face only shows with motion, and the test clips hardly move."""
    for index, frame in enumerate(frames):
        dx = amplitude * np.sin(2 * np.pi * index / period)
        dy = 0.3 * amplitude * np.sin(4 * np.pi * index / period)
        yield cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]), (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def nod_frames(frames: Iterable[np.ndarray], amplitude: float = 45.0, period: float = 20.0, squash: float = 0.2) -> Iterator[np.ndarray]:
    """The same clip with a nod and a bow: about 14 px per frame at the fastest, the face foreshortened about the eye line. A 2D
    warp, not a real head. The crop follows the eyes, so this is what moves the lower face against a delta from an earlier frame."""
    for index, frame in enumerate(frames):
        phase = np.sin(2 * np.pi * index / period)
        scale = 1.0 - squash * (0.5 + 0.5 * phase)
        pivot = 0.4 * frame.shape[0]
        yield cv2.warpAffine(frame, np.float32([[1, 0, 0], [0, scale, pivot * (1.0 - scale) + amplitude * phase]]), (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def sway_choice(paths: Sequence[Path]) -> list[Path]:
    """At most ``SWAY_CLIPS`` of the clips, evenly spread."""
    if len(paths) <= SWAY_CLIPS:
        return list(paths)
    return [paths[index * len(paths) // SWAY_CLIPS] for index in range(SWAY_CLIPS)]


_NOT_RENDER_STEPS = frozenset({"landmarks", "segmentation", "total", "neural", "compose"})  # waits, the stage itself, the sum
# A 2D nod at 14 px per frame leaves the landmark model behind, whatever draws on it: that says nothing about a model.
NOD_RULES = {**EVENT_RULES, "stage_jump": float("inf"), "output_jump": float("inf"), "stage_offset": float("inf"), "lip_offset": float("inf")}


def run_quality_clip(path: Path, options: RunOptions, tools: Tools, paired: Path | None = None, rendered: Path | None = None, sway: bool = False, nod: bool = False) -> ClipResult:
    """One clip through the pipeline (or a finished video), measured frame by frame."""
    name = path.stem + ("+sway" if sway else "") + ("+nod" if nod else "")
    fps = file_fps(path)
    crops = None if options.out_dir is None else options.out_dir / "events"
    has_model = options.model_dir is not None
    if rendered is not None:
        mode = MODE_PAIRED if paired is not None else MODE_INPUT
    else:
        mode = MODE_CLASSIC if has_model else MODE_INPUT
    settings = _with_model(options.settings, options.strength) if has_model else _without_model(options.settings)
    probe = StageProbe()
    stage = build_stage(options.model_dir, tools) if has_model else None
    pipeline = _build_pipeline(settings, tools, probe, stage)
    raw_bare_tracker = tools.tracker_factory(0.0)
    raw_ref_tracker = tools.tracker_factory(0.0) if mode == MODE_CLASSIC or paired is not None else None
    raw_out_tracker = tools.tracker_factory(0.0)
    paired_frames = None if paired is None else read_frames(paired, options.frames_per_clip)
    rendered_frames = None if rendered is None else read_frames(rendered, options.frames_per_clip)

    def track_out(frame: np.ndarray, index: int) -> np.ndarray | None:
        return raw_out_tracker.detect(frame, _timestamp(index, fps))[0]

    embedder = None if tools.embedder is None else _CosineProbe(tools.embedder)
    analyzer = ClipAnalyzer(name, fps, mode, track_output=track_out, crops_dir=crops, embedder=embedder, rules=NOD_RULES if nod else None)
    serving = stage is None
    weak_frames = 0
    render_ms: list[float] = []
    try:
        source = read_frames(path, options.frames_per_clip)
        source = sway_frames(source) if sway else nod_frames(source) if nod else source
        for index, bare in enumerate(source):
            ts = _timestamp(index, fps)
            probe.classic = None
            result = pipeline.process(bare, ts)
            classic = probe.classic if probe.classic is not None else result.output
            if index >= SETTLE_FRAMES:
                render_ms.append(sum(ms for step, ms in result.timings_ms.items() if step not in _NOT_RENDER_STEPS))
            if stage is not None:
                if not serving and probe.landmarks is not None:
                    wait_until_serving(stage)
                    serving = True
                wait_for_stage(stage)
                if index >= SETTLE_FRAMES and probe.landmarks is not None and stage.health < 0.5 * geometry.soft_gate(probe.landmarks, bare.shape):
                    weak_frames += 1
            pair = None if paired_frames is None else next(paired_frames, None)
            if paired_frames is not None and pair is None:
                break  # the paired video is shorter: only pairs are compared
            if rendered_frames is not None:
                output = next(rendered_frames, None)
                if output is None:
                    break
                reference = bare if pair is None else pair
            else:
                output = result.output
                reference = classic if mode == MODE_CLASSIC else bare
            raw_bare = raw_bare_tracker.detect(bare, ts)[0]
            raw_ref = raw_bare if raw_ref_tracker is None else raw_ref_tracker.detect(reference, ts)[0]
            ref = make_frame_ref(index, bare, reference, probe.landmarks, raw_bare, raw_ref, probe.hair_mask, probe.face_skin_mask, options.may_edit_eyes)
            analyzer.feed(ref, bare, reference, output, pair)
        final_stats = None if stage is None else stage.stats()
    finally:
        pipeline.close()
        for tracker in (raw_bare_tracker, raw_ref_tracker, raw_out_tracker):
            if tracker is not None:
                tracker.close()
    outcome = analyzer.finish()
    outcome.metrics["classic_render_ms"] = float(np.median(render_ms)) if render_ms else float("nan")
    if final_stats is not None:
        outcome.metrics["paste_drift_p99"] = _percentile(np.array(final_stats["drift"], np.float64), 99) if final_stats["drift"] else float("nan")
        outcome.metrics["motion_weak_frames"] = float(final_stats["motion_weak_frames"])
        _add_stage_events(outcome, final_stats)
        if weak_frames:
            # The model was not really on for these frames, so what was measured there says nothing about it.
            outcome.events.append(ClipEvent(outcome.name, "health_fall", outcome.frames - 1, outcome.frames - 1, float(weak_frames)))
    return outcome


def _add_stage_events(outcome: ClipResult, info: dict[str, Any]) -> None:
    for kind, key in (("result_rejected", "rejected_results"), ("health_fall", "health_falls"), ("worker_restart", "restarts")):
        for _ in range(int(info.get(key, 0))):
            outcome.events.append(ClipEvent(outcome.name, kind, outcome.frames - 1, outcome.frames - 1, 1.0))
    if info.get("phase") in ("waiting", "failed", "off"):  # it failed and has not been restarted yet
        outcome.events.append(ClipEvent(outcome.name, "worker_restart", outcome.frames - 1, outcome.frames - 1, 1.0))


def _timestamp(index: int, fps: float) -> int:
    return round(index * 1000.0 / fps)


class _CosineProbe:
    """Cosine of output against bare, and of reference against bare, on every ``step``-th frame."""

    def __init__(self, embedder: Any, step: int = 4) -> None:
        self._embedder = embedder
        self._step = step

    def __call__(self, index: int, bare: np.ndarray, raw_bare: np.ndarray, reference: np.ndarray, raw_ref: np.ndarray | None, output: np.ndarray, raw_out: np.ndarray) -> tuple[float, float]:
        if index % self._step != 0:
            return float("nan"), float("nan")
        base = self._embedder.embed(bare, raw_bare)
        if base is None:
            return float("nan"), float("nan")
        out = self._embedder.embed(output, raw_out)
        ref = None if raw_ref is None else self._embedder.embed(reference, raw_ref)
        return (float("nan") if out is None else float(base @ out)), (float("nan") if ref is None else float(base @ ref))


# --------------------------------------------------------------------------
# What only a running model can show
# --------------------------------------------------------------------------


def run_strength_checks(path: Path, options: RunOptions, tools: Tools, frames: int = 60) -> dict[str, Any]:
    """Strength 0 and health 0 give the classic output, bit for bit; the applied change grows with the strength."""
    amounts: dict[float, float] = {}
    identical = True
    fps = file_fps(path)
    for strength in (0.0, 0.25, 0.5, 1.0):
        probe = StageProbe()
        stage = build_stage(options.model_dir, tools)
        pipeline = _build_pipeline(_with_model(options.settings, strength), tools, probe, stage)
        changes = []
        serving = strength <= 0.0
        try:
            for index, bare in enumerate(read_frames(path, frames)):
                probe.classic = None
                result = pipeline.process(bare, _timestamp(index, fps))
                classic = probe.classic if probe.classic is not None else result.output
                same = bool(np.array_equal(result.output, classic))
                if strength == 0.0 or stage.health <= 0.0:
                    identical = identical and same
                if not serving and probe.landmarks is not None:
                    wait_until_serving(stage)
                    serving = True
                wait_for_stage(stage)
                if index >= SETTLE_FRAMES:
                    changes.append(float(np.mean(cv2.absdiff(result.output, classic))))
        finally:
            pipeline.close()
        amounts[strength] = float(np.mean(changes)) if changes else float("nan")
    grows = amounts[0.0] == 0.0 and amounts[0.25] < amounts[0.5] < amounts[1.0]
    return {"zero_identical": identical, "monotonic": bool(grows), "mean_change": {str(k): _rounded(v) for k, v in amounts.items()}}


class _JumpClock:
    """The real clock plus jumps: the restart waits of the drill are skipped, the waits themselves are tested elsewhere."""

    def __init__(self, base: Callable[[], float]) -> None:
        self._base = base
        self.offset = 0.0

    def __call__(self) -> float:
        return self._base() + self.offset


DRILL_SUCCESS_FRAMES = 200  # a recovery must show within this many frames after the restart wait
DRILL_FADE_FRAMES = 60  # the classic output must be reached within this many frames
DRILL_STEP_FLOOR = 0.15  # levels: a frame step below this is noise of the measure


def run_fade_drill(path: Path, options: RunOptions, tools: Tools) -> dict[str, Any]:
    """Kill the worker, force a run error and bad outputs. Every frame must come, smoothly, down to classic and back."""
    from .neural import worker

    fps = file_fps(path)
    frames = list(read_frames(path, options.frames_per_clip))
    if len(frames) < 2:
        raise EvaluationError(f"動画が短すぎます: {path.name}")
    probe = StageProbe()
    clock = _JumpClock(tools.now)
    stage = build_stage(options.model_dir, tools, clock)
    pipeline = _build_pipeline(_with_model(options.settings, 1.0), tools, probe, stage)
    counter = {"index": 0}
    last: dict[str, np.ndarray | None] = {"output": None, "classic": None}

    def step() -> tuple[float, float, bool]:
        """One frame. Frame step of the output and of the classic output, and whether they are the same."""
        index = counter["index"]
        counter["index"] += 1
        period = 2 * len(frames) - 2
        position = index % period
        frame = frames[position if position < len(frames) else period - position]
        probe.classic = None
        result = pipeline.process(frame, _timestamp(index, fps))
        classic = probe.classic if probe.classic is not None else result.output
        step_out = step_classic = 0.0
        if last["output"] is not None:
            step_out = float(np.mean(cv2.absdiff(_half(result.output), _half(last["output"]))))
            step_classic = float(np.mean(cv2.absdiff(_half(classic), _half(last["classic"]))))
        last["output"], last["classic"] = result.output, classic
        wait_for_stage(stage, 0.05)
        return step_out, step_classic, bool(np.array_equal(result.output, classic))

    def rise(limit: int) -> bool:
        """Back at the strength this face allows. A turned or small face allows less than 1."""
        for _ in range(limit):
            step()
            allowed = 1.0 if probe.landmarks is None else geometry.soft_gate(probe.landmarks, last["output"].shape)
            if allowed > 0.0 and stage.health >= allowed - 0.01:
                return True
            info = stage.stats()
            if info["phase"] == "waiting":
                clock.offset += 121.0
        return False

    results: dict[str, Any] = {"cases": {}}
    ok = True
    try:
        for _ in range(30):
            step()
            if probe.landmarks is not None:
                break
        wait_until_serving(stage)
        if not rise(DRILL_SUCCESS_FRAMES):
            # A model whose every result is rejected, say. That is a failed drill, not a tool that could not run.
            results["ok"] = False
            results["cases"]["start"] = {"classic_reached": True, "recovered": False}
            results["frames"] = counter["index"]
            return results

        def kill() -> None:
            stage.kill_worker()

        def run_error() -> None:
            stage.backend.drill(worker.DRILL_RUN_ERROR)

        def bad_output() -> None:
            stage.backend.drill(worker.DRILL_BAD_OUTPUT)

        for name, trigger, repeat in (("kill", kill, 1), ("run_error", run_error, 1), ("bad_output", bad_output, 3)):
            steps_out, steps_classic = [], []
            for _ in range(repeat):
                trigger()
                out, cl, _same = step()
                steps_out.append(out)
                steps_classic.append(cl)
            reached = False
            for _ in range(DRILL_FADE_FRAMES):
                out, cl, same = step()
                steps_out.append(out)
                steps_classic.append(cl)
                if same and stage.health <= 0.0:
                    reached = True
                    break
            fade_steps = max(steps_out, default=0.0)
            base_steps = max(max(steps_classic, default=0.0), DRILL_STEP_FLOOR)
            recovered = rise(DRILL_SUCCESS_FRAMES)
            case_ok = reached and recovered and fade_steps <= 1.5 * base_steps
            results["cases"][name] = {"classic_reached": reached, "recovered": recovered, "step_ratio": _rounded(fade_steps / base_steps), "fade_frames": len(steps_out)}
            ok = ok and case_ok
        results["frames"] = counter["index"]
    finally:
        pipeline.close()
    results["ok"] = ok
    return results


_KERNEL_SCRIPT = r"""
import json, os, sys
import numpy as np
import onnxruntime as ort
path, size, folder = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if "DmlExecutionProvider" not in ort.get_available_providers():
    print(json.dumps({"available": False})); sys.exit(0)
so = ort.SessionOptions()
so.enable_profiling = True
so.profile_file_prefix = os.path.join(folder, "profile")
so.enable_mem_pattern = False
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.log_severity_level = 3
session = ort.InferenceSession(path, so, providers=["DmlExecutionProvider"])
for _ in range(3):
    session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, size, size), np.float32)})
profile = session.end_profiling()
events = json.load(open(profile, encoding="utf-8"))
nodes = [e for e in events if e.get("cat") == "Node" and e.get("args", {}).get("provider")]
other = sorted({e["args"].get("op_name", e["name"]) for e in nodes if e["args"]["provider"] != "DmlExecutionProvider"})
print(json.dumps({"available": True, "providers": session.get_providers(), "nodes": len(nodes), "not_on_gpu": other}))
"""


def check_kernels(model_file: Path, size: int) -> dict[str, Any]:
    """From ORT's profile, in a process of its own: does any node run off the GPU?"""
    with tempfile.TemporaryDirectory() as folder:
        try:
            done = subprocess.run(
                [sys.executable, "-c", _KERNEL_SCRIPT, str(model_file), str(size), folder],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("the GPU check did not run: %s", exc)
            return {"available": False}
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode != 0 or not lines:
        # To the log only: a traceback carries paths, and the report must not.
        logger.warning("the GPU check failed: %s", (done.stderr or "")[-300:])
        return {"available": False}
    return json.loads(lines[-1])


def process_vram_mb(pid: int) -> float | None:
    """``GPU Process Memory\\Dedicated Usage`` of a process and its children, Windows only. None if it cannot be read.

    The children count because the ``python.exe`` of a venv only starts the real interpreter as its child.
    """
    if sys.platform != "win32":
        return None
    command = (
        f"$ids = @({pid}) + @(Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}' | ForEach-Object {{ $_.ProcessId }}); "
        "$sum = 0.0; $found = $false; "
        "foreach ($id in $ids) { try { "
        "$samples = (Get-Counter \"\\GPU Process Memory(pid_${id}_*)\\Dedicated Usage\" -ErrorAction Stop).CounterSamples; "
        "foreach ($sample in $samples) { $sum += $sample.CookedValue; $found = $true } "
        "} catch {} }; "
        "if ($found) { $sum }"
    )
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=30, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        text = done.stdout.strip()
        return float(text.split()[-1]) / (1024.0 * 1024.0) if done.returncode == 0 and text else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


# --------------------------------------------------------------------------
# Real time: frames at the clip's rate, threads arranged as in runner.py, no virtual camera
# --------------------------------------------------------------------------


class _Mailbox:
    """The newest frame: the capture thread overwrites, the output thread takes."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: tuple[int, np.ndarray, int] | None = None
        self._closed = False

    def put(self, seq: int, frame: np.ndarray, ts: int) -> None:
        with self._cond:
            self._item = (seq, frame, ts)
            self._cond.notify_all()

    def take_newer(self, seq: int) -> tuple[int, np.ndarray, int] | None:
        with self._cond:
            self._cond.wait_for(lambda: self._closed or (self._item is not None and self._item[0] > seq), 0.5)
            if self._closed or self._item is None or self._item[0] <= seq:
                return None
            return self._item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


def _loop_frames(frames: list[np.ndarray], index: int) -> np.ndarray:
    """Forwards and backwards, so that a short clip loops without a cut."""
    period = max(2 * len(frames) - 2, 1)
    position = index % period
    return frames[position if position < len(frames) else period - position]


@dataclass
class RealtimeResult:
    seconds: float
    fps_in: float
    fps_out: float
    process_ms: list[float]
    stats: dict[str, Any] | None = None
    vram_mb: float | None = None


REALTIME_WARMUP_S = 3.0  # not counted: the models and the fades start up


def _deliver_like_the_app(output: np.ndarray) -> None:
    """Takes ``DELIVERY_MS`` on the output thread, doing the kind of work the delivery does."""
    until = time.perf_counter() + DELIVERY_MS / 1000.0
    cv2.cvtColor(output, cv2.COLOR_BGR2YUV_I420)
    strip = output[:64]
    while time.perf_counter() < until:
        cv2.cvtColor(strip, cv2.COLOR_BGR2YUV_I420)


def run_realtime(
    frames: list[np.ndarray],
    settings: MakeupSettings,
    fps: float,
    seconds: float,
    tools: Tools,
    stage: Any | None = None,
    probe: StageProbe | None = None,
    on_frame: Callable[[int, np.ndarray, Any, StageProbe | None], None] | None = None,
    clip_for_frame: Callable[[int], list[np.ndarray]] | None = None,
    deliver: bool = False,
) -> RealtimeResult:
    """The real pipeline at the real frame rate. Delivered fps counts distinct output frames, by the clock."""
    warmup_s = REALTIME_WARMUP_S
    pipeline = _build_pipeline(settings, tools, probe, stage, background_masks=True)
    box = _Mailbox()
    stop = threading.Event()
    delivered: list[float] = []
    process_ms: list[float] = []
    vram: list[float] = []

    def capture() -> None:
        started = time.perf_counter()
        index = 0
        while not stop.is_set():
            target = started + index / fps
            delay = target - time.perf_counter()
            if delay > 0.0:
                stop.wait(delay)
                if stop.is_set():
                    return
            source = frames if clip_for_frame is None else clip_for_frame(index)
            frame = _loop_frames(source, index)
            ts = round((time.perf_counter() - started) * 1000.0)
            pipeline.prepare(frame, ts)
            box.put(index + 1, frame, ts)
            index += 1

    def output() -> None:
        seq = 0
        while not stop.is_set():
            item = box.take_newer(seq)
            if item is None:
                continue
            seq, frame, ts = item
            began = time.perf_counter()
            result = pipeline.process(frame, ts)
            process_ms.append((time.perf_counter() - began) * 1000.0)
            if deliver:
                _deliver_like_the_app(result.output)
            delivered.append(time.perf_counter())
            if on_frame is not None:
                on_frame(seq - 1, frame, result, probe)

    threads = [threading.Thread(target=capture, name="eval-capture", daemon=True), threading.Thread(target=output, name="eval-output", daemon=True)]
    began = time.perf_counter()
    for thread in threads:
        thread.start()
    try:
        vram_at = began + min(seconds * 0.5 + warmup_s, seconds + warmup_s - 1.0)
        while time.perf_counter() - began < seconds + warmup_s:
            stop.wait(0.2)
            if stage is not None and not vram and time.perf_counter() >= vram_at:
                process = getattr(stage.backend, "_process", None)
                pid = getattr(process, "pid", None)
                value = None if pid is None else process_vram_mb(pid)
                vram.append(float("nan") if value is None else value)
        info = None if stage is None else stage.stats()
    finally:
        stop.set()
        box.close()
        for thread in threads:
            thread.join(timeout=5.0)
        pipeline.close()
    counted = [t for t in delivered if t >= began + warmup_s]
    fps_out = (len(counted) - 1) / (counted[-1] - counted[0]) if len(counted) > 1 else 0.0
    return RealtimeResult(seconds, fps, fps_out, process_ms[len(process_ms) // 10 :], info, vram[0] if vram and np.isfinite(vram[0]) else None)


class SoakAnalysis:
    """A lighter analysis on its own thread: no landmark re-detection, frames it cannot keep up with are dropped."""

    def __init__(self, fps: float, mode: str, may_edit_eyes: bool, crops_dir: Path | None) -> None:
        self._queue: deque[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]] = deque(maxlen=3)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._fps = fps
        self._mode = mode
        self._may_edit_eyes = may_edit_eyes
        self.analyzer = ClipAnalyzer("soak", fps, mode, track_output=None, crops_dir=crops_dir)
        self.seen = 0
        self.dropped = 0
        self._last_index: int | None = None
        self._thread = threading.Thread(target=self._run, name="eval-soak", daemon=True)
        self._thread.start()

    def offer(self, index: int, bare: np.ndarray, classic: np.ndarray, output: np.ndarray, probe: StageProbe) -> None:
        self.seen += 1
        if len(self._queue) == self._queue.maxlen:
            self.dropped += 1
        self._queue.append((index, bare, classic, output, probe.landmarks, probe.hair_mask, probe.face_skin_mask))
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set() or self._queue:
            try:
                index, bare, classic, output, landmarks, hair, skin = self._queue.popleft()
            except IndexError:
                self._wake.wait(0.1)
                self._wake.clear()
                continue
            if self._last_index is not None and index != self._last_index + 1:
                self.analyzer.break_continuity()
            self._last_index = index
            reference = classic if self._mode == MODE_CLASSIC else bare
            self.analyzer.feed(make_frame_ref(index, bare, reference, landmarks, None, None, hair, skin, self._may_edit_eyes), bare, reference, output)

    def finish(self) -> ClipResult:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=30.0)
        return self.analyzer.finish()


def run_soak(paths: Sequence[Path], options: RunOptions, tools: Tools, minutes: float) -> dict[str, Any]:
    """Real time for ``minutes``: looped clips, the real worker. Counts breakdowns; say what it could not see."""
    clips = [list(read_frames(path, options.frames_per_clip)) for path in paths]
    clips = [clip for clip in clips if len(clip) > 1]
    if not clips:
        raise EvaluationError("長時間の実行に使える動画がありません")
    fps = file_fps(paths[0])
    per_clip = max(int(minutes * 60.0 * fps / len(clips)), 1)
    has_model = options.model_dir is not None
    settings = _with_model(options.settings, options.strength) if has_model else _without_model(options.settings)
    probe = StageProbe()
    # This run's own analysis costs frames (22 fps instead of 30), which the stage must not take for its own fault: then it would
    # pause, and the run would say nothing about it.
    stage = build_stage(options.model_dir, tools, keep_frame_rate=False) if has_model else None
    crops = None if options.out_dir is None else options.out_dir / "events"
    analysis = SoakAnalysis(fps, MODE_CLASSIC if has_model else MODE_INPUT, options.may_edit_eyes, crops)

    def on_frame(index: int, frame: np.ndarray, result: Any, used: StageProbe | None) -> None:
        classic = used.classic if used is not None and used.classic is not None else result.output
        analysis.offer(index, frame, classic, result.output, used)

    outcome = run_realtime(
        clips[0], settings, fps, minutes * 60.0, tools, stage=stage, probe=probe, on_frame=on_frame,
        clip_for_frame=lambda index: clips[min(index // per_clip, len(clips) - 1)],
    )
    result = analysis.finish()
    if outcome.stats is not None:
        _add_stage_events(result, outcome.stats)
    counts: dict[str, int] = {}
    for event in result.events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    return {
        "minutes": minutes,
        "fps_out": outcome.fps_out,
        "frames_seen": analysis.seen,
        "frames_dropped_by_analysis": analysis.dropped,
        "events": result.events,
        "event_counts": counts,
        "stats": _stats_summary(outcome.stats),
    }


def _stats_summary(info: dict[str, Any] | None) -> dict[str, Any] | None:
    if info is None:
        return None
    return {key: info.get(key) for key in ("phase", "status", "good_results", "rejected_results", "restarts", "overloads", "health_falls", "frames_by_age", "motion_weak_frames", "lost_frames", "thrifty_frames")}


def realtime_clips(paths: Sequence[Path], results: Sequence[ClipResult]) -> list[Path]:
    """The clips the clock is read on: the one whose classic effects take longest, and a middling one. The stage costs frames only
    where those already fill the frame time. A model lost 0.14 fps on the first clip of the list and 2.5 on another, and the
    largest face is not the answer either: on a measured run it lost 0 where a smaller face with more hair lost 1.1. ``results``
    must be from the preset the clock runs at: at ``off`` all clips cost the same (3.1 ms), and that chose other clips than ``natural``."""
    by_name = {path.stem: path for path in paths}
    timed = sorted(
        ((clip.metrics["classic_render_ms"], by_name[clip.name]) for clip in results if clip.name in by_name and np.isfinite(clip.metrics.get("classic_render_ms", float("nan")))),
        key=lambda item: item[0],
    )
    if not timed:
        return list(paths[:1])
    return list(dict.fromkeys((timed[-1][1], timed[len(timed) // 2][1])))


def render_costs(paths: Sequence[Path], settings: MakeupSettings, options: RunOptions, tools: Tools) -> list[ClipResult]:
    """How long the classic effects take on each clip at ``settings``, for ``realtime_clips`` when the quality pass ran at another
    preset: a model evaluated at ``off`` says nothing about which clip is heavy at the budget's ``natural``."""
    limit = SETTLE_FRAMES + RENDER_COST_FRAMES
    if options.frames_per_clip is not None:
        limit = min(limit, options.frames_per_clip)
    out = []
    for path in paths:
        pipeline = _build_pipeline(_without_model(settings), tools, None, None)
        render_ms: list[float] = []
        try:
            for index, bare in enumerate(read_frames(path, limit)):
                result = pipeline.process(bare, _timestamp(index, 30.0))
                if index >= min(SETTLE_FRAMES, limit // 2):
                    render_ms.append(sum(ms for step, ms in result.timings_ms.items() if step not in _NOT_RENDER_STEPS))
        finally:
            pipeline.close()
        out.append(ClipResult(path.stem, len(render_ms), len(render_ms), 30.0, {"classic_render_ms": float(np.median(render_ms)) if render_ms else float("nan")}, [], {}))
    return out


def measure_realtime(
    paths: Sequence[Path],
    options: RunOptions,
    tools: Tools,
    seconds: float,
    also_settings: dict[str, MakeupSettings] | None = None,
    also_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """The classic pipeline, with the model, and the classic pipeline again, for each clip and setting. Delivered fps, timing, GPU memory.

    The budget is the contract's, at ``natural``, and the worst case counts. ``also_settings`` adds to what is under evaluation,
    which may be far lighter (a stand-in trained from ``off``: 6 ms per frame against 31), and is read on ``also_paths``, the clips
    that are heavy at those settings. The top level is the case that lost the most.
    """
    if options.model_dir is None:
        return _measure_classic(paths[0], options, tools, seconds)
    settings = {"": options.settings, **(also_settings or {})}
    cases = [_measure_model_case(path, name, chosen, options, tools, seconds) for name, chosen in settings.items() for path in (paths if name == "" or also_paths is None else also_paths)]
    worst = max(cases, key=lambda case: case["fps_loss"])
    out = dict(worst)
    out["budget_pauses"] = sum(case.get("budget_pauses", 0) for case in cases)
    out["cases"] = [{key: case.get(key) for key in ("clip", "preset", "classic_fps", "classic_fps_after", "model_fps", "fps_loss", "fps_loss_runs", "budget_pauses", "lost_frames", "thrifty_frames", "miss_pct", "vram_mb", "rtt_ms", "stage_ms", "model_process_ms")} for case in cases]
    for key in ("rtt_ms", "stage_ms"):
        timed = [case[key] for case in cases if case[key] is not None]
        out[key] = max(timed, key=lambda timing: timing["p95"]) if timed else None
    misses = [case["miss_pct"] for case in cases if case["miss_pct"] is not None]
    out["miss_pct"] = max(misses) if misses else None
    memory = [case["vram_mb"] for case in cases if case["vram_mb"] is not None]
    out["vram_mb"] = max(memory) if memory else None
    return out


def _measure_classic(path: Path, options: RunOptions, tools: Tools, seconds: float) -> dict[str, Any]:
    frames = _realtime_frames(path, options)
    fps = file_fps(path)
    classic = run_realtime(frames, _without_model(options.settings), fps, seconds, tools, deliver=True)
    return {"clip": path.stem, "seconds": seconds, "fps_in": fps, "classic_fps": classic.fps_out, "classic_process_ms": _timing(classic.process_ms)}


def _realtime_frames(path: Path, options: RunOptions) -> list[np.ndarray]:
    frames = list(read_frames(path, options.frames_per_clip))
    if len(frames) < 2:
        raise EvaluationError(f"動画が短すぎます: {path.name}")
    return frames


def _measure_model_case(path: Path, preset: str, settings: MakeupSettings, options: RunOptions, tools: Tools, seconds: float) -> dict[str, Any]:
    frames = _realtime_frames(path, options)
    fps = file_fps(path)
    # Classic on both sides of each model run: what else the machine was doing drifts, and the runs show how far.
    classic = [run_realtime(frames, _without_model(settings), fps, seconds, tools, deliver=True)]
    model_runs: list[RealtimeResult] = []
    losses: list[float] = []
    for _ in range(max(options.realtime_rounds, 1)):
        stage = build_stage(options.model_dir, tools)
        model_runs.append(run_realtime(frames, _with_model(settings, options.strength), fps, seconds, tools, stage=stage, deliver=True))
        classic.append(run_realtime(frames, _without_model(settings), fps, seconds, tools, deliver=True))
        losses.append(0.5 * (classic[-2].fps_out + classic[-1].fps_out) - model_runs[-1].fps_out)
    worst_run = int(np.argmax(losses))
    infos = [run.stats or {} for run in model_runs]
    ages: dict[int, int] = {}
    for info in infos:
        for age, count in info.get("frames_by_age", {}).items():
            ages[int(age)] = ages.get(int(age), 0) + int(count)
    total = sum(ages.values())
    memory = [run.vram_mb for run in model_runs if run.vram_mb is not None]
    return {
        "clip": path.stem,
        "preset": preset,
        "seconds": seconds,
        "fps_in": fps,
        "classic_fps": float(np.mean([run.fps_out for run in classic])),
        "classic_fps_before": classic[worst_run].fps_out,
        "classic_fps_after": classic[worst_run + 1].fps_out,
        "classic_process_ms": _timing(value for run in classic for value in run.process_ms),
        "model_fps": model_runs[worst_run].fps_out,
        "fps_loss": losses[worst_run],
        "fps_loss_runs": [_rounded(loss, 3) for loss in losses],
        "budget_pauses": sum(int(info.get("overloads", 0)) for info in infos),
        "lost_frames": sum(int(info.get("lost_frames", 0)) for info in infos),
        "thrifty_frames": sum(int(info.get("thrifty_frames", 0)) for info in infos),
        "model_process_ms": _timing(value for run in model_runs for value in run.process_ms),
        "rtt_ms": _timing(value for info in infos for value in info.get("rtt_ms", [])),
        "stage_ms": _timing(value for info in infos for value in info.get("stage_ms", [])),
        # A frame that sent no job on purpose has age 2 without anything being late.
        "miss_pct": 100.0 * max(sum(v for k, v in ages.items() if k >= 2) - sum(int(info.get("thrifty_frames", 0)) for info in infos), 0) / total if total else None,
        "frames_by_age": ages,
        "vram_mb": max(memory) if memory else None,
        "stats": _stats_summary(infos[worst_run]),
    }


def _timing(values: Iterable[float]) -> dict[str, float] | None:
    data = np.array(list(values), np.float64)
    if data.size == 0:
        return None
    return {"median": _rounded(float(np.median(data)), 3), "p95": _rounded(float(np.percentile(data, 95)), 3), "p99": _rounded(float(np.percentile(data, 99)), 3), "max": _rounded(float(data.max()), 3)}


# --------------------------------------------------------------------------
# Metrics over clips, verdict
# --------------------------------------------------------------------------

STATUS_MEASURED = "measured"
STATUS_NOT_RUN = "not_run"
STATUS_NOT_APPLICABLE = "not_applicable"  # this mode has no such measure: never counts for or against a verdict
STATUS_INFORMATION = "information"  # shown for review: never counts for or against a verdict


def _meets(op: str, value: float, limit: float) -> bool:
    return value <= limit if op == "le" else value >= limit


def _entry(threshold: Threshold, value: Any = None, worst: Any = None, worst_clip: str | None = None, status: str = STATUS_NOT_RUN, passed: bool | None = None, **extra: Any) -> dict[str, Any]:
    entry = {
        "value": value, "worst": worst, "worst_clip": worst_clip, "threshold": threshold.value, "worst_threshold": threshold.worst,
        "passed": passed, "status": status, "provisional": threshold.provisional, "required": threshold.required,
    }
    override = extra.pop("threshold_override", None)
    if override is not None:
        entry["threshold"], entry["worst_threshold"] = override
    entry.update(extra)
    return entry


def aggregate_clip_metric(threshold: Threshold, clips: Sequence[ClipResult], mode: str = MODE_CLASSIC) -> dict[str, Any]:
    """Median and worst over clips. With no worst limit, the single limit applies to the worst clip."""
    value_limit, worst_limit = threshold.limits(mode)
    values = {clip.name: clip.metrics.get(threshold.key, float("nan")) for clip in clips}
    finite = {name: value for name, value in values.items() if np.isfinite(value)}
    if not finite:
        return _entry(threshold)
    pick = max if threshold.op == "le" else min
    worst_clip = pick(finite, key=finite.__getitem__)
    worst = finite[worst_clip]
    median = float(np.median(list(finite.values())))
    if value_limit is None:
        passed = None
    elif worst_limit is None:
        passed = _meets(threshold.op, worst, float(value_limit))
    else:
        passed = _meets(threshold.op, median, float(value_limit)) and _meets(threshold.op, worst, worst_limit)
    return _entry(threshold, _rounded(median), _rounded(worst), worst_clip, STATUS_MEASURED, passed, threshold_override=(value_limit, worst_limit), clips=len(finite))


_CLIP_KEYS = (
    "flicker_ratio", "flicker_block_ratio", "flicker_outside_ratio", "delta_flicker", "mouth_interior_max", "eye_interior_max",
    "outside_max", "mouth_band_edge_corr", "mouth_inner_edge_corr", "eye_edge_corr", "background_change", "landmark_shift_p99",
    "identity_geometry_p99", "identity_structure_p01", "delta_abs_p99", "mouth_latency_frames", "paint_lag_frames", "paint_lag_local_frames", "paint_offset",
    "warp_flow_px", "seam_edge", "side_cast", "skin_cast", "rim_change", "frame_locked_ratio", "paste_drift_p99",
)


def _identity_cosine(clips: Sequence[ClipResult], mode: str = MODE_CLASSIC) -> dict[str, Any]:
    """The output against the bare frame. How far it fell below classic is shown beside it, for review."""
    entry = aggregate_clip_metric(THRESHOLD_BY_KEY["identity_cosine_p01"], clips, mode)
    if mode == MODE_CLASSIC:
        drops = {
            clip.name: clip.metrics["identity_cosine_p01"] - clip.metrics["identity_cosine_ref_p01"]
            for clip in clips
            if np.isfinite(clip.metrics.get("identity_cosine_p01", float("nan"))) and np.isfinite(clip.metrics.get("identity_cosine_ref_p01", float("nan")))
        }
        if drops:
            entry["below_classic_median"] = _rounded(float(np.median(list(drops.values()))))
            entry["below_classic_worst"] = _rounded(min(drops.values()))
    return entry


def _breaks(clips: Sequence[ClipResult], soak: dict[str, Any] | None) -> tuple[dict[str, Any], list[ClipEvent]]:
    """A short run cannot show a low rate. It can only show a high one: more events than the limit allows at all."""
    threshold = THRESHOLD_BY_KEY["breaks_per_30min"]
    events = [event for clip in clips for event in clip.events]
    seconds = sum(clip.frames / max(clip.fps, 1.0) for clip in clips)
    limit = float(threshold.value)
    if soak is not None and soak["minutes"] >= MIN_SOAK_MINUTES:
        # Both count: the soak has no landmarks to look at, and a clean soak must not excuse what the quality pass found.
        soak_events = sum(soak["event_counts"].values())
        count = soak_events + len(events)
        rate = count / (soak["minutes"] + seconds / 60.0) * 30.0
        entry = _entry(threshold, _rounded(rate, 3), _rounded(rate, 3), None, STATUS_MEASURED, rate <= limit, soak_minutes=soak["minutes"], soak_events=soak_events, quality_pass_events=len(events))
        return entry, events + list(soak["events"])
    soak_events = 0 if soak is None else sum(soak["event_counts"].values())
    total_seconds = seconds + (0.0 if soak is None else soak["minutes"] * 60.0)
    count = len(events) + soak_events
    allowed = limit * max(total_seconds / 1800.0, 1.0)
    projected = count / max(total_seconds, 1e-6) * 1800.0
    extra = {"events": count, "seconds": _rounded(total_seconds, 1), "projected_per_30min": _rounded(projected, 2)}
    if count > allowed:
        return _entry(threshold, _rounded(projected, 2), None, None, STATUS_MEASURED, False, **extra), events + ([] if soak is None else list(soak["events"]))
    return _entry(threshold, _rounded(projected, 2), None, None, STATUS_NOT_RUN, None, **extra), events + ([] if soak is None else list(soak["events"]))


def _bool_entry(key: str, value: bool | None, **extra: Any) -> dict[str, Any]:
    threshold = THRESHOLD_BY_KEY[key]
    if value is None:
        return _entry(threshold, **extra)
    return _entry(threshold, bool(value), None, None, STATUS_MEASURED, bool(value), **extra)


def _number_entry(key: str, value: float | None, **extra: Any) -> dict[str, Any]:
    threshold = THRESHOLD_BY_KEY[key]
    if value is None or not np.isfinite(value):
        return _entry(threshold, **extra)
    return _entry(threshold, _rounded(float(value), 3), None, None, STATUS_MEASURED, _meets(threshold.op, float(value), float(threshold.value)), **extra)


def read_detector_report(path: Path) -> dict[str, Any]:
    """A score from a detector the user ran themselves. Read only; see the module docstring."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        output, reference = float(data["output_score"]), float(data["reference_score"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EvaluationError(f"検出器の結果のファイルを読めませんでした: {path.name}（{exc}）") from exc
    if not (np.isfinite(output) and np.isfinite(reference)):
        raise EvaluationError(f"検出器の結果の値が数値ではありません: {path.name}")
    return {"output_score": output, "reference_score": reference, "detector": str(data.get("detector", ""))[:100]}


def build_metrics(
    mode: str,
    clips: Sequence[ClipResult],
    soak: dict[str, Any] | None = None,
    realtime: dict[str, Any] | None = None,
    strength: dict[str, Any] | None = None,
    drill: dict[str, Any] | None = None,
    kernels: dict[str, Any] | None = None,
    detector: dict[str, Any] | None = None,
    may_edit_eyes: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[ClipEvent]]:
    metrics: dict[str, dict[str, Any]] = {}
    for key in _CLIP_KEYS:
        metrics[key] = aggregate_clip_metric(THRESHOLD_BY_KEY[key], clips, mode)
    metrics["identity_cosine_p01"] = _identity_cosine(clips, mode)
    paired = aggregate_clip_metric(THRESHOLD_BY_KEY["paired_abs"], clips, mode) if any(np.isfinite(clip.metrics.get("paired_abs", float("nan"))) for clip in clips) else _entry(THRESHOLD_BY_KEY["paired_abs"])
    paired.update(passed=None, status=STATUS_INFORMATION if paired["status"] == STATUS_MEASURED else STATUS_NOT_RUN)
    metrics["paired_abs"] = paired
    metrics["breaks_per_30min"], events = _breaks(clips, soak)
    deepfake = THRESHOLD_BY_KEY["deepfake_margin"]
    if detector is None:
        metrics["deepfake_margin"] = _entry(deepfake, status=STATUS_NOT_RUN)
    else:
        margin = detector["output_score"] - detector["reference_score"]
        metrics["deepfake_margin"] = _entry(deepfake, _rounded(margin), status=STATUS_INFORMATION, detector=detector["detector"])
    kernel_ok = None
    if kernels is not None and kernels.get("available"):
        kernel_ok = not kernels.get("not_on_gpu")
    metrics["all_kernels_dml"] = _bool_entry("all_kernels_dml", kernel_ok, **({} if kernels is None else {"detail": kernels}))
    rt = realtime or {}
    metrics["rtt_p95_ms"] = _number_entry("rtt_p95_ms", (rt.get("rtt_ms") or {}).get("p95"))
    metrics["rtt_p99_ms"] = _number_entry("rtt_p99_ms", (rt.get("rtt_ms") or {}).get("p99"))
    metrics["stage_p95_ms"] = _number_entry("stage_p95_ms", (rt.get("stage_ms") or {}).get("p95"))
    metrics["fps_loss"] = _number_entry("fps_loss", rt.get("fps_loss"), classic_fps=rt.get("classic_fps"), model_fps=rt.get("model_fps"), runs=rt.get("fps_loss_runs"), budget_pauses=rt.get("budget_pauses"))
    if rt.get("budget_pauses"):  # the stage gave up to keep the frame rate: what the clock read then is the classic's, not the model's
        metrics["fps_loss"].update(passed=False)
    metrics["miss_pct"] = _number_entry("miss_pct", rt.get("miss_pct"))
    metrics["vram_mb"] = _number_entry("vram_mb", rt.get("vram_mb"))
    metrics["strength_zero_identical"] = _bool_entry("strength_zero_identical", None if strength is None else strength["zero_identical"])
    metrics["strength_monotonic"] = _bool_entry("strength_monotonic", None if strength is None else strength["monotonic"], **({} if strength is None else {"mean_change": strength["mean_change"]}))
    metrics["fade_drill"] = _bool_entry("fade_drill", None if drill is None else drill["ok"], **({} if drill is None else {"cases": drill["cases"]}))
    for key, entry in metrics.items():
        threshold = THRESHOLD_BY_KEY[key]
        if mode not in threshold.modes or (may_edit_eyes and key in ("eye_interior_max", "eye_edge_corr")):
            entry.update(status=STATUS_NOT_APPLICABLE, passed=None)
    return metrics, events


def decide_verdict(metrics: dict[str, dict[str, Any]], measured_frames: int, require_frames: int = MIN_PASS_FRAMES) -> tuple[str, list[str]]:
    """``fail``: something measured misses. ``incomplete``: nothing failed, but something required has no data. Else ``pass``."""
    failed = [key for key, entry in metrics.items() if entry["status"] == STATUS_MEASURED and entry["required"] and entry["passed"] is False]
    missing = [key for key, entry in metrics.items() if entry["status"] == STATUS_NOT_RUN and entry["required"]]
    notes = []
    if failed:
        return "fail", notes
    if measured_frames < require_frames:
        notes.append(f"測ったフレームが {measured_frames} で、合格に必要な {require_frames} に足りません")
        return "incomplete", notes
    if missing:
        return "incomplete", notes
    return "pass", notes


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

VERDICT_LABELS = {"pass": "合格", "fail": "不合格", "incomplete": "未完了（足りない項目があります）"}
STATUS_LABELS = {
    STATUS_MEASURED: "測定",
    STATUS_NOT_RUN: "未実施",
    STATUS_NOT_APPLICABLE: "対象外",
    STATUS_INFORMATION: "参考",
}
MODE_LABELS = {
    MODE_CLASSIC: "モデルの出力を、同じ実行の従来の出力と比べる",
    MODE_INPUT: "出力を入力の動画と比べる",
    MODE_PAIRED: "出力を対の参照動画と比べる",
}
REPORT_NOTES = (
    "これらの閾値はまだ最初の提案です。本物のメイクのモデルがない状態で、CREMA-D の女性 11 人の 44 本への注入実験から決めました。",
    "合格は「崩れを見つけられなかった」ことを意味します。メイクの見た目の良さは示しません。",
    "検出器のスコアは、利用者が自分で確認するための参考情報です。判定には使いません。良いスコアは何も保証しません。このツールは加工を隠すためのものではありません。",
)


EVENT_RULE_LABELS = {
    "spike_factor": "一瞬の変化: 直前 30 フレームの中央値の何倍で崩れとするか",
    "spike_floor": "一瞬の変化: 下限（階調、ブロック平均）",
    "background_block": "背景: ブロック平均の上限（階調）",
    "background_mean": "背景: 遠い背景全体の平均の上限（階調）",
    "teeth_edge_corr": "歯: エッジの相関の下限（モデルあり）",
    "eye_edge_corr": "目: エッジの相関の下限（モデルあり）",
    "input_teeth_edge_corr": "歯: エッジの相関の下限（入力と比べるとき）",
    "input_eye_edge_corr": "目: エッジの相関の下限（入力と比べるとき）",
    "stage_jump": "描画に使った点の飛び（×顔の大きさ）",
    "output_jump": "出力から検出した点の飛び（×顔の大きさ）",
    "stage_offset": "描画に使った点のずれ（×顔の大きさ、3 フレーム続く）",
    "lip_offset": "唇の外に出た色の割合（3 フレーム続く）",
    "lag_frames": "顔全体のメイクの遅れ（フレーム、15 フレームの窓）",
    "lag_local_frames": "一部のメイクの遅れ（フレーム、15 フレームの窓）",
    "lip_min_strength": "唇の色を測る下限（階調）",
    "frozen_output": "固まり: 出力の変化の上限（階調、3 フレーム続く）",
    "frozen_input": "固まり: 入力の変化の下限（階調）",
    "identity_landmarks": "顔の形: 点のずれの上限（×顔の大きさ、3 フレーム続く）",
    "identity_structure": "顔の構造: エッジの相関の下限（3 フレーム続く）",
}


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "はい" if value else "いいえ"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _format_limit(threshold: Threshold, mode: str = MODE_CLASSIC) -> str:
    if threshold.op == "true":
        return "はい"
    value, worst = threshold.limits(mode)
    if value is None:
        return "なし（判定に使いません）"
    sign = "≦" if threshold.op == "le" else "≧"
    if worst is None:
        return f"最悪のクリップで {sign} {value:g}"
    return f"中央値 {sign} {value:g}、最悪 {sign} {worst:g}"


def _pass_mark(entry: dict[str, Any]) -> str:
    if entry["status"] != STATUS_MEASURED:
        return STATUS_LABELS[entry["status"]]
    return "OK" if entry["passed"] else "NG"


def render_report_ja(report: dict[str, Any]) -> str:
    """The report in Japanese, every threshold shown."""
    lines = ["# 評価の結果", "", f"**判定: {VERDICT_LABELS[report['verdict']]}**", ""]
    model = report.get("model")
    conditions = report["conditions"]
    if model:
        lines.append(f"- モデル: {model['name']} {model['version']}（代用品: {'はい' if model['stand_in'] else 'いいえ'}）")
        lines.append(f"- 通話で使える: {'はい' if report['eligible_for_calls'] else 'いいえ'}")
    else:
        lines.append("- モデル: なし（従来のメイクだけ）")
    lines.extend(
        [
            f"- 比べ方: {MODE_LABELS[conditions['mode']]}",
            f"- 設定: {conditions['preset']}",
            f"- クリップ: {len(conditions['clips'])} 本、{conditions['frames']} フレーム、大きさ {conditions['frame_size'][0]}x{conditions['frame_size'][1]}",
            f"- 長時間の実行: {conditions['soak_minutes']} 分",
            f"- 日時: {report['date']}",
            f"- 使った学習済みモデル: {', '.join(p['name'] + '（' + p['licence'] + '）' for p in conditions['pretrained']) or 'なし'}",
            "",
        ]
    )
    for note in report["notes"]:
        lines.append(f"- {note}")
    if report["notes"]:
        lines.append("")
    lines.extend(["## 指標と閾値", "", "| 指標 | キー | 中央値など | 最悪 | 最悪のクリップ | 閾値 | 結果 |", "|---|---|---|---|---|---|---|"])
    for threshold in THRESHOLDS:
        entry = report["metrics"][threshold.key]
        unit = f" {threshold.unit}" if threshold.unit else ""
        mark = _pass_mark(entry)
        flag = "（暫定）" if threshold.provisional and threshold.value is not None else ""
        lines.append(f"| {threshold.label} | `{threshold.key}` | {_format_value(entry['value'])}{unit if entry['value'] is not None else ''} | {_format_value(entry['worst'])} | {entry['worst_clip'] or '-'} | {_format_limit(threshold, conditions['mode'])}{flag} | {mark} |")
    lines.extend(_realtime_lines(report))
    lines.extend(["", "## 崩れの記録", ""])
    events = report["events"]
    if not events:
        lines.append("見つかりませんでした。")
    else:
        lines.extend(["| クリップ | 種類 | フレーム | ピーク | 切り抜き |", "|---|---|---|---|---|"])
        for event in events:
            crops = "、".join(event["crops"]) or "-"
            frames = f"{event['start_frame']}" if event["start_frame"] == event["end_frame"] else f"{event['start_frame']}〜{event['end_frame']}"
            lines.append(f"| {event['clip']} | {event['label']} | {frames} | {_format_value(event['peak'])} | {crops} |")
    lines.extend(["", "## 崩れと数える基準", "", "| 基準 | キー | 値 |", "|---|---|---|"])
    for key, value in report.get("event_rules", EVENT_RULES).items():
        lines.append(f"| {EVENT_RULE_LABELS.get(key, key)} | `{key}` | {value:g} |")
    lines.extend(["", "## 注意", ""])
    lines.extend(f"- {note}" for note in REPORT_NOTES)
    lines.append("")
    return "\n".join(lines)


def _realtime_lines(report: dict[str, Any]) -> list[str]:
    """The numbers from the clock. They depend on what else the machine was doing, and the report says so."""
    realtime, soak = report.get("realtime"), report.get("soak")
    if not realtime and not soak:
        return []
    lines = ["", "## 実際の時間での計測", ""]
    if realtime:
        for case in realtime.get("cases", []):
            runs = "、".join(f"{loss:.2f}" for loss in case.get("fps_loss_runs") or [])
            extra = f"（各回 {runs}）" if runs else ""
            if case.get("budget_pauses"):
                extra += f"、フレームレートを守るために止まった回数 {case['budget_pauses']}"
            lines.append(f"- {case['clip']}（{case['preset'] or '評価した設定'}）: 従来 {case['classic_fps']:.2f} fps → モデルあり {case['model_fps']:.2f} fps、低下 {case['fps_loss']:.2f}{extra}")
        lines.append(f"- 動画: {realtime['clip']}、入力 {realtime['fps_in']:.2f} fps、{realtime['seconds']:g} 秒。仮想カメラは開いていません。送信の分は {DELIVERY_MS:g} ms と見積もって足しています")
        lines.append(f"- 従来のメイクだけ: {realtime['classic_fps']:.2f} fps、処理時間（中央値、p95、p99、最大）{_timing_text(realtime.get('classic_process_ms'))}")
        if "model_fps" in realtime:
            lines.append(f"- モデルあり: {realtime['model_fps']:.2f} fps、処理時間 {_timing_text(realtime.get('model_process_ms'))}")
            lines.append(f"- モデルの往復 {_timing_text(realtime.get('rtt_ms'))}、カメラスレッド上の処理 {_timing_text(realtime.get('stage_ms'))}")
    if soak:
        lines.append(f"- 長時間の実行: {soak['minutes']:g} 分、{soak['fps_out']:.2f} fps、{soak['frames_seen']} フレーム、分析が間に合わず飛ばしたフレーム {soak['frames_dropped_by_analysis']}、崩れ {sum(soak['event_counts'].values())} 件")
    lines.append("- これらの数字は、そのときの機械の混み具合に左右されます。他の重い処理を止めて測ってください。")
    return lines


def _timing_text(timing: dict[str, Any] | None) -> str:
    if not timing:
        return "-"
    return f"{timing['median']:g}、{timing['p95']:g}、{timing['p99']:g}、{timing['max']:g} ms"


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp)
        raise


def _git_changes() -> int | None:
    """Files that differ from the commit in ``_git_hash``, or None if git cannot tell: a hash alone would claim too much."""
    try:
        done = subprocess.run(
            ["git", "status", "--porcelain", "--", "."], cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, timeout=30, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return len(done.stdout.splitlines()) if done.returncode == 0 else None


def _git_hash() -> str | None:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=10, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return done.stdout.strip() or None if done.returncode == 0 else None
    except OSError:
        return None


def _gpu_name() -> str | None:
    if sys.platform != "win32":
        return None
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '; '"],
            capture_output=True, text=True, timeout=30, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return done.stdout.strip() or None
    except OSError:
        return None


def _ort_version() -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return None
    for name in ("onnxruntime-directml", "onnxruntime"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return None


def build_report(
    model: dict[str, Any] | None,
    mode: str,
    preset: str,
    clips: Sequence[ClipResult],
    metrics: dict[str, dict[str, Any]],
    events: Sequence[ClipEvent],
    soak: dict[str, Any] | None,
    realtime: dict[str, Any] | None,
    pretrained: Sequence[dict[str, str]],
    strength: float = 1.0,
) -> dict[str, Any]:
    measured = sum(clip.measured_frames for clip in clips)
    verdict, notes = decide_verdict(metrics, measured)
    if model is not None and strength < FULL_STRENGTH and verdict == "pass":
        # The slider goes up to 1: a pass at a lower strength says nothing about the higher ones.
        verdict = "incomplete"
        notes.append(f"強さ {strength:g} で測りました。合格にするには強さ {FULL_STRENGTH:g} で測ってください")
    if model is not None and model["stand_in"]:
        notes.append("このモデルは代用品です。判定がどうであれ、通話には使えません。見た目については何も示していません。")
    eligible = bool(model is not None and verdict == "pass" and not model["stand_in"])
    event_counts: dict[str, int] = {}
    for event in events:
        event_counts[event.kind] = event_counts.get(event.kind, 0) + 1
    return {
        "schema": SCHEMA,
        "model": model,
        "verdict": verdict,
        "eligible_for_calls": eligible,
        "date": dt.datetime.now().replace(microsecond=0).isoformat(),
        "conditions": {
            "mode": mode,
            "preset": preset,
            "strength": strength if model is not None else None,
            "clips": [clip.name for clip in clips],
            "frames": sum(clip.frames for clip in clips),
            "measured_frames": measured,
            "frame_size": list(FRAME_SIZE),
            "soak_minutes": 0 if soak is None else soak["minutes"],
            "gpu": _gpu_name() if model is not None else None,
            "onnxruntime": _ort_version(),
            "code": _git_hash(),
            "uncommitted_files": _git_changes(),
            "pretrained": list(pretrained),
        },
        "metrics": metrics,
        "thresholds": {t.key: {"op": t.op, "value": t.limits(mode)[0], "worst": t.limits(mode)[1], "provisional": t.provisional, "required": t.required, "basis": t.basis, "label": t.label, "unit": t.unit} for t in THRESHOLDS},
        "event_rules": dict(EVENT_RULES),
        "event_counts": event_counts,
        "events": [event.as_dict() for event in events],
        "per_clip": {clip.name: {key: _rounded(value) for key, value in clip.metrics.items()} for clip in clips},
        "soak": None if soak is None else {key: value for key, value in soak.items() if key != "events"},
        "realtime": realtime,
        "notes": notes,
    }


def write_report(report: dict[str, Any], out_dir: Path, model_dir: Path | None, update_metadata: bool) -> Path:
    """``evaluation.json`` and ``report_ja.md`` under ``out_dir``; with a model, also its folder and ``model.json``."""
    data = json.dumps(report, ensure_ascii=False, indent=1, default=_json_default).encode("utf-8")
    atomic_write(out_dir / "evaluation.json", data)
    atomic_write(out_dir / "report_ja.md", render_report_ja(report).encode("utf-8"))
    if model_dir is not None and update_metadata:
        atomic_write(model_dir / "evaluation.json", data)
        path = model_dir / "model.json"
        meta = json.loads(path.read_text(encoding="utf-8-sig"))
        meta["evaluation"] = {
            "verdict": report["verdict"],
            "report": "evaluation.json",
            "report_sha256": hashlib.sha256(data).hexdigest(),
            "model_sha256": report["model"]["sha256"],
            "date": report["date"],
        }
        atomic_write(path, (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return out_dir / "evaluation.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return value.name
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return str(value)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def default_clips() -> list[str]:
    return [str(Path(__file__).resolve().parent.parent / ".test-media" / "testvids" / "*.flv")]


def resolve_clips(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(globlib.glob(pattern))
        paths.extend(Path(match) for match in matches)
    return [path for path in dict.fromkeys(paths) if path.is_file()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m makeup_cam.evaluate",
        description="動画を実際のパイプラインに通して、ちらつき、崩れ、本人らしさ、遅れを測り、合否を出します。",
    )
    parser.add_argument("--model", metavar="DIR", type=Path, help="model.onnx と model.json のあるフォルダ。なければ従来のメイクだけを測ります")
    parser.add_argument("--clips", "--video", metavar="GLOB", nargs="+", action="extend", help="動画（ワイルドカード可）。既定は .test-media/testvids/*.flv")
    parser.add_argument("--preset", metavar="NAME", help="プリセットの名前。既定はモデルの input_preset、モデルがなければ natural")
    parser.add_argument("--settings", metavar="FILE", type=Path, help="設定ファイル（JSON）。--preset の代わりに使います")
    parser.add_argument("--out", metavar="DIR", type=Path, help="結果の保存先。崩れの切り抜き（顔が写ります）もここに入ります。git の管理外にしてください。既定はモデルのフォルダ")
    parser.add_argument("--reference", metavar="FILE", type=Path, help="対になる参照動画。--rendered があれば出力の比較先、なければ差の参考値だけに使います")
    parser.add_argument("--rendered", metavar="FILE", type=Path, help="すでに処理した動画を測ります（--clips に 1 本の入力動画が必要）")
    parser.add_argument("--frames-per-clip", metavar="N", type=int, help="1 本あたりのフレーム数の上限。既定は全部")
    parser.add_argument("--strength", metavar="X", type=float, default=1.0, help="測るときのモデルの強さ。既定は 1")
    parser.add_argument("--measure-fps", action="store_true", help="実際の時間で流してフレームレートと処理時間を測ります（仮想カメラは開きません）")
    parser.add_argument("--fps-seconds", metavar="S", type=float, default=15.0, help="--measure-fps の 1 回の長さ（秒）。既定は 15")
    parser.add_argument("--fps-rounds", metavar="N", type=int, default=REALTIME_ROUNDS, help=f"--measure-fps でモデルありを流す回数。いちばん悪い値で判定します。既定は {REALTIME_ROUNDS}")
    parser.add_argument("--soak-minutes", metavar="M", type=float, default=0.0, help="実際の時間で流す長さ（分）。30 分未満なら判定は「未完了」止まりです")
    parser.add_argument("--identity-model", metavar="PATH", type=Path, help="任意の顔認識モデル（FaceRecognizerSF 形式）。同梱もダウンロードもしません")
    parser.add_argument("--identity-licence", metavar="TEXT", default="利用者が確認すること（未記入）", help="--identity-model のライセンス。レポートに書きます")
    parser.add_argument("--deepfake-report", metavar="FILE", type=Path, help="利用者が自分で動かした検出器の結果（JSON）。参考として表示するだけで、判定には使いません")
    parser.add_argument("--no-update-metadata", action="store_true", help="model.json の evaluation を書き換えません")
    parser.add_argument("--no-sway", action="store_true", help="揺らした動画での確認を省きます。動きの少ない動画では「メイクの遅れ」が測れず、判定は「未完了」止まりになります")
    return parser


def check_arguments(args: argparse.Namespace) -> None:
    """Values that would give a report that looks measured and is not. Raises ``EvaluationError``."""
    if not 0.0 <= args.strength <= FULL_STRENGTH:
        raise EvaluationError(f"--strength は 0 から {FULL_STRENGTH:g} の間で指定してください")
    if args.frames_per_clip is not None and args.frames_per_clip < 1:
        raise EvaluationError("--frames-per-clip は 1 以上で指定してください")
    if args.fps_seconds <= 0.0:
        raise EvaluationError("--fps-seconds は 0 より大きい値で指定してください")
    if args.fps_rounds < 1:
        raise EvaluationError("--fps-rounds は 1 以上で指定してください")
    if args.soak_minutes < 0.0:
        raise EvaluationError("--soak-minutes は 0 以上で指定してください")


def out_dir_is_tracked(out_dir: Path) -> bool:
    """Whether what lands in this folder could be added to a git repository: the saved crops show faces.

    Read-only: asks git, changes nothing. With no git, or outside a repository, the answer is no.
    """
    folder = out_dir.resolve()
    while not folder.is_dir() and folder != folder.parent:
        folder = folder.parent
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=folder, capture_output=True, text=True, timeout=10, check=False, creationflags=flags)
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return False
        probe = (out_dir.resolve() / "events" / "x.png").as_posix()
        ignored = subprocess.run(["git", "check-ignore", "-q", probe], cwd=folder, capture_output=True, timeout=10, check=False, creationflags=flags)
        return ignored.returncode == 1  # 0: ignored, 128: git could not tell
    except (OSError, subprocess.TimeoutExpired):
        return False


def _model_summary(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    from .neural import model_files

    try:
        spec = model_files.load_spec(model_dir)
        model_files.check_gate(spec, allow_unevaluated=True)
    except model_files.ModelRefused as exc:
        logger.warning("the model was refused: %s", exc)
        raise EvaluationError(f"モデルを読み込めません: {exc.status}") from exc
    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8-sig"))
    summary = {"name": spec.name, "version": spec.version, "sha256": spec.sha256, "stand_in": spec.stand_in}
    return summary, meta, spec


def _load_settings(args: argparse.Namespace, meta: dict[str, Any] | None) -> tuple[MakeupSettings, str]:
    from .presets import PresetError, load_preset

    if args.settings is not None:
        try:
            return MakeupSettings.load(args.settings), args.settings.name
        except (OSError, ValueError) as exc:
            raise EvaluationError(f"設定ファイルを読み込めませんでした: {args.settings.name}（{exc}）") from exc
    name = args.preset or ((meta or {}).get("training_data") or {}).get("input_preset") or "natural"
    try:
        return load_preset(name), name
    except PresetError as exc:
        raise EvaluationError(f"プリセットを読み込めませんでした: {name}（{exc}）") from exc


def _budget_settings() -> MakeupSettings:
    from .presets import load_preset

    return load_preset(BUDGET_PRESET)


def run_evaluation(args: argparse.Namespace, tools: Tools | None = None) -> dict[str, Any]:
    """Everything the command line does but printing. Raises ``EvaluationError``."""
    tools = Tools() if tools is None else tools
    model_summary, meta, spec = (None, None, None)
    if args.model is not None:
        model_summary, meta, spec = _model_summary(args.model)
    if args.rendered is not None and args.model is not None:
        raise EvaluationError("--rendered と --model は一緒に使えません")
    settings, preset = _load_settings(args, meta)
    paths = resolve_clips(args.clips or default_clips())
    if not paths:
        raise EvaluationError("動画が見つかりません。--clips を確認してください")
    if (args.rendered is not None or args.reference is not None) and len(paths) != 1:
        raise EvaluationError("--rendered と --reference を使うときは、入力の動画を 1 本にしてください")
    out_dir = args.out if args.out is not None else args.model
    if out_dir is None:
        raise EvaluationError("モデルを指定しないときは、--out で保存先を指定してください")
    if out_dir_is_tracked(out_dir):
        raise EvaluationError("保存先が git の管理下にあります。崩れの切り抜きには顔が写るので、git の管理外のフォルダを --out に指定してください")
    pretrained: list[dict[str, str]] = []
    if args.identity_model is not None:
        if tools.embedder is None:
            if not args.identity_model.is_file():
                raise EvaluationError(f"顔認識モデルが見つかりません: {args.identity_model.name}")
            try:
                tools = dataclasses.replace(tools, embedder=SFaceEmbedder(args.identity_model))
            except cv2.error as exc:
                raise EvaluationError(f"顔認識モデルを読み込めませんでした: {args.identity_model.name}") from exc
        pretrained.append({"name": args.identity_model.stem, "licence": args.identity_licence})
    detector = None if args.deepfake_report is None else read_detector_report(args.deepfake_report)
    options = RunOptions(settings, args.model, False if spec is None else spec.may_edit_eyes, out_dir, args.frames_per_clip, args.strength, args.fps_rounds)

    results = []
    for number, path in enumerate(paths, 1):
        print(f"[{number}/{len(paths)}] {path.stem}", flush=True)
        results.append(run_quality_clip(path, options, tools, paired=args.reference, rendered=args.rendered))
    if not any(clip.measured_frames for clip in results):
        if args.frames_per_clip is not None and args.frames_per_clip <= SETTLE_FRAMES + 2:
            raise EvaluationError(f"--frames-per-clip が短すぎます。最初の {SETTLE_FRAMES} フレームは測りません（{SETTLE_FRAMES + 3} 以上で指定してください）")
        raise EvaluationError("顔が写っているフレームを測れませんでした")
    if args.rendered is None and not args.no_sway:
        for path in sway_choice(paths):
            print(f"揺らした動画: {path.stem}", flush=True)
            results.append(run_quality_clip(path, options, tools, sway=True))
        for path in sway_choice(paths)[:NOD_CLIPS]:
            print(f"うなずく動画: {path.stem}", flush=True)
            results.append(run_quality_clip(path, options, tools, nod=True))
    mode = MODE_CLASSIC if args.model is not None else (MODE_PAIRED if args.rendered is not None and args.reference is not None else MODE_INPUT)

    strength = drill = kernels = None
    if args.model is not None:
        print("強さの確認", flush=True)
        strength = run_strength_checks(paths[0], options, tools)
        print("障害の訓練", flush=True)
        drill = run_fade_drill(paths[0], options, tools)
        if tools.backend_factory is None:
            kernels = check_kernels(spec.file, spec.size)
    realtime = None
    if args.measure_fps:
        print("実時間の計測", flush=True)
        also = {} if args.model is None or preset == BUDGET_PRESET else {BUDGET_PRESET: _budget_settings()}
        also_paths = None if not also else realtime_clips(paths, render_costs(paths, also[BUDGET_PRESET], options, tools))
        realtime = measure_realtime(realtime_clips(paths, results), options, tools, args.fps_seconds, also, also_paths)
    soak = None
    if args.soak_minutes > 0.0:
        print(f"長時間の実行（{args.soak_minutes:g} 分）", flush=True)
        soak = run_soak(paths, options, tools, args.soak_minutes)
    metrics, events = build_metrics(mode, results, soak, realtime, strength, drill, kernels, detector, options.may_edit_eyes)
    if args.model is not None:
        # The file must still be the one that was measured.
        from .neural import model_files

        if model_files.sha256_of(spec.file) != spec.sha256:
            raise EvaluationError("評価の途中でモデルのファイルが変わりました")
    report = build_report(model_summary, mode, preset, results, metrics, events, soak, realtime, pretrained, args.strength)
    if args.rendered is not None:
        report["notes"].append("圧縮した動画では、圧縮そのものが背景とちらつきの値を変えます。閾値は無圧縮のフレームで決めたものです。無圧縮に近い形式で保存した動画で測ってください。")
    write_report(report, out_dir, args.model, not args.no_update_metadata)
    return report


def print_summary(report: dict[str, Any], out_dir: Path | None) -> None:
    print()
    print(f"判定: {VERDICT_LABELS[report['verdict']]}")
    print()
    for threshold in THRESHOLDS:
        entry = report["metrics"][threshold.key]
        if entry["status"] == STATUS_NOT_APPLICABLE:
            continue
        worst = "" if entry["worst"] is None else f"、最悪 {_format_value(entry['worst'])}"
        print(f"  [{_pass_mark(entry):>4}] {threshold.label}: {_format_value(entry['value'])}{worst}（閾値 {_format_limit(threshold, report.get('conditions', {}).get('mode', MODE_CLASSIC))}）")
    print()
    for line in _realtime_lines(report)[2:]:
        print(line)
    print(f"崩れの記録: {len(report['events'])} 件")
    for note in report["notes"]:
        print(f"注意: {note}")
    for note in REPORT_NOTES:
        print(f"注意: {note}")
    if out_dir is not None:
        print(f"保存先のファイル: {out_dir / 'evaluation.json'}、{out_dir / 'report_ja.md'}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        check_arguments(args)
        report = run_evaluation(args)
    except EvaluationError as exc:
        print(f"評価できませんでした: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # the exit code says "could not run", and the reason goes to the log
        logger.exception("the evaluation failed")
        print(f"評価できませんでした: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print_summary(report, args.out if args.out is not None else args.model)
    return {"pass": EXIT_PASS, "fail": EXIT_FAIL, "incomplete": EXIT_INCOMPLETE}[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
