"""The learned finishing stage as a pipeline effect (``docs/dev/contract-plan-b.md`` sections 2, 3, 5 and 7).

Order 90: the last step before the pipeline clips and converts the frame. It finishes the classic result:

    frame + clamp(delta) * edge_window (warped to the frame) * weight * strength * ctx.intensity * health * motion

Camera thread (``apply``). It never waits, reads a file, or starts or kills a process: it makes the crop, offers it to
the backend if no job is in flight, and pastes the newest finished delta through the *current* landmarks. That delta
is normally from the frame before (age 1) and lives in crop space, so a late result follows the face and cannot land
on the wrong place. With ``health``, the strength or the intensity at 0 the frame comes back untouched, bit for bit.

Paste thread. The weight and the warped delta need only the landmarks and the masks, and those are final long before
the frame reaches this stage. So the pipeline calls ``start_ahead`` once they are, and a helper thread prepares the
paste while the classic effects run. ``apply`` only adds what is ready. If it is not ready (or belongs to other landmarks)
``apply`` makes it itself, with the same result: it never waits for the helper. Measured on a large face at ``natural``:
the stage takes 1.2 ms on the camera thread instead of 3.4, and the whole frame grows by 1.5 to 2 ms (the rest is the effect
that runs next to the helper, as the two share the GIL; it costs about the same whichever effect that is). That matters when
the classic effects already fill the frame time, where each ms costs frames.

Control thread. Started when the stage is first wanted, never before: with the section off, the strength at 0, or no
face, nothing starts. It checks the gate, starts, watches, kills and restarts the worker. The camera thread only
reads what it publishes (one attribute at a time), so no lock is shared with slow work.

``health`` is the stage's own fade level. All changes are ramps, so the picture never jumps. While it falls the last
good delta keeps following the face. Failures of any kind lead to the same place: the classic output.

``motion`` weakens the paste while the delta does not fit the face. The crop follows the eyes, so a pan costs nothing, but in
a nod or a bow the mouth and the chin move against the eyes, and a delta from an earlier frame lands beside them. Age alone
does not say so: on a still face age 3 fits, and in a nod age 1 does not. So it is measured, for each frame: how far the
face parts are, in crop space, from where they were in the frame the delta came from. ``_Ease`` makes it move without corners:
a level that turns within one frame shows as a flash where the delta is large.

The masks come from a slower model, so they are some frames behind. On a still face that is nothing. In a fast pan it leaves their
edge on the wall, which the crop following the eyes does not help with. So ``MaskTrust`` weakens what the masks add to the face oval
while the face is far from where it was in the frame they saw. The oval itself follows the current landmarks.

Frame rate. The classic effects may fill the frame time, and then all the stage adds costs frames. The pipeline says how many
frames it never got to. Too many, and the stage sends a job only on every second frame, which saves the crop and the copies at the
price of age 2 on half the frames. If frames are still lost it pauses (the overload path). A pause is a failure of the budget, and
the evaluation counts it as one.
"""

from __future__ import annotations

import importlib.util
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np

from .. import regions
from ..effects.base import Effect, FrameContext
from . import geometry, model_files, worker
from .backend import Backend, BackendError, ProcessBackend, Result
from .model_files import ModelRefused, ModelSpec

logger = logging.getLogger(__name__)

NEURAL_WAIT_MS = 0  # measured: waiting up to 8 ms for the current result costs 2 fps at natural

HEALTH_RISE = 1.0 / 15.0  # per frame: 0.5 s
HEALTH_FALL = 1.0 / 10.0  # per frame: 0.33 s
# A result this many frames old starts the fade. Younger late results are reused through the current landmarks.
# Measured with a stand-in on three clips, with moving faces, as the second difference of what the stage adds, in crop
# space: with 1 job in 20 late by 2 frames, reuse shimmers as much as on-time results (0.576 levels against 0.576), and
# fading at each late result adds 7 % (0.615) and dips the finish; with results steadily at age 2 and 3, reuse still
# gives 0.577, and fading gives no finish at all.
STALE_AGE = 4

# How far the face parts may be from the delta's frame: (full strength, nothing), in units of the face scale, as the 90th
# percentile over lips, oval and eyes, allowing for the smoothing's own lag, smoothed over a few frames. Measured on 8 women's
# clips: see docs/dev/contract-plan-b.md section 7.4. Provisional: real heads, not 2D warps, may need other values.
MOTION_FIT = (0.05, 0.10)
MOTION_SMOOTHING = 0.5
MOTION_FALL = 1.0 / 4.0  # per frame: a nod is short, so this must be quicker than ``health``
MOTION_RISE = 1.0 / 15.0
# How much the speed of a level may change per frame. At 1/4, a fall that turned into a rise changed the speed by 0.32 within one
# frame: over a delta of 10 to 12 levels that is 3 levels for one frame, and it was all the breaks of a 30-minute run (22 of 22,
# double application; each at such a corner). 1/8 costs a nod about one frame.
EASE_ACCEL = 1.0 / 8.0
_JOB_POINTS_KEPT = 16

# How far the face may be from where it was in the frame the masks saw: (full trust, none), in units of the face scale, as the
# centre between the eyes plus the change of size. Provisional, from 8 women's clips and 2D pans: see the contract, section 5.
MASK_MOVED = (0.07, 0.14)
_MASK_TRACK_KEPT = 32

WORKER_SILENT_S = 1.0  # a job in flight this long means a driver reset or a hang
RESTART_WAITS_S = (5.0, 30.0, 120.0)
RESTART_WINDOW_S = 600.0
GOOD_RESULTS_AFTER_RESTART = 10
OVERLOAD_WINDOW_S = 5.0
OVERLOAD_SHARE = 0.2
OVERLOAD_MIN_FRAMES = 90  # fewer frames than this (3 s) say nothing: the first runs after a start may be slow
OVERLOAD_PAUSE_S = 30.0
OVERLOAD_MAX_PAUSE_S = 300.0
OVERLOAD_FORGIVE_S = 300.0  # this long without overload: the next pause is short again
# Frames lost from which the stage does less. The budget is 0.5 fps. Next to the limit of the frame time they come in bursts, so a
# pause, which costs the finish for 30 s, takes a longer look than sending fewer jobs, which costs almost nothing.
BUDGET_LOST_FRAMES = 3  # within OVERLOAD_WINDOW_S: a job on every second frame only
BUDGET_PAUSE_FRAMES = 6  # within BUDGET_PAUSE_WINDOW_S, and still after that: pause
BUDGET_PAUSE_WINDOW_S = 10.0
THRIFTY_HOLD_S = 120.0  # this long without such a loss: back to a job on every frame
# Over budget on the camera thread (provisional): twice the 5 ms p95 of the contract's budget, on a machine so busy that
# the stage's own work costs the call frames. A busy CPU showed this (20 ms) while results still came on time.
STAGE_OVER_BUDGET_MS = 10.0
_STAGE_RECENT_KEPT = 150
IDLE_STOP_S = 60.0  # nothing wanted for this long (section off, no face): give the GPU memory back
REARM_GAP_S = 0.25  # apply was not called for this long and the settings changed: the section was off and on again
RESUME_GAP_S = 1.0  # apply was not called for this long: what is left is old, so rise from 0 instead of showing it
AHEAD_RESULT_WAIT_S = 0.015  # the helper waits this long for the job in flight, so that the frame shows age 1 if it can
SYNC_START_TIMEOUT_S = 30.0
SYNC_RESULT_TIMEOUT_S = 2.0
_CONTROL_POLL_S = 0.1
_FRAMES_KEPT = 1024  # per-frame records for the overload check and the statistics
_STAGE_TIMES_KEPT = 8192
_LAST_AGE_KEPT = 16  # in the statistics: this age or older
_NEGLIGIBLE_WEIGHT = 1.0 / 255.0  # with the clamp, under half an 8-bit level: the corners of the crop cost nothing

# Status texts (contract section 11).
STATUS_NO_MODEL = model_files.NO_MODEL_MESSAGE
STATUS_NOT_PASSED = model_files.NOT_PASSED_MESSAGE
STATUS_NO_RUNTIME = "学習モデルの実行に必要な onnxruntime-directml が入っていないため、従来のメイクのまま動いています"
STATUS_NO_GPU = "GPU が使えないため、従来のメイクのまま動いています"
STATUS_OVERLOAD = "処理が間に合わないため、一時的に弱めています"
STATUS_FAILED = "エラーのため止めました。もう一度オンにすると再開します"
STATUS_PREPARING = "準備しています…"
STATUS_RUNNING = "動作中"

# Phases of the control thread.
_IDLE = "idle"  # nothing started: nothing wanted yet, or stopped after being idle
_STARTING = "starting"
_RUNNING = "running"
_WAITING = "waiting"  # before a restart
_OFF = "off"  # refused, no runtime, no GPU: stays off until the section is switched on again
_FAILED = "failed"  # three restarts used up: the same
_CLOSED = "closed"


def _onnxruntime_available() -> bool:
    return importlib.util.find_spec("onnxruntime") is not None


class _Ease:
    """A level that follows a target with a limited speed and a limited change of speed, and slows down before it arrives."""

    def __init__(self, fall: float, rise: float, value: float = 1.0) -> None:
        self._fall = fall
        self._rise = rise
        self._speed = 0.0
        self.value = value

    def step(self, target: float) -> float:
        distance = target - self.value
        braking = min(abs(distance), (2.0 * EASE_ACCEL * abs(distance)) ** 0.5)
        wanted = max(-braking, -self._fall) if distance < 0.0 else min(braking, self._rise)
        self._speed = min(max(wanted, self._speed - EASE_ACCEL), self._speed + EASE_ACCEL)
        self.value = min(max(self.value + self._speed, 0.0), 1.0)
        if abs(target - self.value) < 1e-6 and abs(self._speed) < EASE_ACCEL:
            self.value, self._speed = target, 0.0
        return self.value


class MaskTrust:
    """How much of what the masks add to the oval to believe in this frame: see the module docstring. One step per frame."""

    def __init__(self) -> None:
        self._ease = _Ease(MOTION_FALL, MOTION_RISE)
        self._track: deque[tuple[int, float, float, float]] = deque(maxlen=_MASK_TRACK_KEPT)

    def reset(self) -> None:
        self.__init__()

    def step(self, landmarks: np.ndarray, ts_ms: int | None, masks_ts_ms: int | None) -> float:
        if ts_ms is None or masks_ts_ms is None:  # a source that does not say when: nothing to go by
            return self._ease.value
        scale = max(float(regions.face_scale(landmarks)), 1.0)
        x, y = (regions.eye_center(landmarks, "left") + regions.eye_center(landmarks, "right")) / 2.0
        self._track.append((ts_ms, float(x), float(y), scale))
        then = next((entry for entry in reversed(self._track) if entry[0] <= masks_ts_ms), None)
        if then is None:  # older than anything kept: they may be anywhere
            return self._ease.step(0.0)
        moved = (np.hypot(x - then[1], y - then[2]) + abs(scale - then[3])) / scale
        return self._ease.step(1.0 - geometry._ramp(moved, *MASK_MOVED) if np.isfinite(moved) else 0.0)


@dataclass(frozen=True)
class _Paste:
    """What to add to the frame, before the amount: the delta, warped and weighted for one set of landmarks."""

    y0: int
    y1: int
    x0: int
    x1: int
    weighted: np.ndarray


def _weighted_delta(
    landmarks: np.ndarray,
    hair: np.ndarray | None,
    skin: np.ndarray | None,
    shape: tuple[int, ...],
    spec: ModelSpec,
    delta_of: Callable[[], np.ndarray | None],
    mask_trust: float = 1.0,
) -> _Paste | None:
    """The paste for these landmarks, or None when nothing falls on the frame. Reads the delta only after the weight."""
    matrix = geometry.crop_matrix(landmarks, spec.size)
    roi = geometry.crop_roi(matrix, spec.size, shape)
    if roi is None:
        return None
    weight = geometry.edit_weight(landmarks, roi, hair, skin, spec.may_edit_eyes, mask_trust)
    # Only the rectangle where the weight is more than a mask's noise: most of the crop's corners cost nothing.
    x, y, w, h = cv2.boundingRect(cv2.compare(weight, _NEGLIGIBLE_WEIGHT, cv2.CMP_GT))
    delta = delta_of()
    if w == 0 or h == 0 or delta is None or delta.shape[0] != spec.size:  # a delta kept from before a change of model
        return None
    inverse = cv2.invertAffineTransform(matrix)
    inverse[0, 2] -= roi.x0 + x
    inverse[1, 2] -= roi.y0 + y
    warped = cv2.warpAffine(delta, inverse, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    part = weight[y : y + h, x : x + w]
    cv2.multiply(warped, cv2.merge((part, part, part)), dst=warped)
    return _Paste(roi.y0 + y, roi.y0 + y + h, roi.x0 + x, roi.x0 + x + w, warped)


@dataclass(eq=False)
class _AheadJob:
    """One frame's paste, made on the helper thread. The camera thread reads the results only after ``done``."""

    landmarks: np.ndarray
    hair: np.ndarray | None
    skin: np.ndarray | None
    shape: tuple[int, ...]
    spec: ModelSpec
    backend: Backend | None
    held: Result | None  # the delta the stage holds, for when no newer one comes
    min_tag: int
    mask_trust: float = 1.0
    result: Result | None = None
    paste: _Paste | None = None
    done: bool = False
    failed: bool = False
    abandoned: bool = False  # the camera thread got there first: there is no use in finishing

    def run(self) -> None:
        try:
            if geometry.usable_landmarks(self.landmarks, self.shape):
                self.paste = _weighted_delta(self.landmarks, self.hair, self.skin, self.shape, self.spec, self._newest_delta, self.mask_trust)
            else:
                self.failed = True  # apply decides what happens to these landmarks
        except Exception:  # apply then tries itself, and the pipeline's isolation sees what it raises
            logger.exception("preparing the neural paste failed")
            self.failed = True
        self.done = True

    def _newest_delta(self) -> np.ndarray | None:
        """Normally the result of the frame before. The job in flight is worth a short wait, on this thread only."""
        backend = self.backend
        if backend is not None and backend.in_flight_since() is not None and not self.abandoned:
            backend.wait_for_in_flight(AHEAD_RESULT_WAIT_S)
        latest = None if backend is None else backend.latest()
        self.result = latest if latest is not None and latest.tag >= self.min_tag else self.held
        return None if self.result is None or self.abandoned else self.result.delta


class _AheadThread:
    """Runs the newest ``_AheadJob``. Older ones that were not started are dropped. Started when first needed."""

    def __init__(self) -> None:
        self._jobs: queue.SimpleQueue[_AheadJob | None] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None

    def post(self, job: _AheadJob) -> None:
        """Never waits."""
        self._jobs.put(job)
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="neural-paste", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._jobs.put(None)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            while job is not None and not self._jobs.empty():
                job = self._jobs.get()
            if job is None:
                return
            if not job.abandoned:
                job.run()


class NeuralFinishEffect(Effect):
    """See the module docstring. ``allow_unevaluated`` and ``sync`` are for the offline modes and tests only."""

    name: ClassVar[str] = "neural"
    order: ClassVar[int] = 90
    needs_face: ClassVar[bool] = True

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        allow_unevaluated: bool = False,
        sync: bool = False,
        provider: str = "dml",
        backend_factory: Callable[[ModelSpec], Backend] | None = None,
        clock: Callable[[], float] = time.perf_counter,  # time.monotonic only ticks every 15.6 ms on Windows
        control_thread: bool = True,
        keep_frame_rate: bool = True,  # False only where something else is known to cost the frames: the evaluation's long run
    ) -> None:
        self._model_dir = model_dir
        self._allow_unevaluated = allow_unevaluated
        self._sync = sync
        self._provider = provider
        self._backend_factory = backend_factory
        self._clock = clock
        self._use_control_thread = control_thread
        self._keep_frame_rate = keep_frame_rate
        self._control: threading.Thread | None = None
        self._wake = threading.Event()
        self._ready_or_off = threading.Event()  # only ``sync`` waits on it
        # Written by the control thread, read by the camera thread.
        self._phase = _IDLE
        self._status: str | None = None
        self._backend: Backend | None = None
        self._starting: Backend | None = None  # the one the control thread is waiting for, so that ``close`` can reach it
        self._spec: ModelSpec | None = None
        self._good_needed = 1
        self._no_jobs_until = 0.0
        # Written by the camera thread, read by the control thread.
        self._wanted = False
        self._rearm = False
        self._last_apply_at: float | None = None
        self._last_wanted_at: float | None = None
        self._frames: deque[tuple[float, bool]] = deque(maxlen=_FRAMES_KEPT)  # (when, a job was sent)
        self._stage_recent: deque[float] = deque(maxlen=_STAGE_RECENT_KEPT)  # ms, frames with work to do
        self._lost: deque[float] = deque(maxlen=_FRAMES_KEPT)  # when the pipeline never got to a frame, while the stage worked
        # Written by the control thread, read by the camera thread.
        self._thrifty = False  # a job on every second frame only
        # Camera thread only.
        self._frame_no = 0
        self._health = 0.0
        self._falling = False
        self._motion = _Ease(MOTION_FALL, MOTION_RISE)
        self._drift = 0.0
        self._job_points: deque[tuple[int, np.ndarray]] = deque(maxlen=_JOB_POINTS_KEPT)
        self._points_now: tuple[np.ndarray | None, np.ndarray | None] = (None, None)
        self._mask_trust = MaskTrust()
        self._trust_now: tuple[np.ndarray | None, float] = (None, 1.0)
        self._last_sent_frame = -1
        self._result: Result | None = None
        self._min_tag = 0
        self._last_section: Any = None
        self._was_positive = False
        self._ahead_thread = _AheadThread()
        self._ahead: _AheadJob | None = None
        # Control thread only.
        self._closed = False
        self._failures: deque[float] = deque()
        self._ever_ready = False
        self._restart_at = 0.0
        self._overload_pause = OVERLOAD_PAUSE_S
        self._last_overload_at: float | None = None
        self._last_budget_miss_at = 0.0
        # Statistics, for the status text, the evaluation and the proof scripts.
        self.restarts = 0
        self.overloads = 0
        self.health_falls = 0
        self.pastes_ahead = 0  # frames whose paste was ready when they reached the stage ...
        self.pastes_inline = 0  # ... and frames that made it on the camera thread
        self.frames_by_age: dict[int, int] = {}
        self.stage_ms: deque[float] = deque(maxlen=_STAGE_TIMES_KEPT)
        self.drift: deque[float] = deque(maxlen=_STAGE_TIMES_KEPT)  # how far off the face each paste landed that was not weakened much
        self.motion_weak_frames = 0  # frames the paste was weakened to under half because it did not fit
        self.lost_frames = 0  # frames the pipeline never got to, while the stage worked
        self.thrifty_frames = 0  # frames that sent no job to save time
        self.refused_because: str | None = None

    # ---- what others read -----------------------------------------------------------------------

    @property
    def health(self) -> float:
        return self._health

    @property
    def motion(self) -> float:
        return self._motion.value

    @property
    def mask_trust(self) -> float:
        return self._trust_now[1]

    @property
    def status_message(self) -> str | None:
        """Japanese, for the GUI; None until the stage is first wanted."""
        return self._status

    @property
    def backend(self) -> Backend | None:
        return self._backend

    def stats(self) -> dict[str, Any]:
        """A snapshot for the evaluation tool and the proof scripts. Any thread."""
        backend = self._backend
        return {
            "phase": self._phase,
            "status": self._status,
            "health": self._health,
            "frames": self._frame_no,
            "frames_by_age": dict(self.frames_by_age),
            "stage_ms": _snapshot(self.stage_ms),
            "rtt_ms": [] if backend is None else _snapshot(getattr(backend, "rtt_ms", ())),
            "good_results": 0 if backend is None else backend.good_results,
            "rejected_results": 0 if backend is None else backend.rejected_results,
            "restarts": self.restarts,
            "overloads": self.overloads,
            "health_falls": self.health_falls,
            "pastes_ahead": self.pastes_ahead,
            "pastes_inline": self.pastes_inline,
            "drift": _snapshot(self.drift),
            "motion_weak_frames": self.motion_weak_frames,
            "lost_frames": self.lost_frames,
            "thrifty_frames": self.thrifty_frames,
            "refused_because": self.refused_because,
        }

    def kill_worker(self) -> None:
        """A drill: end the worker the hard way. What follows is the normal failure path. Any thread but the camera's."""
        backend = self._backend
        if backend is not None:
            backend.kill()

    # ---- Effect ---------------------------------------------------------------------------------

    def wants_masks(self, settings: Any) -> bool:
        """Whether this frame's weight will use the masks. A stage that does nothing must not make the app segment."""
        return float(settings.strength) > 0.0 and not self._closed and (self._phase == _RUNNING or self._health > 0.0)

    def start_ahead(self, ctx: FrameContext, settings: Any) -> None:
        """The pipeline calls this when the landmarks and masks are final. Never waits; what it starts may go unused."""
        landmarks, spec = ctx.landmarks, self._spec
        if self._closed or self._sync or landmarks is None or spec is None or self._health <= 0.0 or float(settings.strength) <= 0.0:
            return
        if self._ahead is not None:
            self._ahead.abandoned = True
        backend = self._backend if self._phase == _RUNNING else None
        job = _AheadJob(landmarks, ctx.hair_mask, ctx.face_skin_mask, (ctx.height, ctx.width), spec, backend, self._result, self._min_tag, self._trust_of_masks(ctx, landmarks))
        self._ahead = job
        self._ahead_thread.post(job)

    def apply(self, frame: np.ndarray, ctx: FrameContext, settings: Any) -> np.ndarray:
        started = time.perf_counter()
        now = self._clock()
        strength = min(max(float(settings.strength), 0.0), 1.0)
        self._note_call(settings, strength, now)
        landmarks = ctx.landmarks
        if self._closed or strength <= 0.0 or landmarks is None:
            return frame
        self._want_stage(now)
        if self._sync:
            self._ready_or_off.wait(SYNC_START_TIMEOUT_S)
        backend = self._backend if self._phase == _RUNNING else None
        if backend is None and self._health <= 0.0:
            return frame  # nothing running and nothing to fade out: no work at all, frame untouched
        self._frame(frame, ctx, landmarks, backend, strength, now)
        elapsed = (time.perf_counter() - started) * 1000.0
        self.stage_ms.append(elapsed)
        self._stage_recent.append(elapsed)
        return frame

    def _frame(
        self, frame: np.ndarray, ctx: FrameContext, landmarks: np.ndarray, backend: Backend | None, strength: float, now: float
    ) -> None:
        self._frame_no += 1
        spec = self._spec
        gate = geometry.soft_gate(landmarks, frame.shape)
        working = backend is not None and spec is not None and gate > 0.0 and now >= self._no_jobs_until
        if working and ctx.skipped_frames > 0 and self._health > 0.0 and self._keep_frame_rate:
            self.lost_frames += ctx.skipped_frames
            self._lost.extend([now] * min(ctx.skipped_frames, BUDGET_PAUSE_FRAMES))
        if working and self._thrifty and self._last_sent_frame == self._frame_no - 1:
            self.thrifty_frames += 1
            self._frames.append((now, True))  # no job on purpose: not a job that could not go
        elif working:
            sent = not backend.busy and backend.submit(geometry.extract_crop(frame, landmarks, spec.size), self._frame_no)
            self._frames.append((now, sent))
            if sent:
                self._last_sent_frame = self._frame_no
            if sent and not self._sync:
                self._job_points.append((self._frame_no, self._points_of(landmarks)))
            if self._sync and sent:
                backend.wait_for(self._frame_no, SYNC_RESULT_TIMEOUT_S)
        ahead = self._finished_ahead(landmarks)
        if ahead is None:
            result = self._newest_result(backend)
        else:
            result = self._result = ahead.result
        age = None if result is None else self._frame_no - result.tag
        if age is not None:
            key = min(age, _LAST_AGE_KEPT)
            self.frames_by_age[key] = self.frames_by_age.get(key, 0) + 1
        self._step_health(self._health_target(backend, age, gate, now))
        amount = strength * float(ctx.intensity) * self._health
        if result is None or spec is None or amount <= 0.0:
            return
        if not self._sync and geometry.usable_landmarks(landmarks, frame.shape):
            self._step_motion(landmarks, result.tag, age, spec.temporal_alpha)
            amount *= self._motion.value
            if amount <= 0.0:
                return
        if ahead is not None:
            paste = ahead.paste
            self.pastes_ahead += 1
        elif geometry.usable_landmarks(landmarks, frame.shape):
            paste = _weighted_delta(landmarks, ctx.hair_mask, ctx.face_skin_mask, frame.shape, spec, lambda: result.delta, self._trust_of_masks(ctx, landmarks))
            self.pastes_inline += 1
        else:
            return
        if paste is not None:
            view = frame[paste.y0 : paste.y1, paste.x0 : paste.x1]
            # Where the weight is 0 a pixel stays what it was, bit for bit. Past 0 or 1 is no concern: this is the last
            # step, and the pipeline clips next.
            cv2.scaleAdd(paste.weighted, amount, view, dst=view)

    def _finished_ahead(self, landmarks: np.ndarray) -> _AheadJob | None:
        """This frame's paste if it is ready. Whatever else there is will not be used."""
        job, self._ahead = self._ahead, None
        if job is None:
            return None
        if job.done and not job.failed and job.landmarks is landmarks and job.min_tag == self._min_tag and job.result is not None:
            return job
        job.abandoned = True
        return None

    def reset(self) -> None:
        """The frame size changed or the face is gone: drop the delta and the smoothing, rise from 0 next time."""
        self._result = None
        if self._ahead is not None:
            self._ahead.abandoned = True
            self._ahead = None
        self._health = 0.0
        self._motion = _Ease(MOTION_FALL, MOTION_RISE)
        self._drift = 0.0
        self._job_points.clear()
        self._points_now = (None, None)
        self._mask_trust.reset()
        self._trust_now = (None, 1.0)
        self._min_tag = self._frame_no + 1
        backend = self._backend
        if backend is not None:
            backend.reset()

    def close(self) -> None:
        """Called by ``MakeupPipeline.close``. Returns at once; the worker ends on a daemon thread."""
        self._closed = True
        self._phase = _CLOSED
        self._wake.set()
        self._ready_or_off.set()
        self._ahead_thread.stop()
        backend, self._backend = self._backend, None
        for held in (backend, self._starting):
            if held is not None:
                threading.Thread(target=_close_quietly, args=(held,), name="neural-close", daemon=True).start()

    # ---- camera thread --------------------------------------------------------------------------

    def _note_call(self, section: Any, strength: float, now: float) -> None:
        """Notice the section being switched on again: that gives a stopped stage another chance."""
        gap = None if self._last_apply_at is None else now - self._last_apply_at
        came_back = gap is not None and gap > REARM_GAP_S and section is not self._last_section
        switched_on = strength > 0.0 and not self._was_positive
        if switched_on or came_back:
            self._rearm = True
        if switched_on or (gap is not None and gap > RESUME_GAP_S):
            self.reset()
        self._was_positive = strength > 0.0
        self._last_section = section
        self._last_apply_at = now

    def _want_stage(self, now: float) -> None:
        self._wanted = True
        self._last_wanted_at = now
        if self._phase == _IDLE:
            self._wake.set()
        if self._control is None and self._use_control_thread:
            self._control = threading.Thread(target=self._control_loop, name="neural-control", daemon=True)
            self._control.start()

    def _newest_result(self, backend: Backend | None) -> Result | None:
        """The newest good delta. After a failure, the last one stays for the fade."""
        if backend is not None:
            result = backend.latest()
            if result is not None and result.tag >= self._min_tag:
                self._result = result
        return self._result

    def _health_target(self, backend: Backend | None, age: int | None, gate: float, now: float) -> float:
        if backend is None or age is None or now < self._no_jobs_until:
            return 0.0
        if backend.good_results < self._good_needed:
            return 0.0
        if age >= STALE_AGE and not self._sync:
            return 0.0
        return gate

    def _points_of(self, landmarks: np.ndarray) -> np.ndarray:
        """``geometry.motion_points``, at most once per frame."""
        if self._points_now[0] is not landmarks:
            self._points_now = (landmarks, geometry.motion_points(landmarks))
        return self._points_now[1]

    def _trust_of_masks(self, ctx: FrameContext, landmarks: np.ndarray) -> float:
        """One step per frame, whoever asks first. Offline the masks are made for the frame, and one image must show the full result."""
        if self._sync:
            return 1.0
        if self._trust_now[0] is not landmarks:
            self._trust_now = (landmarks, self._mask_trust.step(landmarks, ctx.timestamp_ms, ctx.masks_timestamp_ms))
        return self._trust_now[1]

    def _step_motion(self, landmarks: np.ndarray, tag: int, age: int, alpha: float) -> None:
        """How well the delta of frame ``tag`` fits these landmarks, and from that ``_motion``."""
        then = next((points for number, points in reversed(self._job_points) if number == tag), None)
        if then is None:  # a job from before a reset; nothing to compare with
            return
        drift = geometry.paste_drift(self._points_of(landmarks), then)
        if age > 0:  # smoothing holds earlier frames too: on average (1 - alpha) / alpha frames older still
            drift *= (age + (1.0 - alpha) / alpha) / age
        self._drift += MOTION_SMOOTHING * (drift - self._drift)
        if self._motion.step(1.0 - geometry._ramp(self._drift, *MOTION_FIT)) >= 0.5:
            self.drift.append(drift)
        else:
            self.motion_weak_frames += 1

    def _step_health(self, target: float) -> None:
        if self._sync:
            self._health = target  # offline: one image must show the model's full result
        elif target > self._health:
            self._health = min(target, self._health + HEALTH_RISE)
        elif target < self._health:
            if target <= 0.0 and not self._falling:
                self.health_falls += 1
            self._health = max(target, self._health - HEALTH_FALL)
        self._falling = target < self._health

    # ---- control thread -------------------------------------------------------------------------

    def _control_loop(self) -> None:
        while not self._closed:
            try:
                self._control_step(self._clock())
            except Exception:  # this thread must outlive any bug here: the stage just fades
                logger.exception("the neural stage's control step failed")
                backend = self._backend
                if backend is not None:
                    self._drop_backend(backend)
                self._set_failed(STATUS_FAILED, None)
            self._wake.wait(_CONTROL_POLL_S)
            self._wake.clear()

    def _control_step(self, now: float) -> None:
        phase = self._phase
        if self._rearm:
            self._rearm = False
            if phase in (_OFF, _FAILED):
                self._failures.clear()
                self._ever_ready = False
                self._ready_or_off.clear()
                self._phase = phase = _IDLE
        if phase == _IDLE:
            if self._wanted:
                self._start_backend(now, restarted=False)
        elif phase == _RUNNING:
            self._watch(now)
        elif phase == _WAITING and now >= self._restart_at:
            self.restarts += 1
            self._start_backend(now, restarted=True)

    def _start_backend(self, now: float, restarted: bool) -> None:
        self._phase = _STARTING
        self._status = STATUS_PREPARING
        if self._backend_factory is None and not _onnxruntime_available():
            self._set_off(STATUS_NO_RUNTIME, "onnxruntime is not installed")
            return
        try:
            spec = self._load_spec()
        except ModelRefused as exc:
            self._set_off(exc.status, str(exc))
            return
        backend = self._make_backend(spec)
        self._starting = backend
        try:
            if self._closed:  # ``close`` may have looked just before ``_starting`` was set
                _close_quietly(backend)
                return
            backend.start()
        except BackendError as exc:
            _close_quietly(backend)
            self._on_start_failed(now, exc)
            return
        except Exception as exc:  # e.g. no shared memory, no process: the same as a worker that would not start
            _close_quietly(backend)
            self._on_start_failed(now, BackendError(repr(exc)))
            return
        finally:
            self._starting = None
        if self._closed:
            _close_quietly(backend)
            return
        self._spec = spec
        self._good_needed = GOOD_RESULTS_AFTER_RESTART if restarted else 1
        self._ever_ready = True
        self._backend = backend
        self._status = STATUS_RUNNING
        self._phase = _RUNNING
        self._ready_or_off.set()

    def _load_spec(self) -> ModelSpec:
        folder = self._model_dir if self._model_dir is not None else model_files.active_model_dir()
        spec = model_files.load_spec(folder)
        model_files.check_gate(spec, self._allow_unevaluated)
        if self._allow_unevaluated:
            logger.warning("unevaluated model in use (offline only): %s-%s", spec.name, spec.version)
        return spec

    def _make_backend(self, spec: ModelSpec) -> Backend:
        if self._backend_factory is not None:
            return self._backend_factory(spec)
        return ProcessBackend(spec, provider=self._provider, clock=self._clock)

    def _on_start_failed(self, now: float, exc: BackendError) -> None:
        if self._closed:
            return  # ``close`` ended it on purpose
        logger.warning("the neural worker did not start: %s", exc)
        if exc.code == worker.CODE_NOT_DML and not self._ever_ready:
            self._set_off(STATUS_NO_GPU, str(exc))
        elif exc.code in (worker.CODE_LOAD_ERROR, worker.CODE_BAD_MODEL):
            self._set_failed(STATUS_FAILED, str(exc))  # the same file will not load any better later
        else:
            self._on_failure(now, str(exc), exc.code)

    def _watch(self, now: float) -> None:
        backend = self._backend
        if backend is None:
            return
        last = self._last_wanted_at
        if last is not None and now - last > IDLE_STOP_S:
            self._stop_idle(backend)
            return
        if backend.failure is not None:
            why, code = backend.failure
            self._drop_backend(backend)
            self._on_failure(now, why, code)
            return
        since = backend.in_flight_since()
        if since is not None and now - since > (SYNC_RESULT_TIMEOUT_S if self._sync else WORKER_SILENT_S):
            self._drop_backend(backend)
            self._on_failure(now, "no answer from the worker", None)
            return
        self._check_overload(backend, now)

    def _stop_idle(self, backend: Backend) -> None:
        self._wanted = False
        self._drop_backend(backend)
        self._status = None
        self._ready_or_off.clear()
        self._phase = _IDLE

    def _drop_backend(self, backend: Backend) -> None:
        self._backend = None
        backend.kill()
        threading.Thread(target=_close_quietly, args=(backend,), name="neural-close", daemon=True).start()

    def _on_failure(self, now: float, why: str, code: int | None) -> None:
        """Go to restart: waits of 5, 30 and 120 s. Those three used up within 10 minutes: stop."""
        logger.warning("the neural stage failed: %s", why)
        self._frames.clear()  # frames lost to the failure say nothing about how fast the next worker is
        self._failures.append(now)
        while self._failures and now - self._failures[0] > RESTART_WINDOW_S:
            self._failures.popleft()
        count = len(self._failures)
        if count > len(RESTART_WAITS_S):
            self._set_failed(STATUS_FAILED, why)
            return
        self._restart_at = now + RESTART_WAITS_S[count - 1]
        self._status = STATUS_NO_GPU if code == worker.CODE_NOT_DML else STATUS_PREPARING
        self._phase = _WAITING

    def _check_overload(self, backend: Backend, now: float) -> None:
        if self._sync or now < self._no_jobs_until:
            return
        if self._status == STATUS_OVERLOAD:
            self._status = STATUS_RUNNING
        if self._last_overload_at is not None and now - self._last_overload_at > OVERLOAD_FORGIVE_S:
            self._overload_pause = OVERLOAD_PAUSE_S
            self._last_overload_at = None
        frames = _snapshot(self._frames)
        recent = [sent for when, sent in frames if now - when <= OVERLOAD_WINDOW_S]
        if len(recent) < OVERLOAD_MIN_FRAMES:
            return
        missed = recent.count(False)
        slow = sum(1 for when in _snapshot(backend.slow_results) if now - when <= OVERLOAD_WINDOW_S)
        stage_p95 = _p95(_snapshot(self._stage_recent))
        over_budget = self._budget_broken(now)
        if over_budget or (missed + slow) / len(recent) > OVERLOAD_SHARE or stage_p95 > STAGE_OVER_BUDGET_MS:
            self.overloads += 1
            self._no_jobs_until = now + self._overload_pause
            self._last_overload_at = now
            self._overload_pause = min(self._overload_pause * 2.0, OVERLOAD_MAX_PAUSE_S)
            self._frames.clear()
            self._lost.clear()
            self._stage_recent.clear()
            backend.slow_results.clear()
            self._status = STATUS_OVERLOAD
            logger.warning(
                "the neural stage is overloaded (%d of %d frames, %.1f ms on the camera thread)", missed + slow, len(recent), stage_p95
            )

    def _budget_broken(self, now: float) -> bool:
        """Too many frames lost: first a job on every second frame only. Still too many after that: True."""
        lost = _snapshot(self._lost)
        if sum(1 for when in lost if now - when <= OVERLOAD_WINDOW_S) >= BUDGET_LOST_FRAMES:
            self._last_budget_miss_at = now
            if not self._thrifty:
                self._thrifty = True
                self._lost.clear()  # what comes next shows whether that was enough
                logger.info("the neural stage costs frames: a job on every second frame from now on")
                return False
        elif self._thrifty and now - self._last_budget_miss_at > THRIFTY_HOLD_S:
            self._thrifty = False
        return self._thrifty and sum(1 for when in lost if now - when <= BUDGET_PAUSE_WINDOW_S) >= BUDGET_PAUSE_FRAMES

    def _set_off(self, status: str, why: str) -> None:
        logger.info("the neural stage stays off: %s", why)
        self.refused_because = why
        self._status = status
        self._phase = _OFF
        self._ready_or_off.set()

    def _set_failed(self, status: str, why: str | None) -> None:
        logger.error("the neural stage stopped: %s", why)
        self._status = status
        self._phase = _FAILED
        self._ready_or_off.set()


def _close_quietly(backend: Backend) -> None:
    try:
        backend.close()
    except Exception:
        logger.exception("closing the neural backend failed")


def _p95(values: list[float]) -> float:
    return sorted(values)[int(len(values) * 0.95)] if values else 0.0


def _snapshot(items: Any) -> list[Any]:
    """A deque that another thread appends to, as a list."""
    for _ in range(5):
        try:
            return list(items)
        except RuntimeError:  # mutated during iteration
            continue
    return []

