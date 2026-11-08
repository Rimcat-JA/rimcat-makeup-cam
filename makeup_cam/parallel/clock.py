"""The place of each frame on the output (contract section 6).

The OBS filter shows the newest write that has ended at each of its exact ticks, so a frame is lost when two writes end
between two ticks. So once the load has settled, each frame gets a place of its own on a grid of the same period as the
filter's ticks, on the same clock. The phase of that grid never moves: a clock error of the camera costs one repeat (a
slip) or one drop per crossing and nothing more. Pure arithmetic, so that it can be tested with a fake camera and a fake clock.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

BURST_S = 0.6  # frames go out as they come: the first frames of new stages build their tables and arrive in a heap
SETTLE_S = 1.5  # from the first frame to the grid, if nothing disturbs the window between BURST_S and this
SETTLE_LATEST_S = 6.0  # disturbed all the time: the grid starts now, with what the last window showed
DISTURBED_S = 0.020  # a frame this far from the window's median is no sign of the load: the window starts over
GUARD_S = 0.009  # time reserved for the conversion (p99 8.6 ms on the real app)
MARGIN_S = 0.015  # spare, for spikes of the stages
SPARE_FRAMES = 3  # frames of delay allowed above the settled one
WINDOW = 30  # frames
CAMERA_PERIOD_S = 1.0 / 30.0
_FEWEST_TO_JUDGE = 5  # frames in the window before one of them can be called far from the rest
# A camera this far from the expected period is a camera of another rate, not one with a clock error.
_OTHER_RATE = 0.01


def declared_interval(fps: float) -> int:
    """The period OBS's queue declares, in units of 100 ns: pyvirtualcam truncates 1e7 / fps."""
    return int(1e7 / fps)


@dataclass(frozen=True)
class Place:
    """Where one frame goes. ``due`` None: as it comes (the first frames). ``drop``: it does not go at all."""

    due: float | None = None
    drop: bool = False
    slips: int = 0


class OutputClock:
    """One per camera clock. ``place`` is called once for each camera frame, in order."""

    def __init__(
        self,
        period_s: float,
        *,
        camera_period_s: float = CAMERA_PERIOD_S,
        burst_s: float = BURST_S,
        settle_s: float = SETTLE_S,
        settle_latest_s: float = SETTLE_LATEST_S,
        disturbed_s: float = DISTURBED_S,
        guard_s: float = GUARD_S,
        margin_s: float = MARGIN_S,
        spare_frames: int = SPARE_FRAMES,
        window: int = WINDOW,
    ) -> None:
        self.period_s = period_s
        self.camera_period_s = camera_period_s
        self.burst_s = burst_s
        self.settle_s = settle_s
        self.settle_latest_s = settle_latest_s
        self.disturbed_s = disturbed_s
        self.guard_s = guard_s
        self.margin_s = margin_s
        self.spare_frames = spare_frames
        self.first_finished: float | None = None
        self.settling: list[tuple[int, float, float]] = []
        self.grid0: float | None = None
        self.k = 0
        self.slips = 0
        self.drops = 0
        self.delay_s: float | None = None  # what the settling showed, margin included
        self._on_grid = False
        self._paced: float | None = None  # the last place given while settling
        self._slack: deque[float] = deque(maxlen=window)

    @property
    def settled(self) -> bool:
        return self._on_grid

    def due(self) -> float:
        """The current place."""
        assert self.grid0 is not None
        return self.grid0 + self.k * self.period_s

    def renew(self) -> None:
        """New stages, or the camera opened again: what the old load showed says nothing about the new one. The delay is
        measured again, and the phase of the grid stays."""
        self._on_grid = False
        self.first_finished = None
        self.settling = []
        self._paced = None
        self._slack.clear()

    def place(self, seq: int, t_camera: float, t_finished: float, now: float) -> Place:
        if self._on_grid:
            self.k += 1
        else:
            if self.first_finished is None:
                self.first_finished = t_finished
            since = t_finished - self.first_finished
            if since <= self.burst_s:
                return Place()
            # Frames that wait for the sender are a heap to clear first: held at one per period it would stay in the delay.
            if self._disturbs(seq, t_finished) or now - t_finished > self.period_s:
                self.settling = []
                self._paced = None
                if since <= self.settle_latest_s:
                    return Place()
            self.settling.append((seq, t_camera, t_finished))
            window_began = self.settling[0][2]
            if t_finished - window_began < self.settle_s - self.burst_s and since <= self.settle_latest_s:
                return Place(self._pace(now))
            self._settle(seq, now)
        due = self.due()
        if now > due - self.guard_s:
            slips = self._catch_up(now + self.guard_s)
            return Place(self.due(), slips=slips)
        self._slack.append(due - t_finished)
        spare = self.guard_s + self.margin_s + self.spare_frames * self.period_s
        if len(self._slack) == self._slack.maxlen and min(self._slack) > spare:
            # More than the spare has been held for a whole window: the camera outruns the filter, or a long stall has
            # passed. The reader cannot take more than one frame per tick, so one goes; the delay stays bounded.
            self.drops += 1
            self.k -= 1
            self._slack.clear()
            return Place(drop=True)
        return Place(due)

    def late_after_convert(self, now: float) -> float:
        """The conversion ended after the frame's place: the next place that can still be met."""
        if not self._on_grid:
            self._paced = now
            return now
        self._catch_up(now)
        return self.due()

    def _pace(self, now: float) -> float:
        """Before the grid: never two writes within one period, as a frame would be lost between two ticks. Only ever later."""
        earliest = now + self.guard_s
        self._paced = earliest if self._paced is None else max(self._paced + self.period_s, earliest)
        return self._paced

    def _disturbs(self, seq: int, t_finished: float) -> bool:
        rows = self.settling
        if len(rows) < _FEWEST_TO_JUDGE:
            return False
        period = self._camera_period(rows)
        lateness = sorted(finished - s * period for s, _, finished in rows)
        return abs(t_finished - seq * period - lateness[len(lateness) // 2]) > self.disturbed_s

    def _catch_up(self, earliest: float) -> int:
        assert self.grid0 is not None
        k = math.ceil((earliest - self.grid0) / self.period_s)
        slips = k - self.k
        self.k = k
        self.slips += slips
        self._slack.clear()
        return slips

    def _settle(self, seq: int, now: float) -> None:
        rows = self.settling
        period = self._camera_period(rows)
        offsets = sorted(t_camera - s * period for s, t_camera, _ in rows)
        phase = offsets[len(offsets) // 2]
        self.delay_s = max(t_finished - (s * period + phase) for s, _, t_finished in rows) + self.guard_s + self.margin_s
        place = max(phase + seq * period + self.delay_s, now + self.guard_s)
        if self._paced is not None:
            place = max(place, self._paced + self.period_s)
        if self.grid0 is None:
            self.grid0 = place
            self.k = 0
        else:
            self.k = math.ceil((place - self.grid0) / self.period_s)
        self.settling = []
        self._paced = None
        self._on_grid = True

    def _camera_period(self, rows: list[tuple[int, float, float]]) -> float:
        """1/30 s, unless this camera clearly runs at another rate: then the formula would count its drift as delay."""
        if len(rows) < 2 or rows[-1][0] == rows[0][0]:
            return self.camera_period_s
        measured = (rows[-1][1] - rows[0][1]) / (rows[-1][0] - rows[0][0])
        return self.camera_period_s if abs(measured / self.camera_period_s - 1.0) < _OTHER_RATE else measured
