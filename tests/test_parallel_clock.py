"""The output clock (contract section 6) with a fake camera and a fake clock, and the conversion to NV12."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from makeup_cam.parallel.clock import (
    BURST_S,
    DISTURBED_S,
    GUARD_S,
    MARGIN_S,
    SETTLE_LATEST_S,
    SETTLE_S,
    SPARE_FRAMES,
    WINDOW,
    OutputClock,
    declared_interval,
)
from makeup_cam.parallel.nv12 import NV12Buffer, to_nv12

P = declared_interval(30) * 1e-7
RENDER_S = 0.050


@dataclass
class Written:
    seq: int
    due: float | None
    at: float


class Rig:
    """A camera on its own clock, frames finished ``RENDER_S`` later, and a sender that deals with one frame at a time."""

    def __init__(self, ppm: float = 0.0, render_s: float = RENDER_S) -> None:
        self.clock = OutputClock(P)
        self.camera_period = (1.0 / 30.0) / (1.0 + ppm * 1e-6)
        self.render_s = render_s
        self.seq = 0
        self.t_camera = 100.0
        self.busy_until = 0.0
        self.written: list[Written] = []
        self.dropped: list[int] = []

    def frame(self, extra_render_s: float = 0.0, *, gap_s: float = 0.0) -> None:
        self.seq += 1
        self.t_camera += self.camera_period + gap_s
        finished = self.t_camera + self.render_s + extra_render_s
        now = max(finished, self.busy_until)
        place = self.clock.place(self.seq, self.t_camera, finished, now)
        if place.drop:
            self.dropped.append(self.seq)
            return
        at = now + 0.006 if place.due is None else place.due
        self.busy_until = at
        self.written.append(Written(self.seq, place.due, at))

    def run(self, seconds: float) -> None:
        for _ in range(int(seconds * 30)):
            self.frame()

    def settled(self) -> Rig:
        self.run(SETTLE_S + 1.0)
        assert self.clock.settled
        return self

    @property
    def delay_s(self) -> float:
        return self.written[-1].at - self.t_camera


def test_the_period_is_the_declared_interval_and_not_a_thirtieth() -> None:
    assert declared_interval(30) == 333333
    assert P == pytest.approx(0.0333333)
    assert P != pytest.approx(1.0 / 30.0, abs=1e-8)


def test_only_the_first_frames_go_as_they_come() -> None:
    rig = Rig()
    rig.run(BURST_S - 0.1)
    assert not rig.clock.settled
    assert all(w.due is None for w in rig.written)


def test_from_the_end_of_the_burst_no_two_writes_are_within_one_period() -> None:
    """A reader opened right after the burst loses nothing: a frame is lost only when two writes end between two ticks."""
    rig = Rig()
    wobble = np.random.default_rng(5).uniform(0.0, 0.0156, 90)  # the file camera's hand-over
    for i in range(90):
        rig.frame(float(wobble[i]))
    assert rig.clock.settled
    placed = [w.at for w in rig.written if w.due is not None]
    assert len(placed) >= 90 - int(BURST_S * 30) - 2
    assert all(b - a >= P * 0.999 for a, b in zip(placed, placed[1:], strict=False))


def test_the_grid_starts_soon_after_the_burst() -> None:
    rig = Rig()
    rig.run(SETTLE_S + 0.1)
    assert rig.clock.settled


def test_a_spike_while_settling_does_not_stay_in_the_delay() -> None:
    """One stall of 0.3 s in the window: the delay is what the load needs, not what the stall took."""
    rig = Rig()
    rig.run(BURST_S + 0.4)
    for _ in range(9):
        rig.frame(0.3)
    rig.run(SETTLE_S + 0.5)
    assert rig.clock.settled
    assert rig.clock.delay_s == pytest.approx(RENDER_S + GUARD_S + MARGIN_S, abs=1e-6)
    rig.run(10.0)
    assert rig.clock.drops == 0


def test_a_load_that_never_calms_down_still_gets_a_grid() -> None:
    rig = Rig()
    wobble = np.random.default_rng(6).uniform(0.0, DISTURBED_S * 4, int(SETTLE_LATEST_S * 30) + 5)
    for extra in wobble:
        rig.frame(float(extra))
    assert rig.clock.settled


def test_a_late_conversion_while_settling_moves_the_pace_only() -> None:
    rig = Rig()
    rig.run(BURST_S + 0.2)
    assert not rig.clock.settled
    assert rig.clock.late_after_convert(200.0) == 200.0
    assert rig.clock.slips == 0
    assert rig.clock._pace(150.0) == pytest.approx(200.0 + P)


def test_renewing_measures_the_delay_again_and_keeps_the_phase() -> None:
    """New stages: their first frames are slow, and that costs neither slips nor drops nor a lasting delay."""
    rig = Rig().settled()
    grid0, slips = rig.clock.grid0, rig.clock.slips
    rig.clock.renew()
    assert not rig.clock.settled
    for _ in range(6):
        rig.frame(0.2)
    rig.render_s = 0.030
    rig.run(SETTLE_S + 0.5)
    assert rig.clock.settled
    assert rig.clock.grid0 == grid0
    assert rig.clock.slips == slips
    assert rig.clock.drops == 0
    assert rig.clock.delay_s == pytest.approx(0.030 + GUARD_S + MARGIN_S, abs=1e-6)
    phase = (rig.clock.due() - grid0) / P
    assert phase == pytest.approx(round(phase), abs=1e-6)
    assert rig.delay_s < 0.030 + GUARD_S + MARGIN_S + P * 1.01


def test_the_delay_holds_the_worst_frame_of_the_settling_plus_the_margins() -> None:
    rig = Rig()
    for i in range(int(SETTLE_S * 30) + 5):
        rig.frame(0.010 if i == int(SETTLE_S * 30) - 10 else 0.0)  # one slow frame in the second half
    assert rig.clock.delay_s == pytest.approx(RENDER_S + 0.010 + GUARD_S + MARGIN_S, abs=1e-6)


def test_a_locked_camera_gets_one_place_per_frame_and_neither_slip_nor_drop() -> None:
    rig = Rig(ppm=0.0).settled()
    k_before = rig.clock.k
    rig.run(60.0)
    assert rig.clock.k - k_before == 1800
    assert rig.clock.slips == 0
    assert rig.clock.drops == 0


def test_the_grid_values_are_exactly_grid0_plus_k_p() -> None:
    rig = Rig(ppm=-1000.0).settled()
    before = len(rig.written)
    rig.run(120.0)
    grid0 = rig.clock.grid0
    placed = [w for w in rig.written[before:] if w.due is not None]
    assert grid0 is not None and placed
    for written in placed:
        k = (written.due - grid0) / P
        assert k == pytest.approx(round(k), abs=1e-6)


def test_at_most_one_write_per_place() -> None:
    rig = Rig(ppm=-1000.0).settled()
    rig.run(120.0)
    dues = [w.due for w in rig.written if w.due is not None]
    assert all(b - a >= P * 0.999 for a, b in zip(dues, dues[1:], strict=False))


def test_a_camera_1000_ppm_slow_gives_one_slip_per_33_seconds_and_no_drop() -> None:
    rig = Rig(ppm=-1000.0).settled()
    before = rig.clock.slips
    rig.run(330.0)
    assert 9 <= rig.clock.slips - before <= 11
    assert rig.clock.drops == 0


def test_a_camera_1000_ppm_fast_drops_at_the_expected_rate_with_the_delay_bounded() -> None:
    rig = Rig(ppm=1000.0).settled()
    settled_delay = rig.delay_s
    rig.run(330.0)
    bound = SPARE_FRAMES * P + 2 * (GUARD_S + MARGIN_S) + WINDOW * P * 0.05
    assert rig.delay_s - settled_delay < bound
    first_needs = (SPARE_FRAMES * P - 0.0) / (1e-3 * (1.0 / 30.0)) / 30.0  # seconds until the spare has filled up
    expected = (330.0 - first_needs) / 33.3
    assert expected * 0.8 <= rig.clock.drops <= expected * 1.2 + 1
    assert rig.clock.slips == 0


def test_a_spike_gives_one_slip_and_no_drop_and_a_second_one_more() -> None:
    rig = Rig().settled()
    delay = rig.delay_s
    rig.frame(0.030)
    rig.run(1.0)
    assert (rig.clock.slips, rig.clock.drops) == (1, 0)
    rig.run(0.5)
    rig.frame(0.065)
    rig.run(1.0)
    assert rig.clock.slips == 2
    assert rig.clock.drops == 0
    assert rig.delay_s - delay < SPARE_FRAMES * P + GUARD_S + MARGIN_S + 0.1


def test_after_a_long_stall_the_delay_returns_within_the_bound_through_drops_one_per_window() -> None:
    rig = Rig().settled()
    delay = rig.delay_s
    for _ in range(12):
        rig.frame(0.4)  # the stages stood still: everything is late and the grid moves on by whole periods
    assert rig.delay_s - delay > SPARE_FRAMES * P
    rig.run(8.0)
    assert rig.delay_s - delay <= SPARE_FRAMES * P + GUARD_S + MARGIN_S + P
    assert rig.clock.drops >= 1
    drops = np.diff(rig.dropped)
    assert (drops >= WINDOW).all()


def test_a_camera_that_returns_catches_up_through_the_slip_rule() -> None:
    rig = Rig().settled()
    k = rig.clock.k
    grid0 = rig.clock.grid0
    rig.frame(gap_s=5.0)  # the placeholders went out off the grid meanwhile
    assert rig.clock.grid0 == grid0
    assert rig.clock.k > k + 100
    assert rig.clock.slips > 100
    assert rig.clock.drops == 0
    rig.run(5.0)
    assert rig.clock.drops == 0


def test_a_slower_preset_may_slip_once_and_the_phase_stays() -> None:
    rig = Rig().settled()
    grid0 = rig.clock.grid0
    rig.render_s += 0.040
    rig.run(10.0)
    assert 1 <= rig.clock.slips <= 3
    assert rig.clock.drops == 0
    assert rig.clock.grid0 == grid0


def test_a_late_conversion_goes_to_the_next_place_and_never_as_soon_as_possible() -> None:
    rig = Rig().settled()
    due = rig.clock.due()
    later = rig.clock.late_after_convert(due + P * 0.5)
    assert later == pytest.approx(due + P)
    assert rig.clock.slips == 1


def test_a_camera_of_another_rate_is_not_taken_for_delay() -> None:
    """A 15 fps camera, whose frames would otherwise seem to be late by up to half the settling window."""
    rig = Rig()
    rig.camera_period = 1.0 / 15.0
    rig.run(SETTLE_S * 2 + 1)
    assert rig.clock.settled
    assert rig.clock.delay_s is not None
    assert rig.clock.delay_s < RENDER_S + GUARD_S + MARGIN_S + 0.01
    assert rig.clock.drops == 0
    assert rig.delay_s < RENDER_S + GUARD_S + MARGIN_S + 2 * P


# ---- the conversion ---------------------------------------------------------------------


def _reference(bgr: np.ndarray) -> np.ndarray:
    """The formula of contract section 2.3, in plain Python integers on arrays, without any table or cv2 call."""
    b, g, r = (bgr[..., i].astype(np.int64) for i in range(3))
    h, w = b.shape
    y = (66 * r + 130 * g + 26 * b + 0x1000) >> 8

    def half(c: np.ndarray) -> np.ndarray:
        s = (c[0::2] + c[1::2] + 1) >> 1
        return (s[:, 0::2] + s[:, 1::2] + 1) >> 1

    rb, gb, bb = half(r), half(g), half(b)
    uv = np.empty((h // 2, w), np.int64)
    uv[:, 0::2] = (112 * bb - 74 * gb - 38 * rb + 0x8000) >> 8
    uv[:, 1::2] = (112 * rb - 94 * gb - 18 * bb + 0x8000) >> 8
    return np.concatenate([y.ravel(), uv.ravel()]).astype(np.uint8)


@pytest.mark.parametrize(
    "frame",
    [
        np.random.default_rng(1).integers(0, 256, (72, 128, 3), np.uint8),
        np.random.default_rng(2).integers(0, 256, (720, 1280, 3), np.uint8),
        np.zeros((72, 128, 3), np.uint8),
        np.full((72, 128, 3), 255, np.uint8),
        np.full((72, 128, 3), (1, 254, 127), np.uint8),
        np.tile(np.arange(128, dtype=np.uint8)[None, :, None], (72, 1, 3)) * 2 + 1,
    ],
    ids=["random", "random-720p", "black", "white", "odd-values", "ramp"],
)
def test_to_nv12_equals_the_formula(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    buffer = NV12Buffer(w, h)
    buffer.fill(frame)
    assert np.array_equal(buffer.data, _reference(frame))


def test_to_nv12_writes_only_the_planes_it_is_given() -> None:
    frame = np.random.default_rng(3).integers(0, 256, (72, 128, 3), np.uint8)
    y = np.full((72, 128), 7, np.uint8)
    uv = np.full((36, 128), 7, np.uint8)
    original = frame.copy()
    to_nv12(frame, y, uv)
    assert np.array_equal(frame, original)
    assert not np.array_equal(y, np.full_like(y, 7))
