"""What the window says about the pipeline of processes. Fake runner, offscreen."""

from __future__ import annotations

from collections.abc import Callable

from test_gui import Harness, isolated_dirs, make_window, qapp  # noqa: F401 (fixtures)

from makeup_cam import parallel
from makeup_cam.runner import PROCESSING_FAILED_MESSAGE, RunnerStatus


def _show(harness: Harness, status: RunnerStatus) -> None:
    harness.window._status_relay.post(status)
    harness.window._on_status_ready()


def test_the_status_bar_says_what_happened_in_full(make_window: Callable[..., Harness]) -> None:  # noqa: F811
    harness = make_window()
    label = harness.window._processing_status
    for message in (parallel.FALLBACK_MESSAGE, parallel.FALLBACK_FINAL_MESSAGE, parallel.OVERLOAD_MESSAGE, parallel.SLOW_MESSAGE):
        _show(harness, RunnerStatus(processing_error=message))
        assert not label.isHidden()
        assert label.text() == message
    _show(harness, RunnerStatus())
    assert label.isHidden()


def test_a_real_processing_error_is_still_shown_as_an_error(make_window: Callable[..., Harness]) -> None:  # noqa: F811
    harness = make_window()
    _show(harness, RunnerStatus(processing_error=PROCESSING_FAILED_MESSAGE))
    assert "エラー" in harness.window._processing_status.text()
    assert harness.window._processing_status.toolTip() == PROCESSING_FAILED_MESSAGE


def test_the_status_bar_counts_lost_frames(make_window: Callable[..., Harness]) -> None:  # noqa: F811
    harness = make_window()
    _show(harness, RunnerStatus(fps=30.0))
    assert "落ちた" not in harness.window._fps_status.text()
    _show(harness, RunnerStatus(fps=30.0, lost_frames=3))
    assert "落ちたフレーム 3" in harness.window._fps_status.text()


def test_the_status_bar_says_the_delay(make_window: Callable[..., Harness]) -> None:  # noqa: F811
    harness = make_window()
    _show(harness, RunnerStatus(processing_ms=12.0))
    assert "遅延" not in harness.window._time_status.text()
    _show(harness, RunnerStatus(processing_ms=12.0, output_delay_ms=91.4))
    assert "遅延 91 ms" in harness.window._time_status.text()
