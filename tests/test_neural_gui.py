"""What the GUI and the command line say about the learned finishing stage. Fake runner, offscreen."""

from __future__ import annotations

from collections.abc import Callable

from test_gui import Harness, isolated_dirs, make_window, qapp  # noqa: F401 (fixtures)

from makeup_cam.__main__ import format_status
from makeup_cam.gui import SECTION_GROUPS
from makeup_cam.runner import RunnerStatus


def _show(harness: Harness, status: RunnerStatus) -> None:
    harness.window._status_relay.post(status)
    harness.window._on_status_ready()


def test_the_status_bar_shows_the_finish_only_when_there_is_something_to_say(make_window: Callable[..., Harness]) -> None:  # noqa: F811
    harness = make_window()
    label = harness.window._neural_status
    assert label.isHidden()
    _show(harness, RunnerStatus(neural_message="モデルがありません"))
    assert not label.isHidden()
    assert "モデルがありません" in label.text()
    _show(harness, RunnerStatus(neural_message=None))
    assert label.isHidden()


def test_the_section_has_its_own_group_in_the_settings() -> None:
    assert any("neural" in sections for _, sections in SECTION_GROUPS)


def test_the_command_line_status_mentions_the_finish_only_when_there_is_something_to_say() -> None:
    assert "仕上げ" not in format_status(RunnerStatus(face_found=True), virtual_camera_enabled=False)
    line = format_status(RunnerStatus(face_found=True, neural_message="動作中"), virtual_camera_enabled=False)
    assert line.endswith("仕上げ 動作中")
