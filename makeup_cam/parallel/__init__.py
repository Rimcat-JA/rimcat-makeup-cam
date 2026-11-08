"""Drawing in a pipeline of processes, so that every camera frame reaches the virtual camera (``docs/dev/contract-fps.md``).

    main process      camera, tracker, segmentation, the plan of each frame; to float in shared memory
    stage 0, 1, 2     the effects, in order, each stage in a process of its own
    sender            mirror, BGR to NV12, one write per place on an exact grid

This module stays light: the stages and the sender import it, and the rest of the app asks it whether to use the engine.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from typing import Any

SWITCH_ENV = "AI_MAKEUP_CAM_PARALLEL"
LOG_ENV = "AI_MAKEUP_CAM_PARALLEL_LOG"  # measurements only: a folder for the logs of all processes
MIN_PHYSICAL_CORES = 6

# What the status line says (contract section 8).
FALLBACK_MESSAGE = "並列処理が止まったため、1 つのプロセスで処理を続けています（フレームが落ちることがあります）。"
FALLBACK_FINAL_MESSAGE = "並列処理を再開できませんでした。1 つのプロセスで処理を続けています（フレームが落ちることがあります）。アプリを再起動すると元に戻ります。"
OVERLOAD_MESSAGE = "CPU が足りないため、1 つのプロセスで処理を続けています（フレームが落ちることがあります）。ほかのアプリを閉じてアプリを再起動すると元に戻ります。"
VCAM_REOPEN_MESSAGE = "仮想カメラを開き直せません。通話アプリのカメラを一度オフにして、オンに戻してください。"
SLOW_MESSAGE = "処理が遅れているため、一部のフレームが落ちています。"

# A stage ends after these effects; the rest of the effects form the last stage (contract section 3.2).
STAGE_ENDS = ("hairshape", "contour")

# The masks each effect reads (contract section 4.3). A stage hands on what a later stage's effect reads and nothing
# else, so a mask missing here would change the picture silently: test_parallel_identity checks this against the code.
MASK_READS: dict[str, tuple[str, ...]] = {
    "reshape": ("hair_mask", "face_skin_mask", "body_skin_mask"),
    "hairshape": ("hair_mask", "face_skin_mask", "body_skin_mask"),
    "hair": ("hair_mask", "face_skin_mask"),
    "skin": ("hair_mask", "face_skin_mask"),
    "contour": ("hair_mask", "face_skin_mask"),
    "blush": ("face_skin_mask",),
    "neural": ("hair_mask", "face_skin_mask"),
}
# The context fields an effect may write (contract section 4.3); these cross to the next stage.
CONTEXT_WRITES: dict[str, tuple[str, ...]] = {
    "reshape": ("landmarks", "hair_mask", "face_skin_mask", "body_skin_mask"),
    "hairshape": ("hair_mask",),
}


def plan_stages(names_in_order: Iterable[str]) -> list[list[str]]:
    """The effect names of each stage. An effect added later joins the stage of the effect it follows."""
    stages: list[list[str]] = [[]]
    for name in names_in_order:
        stages[-1].append(name)
        if name in STAGE_ENDS:
            stages.append([])
    return stages


def physical_cores() -> int:
    """How many CPU cores this process may use; 0 if unknown."""
    if sys.platform != "win32":
        return 0
    import ctypes

    from ..pipeline import _allowed_threads, _threads_of_each_core

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        allowed = _allowed_threads(kernel32)
        return sum(1 for threads in _threads_of_each_core(kernel32) if threads & allowed)
    except (OSError, AttributeError, ValueError):
        return 0


def enabled(no_parallel: bool = False) -> bool:
    """Whether to draw in the stages. ``AI_MAKEUP_CAM_PARALLEL=0`` or ``--no-parallel`` is exactly the single-process path."""
    if no_parallel or sys.platform != "win32":  # kernel semaphores and job objects: only measured on Windows
        return False
    value = os.environ.get(SWITCH_ENV, "").strip()
    if value == "0":
        return False
    return value == "1" or physical_cores() >= MIN_PHYSICAL_CORES


def attach_engine(pipeline: Any, *, no_parallel: bool = False) -> Any:
    """Give ``pipeline`` an engine if the switch says so. Only the window and ``--headless`` call this. Returns it, or None."""
    attach = getattr(pipeline, "attach_engine", None)  # not every pipeline can: a stand-in need not
    if attach is None or not enabled(no_parallel):
        return None
    from .engine import ParallelEngine

    engine = ParallelEngine()
    attach(engine)
    return engine
