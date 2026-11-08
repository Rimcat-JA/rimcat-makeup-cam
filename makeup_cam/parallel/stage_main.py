"""One stage of effects in its own process. ``python -m makeup_cam.parallel.stage_main JSON``. It ends with the main process.

It takes each frame in order, draws its effects in the shared picture, hands on what later stages read, and passes the
slot on. Nothing here decides anything about the picture: settings, intensity and resets arrive with each frame.
"""

from __future__ import annotations

import gc
import json
import os
import signal
import sys
import time
import traceback
from typing import Any

MASK_FIELDS = ("hair_mask", "face_skin_mask", "body_skin_mask")


def main() -> None:
    config: dict[str, Any] = json.loads(sys.argv[1])
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C belongs to the main process, which stops the stages in order

    import cv2
    import numpy as np

    from ..effects import load_builtin_effects
    from ..effects.base import FrameContext
    from ..pipeline import LAST_GEOMETRY_ORDER, _limit_cv2_threads, _use_one_thread_per_core
    from . import shm

    shm.watch_parent(config["parent"])
    shm.keep_full_speed()
    shm.above_normal_process()
    _limit_cv2_threads()
    _use_one_thread_per_core()

    stage: int = config["stage"]
    last: bool = config["last"]
    names = set(config["effects"])
    effects = {cls.name: cls() for cls in load_builtin_effects() if cls.name in names}
    block = shm.Block(shm.Layout(*config["layout"]), config["shm"])
    block.touch()
    height, width = block.layout.height, block.layout.width
    receive = shm.Edge(block.ring(stage), shm.Semaphore(config["receive_semaphore"], create=False))
    send = shm.Edge(block.ring(stage + 1), shm.Semaphore(config["send_semaphore"], create=False))
    reads_later: dict[str, list[str]] = config["reads_later"]  # effect of a later stage -> the masks it reads
    heart = shm.LINE_HEART + stage

    gc.collect()
    gc.freeze()
    print(json.dumps({"ready": stage}), flush=True)

    log_path = config.get("log")
    log: list[Any] = []
    gc_log: list[tuple[int, float]] = []
    if log_path:
        started = [0.0]

        def on_gc(phase: str, info: dict[str, int]) -> None:
            if phase == "start":
                started[0] = time.perf_counter()
            else:
                gc_log.append((info["generation"], round((time.perf_counter() - started[0]) * 1000, 3)))

        gc.callbacks.append(on_gc)
    wall0: float | None = None
    cpu0 = 0.0
    views_of = shm.ArrayViews(block)
    logged: set[tuple[str, str]] = set()
    last_status: str | None = None
    last_wants = False
    sections: dict[str, Any] = {}
    polite = 0.0

    def report_status(steps: list[Any]) -> None:
        nonlocal last_status, last_wants
        neural = effects.get("neural")
        if neural is None:
            return
        status = getattr(neural, "status_message", None)
        if status != last_status:
            last_status = status
            print(json.dumps({"neural_status": status}), flush=True)
        # Whether it will use the masks depends on its worker, which lives here; the main process decides on segmenting.
        section = next((step[1] for step in steps if step[0] == "neural"), None)
        wants = section is not None and bool(neural.wants_masks(section))
        if wants != last_wants:
            last_wants = wants
            print(json.dumps({"neural_wants_masks": wants}), flush=True)

    def draw(slot: int, meta: dict[str, Any], timing: dict[str, float]) -> None:
        picture = image = block.image(slot)
        references = meta["masks"]
        views = [views_of.mask(ref) for ref in references]
        landmarks, z = meta["landmarks"], meta["z"]
        for array in (landmarks, z):
            if array is not None:
                array.setflags(write=False)
        ctx = FrameContext(
            height=height,
            width=width,
            landmarks=landmarks,
            landmarks_z=z,
            hair_mask=views[0],
            face_skin_mask=views[1],
            body_skin_mask=views[2],
            background=views_of.plate(meta["plate"]),
            timestamp_ms=meta["ts_ms"],
            masks_timestamp_ms=meta["masks_ts_ms"],
            skipped_frames=meta["skipped"],
        )
        steps = [(name, keep_section(sections, name, section), *rest) for name, section, *rest in meta["steps"]]
        meta["steps"] = steps
        ahead = [
            (name, effects[name], section)
            for name, section, *_ in steps
            if name in effects and callable(getattr(effects[name], "start_ahead", None))
        ]
        for name, section, intensity, needs_hair, order in steps:
            effect = effects.get(name)
            if effect is None:
                continue
            if ahead and order > LAST_GEOMETRY_ORDER:
                # All geometry is final here. Not when that effect is next itself: nothing would run in the meantime.
                for other_name, other, other_section in ahead:
                    if other is not effect:
                        try:
                            other.start_ahead(ctx, other_section)
                        except Exception as exc:
                            _log_once(logged, other_name, exc)
                ahead = []
            if needs_hair and ctx.hair_mask is None:
                continue
            ctx.intensity = intensity
            begin = time.perf_counter()
            try:
                result = effect.apply(image, ctx, section)
                if not (isinstance(result, np.ndarray) and result.dtype == np.float32 and result.shape == image.shape):
                    raise TypeError(f"apply() must return a float32 array of shape {image.shape}")
                image = result
            except Exception as exc:
                _log_once(logged, name, exc)
            if log_path:
                timing[name] = round((time.perf_counter() - begin) * 1000, 3)
        begin = time.perf_counter()
        if last:
            cv2.max(image, 0.0, dst=image)
            cv2.convertScaleAbs(image, alpha=255.0, dst=block.out(slot))
        else:
            if image is not picture:
                cv2.copyTo(image, None, picture)
            # Whatever this stage changed and a later stage reads goes along.
            needed: set[str] = set()
            for name, *_ in steps:
                needed.update(reads_later.get(name, ()))
            handed_on = list(references)
            for which, field in enumerate(MASK_FIELDS):
                mask = getattr(ctx, field)
                if mask is None or field not in needed:
                    handed_on[which] = None
                elif mask is not views[which]:
                    cv2.copyTo(mask, None, block.slot_mask(slot, which))
                    handed_on[which] = ("slot", slot, which)
            meta["masks"] = handed_on
            meta["landmarks"] = ctx.landmarks
        if log_path:
            timing["_cross"] = round((time.perf_counter() - begin) * 1000, 3)

    try:
        while True:
            block.flags[heart] = shm.now()
            slot = receive.get(100)
            if block.flags[shm.LINE_STOP]:
                break
            if block.flags[shm.LINE_POLITE] != polite:
                polite = block.flags[shm.LINE_POLITE]
                (shm.normal_process if polite else shm.above_normal_process)()
            if slot is None:
                continue
            began = shm.now()
            if wall0 is None:
                wall0, cpu0 = began, time.process_time()
            meta = block.read_meta(slot)
            if meta["reset"]:
                for name, effect in effects.items():
                    try:
                        effect.reset()
                    except Exception as exc:
                        _log_once(logged, name, exc)
            timing: dict[str, float] = {}
            if not meta["through"]:
                draw(slot, meta, timing)
                report_status(meta["steps"])
            ended = shm.now()
            meta["finished"] = ended
            block.write_meta(slot, meta)
            send.put(slot)
            if log_path:
                log.append((meta["seq"], began, ended, timing))
    finally:
        for name, effect in effects.items():
            close = getattr(effect, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    _log_once(logged, name, exc)
    if log_path:
        wall = shm.now() - (wall0 or shm.now())
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "stage": stage,
                    "effects": sorted(names),
                    "cpu_s": time.process_time() - cpu0,
                    "wall_s": wall,
                    "log": log,
                    "gc": gc_log,
                    "memory": shm.memory_mb(),
                },
                handle,
            )
    os._exit(0)  # views into the block are still alive; the system unmaps it


def keep_section(sections: dict[str, Any], name: str, section: Any) -> Any:
    """The object of the last frame if the section did not change. Each frame arrives as a new copy; in one process a section
    stays the same object until it changes, and an effect may go by that (the finish does: a new section gives it another try)."""
    known = sections.get(name)
    if type(known) is type(section) and known == section:
        return known
    sections[name] = section
    return section


def _log_once(logged: set[tuple[str, str]], source: str, exc: BaseException) -> None:
    """Through stderr, to the main process's log. As ``MakeupPipeline._log_once``: the effect is skipped for that frame."""
    key = (source, f"{type(exc).__name__}: {exc}")
    if key in logged or len(logged) >= 256:
        return
    logged.add(key)
    print(f"{source} failed and is skipped for this frame: {key[1]}", file=sys.stderr, flush=True)
    traceback.print_exception(exc, file=sys.stderr)
    sys.stderr.flush()


if __name__ == "__main__":
    main()
