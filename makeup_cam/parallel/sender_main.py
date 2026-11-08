"""The process that owns the virtual camera. ``python -m makeup_cam.parallel.sender_main JSON``. It ends with the main process.

One hot thread gives each finished frame its place on the grid (``clock.py``), converts it and writes it there. One control
thread reads commands from stdin: open, close, attach (a new line of stages), detach. The answers go to stdout, one JSON
per line. It must not import the effects, and it imports all it needs before the control thread starts reading: a thread
blocked reading stdin freezes later DLL loads of the process.
"""

from __future__ import annotations

import gc
import json
import os
import signal
import sys
import threading
import time
from collections import deque
from typing import Any


def main() -> None:
    config: dict[str, Any] = json.loads(sys.argv[1])
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import cv2
    import numpy as np
    import pyvirtualcam

    from ..camera import fit_frame
    from . import shm
    from .clock import SPARE_FRAMES, OutputClock, declared_interval
    from .nv12 import NV12Buffer
    from .obs_queue import ObsQueue

    shm.watch_parent(config["parent"])
    shm.keep_full_speed()
    shm.above_normal_process()
    cv2.setNumThreads(1)
    session = shm.SessionBlock(config["session"])
    wake = shm.Semaphore(config["wake_semaphore"], create=False)
    flags = session.flags
    reply_lock = threading.Lock()
    camera_lock = threading.Lock()
    pending: deque[dict[str, Any]] = deque()
    camera: list[Any] = [None]
    nv12: list[NV12Buffer | None] = [None]
    clock: list[OutputClock | None] = [None]
    renew = [False]  # asked by the control thread, done by the hot thread, which owns the clock

    def reply(message: dict[str, Any]) -> None:
        with reply_lock:
            print(json.dumps(message), flush=True)

    def open_camera(command: dict[str, Any]) -> None:
        width, height, fps = command["width"], command["height"], command["fps"]
        try:
            if command.get("old_queue"):  # a reader holds the queue of a sender that died: the rescue of contract section 8
                device, is_nv12 = ObsQueue(width, height, fps), False
            else:
                device, is_nv12 = open_device(pyvirtualcam, width, height, fps)
        except Exception as exc:
            reply({"ok": False, "error": str(exc)})
            return
        buffer = NV12Buffer(width, height) if is_nv12 else None
        period_s = declared_interval(fps) * 1e-7
        with camera_lock:
            camera[0], nv12[0] = device, buffer
            if clock[0] is None or clock[0].period_s != period_s:
                clock[0] = OutputClock(period_s)
            else:
                renew[0] = True  # the phase never moves: the same clock across a reopen
        reply({"ok": True, "device": str(device.device), "bgr": not is_nv12})

    def close_camera() -> None:
        with camera_lock:
            device, camera[0], nv12[0] = camera[0], None, None
        if device is not None:
            device.close()

    def control() -> None:
        for line in sys.stdin:
            command = json.loads(line)
            operation = command["op"]
            if operation == "open":
                close_camera()
                open_camera(command)
            elif operation == "close":
                close_camera()
                reply({"ok": True, "op": "close"})
            else:  # the hot thread owns the rings
                pending.append(command)
                wake.post()
        os._exit(3)  # the pipe is closed: the main process has gone

    threading.Thread(target=control, name="sender-control", daemon=True).start()
    shm.begin_timer_period()
    shm.set_this_thread_priority(shm.THREAD_PRIORITY_TIME_CRITICAL)  # it runs for under 10 ms of every 33
    gc.collect()
    gc.freeze()
    gc.disable()  # no cycles are made here, and a collection took 10 to 12 ms
    print(json.dumps({"ready": "sender"}), flush=True)

    log_path = config.get("log")
    log: list[Any] = []
    events: list[Any] = []
    prepare_ms: list[float] = []
    wall0: float | None = None
    cpu0 = 0.0
    last_event_log = 0.0

    polite = 0.0
    block: shm.Block | None = None
    receive: shm.Ring | None = None
    done: shm.Edge | None = None

    def forget_line() -> None:
        nonlocal block, receive, done
        if block is not None:
            receive = None
            if done is not None:
                done.semaphore.close()
            done = None
            block.close()
            block = None

    def note(kind: str, *details: Any) -> None:
        """Slips and drops are counted always; their times go to the log at most once per second."""
        nonlocal last_event_log
        moment = shm.now()
        if log_path:
            events.append((kind, *details, round(moment, 4)))
        if moment - last_event_log >= 1.0:
            last_event_log = moment
            print(f"{kind} at {moment:.3f}: {details}", file=sys.stderr, flush=True)

    def write(device: Any, buffer: NV12Buffer | None, frame: Any) -> None:
        try:
            with camera_lock:
                if camera[0] is device:
                    if isinstance(device, ObsQueue):
                        device.commit(frame)
                    else:
                        device.send(frame if buffer is None else buffer.data)
                    flags[shm.SESSION_WRITES] += 1
        except Exception as exc:
            print(f"the virtual camera failed: {exc}", file=sys.stderr, flush=True)
            os._exit(4)  # the main process takes over from here

    def convert(device: Any, buffer: NV12Buffer | None, frame: np.ndarray) -> Any:
        fitted = fit_frame(frame, device.width, device.height)
        if isinstance(device, ObsQueue):
            return device.prepare(fitted)  # into the next buffer, and only ``commit`` shows it: on its place as ever
        if buffer is None:
            return fitted.copy()  # the wider write of BGR, and the frame's slot must not change under it
        buffer.fill(fitted)
        return fitted

    def write_now() -> bool:
        """A frame that takes no place on the grid: a placeholder, or a frame drawn in the main process."""
        frame = session.take_now()
        if frame is None:
            return False
        with camera_lock:
            device, buffer = camera[0], nv12[0]
        if device is not None:
            fitted = convert(device, buffer, frame)
            write(device, buffer, fitted)
            del fitted
        del frame
        session.now_done()
        return True

    def handle_frame(slot: int) -> None:
        nonlocal wall0, cpu0
        assert block is not None and done is not None
        meta = block.read_meta(slot)
        identity, seq, t_camera, finished = meta["id"], meta["seq"], meta["t_camera"], meta["finished"]
        taken = shm.now()
        if wall0 is None:
            wall0, cpu0 = taken, time.process_time()
        with camera_lock:
            device, buffer, grid = camera[0], nv12[0], clock[0]
        if device is None or grid is None:  # the virtual camera is off: the stages still run, for the preview
            flags[shm.SESSION_LAST_ID] = identity
            done.put(slot)
            return
        if renew[0]:
            renew[0] = False
            grid.renew()
        grid.spare_frames = 1 if polite else SPARE_FRAMES  # frames are being lost anyway: the delay need not stay as well
        place = grid.place(seq, t_camera, finished, taken)
        flags[shm.SESSION_SLIPS] = grid.slips
        if place.slips:
            note("slip", seq, place.slips)
        if place.drop:
            flags[shm.SESSION_DROPS] = grid.drops
            flags[shm.SESSION_LAST_ID] = identity
            note("drop", seq)
            done.put(slot)
            if log_path:
                log.append((seq, 0.0, 0.0, finished, 0.0, seq, t_camera))
            return
        if grid.delay_s is not None and flags[shm.SESSION_SETTLED_MS] != grid.delay_s * 1000.0:
            flags[shm.SESSION_SETTLED_MS] = grid.delay_s * 1000.0
            if log_path:
                events.append(("settled", seq, grid.delay_s * 1000.0))
        frame = block.out(slot)
        begin = shm.now()
        fitted = convert(device, buffer, frame_for_camera(frame, meta["mirror"], device.width, device.height))
        converted = shm.now()
        if log_path:
            prepare_ms.append(round((converted - begin) * 1000, 3))
        due = place.due
        if due is not None and converted > due:
            before = grid.slips
            due = grid.late_after_convert(converted)
            flags[shm.SESSION_SLIPS] = grid.slips
            note("slip after converting", seq, grid.slips - before)
        if due is not None:
            while True:
                left = due - shm.now()
                if left <= 0:
                    break
                if left > 0.0025:
                    time.sleep(left - 0.002)
        write_began = shm.now()
        write(device, buffer, fitted)
        write_ended = shm.now()
        flags[shm.SESSION_LAST_ID] = identity
        if t_camera:
            flags[shm.SESSION_DELAY_MS] = (write_ended - t_camera) * 1000.0
        del frame, fitted
        done.put(slot)
        if log_path:
            log.append((seq, write_began, write_ended, finished, due or 0.0, seq, t_camera))

    while True:
        flags[shm.SESSION_HEART] = shm.now()
        wake.wait(100)
        if flags[shm.SESSION_STOP]:
            break
        if flags[shm.SESSION_POLITE] != polite:
            polite = flags[shm.SESSION_POLITE]
            (shm.normal_process if polite else shm.above_normal_process)()
            shm.set_this_thread_priority(shm.THREAD_PRIORITY_HIGHEST if polite else shm.THREAD_PRIORITY_TIME_CRITICAL)
        if flags[shm.SESSION_FLUSH] and not flags[shm.SESSION_FLUSH_DONE]:
            forget_line()  # what is left in the ring is drawn again by the main process
            flags[shm.SESSION_FLUSH_DONE] = 1.0
        while pending:
            command = pending.popleft()
            if command["op"] == "attach":
                forget_line()
                block = shm.Block(shm.Layout(*command["layout"]), command["shm"])
                block.touch()
                receive = shm.Ring(block.ring(shm.EDGE_TO_SENDER))
                done = shm.Edge(block.ring(shm.EDGE_DONE), shm.Semaphore(command["done_semaphore"], create=False))
                if clock[0] is not None:
                    clock[0].renew()  # new stages: their first frames are slow, and that says nothing about the load
            elif command["op"] == "detach":
                forget_line()
            reply({"ok": True, "op": command["op"]})
        write_now()
        if receive is not None:
            slot = receive.get()
            if slot is not None:
                handle_frame(slot)
    if log_path:
        grid = clock[0]
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "prepare_ms": prepare_ms,
                    "log": log,
                    "events": events,
                    "late": grid.slips if grid else 0,
                    "slips": grid.slips if grid else 0,
                    "drops": grid.drops if grid else 0,
                    "settled_ms": grid.delay_s * 1000.0 if grid and grid.delay_s else None,
                    "cpu_s": time.process_time() - cpu0,
                    "wall_s": shm.now() - (wall0 or shm.now()),
                    "memory": shm.memory_mb(),
                },
                handle,
            )
    close_camera()  # last, so that readers see the camera stop
    os._exit(0)


def open_device(backend: Any, width: int, height: int, fps: float) -> tuple[Any, bool]:
    """The camera, and whether it takes NV12. One that cannot goes on with BGR: the same bytes, with a wider write. If it takes
    neither, another program holds it: nothing was refused for its format, and the caller retries and reports that."""
    try:
        return backend.Camera(width, height, fps, fmt=backend.PixelFormat.NV12, print_fps=False), True
    except Exception as refused:
        device = backend.Camera(width, height, fps, fmt=backend.PixelFormat.BGR, print_fps=False)
        print(f"the virtual camera refused NV12 and takes BGR: {refused}", file=sys.stderr, flush=True)
    return device, False


def frame_for_camera(frame: Any, mirror: bool, width: int, height: int) -> Any:
    """What goes to the virtual camera from a finished picture. ``ProcessResult.output`` is this, before the size fits."""
    import cv2

    from ..camera import fit_frame

    return fit_frame(cv2.flip(frame, 1) if mirror else frame, width, height)


if __name__ == "__main__":
    main()
