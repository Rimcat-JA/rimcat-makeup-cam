"""Shared memory, one-way rings and kernel semaphores between the processes (contract section 4). Windows only.

A slot belongs to exactly one process at a time; ownership passes with the ring. No locks. All processes read the time
from the machine-wide QPC, so stamps can be compared across them.
"""

from __future__ import annotations

import ctypes
import json
import os
import pickle
import struct
import sys
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.CreateSemaphoreW.restype = ctypes.c_void_p
_k32.CreateSemaphoreW.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_long, ctypes.c_wchar_p]
_k32.OpenSemaphoreW.restype = ctypes.c_void_p
_k32.OpenSemaphoreW.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_wchar_p]
_k32.ReleaseSemaphore.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p]
_k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
_k32.WaitForSingleObject.restype = ctypes.c_uint
_k32.CloseHandle.argtypes = [ctypes.c_void_p]
_k32.GetCurrentThread.restype = ctypes.c_void_p
_k32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
_k32.GetCurrentProcess.restype = ctypes.c_void_p
_k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
_k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
_k32.OpenProcess.restype = ctypes.c_void_p
_k32.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]

_SEMAPHORE_ALL_ACCESS = 0x1F0003
_SYNCHRONIZE = 0x00100000
_WAIT_OBJECT_0 = 0
_INFINITE = 0xFFFFFFFF
_ABOVE_NORMAL_PRIORITY_CLASS = 0x8000
_NORMAL_PRIORITY_CLASS = 0x20
_PROCESS_POWER_THROTTLING = 4
_THROTTLE_EXECUTION_SPEED = 1
_THROTTLE_IGNORE_TIMER_RESOLUTION = 4
THREAD_PRIORITY_ABOVE_NORMAL = 1
THREAD_PRIORITY_HIGHEST = 2
THREAD_PRIORITY_TIME_CRITICAL = 15

_PAGE = 4096
META_BYTES = 128 * 1024
CONTROL_BYTES = 64 * 1024
RING_LENGTH = 64
_RING_BYTES = 8 * (2 + RING_LENGTH)
N_SLOTS = 8
N_MASK_SETS = 10
N_PLATES = 2
N_MASKS = 3
_N_FLAGS = 64
# The edges of a line, one reader and one writer each: main to stage 0, 0 to 1, 1 to 2, 2 to the sender, the sender to main.
N_EDGES = 5
EDGE_TO_SENDER = 3
EDGE_DONE = 4

# Flags of a line. Heartbeats by stage.
LINE_STOP = 0
LINE_POLITE = 1  # the machine is short of CPU: no priority above the other programs (contract section 8)
LINE_HEART = 8
# Flags of the session, shared with the sender for its whole life.
SESSION_STOP = 0
SESSION_HEART = 1
SESSION_WRITES = 2
SESSION_SLIPS = 3
SESSION_DROPS = 4
SESSION_LAST_ID = 5  # the id of the newest frame the sender has dealt with
SESSION_FLUSH = 6  # asked: forget the line
SESSION_FLUSH_DONE = 7
SESSION_NOW_WRITTEN = 8
SESSION_NOW_TAKEN = 9
SESSION_SETTLED_MS = 10
SESSION_POLITE = 11  # as LINE_POLITE
SESSION_DELAY_MS = 12  # from the camera to the end of the last write

_freq = ctypes.c_longlong()
_k32.QueryPerformanceFrequency(ctypes.byref(_freq))


def _qpc_offset() -> float:
    counter = ctypes.PyDLL("kernel32").QueryPerformanceCounter
    value = ctypes.c_longlong()
    best: tuple[float, float] | None = None
    for _ in range(50):
        a = time.perf_counter()
        counter(ctypes.byref(value))
        b = time.perf_counter()
        if best is None or b - a < best[0]:
            best = (b - a, value.value / _freq.value - (a + b) / 2)
    assert best is not None
    return best[1]


_OFFSET = _qpc_offset()


def now() -> float:
    """Seconds of the machine-wide QPC: all processes share this clock."""
    return time.perf_counter() + _OFFSET


def from_perf_counter(value: float) -> float:
    return value + _OFFSET


class Semaphore:
    """A named kernel semaphore. Wake-up 17 us p50 by measurement, and no CPU while waiting."""

    def __init__(self, name: str, *, create: bool) -> None:
        self.name = name
        self._handle = (
            _k32.CreateSemaphoreW(None, 0, 1 << 20, name) if create else _k32.OpenSemaphoreW(_SEMAPHORE_ALL_ACCESS, 0, name)
        )
        if not self._handle:
            raise OSError(ctypes.get_last_error(), f"semaphore {name}")

    def post(self) -> None:
        if self._handle:
            _k32.ReleaseSemaphore(self._handle, 1, None)

    def wait(self, ms: int) -> bool:
        return _k32.WaitForSingleObject(self._handle, ms) == _WAIT_OBJECT_0

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle:
            _k32.CloseHandle(handle)


class Layout:
    """control | slots: metadata, float32 picture, uint8 picture, 3 masks | mask sets | plates."""

    def __init__(self, height: int, width: int) -> None:
        self.height, self.width = height, width
        image_bytes = _pages(height * width * 3 * 4)
        out_bytes = _pages(height * width * 3)
        self.mask_step = _pages(height * width * 4)
        self.image_at = _pages(META_BYTES)
        self.out_at = self.image_at + image_bytes
        self.masks_at = self.out_at + out_bytes
        self.slot_bytes = self.masks_at + N_MASKS * self.mask_step
        self.slots_at = _pages(CONTROL_BYTES)
        self.sets_at = self.slots_at + N_SLOTS * self.slot_bytes
        self.set_bytes = N_MASKS * self.mask_step
        self.plates_at = self.sets_at + N_MASK_SETS * self.set_bytes
        self.plate_bytes = image_bytes
        self.total = self.plates_at + N_PLATES * self.plate_bytes

    def args(self) -> list[int]:
        return [self.height, self.width]


def _pages(size: int) -> int:
    return (size + _PAGE - 1) // _PAGE * _PAGE


_keep_mapped: list[shared_memory.SharedMemory] = []


def _close_memory(memory: shared_memory.SharedMemory, unlink: bool) -> None:
    """Views must be gone by now. A view that is not leaves the memory mapped until the process ends, which is harmless."""
    try:
        memory.close()
    except BufferError:
        _keep_mapped.append(memory)
    if unlink:
        try:
            memory.unlink()
        except OSError:
            pass


class Block:
    """One shared block per frame size. ``name=None`` makes it; a reader passes the name."""

    def __init__(self, layout: Layout, name: str | None = None) -> None:
        self.layout = layout
        self._shm = shared_memory.SharedMemory(name=name, create=name is None, size=layout.total if name is None else 0)
        self.name = self._shm.name
        self.buf: Any = self._shm.buf
        self.flags: Any = _flags(self.buf)

    def touch(self) -> None:
        """Every page once, so that no frame ever faults one in."""
        np.ndarray((self.layout.total,), np.uint8, self.buf)[::_PAGE].sum()

    def ring(self, edge: int) -> np.ndarray:
        return _ring(self.buf, edge)

    def _slot_at(self, slot: int) -> int:
        return self.layout.slots_at + slot * self.layout.slot_bytes

    def image(self, slot: int) -> np.ndarray:
        h, w = self.layout.height, self.layout.width
        return np.ndarray((h, w, 3), np.float32, self.buf, offset=self._slot_at(slot) + self.layout.image_at)

    def out(self, slot: int) -> np.ndarray:
        h, w = self.layout.height, self.layout.width
        return np.ndarray((h, w, 3), np.uint8, self.buf, offset=self._slot_at(slot) + self.layout.out_at)

    def slot_mask(self, slot: int, which: int) -> np.ndarray:
        offset = self._slot_at(slot) + self.layout.masks_at + which * self.layout.mask_step
        return np.ndarray((self.layout.height, self.layout.width), np.float32, self.buf, offset=offset)

    def set_mask(self, mask_set: int, which: int) -> np.ndarray:
        offset = self.layout.sets_at + mask_set * self.layout.set_bytes + which * self.layout.mask_step
        return np.ndarray((self.layout.height, self.layout.width), np.float32, self.buf, offset=offset)

    def plate(self, index: int) -> np.ndarray:
        offset = self.layout.plates_at + index * self.layout.plate_bytes
        return np.ndarray((self.layout.height, self.layout.width, 3), np.float32, self.buf, offset=offset)

    def write_meta(self, slot: int, meta: dict[str, Any]) -> int:
        """Raises ``ValueError`` rather than overwrite the picture."""
        data = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
        if len(data) + 8 > META_BYTES:
            raise ValueError(f"metadata of {len(data)} bytes does not fit in {META_BYTES - 8}")
        at = self._slot_at(slot)
        self.buf[at : at + 8] = struct.pack("<q", len(data))
        self.buf[at + 8 : at + 8 + len(data)] = data
        return len(data)

    def read_meta(self, slot: int) -> dict[str, Any]:
        at = self._slot_at(slot)
        size = struct.unpack("<q", bytes(self.buf[at : at + 8]))[0]
        return pickle.loads(self.buf[at + 8 : at + 8 + size])

    def close(self, *, unlink: bool = False) -> None:
        self.flags = None
        self.buf = None
        _close_memory(self._shm, unlink)


class ArrayViews:
    """The arrays one stage sees. Effects know a mask or a plate by its identity: the hair effect searches each mask once,
    reshape keeps statistics per plate. So one array object per mask set and generation, and per plate generation. A mask
    that a stage changed is a new array for each frame, as reshape's masks are in-process (contract section 4.4)."""

    def __init__(self, block: Block) -> None:
        self._block = block
        self._sets: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
        self._plates: dict[int, tuple[int, np.ndarray]] = {}

    def mask(self, reference: Any) -> np.ndarray | None:
        if reference is None:
            return None
        if reference[0] == "set":
            _, mask_set, which, generation = reference
            known = self._sets.get((mask_set, which))
            if known is not None and known[0] == generation:
                return known[1]
            view = self._block.set_mask(mask_set, which)
            view.setflags(write=False)
            self._sets[(mask_set, which)] = (generation, view)
            return view
        view = self._block.slot_mask(reference[1], reference[2])
        view.setflags(write=False)
        return view

    def plate(self, reference: Any) -> np.ndarray | None:
        if reference is None:
            return None
        index, generation = reference
        known = self._plates.get(index)
        if known is not None and known[0] == generation:
            return known[1]
        view = self._block.plate(index)
        view.setflags(write=False)
        self._plates[index] = (generation, view)
        return view


# Frames for the sender to write at once, not through a line: placeholders and frames drawn in the main process. They
# have the virtual camera's size, which no line knows and which exists before any line does.
NOW_SLOTS = 3
_NOW_HEADER = 16
_NOW_MAX_FRAME = 3840 * 2160 * 3
_NOW_STEP = _pages(_NOW_HEADER + _NOW_MAX_FRAME)


class SessionBlock:
    """What the main process and the sender share for the sender's whole life: flags, and the frames to write at once."""

    def __init__(self, name: str | None = None) -> None:
        size = _pages(CONTROL_BYTES) + NOW_SLOTS * _NOW_STEP
        self._shm = shared_memory.SharedMemory(name=name, create=name is None, size=size if name is None else 0)
        self.name = self._shm.name
        self.buf: Any = self._shm.buf
        self.flags: Any = _flags(self.buf)

    def _now_at(self, index: int) -> int:
        return _pages(CONTROL_BYTES) + (index % NOW_SLOTS) * _NOW_STEP

    def can_put_now(self) -> bool:
        return int(self.flags[SESSION_NOW_WRITTEN]) - int(self.flags[SESSION_NOW_TAKEN]) < NOW_SLOTS

    def put_now(self, frame: np.ndarray) -> None:
        """One writer at a time; ``can_put_now`` first."""
        h, w = frame.shape[:2]
        if h * w * 3 > _NOW_MAX_FRAME:
            raise ValueError(f"a frame of {w}x{h} is too large for the virtual camera")
        index = int(self.flags[SESSION_NOW_WRITTEN])
        at = self._now_at(index)
        self.buf[at : at + _NOW_HEADER] = struct.pack("<qq", h, w)
        cv2.copyTo(frame, None, np.ndarray((h, w, 3), np.uint8, self.buf, offset=at + _NOW_HEADER))
        self.flags[SESSION_NOW_WRITTEN] = index + 1

    def take_now(self) -> np.ndarray | None:
        """The oldest frame to write at once, in place; ``now_done`` when it is written."""
        index = int(self.flags[SESSION_NOW_TAKEN])
        if int(self.flags[SESSION_NOW_WRITTEN]) <= index:
            return None
        at = self._now_at(index)
        h, w = struct.unpack("<qq", bytes(self.buf[at : at + _NOW_HEADER]))
        return np.ndarray((h, w, 3), np.uint8, self.buf, offset=at + _NOW_HEADER)

    def now_done(self) -> None:
        self.flags[SESSION_NOW_TAKEN] = int(self.flags[SESSION_NOW_TAKEN]) + 1

    def close(self, *, unlink: bool = False) -> None:
        self.flags = None
        self.buf = None
        _close_memory(self._shm, unlink)


def _flags(buf: Any) -> np.ndarray:
    return np.ndarray((_N_FLAGS,), np.float64, buf, offset=CONTROL_BYTES - _N_FLAGS * 8)


def _ring(buf: Any, edge: int) -> np.ndarray:
    return np.ndarray((2 + RING_LENGTH,), np.int64, buf, offset=edge * _RING_BYTES)


class Ring:
    """One writer, one reader, slot numbers. On 64-bit x86 an aligned store of 8 bytes is seen whole."""

    def __init__(self, cells: np.ndarray) -> None:
        self._cells = cells

    def put(self, slot: int) -> None:
        written = int(self._cells[0])
        self._cells[2 + written % RING_LENGTH] = slot
        self._cells[0] = written + 1

    def get(self) -> int | None:
        taken = int(self._cells[1])
        if int(self._cells[0]) <= taken:
            return None
        slot = int(self._cells[2 + taken % RING_LENGTH])
        self._cells[1] = taken + 1
        return slot

    def __len__(self) -> int:
        return int(self._cells[0]) - int(self._cells[1])


class Edge:
    """A ring and the semaphore that wakes its reader. One post for each item, so a wake-up is never lost."""

    def __init__(self, cells: np.ndarray, semaphore: Semaphore) -> None:
        self.ring = Ring(cells)
        self.semaphore = semaphore

    def put(self, slot: int) -> None:
        self.ring.put(slot)
        self.semaphore.post()

    def get(self, ms: int) -> int | None:
        if not self.semaphore.wait(ms):
            return None
        return self.ring.get()


def watch_parent(pid: int) -> None:
    """Ends this process when the main process has ended, however that came about. By process handle: a thread blocked
    reading stdin here would freeze every later DLL load of the process, as the C runtime asks for the type of the
    standard handles."""
    handle = _k32.OpenProcess(_SYNCHRONIZE, 0, pid)

    def run() -> None:
        _k32.WaitForSingleObject(handle, _INFINITE)
        os._exit(3)

    threading.Thread(target=run, name="watch-parent", daemon=True).start()


class _Throttle(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def keep_full_speed() -> bool:
    """Windows treats a process without a window as background: it runs it for efficiency and ignores its timer requests.
    Off for this process. Without it the same work took about 20 % more CPU, and stages were slower (contract section 3.5)."""
    state = _Throttle(1, _THROTTLE_EXECUTION_SPEED | _THROTTLE_IGNORE_TIMER_RESOLUTION, 0)
    return bool(
        _k32.SetProcessInformation(_k32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state))
    )


def above_normal_process() -> bool:
    return bool(_k32.SetPriorityClass(_k32.GetCurrentProcess(), _ABOVE_NORMAL_PRIORITY_CLASS))


def normal_process() -> bool:
    return bool(_k32.SetPriorityClass(_k32.GetCurrentProcess(), _NORMAL_PRIORITY_CLASS))


def set_this_thread_priority(priority: int) -> bool:
    return bool(_k32.SetThreadPriority(_k32.GetCurrentThread(), priority))


def begin_timer_period() -> None:
    ctypes.WinDLL("winmm").timeBeginPeriod(1)


class _MemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def memory_mb() -> dict[str, float]:
    """For the measurement logs."""
    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(_MemoryCounters)
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return {
        "peak_ws_mb": round(counters.PeakWorkingSetSize / 1e6, 1),
        "ws_mb": round(counters.WorkingSetSize / 1e6, 1),
        "private_mb": round(counters.PrivateUsage / 1e6, 1),
    }


def child_command(module: str, args: Any) -> tuple[list[str], dict[str, str]]:
    """How to start a child, and its environment. Inside a venv ``sys.executable`` is a launcher that starts the real
    interpreter as a process of its own: a kill would hit only the launcher. So the real one is started, as
    ``multiprocessing`` does."""
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH", "")) if p)
    executable = sys.executable
    base = getattr(sys, "_base_executable", executable)
    if os.path.normcase(os.path.abspath(base)) != os.path.normcase(os.path.abspath(executable)):
        env["__PYVENV_LAUNCHER__"] = executable
        executable = base
    return [executable, "-m", module, json.dumps(args)], env
