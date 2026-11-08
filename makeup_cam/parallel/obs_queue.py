"""The rescue: writing into the shared queue of OBS Virtual Camera directly (contract section 8).

Used only when the sender has died while a reader holds its queue, as ``pyvirtualcam.Camera()`` then fails. Nothing public
promises this layout, so everything is checked first and after the first write; a failure means no rescue. The normal path
uses the public API alone. ``test_obs_queue_layout`` compares this with ``pyvirtualcam.Camera.send`` on the installed versions.
"""

from __future__ import annotations

import ctypes
import struct
import time
from typing import Any

import numpy as np

from .clock import declared_interval
from .nv12 import to_nv12

QUEUE_NAME = "OBSVirtualCamVideo"
_FILE_MAP_ALL = 0xF001F
_HEADER_BYTES = 48
_FRAME_PAD = 32  # the timestamp, then the aligned frame
_READY = 2
_TYPE_NV12 = 0


class QueueRefused(RuntimeError):
    """What is there is not the queue this code knows, or it did not take the first write."""


def _kernel32() -> Any:
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenFileMappingW.restype = ctypes.c_void_p
    k.OpenFileMappingW.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_wchar_p]
    k.MapViewOfFile.restype = ctypes.c_void_p
    k.MapViewOfFile.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t]
    k.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    return k


def read_header(pointer: int) -> dict[str, Any]:
    raw = ctypes.string_at(pointer, _HEADER_BYTES)
    return {
        "write_idx": struct.unpack_from("<I", raw, 0)[0],
        "read_idx": struct.unpack_from("<I", raw, 4)[0],
        "state": struct.unpack_from("<I", raw, 8)[0],
        "offsets": struct.unpack_from("<3I", raw, 12),
        "type": struct.unpack_from("<I", raw, 24)[0],
        "cx": struct.unpack_from("<I", raw, 28)[0],
        "cy": struct.unpack_from("<I", raw, 32)[0],
        "interval": struct.unpack_from("<Q", raw, 40)[0],
    }


def check_header(header: dict[str, Any], width: int, height: int, fps: float) -> None:
    """Raises ``QueueRefused`` unless the header is the one this code can write to."""
    offsets = header["offsets"]
    size = width * height * 3 // 2 + _FRAME_PAD
    spacing = offsets[1] - offsets[0]
    if header["type"] != _TYPE_NV12 or (header["cx"], header["cy"]) != (width, height):
        raise QueueRefused(f"unknown queue format: {header}")
    if spacing < size or offsets[2] - offsets[1] != spacing:
        raise QueueRefused(f"unknown queue offsets: {header}")
    if header["interval"] != declared_interval(fps) or header["state"] != _READY:
        raise QueueRefused(f"unknown queue state: {header}")


class ObsQueue:
    """The existing queue, opened for writing NV12 frames. Raises ``OSError`` or ``QueueRefused``."""

    device = "OBS Virtual Camera"

    def __init__(self, width: int, height: int, fps: float, *, name: str = QUEUE_NAME) -> None:
        self._k = _kernel32()
        self._head: Any = None
        self._pointer = None
        self._handle = self._k.OpenFileMappingW(_FILE_MAP_ALL, 0, name)
        if not self._handle:
            raise OSError(ctypes.get_last_error(), f"no queue named {name}")
        self._pointer = self._k.MapViewOfFile(self._handle, _FILE_MAP_ALL, 0, 0, 0)
        if not self._pointer:
            error = ctypes.get_last_error()
            self.close()
            raise OSError(error, "the queue cannot be mapped")
        try:
            header = read_header(self._pointer)
            check_header(header, width, height, fps)
        except QueueRefused:
            self.close()
            raise
        self.width, self.height = width, height
        self.offsets = header["offsets"]
        self._head = (ctypes.c_uint32 * 3).from_address(self._pointer)
        self._checked = False

    def prepare(self, bgr: np.ndarray) -> int:
        """Convert and write the next buffer; nothing is visible yet."""
        increment = self._head[0] + 1
        pointer = self._pointer + self.offsets[increment % 3]
        ctypes.c_uint64.from_address(pointer).value = time.perf_counter_ns()
        h, w = self.height, self.width
        y = np.ndarray((h, w), np.uint8, (ctypes.c_ubyte * (h * w)).from_address(pointer + _FRAME_PAD))
        uv = np.ndarray((h // 2, w), np.uint8, (ctypes.c_ubyte * (h * w // 2)).from_address(pointer + _FRAME_PAD + h * w))
        to_nv12(bgr, y, uv)
        return increment

    def commit(self, increment: int) -> None:
        """Makes the prepared frame the one the filter shows: ``write_idx``, ``read_idx``, ``state``, in that order."""
        head = self._head
        head[0] = increment
        head[1] = increment
        head[2] = _READY
        if not self._checked:
            self._checked = True
            if head[0] != increment or head[1] != increment:
                raise QueueRefused("the queue did not take the first write")

    def send(self, bgr: np.ndarray) -> None:
        self.commit(self.prepare(bgr))

    def close(self) -> None:
        """A view that stays open keeps the name taken, and the next camera could not start."""
        self._head = None
        if self._pointer:
            self._k.UnmapViewOfFile(self._pointer)
            self._pointer = None
        if self._handle:
            self._k.CloseHandle(self._handle)
            self._handle = None
