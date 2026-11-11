"""Prepare the Windows ICU dependency before importing either Qt window."""

import sys

if sys.platform == "win32":
    import ctypes

    # Conda can expose an incompatible icuuc.dll through the DLL search path.
    # Retain the system library so Qt resolves its unversioned ICU functions.
    try:
        _system_icu = ctypes.WinDLL("icuuc.dll", winmode=0x00000800)  # LOAD_LIBRARY_SEARCH_SYSTEM32
    except OSError:
        # Older Qt versions may not need ICU; let the normal Qt import decide.
        pass
