# Change evidence: Windows Qt ICU loading

<a id="table-of-contents"></a>
## Table of Contents

- [Change metadata](#change-metadata)
- [Intent](#intent)
- [Affected surface](#affected-surface)
- [Highlighted code changes](#highlighted-code-changes)
- [Reasoning](#reasoning)
- [Verification](#verification)
- [Residual limits](#residual-limits)

## Change metadata

- **Timestamp:** 2026-11-11T15:43:40Z.
- **Change:** Fix the Windows QtCore import failure caused by an incompatible ICU DLL.
- **Delivery mode:** Standard.
- **Evidence trigger:** Bug fix after the working baseline.

[↑ Back to Table of Contents](#table-of-contents)

## Intent

The Windows launcher must reach the settings window when Conda places its own `icuuc.dll` on Python's DLL search path. The recording tool imports the same Qt bindings, so it needs the same initialization order.

[↑ Back to Table of Contents](#table-of-contents)

## Affected surface

- `makeup_cam/_qt_compat.py` initializes the Windows system ICU dependency.
- `makeup_cam/gui.py` and `makeup_cam/record.py` import the compatibility module before PySide6.
- `tests/test_cli.py` adds conflicting-DLL and missing-system-DLL cases. `tests/test_gui.py`, `tests/test_record.py`, and `tests/test_parallel_life.py` follow the same application-before-PySide6 import order.
- `specification.md`, `docs/dev/DTD.md`, and `docs/dev/modules/qt-compat.md` describe the initialization contract.

[↑ Back to Table of Contents](#table-of-contents)

## Highlighted code changes

The following excerpts are taken from the reviewed staged Git diff. The complete test additions and test import-order changes are in the same commit.

```diff
diff --git a/makeup_cam/_qt_compat.py b/makeup_cam/_qt_compat.py
new file mode 100644
--- /dev/null
+++ b/makeup_cam/_qt_compat.py
@@ -0,0 +1,14 @@
+"""Prepare the Windows ICU dependency before importing either Qt window."""
+
+import sys
+
+if sys.platform == "win32":
+    import ctypes
+
+    # Conda can expose an incompatible icuuc.dll through the DLL search path.
+    # Retain the system library so Qt resolves its unversioned ICU functions.
+    try:
+        _system_icu = ctypes.WinDLL("icuuc.dll", winmode=0x00000800)  # LOAD_LIBRARY_SEARCH_SYSTEM32
+    except OSError:
+        # Older Qt versions may not need ICU; let the normal Qt import decide.
+        pass
diff --git a/makeup_cam/gui.py b/makeup_cam/gui.py
--- a/makeup_cam/gui.py
+++ b/makeup_cam/gui.py
@@ -36,0 +37,2 @@ from typing import TYPE_CHECKING, Any, ClassVar, Protocol
+from . import _qt_compat  # noqa: F401  # Initialize Windows DLLs before PySide6.
+
diff --git a/makeup_cam/record.py b/makeup_cam/record.py
--- a/makeup_cam/record.py
+++ b/makeup_cam/record.py
@@ -67,0 +68,2 @@ from typing import Any, Protocol, TypeVar
+from . import _qt_compat  # noqa: F401  # Initialize Windows DLLs before PySide6.
+
```

The added initialization uses a system-directory-only search and keeps the DLL object alive:

```python
try:
    _system_icu = ctypes.WinDLL("icuuc.dll", winmode=0x00000800)  # LOAD_LIBRARY_SEARCH_SYSTEM32
except OSError:
    # Older Qt versions may not need ICU; let the normal Qt import decide.
    pass
```

The subprocess regression registers a conflicting DLL before importing the real entry point and window module, then creates an offscreen widget:

```python
conflicting_dir = os.add_dll_directory(sys.argv[1])
import makeup_cam.__main__
importlib.import_module(sys.argv[2])
from PySide6.QtWidgets import QApplication, QWidget
app = QApplication([])
widget = QWidget()
```

[↑ Back to Table of Contents](#table-of-contents)

## Reasoning

On the affected machine, PySide6 6.12.0's Qt6Core imports unversioned ICU symbols. Conda's ICU 75 DLL exports version-suffixed symbols instead. A normal import selected the Conda DLL and reproduced the reported error; loading the Windows system ICU first made the import succeed. Restricting the preload to the system directory avoids selecting the incompatible search-path entry. The compatibility module suppresses only preload failures, leaving PySide6 to report unrelated import failures normally.

[↑ Back to Table of Contents](#table-of-contents)

## Verification

Before the fix, the two parameterized collision cases failed with `ImportError: DLL load failed while importing QtCore: The specified procedure could not be found.` After the fix:

```bash
.venv/Scripts/python.exe -m pytest -q tests/test_cli.py::test_windows_qt_startup_with_conflicting_icu_dll tests/test_cli.py::test_windows_qt_initialization_tolerates_missing_system_icu
# 3 passed

QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -m pytest -q tests/test_cli.py tests/test_gui.py tests/test_record.py tests/test_parallel_gui.py tests/test_neural_gui.py
# 432 passed, 5 skipped

QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -m pytest -q -rs tests/test_parallel_life.py::test_closing_the_window_ends_everything_within_the_deadline
# 2 skipped: scikit-image is not installed

.venv/Scripts/python.exe -m compileall -q makeup_cam/_qt_compat.py makeup_cam/gui.py makeup_cam/record.py tests/test_cli.py tests/test_gui.py tests/test_record.py tests/test_parallel_life.py
# Exit 0

.venv/Scripts/python.exe -m pip check
# No broken requirements found.
```

`cmd.exe /d /c start_windows.bat --help` exited 0 and displayed CLI usage. A separate temporary harness launched `start_windows.bat` with a synthetic video, `--no-virtual-camera`, `--no-parallel`, an offscreen Qt platform, and a substituted pipeline. The real settings window started and closed with exit code 0; no camera or model download was used.

[↑ Back to Table of Contents](#table-of-contents)

## Residual limits

The preload does not replace a conflicting ICU DLL already loaded by another component before the window module starts. The verified launcher starts a fresh process. The parallel window lifecycle test was skipped because scikit-image is not installed. The full physical-camera and virtual-camera workflow was not exercised in the offscreen smoke test.

[↑ Back to Table of Contents](#table-of-contents)
