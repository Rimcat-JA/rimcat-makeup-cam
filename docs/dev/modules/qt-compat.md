# Qt compatibility module

<a id="table-of-contents"></a>
## Table of Contents

- [Purpose and position](#purpose-and-position)
- [Initialization and limits](#initialization-and-limits)
- [Verification](#verification)

## Purpose and position

`makeup_cam._qt_compat` prepares the Windows DLL loader before the settings window or recording tool imports PySide6. It prevents a Conda ICU DLL with incompatible exports from being selected instead of Windows' system ICU DLL.

[↑ Back to Table of Contents](#table-of-contents)

## Initialization and limits

The module imports only Python's `sys` and `ctypes` and has no public functions or configuration. On Windows, it loads `icuuc.dll` from the system directory and retains the library object. If the system DLL is unavailable, the module leaves the normal PySide6 import to decide whether the application can start. Other operating systems do not load a DLL.

The preload must occur before an incompatible ICU DLL has already been loaded into the process. The module does not unload libraries belonging to another component.

[↑ Back to Table of Contents](#table-of-contents)

## Verification

`tests/test_cli.py::test_windows_qt_startup_with_conflicting_icu_dll` exercises both window modules in fresh Windows subprocesses and creates an offscreen Qt widget with a conflicting DLL on the search path.

[↑ Back to Table of Contents](#table-of-contents)
