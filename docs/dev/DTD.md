# Detailed technical design: Windows Qt initialization

**Status:** Implementation-ready. **Mode:** Standard. **Updated:** 2026-11-11.

**Specification:** [`../../specification.md`](../../specification.md).

**Scope:** Windows DLL initialization for the settings window and recording tool. There are no open product decisions.

<a id="table-of-contents"></a>
## Table of Contents

- [Technical scope and acceptance](#technical-scope-and-acceptance)
- [Architecture and dependencies](#architecture-and-dependencies)
- [File and initialization contract](#file-and-initialization-contract)
- [Runtime behavior and failures](#runtime-behavior-and-failures)
- [Testing and operation](#testing-and-operation)
- [Implementation roadmap](#implementation-roadmap)
- [Implementation Instructions for Coding Agent](#implementation-instructions-for-coding-agent)

## Technical scope and acceptance

Both existing window modules must import PySide6 on Windows when the DLL search path contains an incompatible `icuuc.dll`. The regression must create a Qt widget offscreen, without a camera or model download. CLI options, package versions, camera behavior, persistence, networking, and deployment remain unchanged. The implementation must preserve Python 3.10+ and existing non-Windows behavior.

[↑ Back to Table of Contents](#table-of-contents)

## Architecture and dependencies

`makeup_cam.gui` and `makeup_cam.record` depend on `makeup_cam._qt_compat`; that module depends only on Python's `sys` and `ctypes`. Both windows continue to depend on PySide6. The compatibility module must not import either window, NumPy, OpenCV, or PySide6.

The existing PySide6 6.12.0 installation was observed to import unversioned ICU functions from `icuuc.dll`. Conda's ICU 75 DLL exports version-suffixed functions instead, so it is not an interchangeable implementation. No package version constraint changes are needed.

Python 3.10+ supplies `ctypes.WinDLL(name, winmode=...)`. On Windows, use `winmode=0x00000800`, the `LOAD_LIBRARY_SEARCH_SYSTEM32` flag, to select the system directory rather than an added DLL search directory. Keep the resulting library object in the compatibility module for the process lifetime. References checked on 2026-11-11: [Python ctypes library loading](https://docs.python.org/3/library/ctypes.html#loading-shared-libraries) and [Microsoft LoadLibraryExW flags](https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexw).

[↑ Back to Table of Contents](#table-of-contents)

## File and initialization contract

- `makeup_cam/_qt_compat.py` is an import-only compatibility module. On `sys.platform == "win32"`, it loads `icuuc.dll` through `ctypes.WinDLL` with `LOAD_LIBRARY_SEARCH_SYSTEM32`; the module stores the returned object in a private variable. On other platforms it does not load a DLL. No function, public API, configuration key, structured payload, class, or persistent state is introduced.
- `makeup_cam/gui.py` and `makeup_cam/record.py` import the compatibility module before their PySide6 imports. Existing window APIs and dependency direction stay unchanged.
- `tests/test_cli.py` launches fresh Windows subprocesses with a temporary incompatible `icuuc.dll` registered on the DLL search path. It imports the CLI and each real window module and creates an offscreen Qt widget. A separate test simulates a missing system ICU DLL.
- `tests/test_gui.py`, `tests/test_record.py`, and the GUI subprocess in `tests/test_parallel_life.py` import the application window before importing PySide6 directly, matching the application startup order.

[↑ Back to Table of Contents](#table-of-contents)

## Runtime behavior and failures

The compatibility module catches `OSError` from the preload only. This preserves compatibility with Qt versions that do not require system ICU on machines where the system DLL is missing. PySide6 imports are not caught or retried, so unrelated errors remain visible through the existing CLI error boundary.

The Windows loader can retain a conflicting DLL that another component loaded before this initialization. Loading two DLLs with the same name does not reliably replace that earlier module. The normal launcher uses a fresh process; this change does not attempt to unload libraries owned by other components.

There are no network calls, credentials, new settings, retries, schema changes, migrations, threads, or external services in this flow.

[↑ Back to Table of Contents](#table-of-contents)

## Testing and operation

On Windows, run `.venv\Scripts\python.exe -m pytest -q tests/test_cli.py::test_windows_qt_startup_with_conflicting_icu_dll tests/test_cli.py::test_windows_qt_initialization_tolerates_missing_system_icu`. The collision cases copy the Windows `version.dll` under the conflicting name in a pytest temporary directory, register that directory in a subprocess, import the real application modules, and create an offscreen widget. Both cases must succeed; before this change they reproduce the reported QtCore import error. The remaining test simulates an unsuccessful system ICU preload.

Also run focused CLI and GUI tests, check `python -m makeup_cam --help`, and perform a camera-free startup smoke test. The application retains the existing `start_windows.bat` entry point and dependency installation behavior.

[↑ Back to Table of Contents](#table-of-contents)

## Implementation roadmap

1. Add the failing subprocess regression for both window modules and verify the original QtCore error.
2. Add the import-only compatibility module and import it before PySide6 in both window modules. No public interfaces change.
3. Re-run the regression, relevant CLI and GUI tests, and a Windows launcher smoke test. Record actual results and any environment limitations in the change evidence.
4. Reconcile this document, the specification, and the Qt compatibility module note with the implementation. Preserve unrelated files and dependencies.

[↑ Back to Table of Contents](#table-of-contents)

## Implementation Instructions for Coding Agent

Implement the roadmap in order and do not skip prerequisites. Treat this document as the contract; preserve dependency direction and do not silently change dependencies, APIs, schemas, boundaries, or the chosen loader flags. Run the specified verification as each step permits and update tests alongside implementation. Report incompatible APIs or unmet acceptance criteria. If evidence invalidates the design, update the design before proceeding. Keep deferred work deferred and preserve unrelated repository changes.

[↑ Back to Table of Contents](#table-of-contents)
