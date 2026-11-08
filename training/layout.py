"""Dataset folders (contract section 10) and the capture guide's session folders (docs/capture-guide.md, 5.2 and 7.1).

Needs only the standard library and numpy. No path is ever written into a file by this module.

Beyond the contract's layout a sequence may hold (both optional; a reader that does not know them loses nothing):
- ``masks/000000.png``: uint8, the union of the hair and face-skin masks of that input frame. Without it the face oval
  alone decides what a model may change, which is also what the runtime does when it has no masks.
- ``"consecutive": false`` in the manifest: neighbouring frames are not neighbours in time (recordings matched frame by
  frame), so the temporal losses skip the sequence.
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPOSITORY = Path(__file__).resolve().parent.parent

KINDS = ("3d_render", "recording", "classic_pipeline")
LANDMARK_POINTS = 478
# Sequence names like "cremad-1002-IEO_HAP_HI": what comes before the last dash is who is in it. Held-out numbers are
# reported per group, because the worst person counts.
_GROUP = re.compile(r"^(.*)-[^-]+$")

_SESSION_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})_(main|window|night|mock)_(bare|makeup)$")
_TAKE_NAME = re.compile(r"^step(\d+)_([a-z0-9_]+)_take(\d+)\.json$")


class DatasetError(ValueError):
    """A dataset folder cannot be used. The message is for the person running the tool."""


@dataclass(frozen=True)
class Sequence:
    """One folder of aligned frames."""

    name: str
    root: Path
    kind: str
    input_preset: str
    fps: float
    frames: int
    consecutive: bool  # False: neighbouring pairs are not neighbours in time, so no temporal loss on them
    has_masks: bool

    @property
    def group(self) -> str:
        match = _GROUP.match(self.name)
        return match.group(1) if match else self.name

    def input_path(self, index: int) -> Path:
        return self.root / "input" / f"{index:06d}.png"

    def target_path(self, index: int) -> Path:
        return self.root / "target" / f"{index:06d}.png"

    def mask_path(self, index: int) -> Path:
        return self.root / "masks" / f"{index:06d}.png"

    def landmarks(self) -> np.ndarray:
        return np.load(self.root / "landmarks.npy").astype(np.float32)


def in_tracked_folder(folder: Path) -> bool:
    """Would files here be under version control? Faces and weights must never be; ``.test-media/`` is ignored by git.

    This folder of the repository is decided by the path alone, so it holds without git too. Anywhere else (the rest of the
    repository, another one) git is asked. Read-only. With no git, or outside a work tree, the answer is no.
    """
    resolved = folder.resolve()
    if resolved.is_relative_to(_REPOSITORY / ".test-media"):
        return False
    return resolved.is_relative_to(_REPOSITORY) or _git_would_take(resolved)


def _git_would_take(folder: Path) -> bool:
    existing = folder
    while not existing.is_dir() and existing != existing.parent:
        existing = existing.parent
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=existing, capture_output=True, text=True, timeout=10, check=False, creationflags=flags)
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return False
        probe = (folder / "x.png").as_posix()
        ignored = subprocess.run(["git", "check-ignore", "-q", probe], cwd=existing, capture_output=True, timeout=10, check=False, creationflags=flags)
        return ignored.returncode == 1  # 0: ignored, 128: git could not tell
    except (OSError, subprocess.TimeoutExpired):
        return False


def read_sequence(folder: Path) -> Sequence:
    """Reads and checks one sequence folder. Reads the manifest and the landmarks, not the frames."""
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetError(f"{folder.name}: manifest.json を読めません") from exc
    except ValueError as exc:
        raise DatasetError(f"{folder.name}: manifest.json が JSON として読めません: {exc}") from exc
    kind = manifest.get("kind")
    if kind not in KINDS:
        raise DatasetError(f"{folder.name}: kind は {KINDS} のどれかです（{kind!r}）")
    frames = manifest.get("frames")
    if not isinstance(frames, int) or frames < 1:
        raise DatasetError(f"{folder.name}: frames が正しくありません（{frames!r}）")
    try:
        landmarks = np.load(folder / "landmarks.npy", mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise DatasetError(f"{folder.name}: landmarks.npy を読めません") from exc
    if landmarks.shape != (frames, LANDMARK_POINTS, 2):
        raise DatasetError(f"{folder.name}: landmarks.npy の形が ({frames}, {LANDMARK_POINTS}, 2) ではありません: {landmarks.shape}")
    for side in ("input", "target"):
        for index in (0, frames - 1):
            if not (folder / side / f"{index:06d}.png").is_file():
                raise DatasetError(f"{folder.name}: {side}/{index:06d}.png がありません")
    return Sequence(
        name=folder.name,
        root=folder,
        kind=kind,
        input_preset=str(manifest.get("input_preset", "")),
        fps=float(manifest.get("fps", 30)),
        frames=frames,
        consecutive=bool(manifest.get("consecutive", True)),
        has_masks=(folder / "masks").is_dir(),
    )


def write_manifest(
    folder: Path, kind: str, input_preset: str, fps: float, frames: int, consecutive: bool = True, note: str = ""
) -> None:
    if kind not in KINDS:
        raise DatasetError(f"kind は {KINDS} のどれかです（{kind!r}）")
    manifest = {"kind": kind, "input_preset": input_preset, "fps": fps, "frames": frames, "consecutive": consecutive}
    if note:
        manifest["note"] = note
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def find_sequences(root: Path) -> list[Sequence]:
    """Every sequence directly under ``root``, by name."""
    if not root.is_dir():
        raise DatasetError("データのフォルダがありません")
    # A leading dot marks a sequence still being written.
    folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".") and (p / "manifest.json").is_file())
    if not folders:
        raise DatasetError("データのフォルダに manifest.json を持つシーケンスがありません")
    return [read_sequence(folder) for folder in folders]


def split_sequences(
    sequences: list[Sequence], train_patterns: list[str], heldout_patterns: list[str]
) -> tuple[list[Sequence], list[Sequence]]:
    """(train, held out). Held out always wins, and a whole group goes together, so that no frame of a held-out
    person can reach training by a pattern that names only some of their sequences."""
    heldout_groups = {s.group for s in sequences if _matches(s.name, heldout_patterns)}
    heldout = [s for s in sequences if s.group in heldout_groups]
    train = [s for s in sequences if s.group not in heldout_groups and _matches(s.name, train_patterns)]
    return train, heldout


def data_summary(sequences: list[Sequence]) -> dict:
    """What a model was trained on, for its checkpoint and model.json. Counts and kinds only."""
    return {
        "kinds": sorted({s.kind for s in sequences}),
        "input_presets": sorted({s.input_preset for s in sequences}),
        "sequences": len(sequences),
        "groups": len({s.group for s in sequences}),
        "frames": sum(s.frames for s in sequences),
    }


# --------------------------------------------------------------------------
# The capture guide's session folders: bare and made-up recordings of one day.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Take:
    step: int
    key: str
    take: int
    video: Path
    fps: float
    frames: int


@dataclass(frozen=True)
class SessionPair:
    """The takes of one day and light that exist in both conditions, by step number."""

    date: str
    slot: str
    bare: dict[int, Take]
    makeup: dict[int, Take]

    @property
    def steps(self) -> list[int]:
        return sorted(set(self.bare) & set(self.makeup))

    @property
    def name(self) -> str:
        return f"{self.date}_{self.slot}"


def completed_takes(folder: Path) -> dict[int, Take]:
    """{step: the newest take with ``"status": "completed"`` whose video is there}. The guide's rule, section 5.2."""
    takes: dict[int, Take] = {}
    for sidecar in sorted(folder.glob("step*_take*.json")):
        match = _TAKE_NAME.match(sidecar.name)
        if match is None:
            continue
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if data.get("status") != "completed":
                continue
            video = folder / data["video"]["file"]
            take = Take(
                step=int(match.group(1)),
                key=match.group(2),
                take=int(match.group(3)),
                video=video,
                fps=float(data["fps"]["container"]),
                frames=int(data["frames"]["written"]),
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        if not video.is_file():
            continue
        if take.step not in takes or take.take > takes[take.step].take:
            takes[take.step] = take
    return takes


def find_session_pairs(capture_root: Path) -> list[SessionPair]:
    """Pairs ``sessions/<date>_<slot>_bare`` with ``..._makeup``. ``eval/`` and mock sessions are never used:
    the guide keeps them for the evaluation, section 6, and a model must not be judged on what it trained on."""
    sessions = capture_root / "sessions"
    if not sessions.is_dir():
        raise DatasetError("sessions フォルダがありません（撮影ガイド 7.1）")
    found: dict[tuple[str, str], dict[str, Path]] = {}
    for folder in sessions.iterdir():
        match = _SESSION_NAME.match(folder.name)
        if folder.is_dir() and match and match.group(2) != "mock":
            found.setdefault((match.group(1), match.group(2)), {})[match.group(3)] = folder
    pairs = []
    for (date, slot), sides in sorted(found.items()):
        if len(sides) == 2:
            pairs.append(SessionPair(date, slot, completed_takes(sides["bare"]), completed_takes(sides["makeup"])))
    return pairs
