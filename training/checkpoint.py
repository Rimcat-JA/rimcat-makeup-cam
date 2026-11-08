"""Checkpoints: written atomically, resumed exactly. Needs torch.

A rented machine can disappear at any moment and a laptop can be closed, so a checkpoint is either complete or absent:
it goes to a temporary file and is renamed. Only numbers go in: weights, optimizer state, the step, the config (which
holds no paths) and a summary of the data (kinds and counts).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

LAST = "last.pt"
_STEP_FILE = re.compile(r"^step-(\d+)\.pt$")


def save(folder: Path, step: int, state: dict[str, Any], keep: int) -> Path:
    """Writes ``step-NNNNNNNN.pt`` and ``last.pt``, and removes all but the newest ``keep`` step files."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"step-{step:08d}.pt"
    temporary = folder / f".{path.name}.tmp"
    torch.save(state, temporary)
    os.replace(temporary, path)
    last_temporary = folder / f".{LAST}.tmp"
    torch.save(state, last_temporary)
    os.replace(last_temporary, folder / LAST)
    steps = sorted((p for p in folder.iterdir() if _STEP_FILE.match(p.name)), key=lambda p: p.name)
    for old in steps[: max(0, len(steps) - keep)]:
        old.unlink()
    return path


def load(path: Path) -> dict[str, Any]:
    # No pickled code: torch's restricted loader takes numbers, tensors and plain containers, which is all that goes in.
    # There is no random state to keep either: everything random is derived from (seed, step), see samples.py.
    return torch.load(path, map_location="cpu", weights_only=True)


def to_cpu(state: Any) -> Any:
    """A copy of a state dict, nested ones too, with every tensor on the CPU: a file must load on any device."""
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().clone()
    if isinstance(state, dict):
        return {key: to_cpu(value) for key, value in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(to_cpu(value) for value in state)
    return state
