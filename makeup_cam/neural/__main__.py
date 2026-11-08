"""``python -m makeup_cam.neural``: see which models are there, and choose the active one.

A model is chosen outside the GUI and outside presets (``docs/dev/contract-plan-b.md`` section 6.1).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Sequence

from . import model_files
from .model_files import ModelRefused


def _describe(name: str) -> str:
    folder = model_files.neural_dir() / name
    try:
        spec = model_files.load_spec(folder)
        model_files.check_gate(spec)
    except ModelRefused as exc:
        return f"使えません（{exc.status}）"
    return "評価に合格しています"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m makeup_cam.neural", description="学習モデルの仕上げに使うモデルを確認・選択します")
    parser.add_argument("--use", metavar="NAME", help="使うモデルのフォルダ名（<名前>-<バージョン>）を選びます")
    args = parser.parse_args(argv)
    root = model_files.neural_dir()
    if args.use is not None:
        try:
            model_files.set_active_model(args.use, root)
        except (ValueError, OSError) as exc:
            print(f"選べませんでした: {args.use}（{exc}）", file=sys.stderr)
            return 1
    names = model_files.list_models(root)
    active = model_files.active_model_name(root)
    print(f"モデルの置き場所: {root}")
    if importlib.util.find_spec("onnxruntime") is None:
        print("onnxruntime が入っていません（pip install -r requirements-neural.txt）")
    if not names:
        print("モデルはありません。このアプリにモデルは付属していません。")
    for name in names:
        print(f"{'* ' if name == active else '  '}{name}: {_describe(name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
