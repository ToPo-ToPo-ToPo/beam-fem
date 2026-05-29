"""解析結果の出力先（workspace フォルダ）の管理。

``viz.savefig`` や ``ForceResults.to_csv`` に相対パスを渡すと、ここで解決される
workspace フォルダ（既定 ``./workspace``）の中に保存される。絶対パスを渡した
場合はそのまま使う。出力先は ``set_workspace`` で変更できる。
"""

from __future__ import annotations

import os

_DEFAULT = "workspace"
_current: str | None = None


def set_workspace(path: str) -> str:
    """出力先フォルダを設定する。フォルダは保存時に自動生成される。"""
    global _current
    _current = path
    return _current


def get_workspace() -> str:
    """現在の出力先フォルダ（未設定なら既定 ``workspace``）。"""
    return _current if _current is not None else _DEFAULT


def resolve(filename: str, create: bool = True) -> str:
    """保存用パスを解決する。

    絶対パスはそのまま返す。相対パスは workspace フォルダ内に配置する。
    create=True なら親フォルダを自動生成する。
    """
    if os.path.isabs(filename):
        path = filename
    else:
        path = os.path.join(get_workspace(), filename)
    if create:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    return path
