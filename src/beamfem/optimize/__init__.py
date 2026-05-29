"""構造最適化サブパッケージ。

断面サイジング最適化（解析的感度 + MMA、応力・たわみ制約下の質量最小化）。
"""

from .sections import ScaledSection
from .sizing import SizingProblem, DesignVar, DispLimit
from .driver import minimize_mass, OptResult
from .mma import mmasub, subsolv

__all__ = [
    "ScaledSection",
    "SizingProblem",
    "DesignVar",
    "DispLimit",
    "minimize_mass",
    "OptResult",
    "mmasub",
    "subsolv",
]
