"""構造最適化サブパッケージ。

断面サイジング最適化（解析的感度 + MMA、応力・たわみ制約下の質量最小化）。
"""

from .sections import ScaledSection
from .sizing import SizingProblem, DesignVar, DispLimit
from .driver import minimize_mass, OptResult
from .mma import mmasub, subsolv
from .topology import (
    GroundStructure,
    TopologyResult,
    solve_min_volume,
    generate_members,
    grid_nodes,
    equilibrium_matrix,
)

__all__ = [
    "ScaledSection",
    "SizingProblem",
    "DesignVar",
    "DispLimit",
    "minimize_mass",
    "OptResult",
    "mmasub",
    "subsolv",
    "GroundStructure",
    "TopologyResult",
    "solve_min_volume",
    "generate_members",
    "grid_nodes",
    "equilibrium_matrix",
]
