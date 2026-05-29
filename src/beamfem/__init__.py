"""beamfem: 梁モデルの FEM 解析と構造最適化。

3D Timoshenko 梁要素（各節点6自由度）を基本とし、2D 面内骨組も扱える。
"""

from .material import Material, Section
from .model import Model, Element, UX, UY, UZ, RX, RY, RZ, DOF_PER_NODE
from .solver import solve_static, StaticResult
from .forces import (
    recover_forces,
    ForceResults,
    ElementForces,
    FORCE_COMPONENTS,
    STRESS_COMPONENTS,
)
from .workspace import set_workspace, get_workspace
from .builders import radial_grillage, lump_pressure, Grillage

__version__ = "0.1.0"

__all__ = [
    "Material",
    "Section",
    "Model",
    "Element",
    "solve_static",
    "StaticResult",
    "recover_forces",
    "ForceResults",
    "ElementForces",
    "FORCE_COMPONENTS",
    "STRESS_COMPONENTS",
    "set_workspace",
    "get_workspace",
    "radial_grillage",
    "lump_pressure",
    "Grillage",
    "UX",
    "UY",
    "UZ",
    "RX",
    "RY",
    "RZ",
    "DOF_PER_NODE",
]
