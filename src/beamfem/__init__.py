"""beamfem: 梁モデルの FEM 解析と構造最適化。

3D Timoshenko 梁要素（各節点6自由度）を基本とし、2D 面内骨組も扱える。
"""

from .material import Material, Section
from .model import (
    Model, Element, TrussElement, ShellElement, QuadShellElement,
    UX, UY, UZ, RX, RY, RZ, DOF_PER_NODE,
)
from .solver import solve_static, StaticResult, StructuralMechanismError
from .forces import (
    recover_forces,
    ForceResults,
    ElementForces,
    FORCE_COMPONENTS,
    STRESS_COMPONENTS,
)
from .shell import recover_shell_forces, ShellForceResults, ShellForces
from .workspace import set_workspace, get_workspace
from .builders import radial_grillage, lump_pressure, Grillage

__version__ = "1.0.0rc1"

__all__ = [
    "Material",
    "Section",
    "Model",
    "Element",
    "TrussElement",
    "ShellElement",
    "QuadShellElement",
    "solve_static",
    "StaticResult",
    "StructuralMechanismError",
    "recover_shell_forces",
    "ShellForceResults",
    "ShellForces",
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
