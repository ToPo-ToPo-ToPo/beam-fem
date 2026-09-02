"""beamfem: 梁モデルの FEM 解析と構造最適化。

3D Timoshenko 梁要素（各節点6自由度）を基本とし、2D 面内骨組も扱える。
"""

from .material import Material, Section
from .model import (
    Model, Element, TrussElement, ShellElement, QuadShellElement,
    UX, UY, UZ, RX, RY, RZ, DOF_PER_NODE,
)
from .solver import (
    SparseSolver, SciPyLUSolver, available_sparse_solvers, factorize_static,
    get_sparse_solver, register_sparse_solver, solve_static, StaticResult,
    StructuralMechanismError,
)
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
from .modal import ModalResult, assemble_lumped_mass, solve_modes
from .nonlinear_material import (
    BilinearIsotropicHardening,
    ElasticPerfectlyPlastic,
    UniaxialMaterialModel,
    UniaxialMaterialResponse,
    UniaxialMaterialState,
)
from .nonlinear_truss import (
    CollapseEvent,
    LimitStateReport,
    NonlinearConvergenceError,
    NonlinearDesignEvaluation,
    NonlinearElementResult,
    NonlinearStepResult,
    NonlinearTrussResult,
    NonlinearTrussSubproblem,
    solve_nonlinear_truss,
)

__version__ = "1.0.0rc2"

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
    "SparseSolver",
    "SciPyLUSolver",
    "factorize_static",
    "register_sparse_solver",
    "available_sparse_solvers",
    "get_sparse_solver",
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
    "ModalResult",
    "assemble_lumped_mass",
    "solve_modes",
    "ElasticPerfectlyPlastic",
    "BilinearIsotropicHardening",
    "UniaxialMaterialModel",
    "UniaxialMaterialResponse",
    "UniaxialMaterialState",
    "NonlinearConvergenceError",
    "CollapseEvent",
    "NonlinearElementResult",
    "NonlinearStepResult",
    "NonlinearTrussResult",
    "LimitStateReport",
    "NonlinearDesignEvaluation",
    "NonlinearTrussSubproblem",
    "solve_nonlinear_truss",
    "UX",
    "UY",
    "UZ",
    "RX",
    "RY",
    "RZ",
    "DOF_PER_NODE",
]
