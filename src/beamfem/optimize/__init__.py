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
from .discrete import (
    DiscreteResult,
    solve_discrete_exhaustive,
    solve_discrete_greedy,
)
from .catalogs import SectionOption, SectionCatalog
from .problem import DesignState, LoadCase, LoadCombination, DiscreteStructuralProblem
from .objectives import MassObjective, WeightedImpactObjective, impact_components
from .pareto import ParetoFrontBackend, ParetoPoint, ParetoResult
from .evaluation import StructuralEvaluator, EvaluationResult, CombinationAnalysis
from .persistent_cache import PersistentEvaluationCache, problem_context_checksum
from .constraints import (
    ConstraintRecord,
    RequiredMembers,
    ForbiddenMembers,
    SameSectionGroup,
    MaxSectionTypes,
    ActiveMemberCount,
    SymmetryPairs,
    Connectivity,
    MemberLengthRange,
    SectionSlendernessLimit,
    DisplacementLimit as DiscreteDisplacementLimit,
    RelativeDisplacementLimit,
    StressLimit,
    EulerBucklingLimit,
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
    "DiscreteResult",
    "solve_discrete_exhaustive",
    "solve_discrete_greedy",
    "SectionOption",
    "SectionCatalog",
    "DesignState",
    "LoadCase",
    "LoadCombination",
    "DiscreteStructuralProblem",
    "MassObjective",
    "WeightedImpactObjective",
    "impact_components",
    "ParetoFrontBackend",
    "ParetoPoint",
    "ParetoResult",
    "StructuralEvaluator",
    "EvaluationResult",
    "CombinationAnalysis",
    "PersistentEvaluationCache",
    "problem_context_checksum",
    "ConstraintRecord",
    "RequiredMembers",
    "ForbiddenMembers",
    "SameSectionGroup",
    "MaxSectionTypes",
    "ActiveMemberCount",
    "SymmetryPairs",
    "Connectivity",
    "MemberLengthRange",
    "SectionSlendernessLimit",
    "DiscreteDisplacementLimit",
    "RelativeDisplacementLimit",
    "StressLimit",
    "EulerBucklingLimit",
]
