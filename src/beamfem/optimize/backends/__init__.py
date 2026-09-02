"""Interchangeable discrete optimization backends."""

from .base import (
    DiscreteOptimizerBackend, DiscreteProblemProtocol, HistoryEntry,
    OptimizationResult, SolverLimits,
)
from .exact import ExactBackend
from .greedy import GreedyBackend
from .milp import MILPBackend, MILPFormulation
from .qaoa import QAOABackend, QiskitNotInstalledError
from .sa import SimulatedAnnealingBackend
from .sequential import SequentialQUBOOptimizer

__all__ = [
    "DiscreteOptimizerBackend", "DiscreteProblemProtocol", "HistoryEntry",
    "OptimizationResult", "SolverLimits", "ExactBackend", "GreedyBackend",
    "MILPBackend", "MILPFormulation",
    "SimulatedAnnealingBackend", "QAOABackend", "QiskitNotInstalledError",
    "SequentialQUBOOptimizer",
]
