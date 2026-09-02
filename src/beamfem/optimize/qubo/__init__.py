"""QUBO construction primitives independent of a quantum SDK."""

from .candidate_selection import Candidate, select_candidates
from .encoding import BinaryEncoding, OneHotEncoding
from .model import QUBOModel, QUBOSolution
from .local import DesignMove, LocalQUBOBuilder, LocalQUBOProblemAdapter
from .penalties import AdaptivePenalty
from .trust_region import TrustRegion

__all__ = ["Candidate", "select_candidates", "BinaryEncoding", "OneHotEncoding",
           "QUBOModel", "QUBOSolution", "AdaptivePenalty", "TrustRegion",
           "DesignMove", "LocalQUBOBuilder", "LocalQUBOProblemAdapter"]
