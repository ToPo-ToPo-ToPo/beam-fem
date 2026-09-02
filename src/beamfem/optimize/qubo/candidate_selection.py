"""Cheap, deterministic screening of member-change candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Candidate:
    index: int
    score: float
    metadata: Mapping[str, Any]


def select_candidates(records: Iterable[Mapping[str, Any]], limit: int,
                      weights: Mapping[str, float] | None = None) -> tuple[Candidate, ...]:
    """Rank candidates from normalized engineering indicators.

    Recognized indicators are ``mass_saving``, ``utilization``,
    ``strain_energy``, ``buckling_margin``, ``connectivity`` and
    ``recent_improvement``. Callers may override their weights.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    coefficients = {
        "mass_saving": 1.0, "utilization": 0.7, "strain_energy": 0.5,
        "buckling_margin": 0.4, "connectivity": 0.3, "recent_improvement": 0.6,
    }
    if weights:
        coefficients.update(weights)
    candidates = []
    for position, record in enumerate(records):
        index = int(record.get("index", position))
        score = sum(float(record.get(key, 0.0)) * weight for key, weight in coefficients.items())
        candidates.append(Candidate(index, score, dict(record)))
    return tuple(sorted(candidates, key=lambda item: (-item.score, item.index))[:limit])
