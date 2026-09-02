"""Adaptive Hamming trust region using predicted/actual improvement ratio."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrustRegion:
    radius: int = 2
    minimum: int = 1
    maximum: int = 8
    shrink_threshold: float = 0.25
    expand_threshold: float = 0.75
    history: list[dict[str, float | int | bool]] = field(default_factory=list, init=False)

    def __post_init__(self):
        if not 1 <= self.minimum <= self.radius <= self.maximum:
            raise ValueError("trust-region radii must satisfy 1 <= min <= radius <= max")

    def update(self, predicted_improvement: float, actual_improvement: float) -> tuple[float, bool]:
        if predicted_improvement <= 0:
            rho = float("-inf")
        else:
            rho = actual_improvement / predicted_improvement
        accepted = actual_improvement > 0 and rho >= self.shrink_threshold
        old = self.radius
        if rho < self.shrink_threshold:
            self.radius = max(self.minimum, self.radius - 1)
        elif rho >= self.expand_threshold and actual_improvement > 0:
            self.radius = min(self.maximum, self.radius + 1)
        self.history.append({"old_radius": old, "radius": self.radius,
                             "rho": rho, "accepted": accepted})
        return rho, accepted

    def contains(self, base: tuple[int, ...], candidate: tuple[int, ...]) -> bool:
        return sum(a != b for a, b in zip(base, candidate)) <= self.radius
