"""Adaptive penalty scheduling for feasibility restoration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AdaptivePenalty:
    value: float = 10.0
    minimum: float = 1e-6
    maximum: float = 1e12
    increase: float = 2.0
    decrease: float = 0.8
    target_feasible_rate: float = 0.3
    history: list[float] = field(default_factory=list, init=False)

    def __post_init__(self):
        if not 0 <= self.target_feasible_rate <= 1 or self.value <= 0:
            raise ValueError("invalid adaptive penalty settings")
        self.value = min(self.maximum, max(self.minimum, float(self.value)))

    def update(self, feasible_rate: float) -> float:
        if not 0 <= feasible_rate <= 1:
            raise ValueError("feasible_rate must be between zero and one")
        if feasible_rate < self.target_feasible_rate:
            self.value *= self.increase
        elif feasible_rate > min(1.0, self.target_feasible_rate + 0.4):
            self.value *= self.decrease
        self.value = min(self.maximum, max(self.minimum, self.value))
        self.history.append(self.value)
        return self.value
