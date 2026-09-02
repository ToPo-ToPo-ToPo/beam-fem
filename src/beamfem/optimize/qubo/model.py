"""Solver-neutral quadratic unconstrained binary optimization model."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class QUBOSolution:
    bits: tuple[int, ...]
    energy: float
    metadata: dict | None = None


@dataclass(frozen=True)
class QUBOModel:
    """QUBO ``c + linear*x + sum(i<j) quadratic[i,j]*x_i*x_j``.

    Diagonal entries passed to ``quadratic`` are folded into the linear term,
    because x²=x for binary variables.  Lower/upper triangular inputs are
    combined, making the stored representation unambiguous.
    """

    linear: np.ndarray
    quadratic: np.ndarray
    constant: float = 0.0
    variable_names: tuple[str, ...] = ()

    def __post_init__(self):
        linear = np.asarray(self.linear, dtype=float).copy()
        quadratic = np.asarray(self.quadratic, dtype=float).copy()
        if linear.ndim != 1 or quadratic.shape != (linear.size, linear.size):
            raise ValueError("quadratic must be square and match linear")
        if not np.all(np.isfinite(linear)) or not np.all(np.isfinite(quadratic)):
            raise ValueError("QUBO coefficients must be finite")
        linear += np.diag(quadratic)
        upper = np.triu(quadratic, 1) + np.tril(quadratic, -1).T
        names = self.variable_names or tuple(f"x{i}" for i in range(linear.size))
        if len(names) != linear.size or len(set(names)) != len(names):
            raise ValueError("variable_names must be unique and match model size")
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "quadratic", upper)
        object.__setattr__(self, "constant", float(self.constant))
        object.__setattr__(self, "variable_names", tuple(names))

    @property
    def n_variables(self) -> int:
        return int(self.linear.size)

    def energy(self, bits: Sequence[int]) -> float:
        x = np.asarray(bits, dtype=float)
        if x.shape != (self.n_variables,) or np.any((x != 0) & (x != 1)):
            raise ValueError("bits must contain exactly n_variables binary values")
        return float(self.constant + self.linear @ x + x @ self.quadratic @ x)

    def energies(self, samples: Iterable[Sequence[int]]) -> np.ndarray:
        return np.asarray([self.energy(bits) for bits in samples])

    def exact_solution(self, max_variables: int = 24) -> QUBOSolution:
        if self.n_variables > max_variables:
            raise ValueError(f"exact QUBO solve limited to {max_variables} variables")
        best = min(product((0, 1), repeat=self.n_variables), key=self.energy)
        return QUBOSolution(tuple(best), self.energy(best), {"method": "enumeration"})

    def normalized(self, target: float = 1.0) -> tuple["QUBOModel", float]:
        largest = max(
            abs(self.constant),
            float(np.max(np.abs(self.linear), initial=0.0)),
            float(np.max(np.abs(self.quadratic), initial=0.0)),
        )
        scale = 1.0 if largest == 0.0 else float(target) / largest
        return QUBOModel(self.linear * scale, self.quadratic * scale,
                         self.constant * scale, self.variable_names), scale

    def with_penalty(self, linear: Sequence[float], quadratic: np.ndarray,
                     constant: float = 0.0, weight: float = 1.0) -> "QUBOModel":
        return QUBOModel(
            self.linear + weight * np.asarray(linear, dtype=float),
            self.quadratic + weight * np.asarray(quadratic, dtype=float),
            self.constant + weight * float(constant), self.variable_names,
        )
