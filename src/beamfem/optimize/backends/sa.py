"""Reproducible simulated annealing for QUBO master problems."""

from __future__ import annotations

from math import exp, isfinite
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from ..qubo import QUBOModel, QUBOSolution
from .base import (
    OptimizationResult, SolverLimits, design_values, evaluate_problem,
    evaluation_constraints, evaluation_feasible, evaluation_objective, make_design,
)


Decoder = Callable[[Sequence[int]], Sequence[int] | Any]


def _resolve_qubo(problem: Any, initial_design: Any, supplied: QUBOModel | None,
                  decoder: Decoder | None) -> tuple[QUBOModel, Decoder]:
    if supplied is not None:
        return supplied, decoder or (lambda bits: bits)
    builder = getattr(problem, "build_qubo", None)
    if builder is None:
        raise TypeError("QUBO backend requires qubo=... or problem.build_qubo(initial_design)")
    built = builder(initial_design)
    if isinstance(built, QUBOModel):
        return built, decoder or (lambda bits: bits)
    model, built_decoder = built
    return model, decoder or built_decoder


class SimulatedAnnealingBackend:
    def __init__(self, qubo: QUBOModel | None = None, decoder: Decoder | None = None,
                 sweeps: int = 2_000, restarts: int = 8, seed: int = 0,
                 initial_temperature: float | None = None,
                 final_temperature: float = 1e-3):
        if (
            sweeps < 1 or restarts < 1 or not isfinite(float(final_temperature))
            or final_temperature <= 0
            or (initial_temperature is not None and (
                not isfinite(float(initial_temperature)) or initial_temperature <= 0
            ))
        ):
            raise ValueError("invalid annealing settings")
        self.qubo, self.decoder = qubo, decoder
        self.sweeps, self.restarts, self.seed = int(sweeps), int(restarts), int(seed)
        self.initial_temperature = initial_temperature
        self.final_temperature = float(final_temperature)

    def solve_qubo(self, model: QUBOModel, initial_bits: Sequence[int] | None = None) -> QUBOSolution:
        rng = np.random.default_rng(self.seed)
        scale = max(float(np.max(np.abs(model.linear), initial=0.0)),
                    float(np.max(np.abs(model.quadratic), initial=0.0)), 1.0)
        t0 = float(self.initial_temperature or 2.0 * scale)
        best_bits, best_energy = None, float("inf")
        accepted = 0
        for restart in range(self.restarts):
            if restart == 0 and initial_bits is not None:
                bits = np.asarray(initial_bits, dtype=int).copy()
            else:
                bits = rng.integers(0, 2, size=model.n_variables)
            energy = model.energy(bits)
            if energy < best_energy:
                best_bits, best_energy = bits.copy(), energy
            for sweep in range(self.sweeps):
                progress = sweep / max(1, self.sweeps - 1)
                temperature = t0 * (self.final_temperature / t0) ** progress
                i = int(rng.integers(model.n_variables))
                trial = bits.copy(); trial[i] ^= 1
                trial_energy = model.energy(trial)
                delta = trial_energy - energy
                if delta <= 0 or rng.random() < exp(-delta / temperature):
                    bits, energy = trial, trial_energy
                    accepted += 1
                    if energy < best_energy:
                        best_bits, best_energy = bits.copy(), energy
        return QUBOSolution(tuple(int(v) for v in best_bits), float(best_energy),
            {"sweeps": self.sweeps, "restarts": self.restarts, "seed": self.seed,
             "accepted_moves": accepted})

    def solve(self, problem: Any, initial_design: Any | None = None,
              limits: SolverLimits | None = None) -> OptimizationResult:
        started = perf_counter()
        template = initial_design if initial_design is not None else problem.initial_design
        model, decoder = _resolve_qubo(problem, template, self.qubo, self.decoder)
        solution = self.solve_qubo(model)
        decoded = decoder(solution.bits)
        if isinstance(decoded, (tuple, list, np.ndarray)):
            design = make_design(problem, decoded, template)
        else:
            design = decoded
        evaluation = evaluate_problem(problem, design)
        metadata = dict(solution.metadata or {})
        metadata.update({"qubo_energy": solution.energy, "qubo_bits": solution.bits,
                         "qubo_variables": model.n_variables})
        return OptimizationResult(design, evaluation_objective(evaluation), evaluation_feasible(evaluation),
            evaluation_constraints(evaluation), self.sweeps * self.restarts, 1,
            perf_counter()-started, "sa", "success", "annealing complete",
            metadata, evaluation=evaluation)
