"""Optional Qiskit QAOA backend.

Qiskit is imported only when this backend runs, so beamfem and all classical
backends remain usable without quantum dependencies.
"""

from __future__ import annotations

from time import perf_counter
import math
from typing import Any, Callable, Sequence

import numpy as np

from ..qubo import QUBOModel, QUBOSolution
from .base import (
    OptimizationResult, SolverLimits, evaluate_problem, evaluation_constraints,
    evaluation_feasible, evaluation_objective, make_design,
)
from .sa import Decoder, _resolve_qubo


class QiskitNotInstalledError(ImportError):
    pass


def _qiskit_components():
    try:
        from qiskit.primitives import StatevectorSampler
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit_optimization.minimum_eigensolvers import QAOA
        from qiskit_optimization.optimizers import COBYLA
        from qiskit_optimization.utils import algorithm_globals
    except ImportError as exc:
        raise QiskitNotInstalledError(
            "QAOA requires optional packages: qiskit and qiskit-optimization"
        ) from exc
    return (StatevectorSampler, COBYLA, QuadraticProgram,
            MinimumEigenOptimizer, QAOA, algorithm_globals)


class QAOABackend:
    def __init__(self, qubo: QUBOModel | None = None, decoder: Decoder | None = None,
                 sampler: Any | None = None, sampler_factory: Callable[[], Any] | None = None,
                 optimizer: Any | None = None, reps: int = 1, shots: int | None = 1024,
                 seed: int = 0, pass_manager: Any | None = None, maxiter: int = 100,
                 execution_label: str | None = None,
                 noise_model_description: str | None = None,
                 execution_metadata_provider: Callable[[Any, Any], dict[str, Any]] | None = None,
                 exact_reference_max_variables: int = 20):
        if reps < 1:
            raise ValueError("reps must be positive")
        if maxiter < 1:
            raise ValueError("maxiter must be positive")
        if shots is not None and shots < 1:
            raise ValueError("shots must be positive or None")
        if exact_reference_max_variables < 0:
            raise ValueError("exact_reference_max_variables cannot be negative")
        self.qubo, self.decoder = qubo, decoder
        self.sampler, self.sampler_factory = sampler, sampler_factory
        self.optimizer, self.reps, self.shots = optimizer, int(reps), shots
        self.seed, self.pass_manager, self.maxiter = int(seed), pass_manager, int(maxiter)
        self.execution_label = execution_label
        self.noise_model_description = noise_model_description
        self.execution_metadata_provider = execution_metadata_provider
        self.exact_reference_max_variables = int(exact_reference_max_variables)

    def solve_qubo(self, model: QUBOModel) -> QUBOSolution:
        (StatevectorSampler, COBYLA, QuadraticProgram,
         MinimumEigenOptimizer, QAOA, algorithm_globals) = _qiskit_components()
        algorithm_globals.random_seed = self.seed
        qp = QuadraticProgram("beamfem_qubo")
        for name in model.variable_names:
            qp.binary_var(name=name)
        linear = {name: float(model.linear[i]) for i, name in enumerate(model.variable_names)
                  if model.linear[i] != 0}
        quadratic = {(model.variable_names[i], model.variable_names[j]): float(model.quadratic[i, j])
                     for i in range(model.n_variables) for j in range(i + 1, model.n_variables)
                     if model.quadratic[i, j] != 0}
        qp.minimize(constant=model.constant, linear=linear, quadratic=quadratic)
        sampler = self.sampler or (self.sampler_factory() if self.sampler_factory else None)
        if sampler is None:
            default_shots = self.shots if self.shots is not None else 1024
            sampler = StatevectorSampler(default_shots=default_shots, seed=self.seed)
        optimizer = self.optimizer or COBYLA(maxiter=self.maxiter)
        kwargs = {"sampler": sampler, "optimizer": optimizer, "reps": self.reps}
        if self.pass_manager is not None:
            kwargs["pass_manager"] = self.pass_manager
        qaoa = QAOA(**kwargs)
        quantum_started = perf_counter()
        try:
            result = MinimumEigenOptimizer(qaoa).solve(qp)
        except Exception as exc:
            raise RuntimeError(f"QAOA execution failed: {exc}") from exc
        quantum_wall_time = perf_counter() - quantum_started
        samples = list(getattr(result, "samples", None) or [])
        if samples:
            checked_samples = []
            for sample in samples:
                raw = np.asarray(sample.x, dtype=float)
                if raw.shape != (model.n_variables,) or not np.all(np.isfinite(raw)):
                    raise RuntimeError("QAOA returned a non-finite or malformed sample")
                sample_bits = tuple(int(round(v)) for v in raw)
                if any(bit not in (0, 1) for bit in sample_bits):
                    raise RuntimeError("QAOA returned values outside the binary domain")
                checked_samples.append((model.energy(sample_bits), sample, sample_bits))
            _, selected, bits = min(checked_samples, key=lambda item: item[0])
            probability = float(selected.probability)
            if not math.isfinite(probability):
                probability = None
        else:
            raw = np.asarray(result.x, dtype=float)
            if raw.shape != (model.n_variables,) or not np.all(np.isfinite(raw)):
                raise RuntimeError("QAOA returned a non-finite or malformed bit vector")
            bits = tuple(int(round(v)) for v in raw)
            probability = None
        if len(bits) != model.n_variables or any(bit not in (0, 1) for bit in bits):
            raise RuntimeError("QAOA returned values outside the binary domain")
        fval = float(result.fval)
        if not math.isfinite(fval):
            fval = None
        metadata = {"reps": self.reps, "shots": self.shots, "seed": self.seed,
                    "maxiter": self.maxiter,
                    "qiskit_status": str(result.status), "fval": fval,
                    "selected_probability": probability, "samples": len(samples),
                    "execution_label": self.execution_label or type(sampler).__name__,
                    "noise_model": self.noise_model_description,
                    "qaoa_wall_time": quantum_wall_time,
                    "quantum_execution_time": None}
        minimum_result = getattr(result, "min_eigen_solver_result", None)
        circuit = getattr(minimum_result, "optimal_circuit", None)
        if minimum_result is not None:
            optimizer_time = getattr(minimum_result, "optimizer_time", None)
            evaluations = getattr(minimum_result, "cost_function_evals", None)
            metadata["classical_optimizer_time"] = None if optimizer_time is None else float(optimizer_time)
            metadata["cost_function_evaluations"] = None if evaluations is None else int(evaluations)
        if circuit is not None:
            metadata["logical_qubits"] = int(circuit.num_qubits)
            metadata["logical_circuit_depth"] = int(circuit.depth())
            metadata["logical_two_qubit_gates"] = sum(
                1 for instruction in circuit.data if len(instruction.qubits) == 2
            )
            metadata["logical_gate_counts"] = {str(k): int(v) for k, v in circuit.count_ops().items()}
        if model.n_variables <= self.exact_reference_max_variables:
            exact = model.exact_solution(max_variables=self.exact_reference_max_variables)
            metadata["exact_reference_energy"] = exact.energy
            metadata["energy_gap_to_exact"] = model.energy(bits) - exact.energy
        if self.execution_metadata_provider is not None:
            supplied = dict(self.execution_metadata_provider(result, sampler))
            metadata.update(supplied)
        return QUBOSolution(bits, model.energy(bits), metadata)

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
            evaluation_constraints(evaluation), evaluations=1, runtime=perf_counter()-started,
            backend="qaoa", status="success", message="QAOA complete",
            solver_metadata=metadata, evaluation=evaluation)
