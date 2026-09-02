"""Optional Qiskit QAOA backend.

Qiskit is imported only when this backend runs, so beamfem and all classical
backends remain usable without quantum dependencies.
"""

from __future__ import annotations

from time import perf_counter
import math
import inspect
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..qubo import QUBOModel, QUBOSolution
from .base import (
    OptimizationResult, SolverLimits, evaluate_problem, evaluation_constraints,
    evaluation_feasible, evaluation_objective, make_design,
)
from .sa import Decoder, _resolve_qubo


class QiskitNotInstalledError(ImportError):
    pass


class IndependentReadoutMitigator:
    """Small-problem independent bit-flip readout mitigation.

    ``p01`` is P(measured 1 | prepared 0) and ``p10`` is P(measured 0 |
    prepared 1).  The tensor-product response matrix is inverted, negative
    quasi-probabilities are clipped, and the result is normalized.  This local
    implementation is intentionally limited to modest QUBOs; hardware-specific
    calibration services can be supplied through the same callable hook.
    """

    def __init__(self, p01: float, p10: float | None = None, *, max_qubits: int = 12):
        p10 = p01 if p10 is None else p10
        if not all(math.isfinite(float(value)) and 0.0 <= value < 0.5 for value in (p01, p10)):
            raise ValueError("readout error probabilities must be finite in [0, 0.5)")
        if max_qubits < 1:
            raise ValueError("max_qubits must be positive")
        self.p01, self.p10, self.max_qubits = float(p01), float(p10), int(max_qubits)

    def __call__(self, distribution: Mapping[str, float]) -> dict[str, float]:
        if not distribution:
            return {}
        widths = {len(key) for key in distribution}
        if len(widths) != 1 or any(set(key) - {"0", "1"} for key in distribution):
            raise ValueError("readout distribution keys must be equal-width bit strings")
        n = widths.pop()
        if n > self.max_qubits:
            raise ValueError(f"readout mitigation limited to {self.max_qubits} qubits")
        observed = np.zeros(2**n, dtype=float)
        for key, value in distribution.items():
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError("readout distribution probabilities must be finite and non-negative")
            observed[int(key, 2)] = float(value)
        single = np.array([[1.0 - self.p01, self.p10], [self.p01, 1.0 - self.p10]])
        response = single
        for _ in range(1, n):
            response = np.kron(response, single)
        corrected = np.linalg.solve(response, observed)
        corrected = np.maximum(corrected, 0.0)
        total = float(corrected.sum())
        if total <= 0.0 or not math.isfinite(total):
            raise RuntimeError("readout mitigation produced an invalid distribution")
        corrected /= total
        return {format(index, f"0{n}b"): float(value)
                for index, value in enumerate(corrected) if value > 0.0}


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
                 exact_reference_max_variables: int = 20,
                 cvar_alpha: float | None = None,
                 readout_mitigator: Callable[[Mapping[str, float]], Mapping[str, float]] | None = None):
        if reps < 1:
            raise ValueError("reps must be positive")
        if maxiter < 1:
            raise ValueError("maxiter must be positive")
        if shots is not None and shots < 1:
            raise ValueError("shots must be positive or None")
        if exact_reference_max_variables < 0:
            raise ValueError("exact_reference_max_variables cannot be negative")
        if cvar_alpha is not None and (
            not math.isfinite(float(cvar_alpha)) or not 0.0 < cvar_alpha <= 1.0
        ):
            raise ValueError("cvar_alpha must be finite in (0, 1]")
        self.qubo, self.decoder = qubo, decoder
        self.sampler, self.sampler_factory = sampler, sampler_factory
        self.optimizer, self.reps, self.shots = optimizer, int(reps), shots
        self.seed, self.pass_manager, self.maxiter = int(seed), pass_manager, int(maxiter)
        self.execution_label = execution_label
        self.noise_model_description = noise_model_description
        self.execution_metadata_provider = execution_metadata_provider
        self.exact_reference_max_variables = int(exact_reference_max_variables)
        self.cvar_alpha = None if cvar_alpha is None else float(cvar_alpha)
        self.readout_mitigator = readout_mitigator

    def solve_qubo(self, model: QUBOModel, *, maxiter: int | None = None) -> QUBOSolution:
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
        effective_maxiter = self.maxiter if maxiter is None else min(self.maxiter, int(maxiter))
        optimizer = self.optimizer or COBYLA(maxiter=effective_maxiter)
        kwargs = {"sampler": sampler, "optimizer": optimizer, "reps": self.reps}
        if self.cvar_alpha is not None:
            kwargs["aggregation"] = self.cvar_alpha
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
        raw_distribution: dict[str, float] = {}
        minimum_result = getattr(result, "min_eigen_solver_result", None)
        eigenstate = getattr(minimum_result, "eigenstate", None)
        if eigenstate is not None:
            binary_probabilities = getattr(eigenstate, "binary_probabilities", None)
            source = binary_probabilities() if callable(binary_probabilities) else eigenstate
            if isinstance(source, Mapping):
                for key, value in source.items():
                    bitstring = str(key)
                    probability_value = float(value)
                    if len(bitstring) == model.n_variables and not (set(bitstring) - {"0", "1"}) \
                            and math.isfinite(probability_value) and probability_value >= 0.0:
                        raw_distribution[bitstring] = probability_value
        if not raw_distribution and samples:
            for _, sample, sample_bits in checked_samples:
                sample_probability = float(sample.probability)
                if math.isfinite(sample_probability) and sample_probability >= 0.0:
                    raw_distribution["".join(str(bit) for bit in sample_bits)] = sample_probability
        mitigated_distribution = None
        if self.readout_mitigator is not None:
            mitigated = dict(self.readout_mitigator(raw_distribution))
            if any(len(key) != model.n_variables or set(key) - {"0", "1"}
                   or not math.isfinite(float(value)) or value < 0.0
                   for key, value in mitigated.items()):
                raise RuntimeError("readout mitigator returned an invalid distribution")
            total = sum(float(value) for value in mitigated.values())
            if total <= 0.0 or not math.isfinite(total):
                raise RuntimeError("readout mitigator returned an empty distribution")
            mitigated_distribution = {key: float(value) / total for key, value in mitigated.items()}
        raw_counts = None
        if self.shots is not None and raw_distribution:
            raw_counts = {key: int(round(value * self.shots)) for key, value in raw_distribution.items()}

        metadata = {"reps": self.reps, "shots": self.shots, "seed": self.seed,
                    "maxiter": effective_maxiter, "requested_maxiter": self.maxiter,
                    "qiskit_status": str(result.status), "fval": fval,
                    "selected_probability": probability, "samples": len(samples),
                    "execution_label": self.execution_label or type(sampler).__name__,
                    "noise_model": self.noise_model_description,
                    "qaoa_wall_time": quantum_wall_time,
                    "quantum_execution_time": None,
                    "queue_time": None,
                    "quantum_timing": {"queue_time": None, "execution_time": None,
                                       "total_wall_time": quantum_wall_time,
                                       "source": "local_wall_clock"},
                    "objective_aggregation": "expectation" if self.cvar_alpha is None else "cvar",
                    "cvar_alpha": self.cvar_alpha,
                    "raw_distribution": raw_distribution,
                    "raw_counts": raw_counts,
                    "raw_counts_source": "probability_times_configured_shots" if raw_counts is not None else None,
                    "readout_mitigation_applied": self.readout_mitigator is not None,
                    "mitigated_distribution": mitigated_distribution}
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
            queue = metadata.get("queue_time")
            execution = metadata.get("quantum_execution_time")
            for label, value in (("queue_time", queue), ("quantum_execution_time", execution)):
                if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                    raise RuntimeError(f"execution metadata {label} must be finite and non-negative")
            metadata["quantum_timing"] = {
                "queue_time": None if queue is None else float(queue),
                "execution_time": None if execution is None else float(execution),
                "total_wall_time": quantum_wall_time,
                "source": "execution_metadata_provider",
            }
        return QUBOSolution(bits, model.energy(bits), metadata)

    def solve(self, problem: Any, initial_design: Any | None = None,
              limits: SolverLimits | None = None) -> OptimizationResult:
        started = perf_counter()
        template = initial_design if initial_design is not None else problem.initial_design
        model, decoder = _resolve_qubo(problem, template, self.qubo, self.decoder)
        accepts_budget = "maxiter" in inspect.signature(self.solve_qubo).parameters
        solution = (
            self.solve_qubo(model, maxiter=limits.max_iterations)
            if limits is not None and limits.max_iterations is not None and accepts_budget
            else self.solve_qubo(model)
        )
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
