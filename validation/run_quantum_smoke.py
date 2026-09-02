"""Run a reproducible noisy-Aer QAOA smoke test and write strict JSON evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np

from beamfem.optimize.backends import QAOABackend
from beamfem.optimize.qubo import QUBOModel


def run(seed: int = 11, shots: int = 256, maxiter: int = 10) -> dict:
    import qiskit
    import qiskit_aer
    import qiskit_optimization
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit_aer.primitives import SamplerV2

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.002, 1), ["h", "rx", "rz"])
    noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx"])
    sampler = SamplerV2(
        default_shots=shots,
        seed=seed,
        options={"backend_options": {"noise_model": noise}},
    )
    pass_manager = generate_preset_pass_manager(
        backend=AerSimulator(noise_model=noise),
        optimization_level=1,
        seed_transpiler=seed,
    )
    model = QUBOModel(
        np.array([-1.0, -0.5]),
        np.array([[0.0, 0.25], [0.0, 0.0]]),
    )
    solution = QAOABackend(
        sampler=sampler,
        reps=1,
        maxiter=maxiter,
        shots=shots,
        seed=seed,
        pass_manager=pass_manager,
        execution_label="Aer SamplerV2 noisy smoke",
        noise_model_description="depolarizing p1=0.002 p2=0.01",
    ).solve_qubo(model)
    return {
        "evidence_schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "qiskit": qiskit.__version__,
            "qiskit_aer": qiskit_aer.__version__,
            "qiskit_optimization": qiskit_optimization.__version__,
        },
        "execution": "local_noisy_simulator",
        "hardware_execution_performed": False,
        "hardware_reason": "No provider credentials or paid hardware authorization supplied.",
        "bits": list(solution.bits),
        "qubo_energy": solution.energy,
        "metadata": solution.metadata,
        "passed": solution.metadata["energy_gap_to_exact"] == 0.0,
        "claim": "Smoke test only; no quantum advantage claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("quantum_evidence.json"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--maxiter", type=int, default=10)
    args = parser.parse_args()
    evidence = run(args.seed, args.shots, args.maxiter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"noisy QAOA passed={evidence['passed']}; output={args.output}")
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
