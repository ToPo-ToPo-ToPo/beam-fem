import json

import pytest


def test_noisy_aer_qaoa_evidence_is_finite_and_exact_for_tiny_qubo(tmp_path):
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_optimization")
    from validation.run_quantum_smoke import run

    evidence = run(seed=11, shots=128, maxiter=6)
    assert evidence["passed"] is True
    assert evidence["hardware_execution_performed"] is False
    assert evidence["metadata"]["samples"] > 0
    assert evidence["metadata"]["logical_qubits"] == 2
    assert evidence["metadata"]["energy_gap_to_exact"] == 0.0
    json.dumps(evidence, allow_nan=False)
