# Quantum backend validation protocol

The quantum backend is an optional QUBO solver inside a hybrid workflow. FEM
acceptance checks remain classical and authoritative. A lower QUBO energy does
not by itself establish a safe or feasible structural design.

## Supported execution modes

- Qiskit `StatevectorSampler` for deterministic development checks;
- Qiskit Aer `SamplerV2` with finite shots and user-defined noise models;
- injected Sampler V2 and transpilation pass manager for provider hardware;
- injected metadata callback for provider job ID, backend name, calibration
  timestamp, queue time, execution time, physical depth, and error data.

The common backend records shots, reps, seed, optimizer limit, selected-sample
probability, logical qubits/depth/gate counts, QAOA wall time, exact energy gap
for small QUBOs, execution label, and noise description. Provider-only metrics
are never fabricated; unavailable fields remain null.

## Reproducible noisy simulation

```bash
python validation/run_quantum_smoke.py \
  --output validation/quantum_evidence.json --shots 256 --maxiter 10
```

The committed evidence uses Qiskit Aer depolarizing noise with one-qubit
probability 0.002 and two-qubit probability 0.01. The two-variable reference
QUBO is also solved exactly, and the smoke test passes only when the selected
energy has zero gap to that reference. This is an integration test, not a
performance or quantum-advantage claim.

## Hardware release gate

Hardware execution requires credentials, provider terms, possible paid usage,
and an explicitly selected backend. These are external authorizations and are
not inferred from repository access. A hardware evidence record must include:

- provider and immutable job identifier;
- backend and calibration timestamp;
- transpiler seed, pass-manager configuration, physical layout and circuit
  depth;
- shots, raw counts or provider result reference;
- queue, execution, and total wall time;
- the matched SA/classical evaluation budget;
- final FEM score, mass, feasibility, and governing constraint.

Until such an authorized run is attached, the release report must state
`hardware_execution_performed=false`. Product functionality must not depend on
hardware availability; configured classical fallback remains mandatory.
