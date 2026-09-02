# Quantum truss benchmarks

This directory provides deterministic `small`, `medium`, and `large` planar
truss inputs. They use the public versioned interchange schema and are not tied
to a specific FEM, MILP, SA, or QAOA implementation.

Generate an input document:

```bash
python -m benchmarks.quantum_truss.generate_cases small /tmp/truss-small.json
```

Validate and record a dry-run audit:

```bash
python -m benchmarks.quantum_truss.runner \
  --size small --solver-name dry-run --output /tmp/result.json
```

An optimization backend can be plugged in with
`--solver-factory package.module:function`. The callable receives the validated
problem mapping and a settings mapping, and may return any dataclass, mapping,
or numpy-backed result supported by `beamfem.io.to_serializable`.

The generated sizes are intended for different validation goals:

| Case | Bays | Nodes | Candidate members | Purpose |
|---|---:|---:|---:|---|
| small | 3 | 8 | 16 | exact/MILP reference and QAOA |
| medium | 10 | 22 | 51 | repeated-seed solver comparison |
| large | 40 | 82 | 201 | runtime and memory scaling |

Compare the common FEM results from Greedy, SA, and QAOA in one run:

```bash
python -m benchmarks.quantum_truss.compare /tmp/truss-small.json \
  --output-json /tmp/comparison.json --output-csv /tmp/comparison.csv
```

Each row contains QUBO energy (where applicable), FEM score, mass,
feasibility, governing constraint, evaluation count, and runtime. Add
`--backends exact greedy sa qaoa` only when the discrete state space is small
enough for exhaustive enumeration. MILP is intentionally not included here:
the common MILP backend requires an explicit valid linear formulation and does
not silently linearize the nonlinear FEM constraints.
