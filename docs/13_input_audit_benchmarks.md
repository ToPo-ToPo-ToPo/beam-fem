# Input, audit, and benchmark interfaces

Discrete optimization inputs use a versioned, SI-only JSON schema. JSON has no
optional runtime dependency; YAML files use the same schema and require
`PyYAML`.

```python
from beamfem.io import load_problem_spec

problem = load_problem_spec("truss.json")
print(problem.schema_version)
```

Schema version `1.0` defines materials, section catalogs, nodes, candidate
members, supports, load cases, load combinations, constraints, and the
objective. Validation checks duplicate identifiers and broken references before
an FEM model is created. `diagnose_problem_spec` additionally reports suspicious
but not necessarily invalid conditions such as duplicate members, isolated
nodes, coincident node coordinates, and possibly insufficient supports. The FEM
solver remains responsible for the definitive stiffness-rank check.

## Result and audit output

`write_result_json` accepts dictionaries, dataclasses, numpy values, or a
backend result object and writes an atomic, versioned result. An
`AuditMetadata` record can be attached to preserve:

- beamfem and Python versions;
- Git commit and dirty-worktree state;
- solver name, seed, and complete settings;
- timestamp, platform, and warnings.

`write_result_csv` emits one flattened summary row for aggregation. Nested
arrays are encoded as JSON in their CSV cell. Non-finite numeric values are
rejected so that invalid solver results cannot silently become non-standard
JSON.

## Scalable benchmark cases

`benchmarks/quantum_truss` generates deterministic small, medium, and large
two-chord trusses. Its runner accepts any callable with this contract:

```python
def solve(problem: Mapping[str, Any], settings: Mapping[str, Any]) -> Any:
    ...
```

This deliberately keeps FEM and optimizer packages out of the benchmark data
layer. Exact, MILP, SA, and QAOA adapters can therefore consume identical
validated documents. See `benchmarks/quantum_truss/README.md` for commands.
