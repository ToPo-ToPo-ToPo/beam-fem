# Input, audit, and benchmark interfaces

Discrete optimization inputs use a versioned, SI-only JSON schema. JSON has no
optional runtime dependency; YAML files use the same schema and require
`PyYAML`.

```python
from beamfem.io import load_problem_spec

problem = load_problem_spec("truss.json")
print(problem.schema_version)
```

Both supported schema versions define materials, section catalogs, nodes,
candidate members, supports, load cases, load combinations, constraints, and
the objective. Version `1.0` is the compatibility format; a missing
`member_type` means `frame`. Version `2.0` is the current format and additionally
requires `metadata.model_id`, explicit `analysis` (`dimension`,
`element_formulation`, and `linearity: linear_elastic`), and a `governance`
record whose `external_review_required` flag is true. Use
`migrate_v1_to_v2()` for the deterministic forward migration. Reverse and
unknown-version migrations are rejected.

The portable format is strictly SI:

| Quantity | Unit |
| --- | --- |
| coordinates, translational displacement limits, member-length limits | m |
| rotational displacement limits | rad |
| nodal force | N |
| Young's modulus, stress/strength | Pa |
| area / second moment / torsion constant | m² / m⁴ / m⁴ |
| density | kg/m³ |
| self-weight acceleration | m/s² |
| cost and carbon factors | user-defined per kg, consistently across a run |
| slenderness | dimensionless |

JSON and YAML have identical semantics. Unit labels other than the literal
`"SI"`, non-finite numbers, unknown constraint fields/types, duplicate IDs,
invalid DOFs, and broken node/member/load-combination references fail before an
FEM model is created. The core Python modelling API remains unit-consistent but
does not perform runtime unit conversion; the SI requirement described here is
specific to the portable optimization input boundary.

Supported portable constraint records are:

| `type` | Required fields | Optional fields |
| --- | --- | --- |
| `stress` | — | `tension`, `compression`, `members`, `combinations` |
| `euler_buckling` | — | `effective_length_factor`, `axis`, `members`, `combinations` |
| `displacement` | `limit` | `nodes`, `dofs`, `combinations` |
| `relative_displacement` | `node_a`, `node_b`, `dof`, `limit` | `combinations` |
| `required_members`, `forbidden_members`, `same_section_group` | `members` | — |
| `max_section_types` | `maximum` | `members`, `include_off` |
| `active_member_count` | `minimum` and/or `maximum` | `members` |
| `symmetry_pairs` | `pairs` | — |
| `connectivity` | `nodes` | — |
| `member_length_range` | `minimum` and/or `maximum` | `members` |
| `section_slenderness` | `maximum` | `members` |

Every record may have an optional unique `id`. `section_slenderness` compares
the selected catalog entry's dimensionless `slenderness` value with the limit;
it deliberately does not infer plate dimensions, a design code, or a section
classification. Every catalog option used by that constraint must provide the
value. `diagnose_problem_spec` additionally reports suspicious but not
necessarily invalid conditions such as duplicate members, isolated nodes,
coincident node coordinates, and possibly insufficient supports. The FEM solver
remains responsible for the definitive stiffness-rank check.

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
