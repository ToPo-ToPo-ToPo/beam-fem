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

## Versioned external CSV catalogs

Common catalogs can remain outside a problem file. References are resolved
relative to the JSON/YAML file and require a version. An optional SHA-256 pins
the exact bytes:

```json
{"external_catalogs": {
  "materials": {"steel": {"path": "catalogs/materials.csv", "version": "2026.1"}},
  "sections": {"round_bar": {"path": "catalogs/round.csv", "version": "2026.1"}}
}}
```

Material CSV requires `id,E,density`; section CSV requires `id,area`. Optional
columns use inline field names and SI units. Inline and external definitions
may coexist but cannot define the same name. The loader records version, path,
and actual SHA-256 in `catalog_sources`. Use `load_problem_spec(path)` so
relative references have an unambiguous base directory.

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

## Persistent FEM cache

`DiscreteStructuralProblem.enable_persistent_cache(path)` はFEM評価をプロセスや
実行をまたいで再利用する。完全な問題定義のcontext checksum、payload SHA-256、
原子的置換を用い、モデル、カタログ、荷重、組合せ、制約、目的関数が変われば
context mismatchとして拒否する。疎行列を含む結果の保持にはpickleを使うため、
ローカルでbeamfemが生成したcacheだけを使用する。checksumは完全性検査であり、
第三者に対する真正性署名ではない。

HTML reports embed SVG utilization and objective-history plots.
`write_design_pdf` creates a dependency-free PDF summary, while
`write_comparison_report` compares two stored runs. CLI equivalents are
`--html-report`, `--pdf-report`, and paired `--compare-with` /
`--comparison-report`.

## REST API

`beamfem-api --host 127.0.0.1 --port 8080` serves `GET /health`,
`POST /v1/validate`, and `POST /v1/optimize` (`greedy` or `exact`). Responses
have request IDs and structured errors. `--bearer-token` enables static-token
checking and request bodies are limited. Public deployment must put the WSGI
callable behind HTTPS, managed authentication, rate limits, supervision, and
appropriate network controls; the bundled server targets local/private use.

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
