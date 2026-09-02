# Changelog

## 1.0.0rc2

- Closed every automatable Phase 0–8 acceptance gap: repeated-seed statistics,
  legacy milestone regression, resource limits, release decision evidence,
  dependency security audit, and verified retention/rollback archives.
- Added persistent process-isolated FEM candidate evaluation and a measured
  medium-case 3x performance gate with bitwise QUBO equivalence.
- Made small, medium, and large acceptance cases physically comparable and
  required all selected benchmark designs to pass the common FEM evaluator.
- Added MILP catalog-state mapping, load combinations, Euler capacities, and
  auditable common-FEM repair for lower-bound MILP candidates.
- Exposed fabrication, topology, relative-displacement, and explicit section
  slenderness constraints through strict Schema v2 JSON/YAML input.
- Expanded Linux Python coverage and full macOS/Windows tests, and added a
  locked dependency vulnerability-audit CI job.

## 1.0.0rc1

- Added native 2D/3D axial truss elements and mixed truss/frame analysis.
- Added reusable sparse factorization for multiple load combinations and
  explicit mechanism diagnostics.
- Added Schema v2 with member formulation and governance metadata plus safe v1
  migration.
- Added Exact, Greedy, equilibrium-capacity MILP, multi-start SA, local QUBO,
  and current-Qiskit QAOA workflows with final FEM revalidation.
- Added integrity-checked optimizer checkpoints, run manifests, HTML reports,
  dependency inventory, and artifact checksums.
- Added traceable verification rules and a deliberately limited,
  non-certifying ANSI/AISC 360-22 axial LRFD preview.
- Added closed-form V&V, exact/MILP reference, performance, and noisy-Aer
  evidence packages.
- Added deterministic process-isolated candidate FEM evaluation with reusable
  workers for medium and larger local-QUBO optimization runs.
- Added Linux/Python matrix CI, macOS/Windows smoke CI, and quantum smoke CI.

Known release blockers for final `1.0.0` are recorded in
`RELEASE_CHECKLIST.md` and `validation/README.md`; in particular, external
structural-engineering review and real project pilot approval cannot be
replaced by automated tests.
