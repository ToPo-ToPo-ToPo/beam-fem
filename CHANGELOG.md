# Changelog

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
- Added Linux/Python matrix CI, macOS/Windows smoke CI, and quantum smoke CI.

Known release blockers for final `1.0.0` are recorded in
`RELEASE_CHECKLIST.md` and `validation/README.md`; in particular, external
structural-engineering review and real project pilot approval cannot be
replaced by automated tests.
