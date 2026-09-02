# Product requirements and release gates

This document fixes the scope of the first production candidate. Passing these
automated gates does **not** make beamfem a certified design authority; a
licensed structural engineer remains responsible for the model, load basis,
code selection, and final design.

## Supported scope

- SI-unit, linear-elastic static analysis of 2D/3D trusses and frames;
- explicit pin-jointed axial truss and Timoshenko frame member types;
- mixed truss/frame models with nodal loads, self-weight, load cases, and
  linear load combinations;
- discrete section selection and optional-member topology decisions;
- mass, cost, carbon, stress, displacement, axial buckling, slenderness, and
  fabrication/topology constraints;
- Exact, Greedy, explicit-formulation MILP, SA, and local-QUBO/QAOA backends;
- reproducible JSON/YAML input, checkpoint, audit, and machine-readable report.

The release candidate excludes nonlinear geometry/materials, plastic or fatigue
design, dynamics, fire/seismic system qualification, connections, foundations,
construction sequence, and automatic regulatory approval.

## Design-rule policy

The core check engine is code-neutral. A generic validation rule set is usable
for research and regression tests. A named code adapter is released only as a
clearly versioned preview after its equations, clauses, errata, units, and
published examples have independent review. Unsupported clauses must return an
explicit `not_applicable` or `not_implemented` result, never an assumed pass.

The first named preview target is the axial-member LRFD subset of ANSI/AISC
360-22 because the publisher makes the current specification publicly
available. This selection does not imply applicability to projects governed by
Japanese or other jurisdictions.

## Numerical acceptance gates

The machine-readable thresholds live in
`validation/acceptance_v1.json`. Release evidence must record the exact machine,
Python/dependency versions, Git commit, seed, input checksum, and elapsed time.

- textbook displacement and member-force comparisons: relative error <= 0.5%;
- normalized force-equilibrium residual: <= 1e-8;
- rigid-rotation invariance: relative difference <= 1e-10;
- small discrete cases: Exact and valid MILP solutions must have zero objective
  gap within numerical tolerance and pass independent FEM re-evaluation;
- stochastic backends: report every seed, feasibility rate, best/median/worst
  objective, and evaluation budget; no single-run success claim;
- singular, non-finite, zero-length, duplicate, and unsupported models must fail
  deterministically with a machine-readable diagnostic;
- JSON audit and report output must contain no NaN or infinity.

Performance gates are measured against a recorded baseline rather than an
unqualified wall-clock promise. The medium case must show at least a 3x speedup
from factorization reuse/parallel candidate evaluation when the benchmark has
enough work to amortize process startup. The large case must respect configured
time, memory, and evaluation limits and produce a resumable checkpoint.

## Release stages

1. **Analysis alpha**: truss element and V&V gates pass.
2. **Optimization beta**: exact/MILP references and repeated-seed comparisons
   pass; all selected designs are independently re-evaluated.
3. **Product RC**: schema migration, checkpoint, reports, cross-platform CI,
   security/dependency inventory, and pilot inputs pass.
4. **v1.0**: independent structural-engineering review closes all blocking
   findings. Until this external gate is signed, builds remain release
   candidates regardless of automated test status.

## Required human approvals

- model and boundary-condition review;
- governing load and combination review;
- design-code edition and jurisdiction review;
- spot checks of governing members and combinations;
- approval of every limitation or unsupported check;
- final design approval by the responsible engineer.
