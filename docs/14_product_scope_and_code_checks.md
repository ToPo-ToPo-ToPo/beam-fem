# Product scope, code checks, and review gate

## Status

The discrete optimization workflow is a research and preliminary-design tool.
It does **not** certify a structure and does not claim formal compliance with
AIJ, JIS, ANSI/AISC, building regulations, contract requirements, or any other
standard. Every deliverable must retain the external-review gate. A responsible
structural engineer must independently confirm the model, actions, combinations,
member checks, connections, stability, detailing, and governing law before use.

## Supported analysis envelope

- SI-unit, linear-elastic static 2D/3D axial-truss, Timoshenko-frame, and mixed
  truss/frame models built through the portable optimization adapter;
- discrete section/topology candidates and static load combinations;
- mass/cost/carbon objectives exposed by the optimization problem;
- stress, absolute/relative displacement, idealized Euler buckling, explicitly
  supplied section-slenderness ratios, member count/length/connectivity,
  same-section, section-type-count, required/forbidden-member, and symmetry
  constraints;
- exact, greedy/multi-start, explicit equilibrium/capacity MILP, SA, local QUBO,
  and optional QAOA optimization backends, with common FEM re-evaluation;
- versioned inputs, deterministic seed/settings, checksums, run manifests, and
  external-review-required reports.

The explicit MILP formulation covers truss equilibrium and axial capacity;
displacement/compatibility and all other constraints are accepted only after
common FEM re-evaluation. Exact enumeration is a small-problem reference.
QUBO/QAOA uses a trust-region local surrogate and is not an exact global FEM
encoding. Outside the validated envelope include material/geometric nonlinearity,
connection design, fatigue, fracture, fire, seismic qualification, fabrication
tolerances, soil-structure interaction, accidental actions, construction stages,
and regulatory approval. Absence from this list must not be interpreted as
support.

## Traceable rule engine

`beamfem.validation.RuleSet` records, for every evaluated limit state:

- ruleset/rule identifier and version;
- document edition, section, equation, and source URL;
- each coefficient, its meaning, value, and citation;
- symbolic and substituted expressions;
- demand, capacity, utilization, assumptions, and omissions;
- `external_review_required=True` and `approval_eligible=False`.

`verification_axial_steel_ruleset()` is an internal validation ruleset. Its
simple yielding and Euler equations are intended for regression/verification,
not design-code compliance.

### ANSI/AISC 360-22 preview adapter

`aisc360_22_axial_lrfd_preview_ruleset()` is a deliberately incomplete preview
of gross-section tensile yielding (D2-1) and flexural buckling of a user-confirmed
nonslender axial compression member (E3-1 through E3-4). The source metadata
points to AISC's official [ANSI/AISC 360 current-standard page](https://www.aisc.org/aisc/publications/current-standards/aisc-360/).
The implementation records that the official [January 23, 2025 ANSI/AISC 360-22
errata](https://www.aisc.org/globalassets/aisc/publications/revisions-and-errata/errata_360-22_1st-printing_01.23.2025.pdf)
was checked; the published errata does not list changes to the implemented D2/E3
equations.

The preview omits, at minimum, net-section rupture, connections, slender
elements, torsional/flexural-torsional buckling, system stability, combined
actions, fatigue, seismic design, and all applicability determinations. A passing
preview check therefore cannot become a design approval.

## Input migration

Schema v2 adds explicit `metadata`, `analysis`, and `governance` records. The v1
to v2 migration preserves explicit truss/frame member types and treats missing
legacy member types as `frame`; it never silently relabels an old frame
calculation as axial truss.
Migration always enables `external_review_required`. Reverse migrations and
unknown versions fail explicitly.

## Reproducibility artifacts

A product run should archive together:

1. original and migrated input;
2. result JSON and preliminary HTML report;
3. run manifest with checksum, seed, solver settings, checkpoints, and artifacts;
4. audit metadata including Git commit and dirty state;
5. dependency inventory and SHA-256 checksums;
6. complete code-check trace and signed external review record.
