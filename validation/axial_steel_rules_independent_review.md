# Independent implementation review: axial steel rules

Review date: 2026-09-02

Reviewed file: `src/beamfem/validation/axial_steel_rules.py`
Disposition: implementation findings resolved; professional approval remains pending

This is an implementation review, not professional approval of the design
rules. The findings below were corrected in the implementation and covered by
automated regression tests. They remain listed to preserve the review trail.

## Formula and boundary checks

- The verification tension capacity `Fy A` and Euler capacity
  `pi² E I / (K L)²` are dimensionally correct for a consistent unit system.
- The preview tensile yielding resistance `0.90 Fy Ag` agrees with the LRFD
  factor and gross-section yielding expression stated for AISC Section D2.
- The compression implementation correctly forms `r=sqrt(I/A)`,
  `Fe=pi²E/(KL/r)²`, uses the E3-2 branch when `Fy/Fe <= 2.25`, and otherwise
  uses `Fcr=0.877Fe`. The exact boundary is assigned to E3-2 as specified.
- `phi_c=0.90`, compression-negative/tension-positive sign handling, and
  pass-at-utilization-equal-to-one behavior are internally consistent.
- The explicit gates for nonslender elements, flexural-buckling control, and
  gross-yielding-only scope correctly return `not_verified` when missing.

Primary comparison sources were AISC's current-standard page, 16th Edition
Basic Design Values Cards, and current revisions/errata index:

- https://www.aisc.org/aisc/publications/current-standards/aisc-360/
- https://www.aisc.org/aisc/publications/steel-construction-manual/basic-design-values-cards-for-16th-edition/
- https://www.aisc.org/aisc/publications/revisions-and-errata/

## Resolved implementation findings

1. **Governing buckling axis was under-specified.** The input had one
   `inertia_min` and one `effective_length_factor`. For a member with different
   effective lengths by principal axis, combining minimum inertia with a factor
   entered for another axis is not a defined AISC check and can miss the
   governing `KL/r`. The preview now requires the explicit
   `governing_axis_confirmed` assertion and returns `not_verified` if absent.

2. **The citation URL did not identify the cited normative text.** Results said
   the document is ANSI/AISC 360-22 and cite D2/E3 equations, but `source_url`
   points to the Basic Design Values Cards landing page. AISC states those cards
   are simplified aids and are not a replacement for the Specification. The
   trace now points to AISC's controlled current-standard page, retains the
   exact ANSI/AISC 360-22 edition/clauses/equations, and records the applicable
   January 23, 2025 errata review.

## Resolved robustness and documentation findings

1. **Direct-API units were not declared.** The input and result trace now carry
   an explicit SI unit system (`N, m, Pa, m², m⁴`) and reject other labels.

2. **Finite but extreme inputs could crash.** Exponential and slenderness
   calculations are now guarded and return finite, machine-readable outcomes.

3. **Zero axial demand was inconsistent.** Both tension and compression preview
   rules now return `not_applicable` before applicability assertions are read.

## Scope confirmed as intentionally omitted

Net-section rupture, connections, slender-element reductions, E4/E5/E6/E7,
torsional and flexural-torsional buckling, combined actions, system stability,
fatigue, and seismic checks are clearly identified as omissions. The preview
must remain external-review-required and must not be presented as AISC
compliance. The software findings above are resolved, but the separate licensed
structural-engineer approval gate is intentionally still pending.
