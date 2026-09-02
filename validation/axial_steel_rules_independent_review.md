# Independent implementation review: axial steel rules

Review date: 2026-09-02

Reviewed file: `src/beamfem/validation/axial_steel_rules.py`
Disposition: formulas verified; release-blocking scope/traceability items remain

This is an implementation review, not professional approval of the design
rules. No changes were made to the reviewed rule implementation.

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

## Release-blocking findings

1. **The governing buckling axis is under-specified.** The input has one
   `inertia_min` and one `effective_length_factor`. For a member with different
   effective lengths by principal axis, combining minimum inertia with a factor
   entered for another axis is not a defined AISC check and can miss the
   governing `KL/r`. Require either per-axis `(I, K, L)` values and take the
   minimum capacity, or an explicit user assertion that the supplied pair is
   the governing effective slenderness.

2. **The citation URL does not identify the cited normative text.** Results say
   the document is ANSI/AISC 360-22 and cite D2/E3 equations, but `source_url`
   points to the Basic Design Values Cards landing page. AISC states those cards
   are simplified aids and are not a replacement for the Specification. The
   trace should point to the exact controlled Specification source/edition (and
   retain the errata record); cards may remain a secondary aid.

## Robustness and documentation findings

1. **Direct-API units are not declared or enforced.** Result units are always
   labelled `N`. The repository convention is SI, so the input class should
   explicitly state `N, m, Pa, m², m⁴`, or carry a unit-system identifier.
   Otherwise a dimensionally consistent imperial input is numerically accepted
   but mislabeled as newtons.

2. **Finite but extreme inputs can crash rather than return a failed/not-verified
   check.** For example, a finite `length=1e308` reaches an underflowed `Fe=0`
   and raises `ZeroDivisionError`. Add bounded numerical validation or guarded
   zero/overflow handling. This is not a realistic member, but schema policy
   currently promises finite-input rejection/diagnosis rather than an
   unstructured evaluator failure.

3. At zero axial force the tension preview still requires the
   `gross_section_yielding_only_confirmed` assertion and may return
   `not_verified`; the compression rule returns `not_applicable`. Decide and
   document whether a zero-demand member should be `pass` or `not_applicable`
   consistently. This does not create an unsafe capacity.

## Scope confirmed as intentionally omitted

Net-section rupture, connections, slender-element reductions, E4/E5/E6/E7,
torsional and flexural-torsional buckling, combined actions, system stability,
fatigue, and seismic checks are clearly identified as omissions. The preview
must remain external-review-required and must not be presented as AISC
compliance until those gates and the findings above are resolved.
