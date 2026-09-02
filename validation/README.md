# Release validation package

This directory contains the non-code evidence required for a production
release. Automated tests may produce a release candidate, but they cannot sign
the independent structural-engineering review.

## Required evidence

1. Run the complete test and benchmark matrix on the candidate commit.
2. Record the machine, operating system, Python and dependency versions.
3. Attach reference calculations for the governing V&V cases.
4. Complete at least two anonymized pilot models with independently checked
   reactions, member forces, utilization and governing combinations.
5. Have a reviewer who did not implement the feature complete
   `independent_review_template.json`.
6. Resolve every blocking finding and repeat affected tests.

## Automated closed-form reference evidence

The deterministic 2D triangle, 3D tripod, and mixed frame/truss fixtures are
stored in `reference_cases/`. Run them from the repository root with:

```bash
PYTHONPATH=src python validation/run_reference_cases.py
```

This rewrites `reference_evidence.json` with displacements, axial forces,
global equilibrium residuals, and the planar rotation-invariance error. The
expected values in each fixture are closed-form hand-calculation values rather
than snapshots copied from the FEM output.

`acceptance_v1.json` is machine-readable policy. The review and pilot templates
intentionally start in `pending` state; changing them to `approved` without real
review evidence is not permitted.

## Release decision

- Automated gates pass, human review pending: `1.0.0rcN`
- Automated gates and signed independent review pass: eligible for `1.0.0`
- Any safety or numerical gate fails: no release

The release package must retain input and output checksums, exact Git commit,
dirty-tree state, seed, solver settings, warnings, and known limitations.
