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
5. Have a reviewer who did not implement the feature copy and complete
   `independent_review_template.json` in the controlled external approval
   system; never edit the repository template into an approved record.
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

`run_release_gate.py` never trusts approval JSON stored inside the source
repository. Signed records must be supplied explicitly with
`--independent-review` and repeated `--pilot-record` paths from controlled
external storage. It validates more than the `status` field. An approved review
must identify an independent qualified reviewer, match the exact candidate
commit, close every review check and blocking finding, and carry an external
signature/approval reference. Each approved pilot must likewise match the
candidate commit, use unique identification, pass every comparison, and provide
input/output artifacts whose SHA-256 digests are verified by the gate. A
status-only edit cannot make a build eligible for `v1.0`.

```bash
PYTHONPATH=src python validation/run_release_gate.py \
  --independent-review /controlled/approvals/review.json \
  --pilot-record /controlled/approvals/pilot-1.json \
  --pilot-record /controlled/approvals/pilot-2.json
```

Without these external records the command may pass the automated RC gates but
must report `external_approvals_passed=false` and `release_stage=release-candidate`.

## Release decision

- Automated gates pass, human review pending: `1.0.0rcN`
- Automated gates and signed independent review pass: eligible for `1.0.0`
- Any safety or numerical gate fails: no release

The release package must retain input and output checksums, exact Git commit,
dirty-tree state, seed, solver settings, warnings, and known limitations.

The narrow AISC preview formulas also have an independent hand-calculation
regression artifact. Generate it with:

```bash
PYTHONPATH=src python validation/run_code_check_reference.py
```

This evidence confirms arithmetic and trace metadata only. It deliberately
retains the external-review-required flag and does not constitute an official
AISC example approval.

Install the optional `external-validation` dependency and compare the linear
and bilinear truss implementations with OpenSeesPy using:

```bash
PYTHONPATH=src python validation/run_opensees_crosscheck.py
```

The checked evidence records the independent solver name/version, both result
sets, and numerical tolerances. It complements rather than replaces the signed
engineering review and real-project pilots.

The large-case endurance gate repeatedly evaluates deterministic design
variants while recording wall time, result repeatability, Python allocations,
and process RSS:

```bash
PYTHONPATH=src python -m benchmarks.endurance_acceptance \
  --seconds 30 --evaluations 100 \
  --output validation/endurance_evidence.json
```
