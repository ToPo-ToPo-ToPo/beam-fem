# Release checklist

No production/preliminary-design release is complete until every applicable box
is checked and the named reviewer evidence is archived.

## Engineering scope

- [ ] Supported element formulation, dimensionality, units, and linearity match the input.
- [ ] Loads and combinations were independently reviewed.
- [ ] Boundary conditions and stiffness-rank/mechanism diagnostics were reviewed.
- [ ] Optimized design was re-evaluated by the common FEM evaluator.
- [ ] Every governing utilization, failed check, assumption, omission, and warning is visible.
- [ ] Connection, stability, fatigue, seismic, fire, construction-stage, and regulatory gaps are dispositioned.

## Rule governance

- [ ] Ruleset version and every source edition/section/equation are frozen in output.
- [ ] Coefficient provenance is complete; no unexplained constants remain.
- [ ] Current official errata were reviewed and the review date is recorded.
- [ ] Preview/verification rules are labeled and no unsupported compliance claim appears.
- [ ] An independent structural engineer completed and signed the external review gate.

## Reproducibility and software

- [ ] Schema migration is archived and checksum-verified.
- [ ] Seed, solver settings, limits, backend fallback, and Git commit are archived.
- [ ] Run manifest is complete; no stale or incompatible checkpoint was resumed.
- [ ] Result JSON, HTML report, and dependency audit SHA-256 values match.
- [ ] SBOM-like dependency inventory and lock file are archived.
- [ ] Unit, integration, randomized/property, abnormal-input, and regression tests pass.
- [ ] Optional QAOA absence/failure follows the declared fallback policy.

## Release decision

- [ ] Release limitations and known issues are attached.
- [ ] Security/dependency review is complete.
- [ ] Release approver is not the sole implementation author.
- [ ] Rollback and artifact-retention procedures are tested.
