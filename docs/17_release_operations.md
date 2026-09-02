# Release operations, retention, and rollback

Every release candidate is retained as an immutable bundle containing the
validated input, result JSON, HTML report, run manifest, dependency audit, and
release-gate evidence. The bundle manifest records a SHA-256 digest and byte
count for every artifact plus an explicit retention deadline.

`beamfem.io.create_release_archive` writes the bundle atomically and verifies it
before returning. `verify_release_archive` rejects missing, duplicate,
undeclared, path-traversal, size-mismatched, or checksum-mismatched members.
`restore_release_archive` verifies again and restores into a dedicated empty
directory without overwriting existing files.

Recommended retention is at least seven years for artifacts used in a real
project, subject to the governing contract and jurisdiction. Research-only
records may use a shorter organization-approved period. Retention deletion is
an administrative action outside beamfem; this package never deletes an
archive automatically.

## Rollback drill

1. Select the last independently approved archive and verify its checksums.
2. Restore it into a new, empty directory.
3. Verify the restored run manifest and dependency lock.
4. Re-run the reference, optimization, and release-gate suites at the recorded
   Git commit.
5. Compare the regenerated result hashes with the archive.
6. Have the responsible engineer approve replacement of the rejected result.

The automated round-trip, tamper, and no-overwrite tests exercise steps 1–3.
Steps 4–6 remain part of the project release decision because changing an
engineering deliverable requires project authority.
