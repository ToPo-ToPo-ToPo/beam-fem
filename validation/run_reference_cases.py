"""Run hand-calculated reference cases and write JSON evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from beamfem.validation.reference_cases import run_reference_suite, write_reference_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).parent / "reference_cases")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "reference_evidence.json")
    args = parser.parse_args()
    paths = sorted(args.fixtures.glob("*.json"))
    if not paths:
        parser.error(f"no JSON fixtures found in {args.fixtures}")
    evidence = run_reference_suite(paths)
    write_reference_evidence(evidence, args.output)
    print(f"reference cases: {len(paths)}; passed={evidence['passed']}; output={args.output}")
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
