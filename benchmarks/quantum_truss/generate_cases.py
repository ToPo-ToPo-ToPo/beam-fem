"""Generate deterministic two-chord truss optimization inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CASE_SIZES = {
    "small": 3,
    "medium": 10,
    "large": 40,
}


def generate_case(size: str, *, bay_length: float = 1.5, height: float = 1.2) -> dict[str, Any]:
    """Create a schema-v1 planar ground structure for benchmarking.

    ``small`` reproduces the 8-node/16-member topology used by the legacy
    quantum-truss experiment. Larger cases repeat the same well-conditioned
    pattern without embedding solver-specific objects.
    """

    if size not in CASE_SIZES:
        raise ValueError(f"size must be one of {sorted(CASE_SIZES)}")
    bays = CASE_SIZES[size]
    nodes = []
    for row, y in (("b", 0.0), ("t", height)):
        nodes.extend(
            {"id": f"{row}{i}", "xyz": [i * bay_length, y]}
            for i in range(bays + 1)
        )

    members: list[dict[str, Any]] = []

    def add_member(a: str, b: str) -> None:
        members.append(
            {
                "id": f"m{len(members)}",
                "nodes": [a, b],
                "material": "steel",
                "catalog": "round_bar",
            }
        )

    for i in range(bays):
        add_member(f"b{i}", f"b{i + 1}")
        add_member(f"t{i}", f"t{i + 1}")
    for i in range(bays + 1):
        add_member(f"b{i}", f"t{i}")
    for i in range(bays):
        add_member(f"b{i}", f"t{i + 1}")
        add_member(f"t{i}", f"b{i + 1}")

    return {
        "schema_version": "1.0",
        "name": f"quantum-truss-{size}",
        "units": "SI",
        "materials": {
            "steel": {
                "E": 2.05e11,
                "density": 7850.0,
                "tension_allowable": 1.5e8,
                "compression_allowable": 1.2e8,
            }
        },
        "section_catalogs": {
            "round_bar": [
                {"id": "S", "area": 2.0e-4, "I": 3.2e-9},
                {"id": "M", "area": 4.0e-4, "I": 1.27e-8},
                {"id": "L", "area": 7.0e-4, "I": 3.90e-8},
                {"id": "XL", "area": 1.0e-3, "I": 7.96e-8},
            ]
        },
        "nodes": nodes,
        "members": members,
        "supports": [
            {"node": "b0", "dofs": ["UX", "UY"]},
            {"node": f"b{bays}", "dofs": ["UY"]},
        ],
        "load_cases": {
            "gravity": [
                {"node": f"t{i}", "force": [0.0, -20_000.0]}
                for i in range(1, bays + 1)
            ],
            "wind": [
                {"node": f"t{i}", "force": [4_000.0, -5_000.0]}
                for i in range(1, bays + 1)
            ],
        },
        "load_combinations": {
            "ultimate_gravity": {"gravity": 1.5},
            "ultimate_wind": {"gravity": 1.2, "wind": 1.5},
        },
        "constraints": [
            {"type": "stress"},
            {"type": "euler_buckling", "effective_length_factor": 1.0},
            {"type": "displacement", "limit": 0.025},
        ],
        "objective": {"type": "mass"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("size", choices=sorted(CASE_SIZES))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(generate_case(args.size), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
