"""Convert the portable schema into the common discrete FEM problem."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from ..material import Material, Section
from ..model import Model, UX, UY, UZ, RX, RY, RZ
from ..optimize.catalogs import SectionCatalog, SectionOption
from ..optimize.constraints import (
    ActiveMemberCount,
    Connectivity,
    DisplacementLimit,
    EulerBucklingLimit,
    ForbiddenMembers,
    MaxSectionTypes,
    MemberLengthRange,
    RelativeDisplacementLimit,
    RequiredMembers,
    SameSectionGroup,
    SectionSlendernessLimit,
    StressLimit,
    SymmetryPairs,
)
from ..optimize.objectives import MassObjective, WeightedImpactObjective
from ..optimize.problem import (
    DesignState,
    DiscreteStructuralProblem,
    LoadCase,
    LoadCombination,
)
from .schema import ProblemSpec, validate_problem_spec


DOF_NAMES = {"UX": UX, "UY": UY, "UZ": UZ, "RX": RX, "RY": RY, "RZ": RZ}


@dataclass(frozen=True)
class BuiltProblem:
    """Problem plus stable portable-id to internal-index mappings."""

    problem: DiscreteStructuralProblem
    node_ids: Mapping[str, int]
    member_ids: Mapping[str, int]


def _section(entry: Mapping[str, Any]) -> Section:
    area = float(entry["area"])
    inertia = float(entry.get("I", 0.0))
    # Version 1 uses a single inertia.  Treat it as both principal inertias;
    # the circular-equivalent edge distance enables conservative stress output.
    radius = math.sqrt(area / math.pi)
    return Section(
        A=area,
        Iy=inertia,
        Iz=inertia,
        J=float(entry.get("J", 2.0 * inertia)),
        ky=float(entry.get("ky", 0.9)),
        kz=float(entry.get("kz", 0.9)),
        cy=float(entry.get("cy", radius)),
        cz=float(entry.get("cz", radius)),
        name=str(entry["id"]),
    )


def build_discrete_problem(spec: ProblemSpec | Mapping[str, Any]) -> BuiltProblem:
    """Build a SI ``DiscreteStructuralProblem`` from a validated portable document.

    ``member_type`` は ``frame``（既定、後方互換）または ``truss``。OFF状態は
    portable catalogに無くても各カタログの先頭へ追加する。
    """

    validated = spec if isinstance(spec, ProblemSpec) else validate_problem_spec(spec)
    data = validated.data
    model = Model()
    node_ids: dict[str, int] = {}
    is_2d = True
    for item in data["nodes"]:
        xyz = list(item["xyz"])
        is_2d = is_2d and len(xyz) == 2
        xyz += [0.0] * (3 - len(xyz))
        node_ids[item["id"]] = model.add_node(*xyz[:3])

    materials: dict[str, Material] = {}
    for name, item in data["materials"].items():
        materials[name] = Material(
            E=float(item["E"]),
            nu=float(item.get("nu", 0.3)),
            rho=float(item["density"]),
            name=name,
        )

    catalog_templates: dict[tuple[str, str], SectionCatalog] = {}
    for material_name, material in materials.items():
        mat_data = data["materials"][material_name]
        for catalog_name, entries in data["section_catalogs"].items():
            options = [SectionOption("OFF", None)]
            options.extend(
                SectionOption(
                    name=str(entry["id"]),
                    section=_section(entry),
                    material=material,
                    tensile_strength=float(entry.get(
                        "tension_allowable", mat_data.get("tension_allowable")
                    )) if entry.get("tension_allowable", mat_data.get("tension_allowable")) is not None else None,
                    compressive_strength=float(entry.get(
                        "compression_allowable", mat_data.get("compression_allowable")
                    )) if entry.get("compression_allowable", mat_data.get("compression_allowable")) is not None else None,
                    slenderness_ratio=float(entry["slenderness"])
                    if entry.get("slenderness") is not None else None,
                    cost_per_kg=float(entry.get("cost_per_kg", mat_data.get("cost_per_kg", 0.0))),
                    carbon_per_kg=float(entry.get("carbon_per_kg", mat_data.get("carbon_per_kg", 0.0))),
                )
                for entry in entries
            )
            catalog_templates[(material_name, catalog_name)] = SectionCatalog(
                f"{material_name}:{catalog_name}", options
            )

    catalogs: list[SectionCatalog] = []
    member_ids: dict[str, int] = {}
    for member in data["members"]:
        material_name = member["material"]
        catalog = catalog_templates[(material_name, member["catalog"])]
        active = [option for option in catalog.options if option.active]
        add = model.add_truss if member.get("member_type", "frame") == "truss" else model.add_element
        index = add(
            node_ids[member["nodes"][0]], node_ids[member["nodes"][1]],
            materials[material_name], active[-1].section,
        )
        member_ids[member["id"]] = index
        catalogs.append(catalog)

    if is_2d:
        model.fix_to_plane_xy()
    for support in data.get("supports", []):
        unknown = set(support["dofs"]) - set(DOF_NAMES)
        if unknown:
            raise ValueError(f"unknown support DOFs: {sorted(unknown)}")
        model.fix(node_ids[support["node"]], [DOF_NAMES[name] for name in support["dofs"]])

    load_cases: list[LoadCase] = []
    for name, loads in data["load_cases"].items():
        nodal: dict[tuple[int, int], float] = {}
        for load in loads:
            force = list(load["force"])
            for dof, value in enumerate(force):
                if value:
                    key = (node_ids[load["node"]], dof)
                    nodal[key] = nodal.get(key, 0.0) + float(value)
        load_cases.append(LoadCase(name, nodal))
    combinations = [
        LoadCombination(name, factors)
        for name, factors in data["load_combinations"].items()
    ]

    constraints = []
    for index, item in enumerate(data.get("constraints", [])):
        kind = item.get("type")
        cid = str(item.get("id", f"{kind}_{index}"))
        selected_members = item.get("members")
        members = None if selected_members is None else tuple(member_ids[name] for name in selected_members)
        combinations_filter = item.get("combinations")
        combinations_filter = None if combinations_filter is None else tuple(combinations_filter)
        if kind == "stress":
            constraints.append(StressLimit(
                tension=item.get("tension"), compression=item.get("compression"),
                members=members, combinations=combinations_filter, constraint_id=cid,
            ))
        elif kind == "euler_buckling":
            constraints.append(EulerBucklingLimit(
                effective_length_factor=float(item.get("effective_length_factor", 1.0)),
                axis=str(item.get("axis", "min")), members=members,
                combinations=combinations_filter, constraint_id=cid,
            ))
        elif kind == "displacement":
            limit = float(item["limit"])
            target_nodes = item.get("nodes", list(node_ids))
            target_dofs = item.get("dofs", ["UX", "UY"] if is_2d else ["UX", "UY", "UZ"])
            for node_name in target_nodes:
                for dof_name in target_dofs:
                    constraints.append(DisplacementLimit(
                        node=node_ids[node_name], dof=DOF_NAMES[dof_name], maximum=limit,
                        combinations=combinations_filter,
                        constraint_id=f"{cid}:{node_name}:{dof_name}",
                    ))
        elif kind == "relative_displacement":
            constraints.append(RelativeDisplacementLimit(
                node_a=node_ids[item["node_a"]], node_b=node_ids[item["node_b"]],
                dof=DOF_NAMES[item["dof"]], maximum=float(item["limit"]),
                combinations=combinations_filter, constraint_id=cid,
            ))
        elif kind == "required_members":
            constraints.append(RequiredMembers(members or (), cid))
        elif kind == "forbidden_members":
            constraints.append(ForbiddenMembers(members or (), cid))
        elif kind == "same_section_group":
            constraints.append(SameSectionGroup(members or (), cid))
        elif kind == "max_section_types":
            constraints.append(MaxSectionTypes(
                maximum=int(item["maximum"]), members=members,
                include_off=bool(item.get("include_off", False)), constraint_id=cid,
            ))
        elif kind == "active_member_count":
            constraints.append(ActiveMemberCount(
                minimum=int(item.get("minimum", 0)),
                maximum=None if item.get("maximum") is None else int(item["maximum"]),
                members=members, constraint_id=cid,
            ))
        elif kind == "symmetry_pairs":
            constraints.append(SymmetryPairs(
                ((member_ids[a], member_ids[b]) for a, b in item["pairs"]), cid,
            ))
        elif kind == "connectivity":
            constraints.append(Connectivity(
                (node_ids[name] for name in item["nodes"]), cid,
            ))
        elif kind == "member_length_range":
            constraints.append(MemberLengthRange(
                minimum=float(item.get("minimum", 0.0)),
                maximum=float(item.get("maximum", math.inf)),
                members=members, constraint_id=cid,
            ))
        elif kind == "section_slenderness":
            constraints.append(SectionSlendernessLimit(
                maximum=float(item["maximum"]), members=members, constraint_id=cid,
            ))
        else:
            raise ValueError(f"unsupported constraint type: {kind!r}")

    objective_data = data["objective"]
    if objective_data["type"] == "mass":
        objective = MassObjective()
    else:
        objective = WeightedImpactObjective(
            mass_weight=float(objective_data.get("mass_weight", 0.0)),
            cost_weight=float(objective_data.get("cost_weight", 1.0 if objective_data["type"] == "cost" else 0.0)),
            carbon_weight=float(objective_data.get("carbon_weight", 1.0 if objective_data["type"] == "co2" else 0.0)),
        )

    initial = DesignState(len(catalog) - 1 for catalog in catalogs)
    problem = DiscreteStructuralProblem(
        model=model,
        catalogs=catalogs,
        load_cases=load_cases,
        load_combinations=combinations,
        constraints=constraints,
        objective=objective,
        initial_design=initial,
        self_weight=data.get("self_weight"),
    )
    return BuiltProblem(problem, node_ids, member_ids)
