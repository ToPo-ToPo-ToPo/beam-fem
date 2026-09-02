"""History-dependent uniaxial constitutive models for nonlinear trusses.

All quantities use a consistent unit system.  With SI input, stress and
moduli are Pa, strain is dimensionless, and dissipated energy density is J/m3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import math


@dataclass(frozen=True)
class UniaxialMaterialState:
    """Committed material-point history at the end of a converged increment."""

    strain: float = 0.0
    stress: float = 0.0
    plastic_strain: float = 0.0
    equivalent_plastic_strain: float = 0.0
    dissipated_energy_density: float = 0.0


@dataclass(frozen=True)
class UniaxialMaterialResponse:
    """Stress update and algorithmic tangent returned by a material model."""

    stress: float
    tangent: float
    state: UniaxialMaterialState
    yielded: bool
    plastic_multiplier_increment: float


@runtime_checkable
class UniaxialMaterialModel(Protocol):
    """Interface required by the incremental nonlinear truss solver."""

    E: float

    def initial_state(self) -> UniaxialMaterialState: ...

    def update(
        self, strain: float, committed: UniaxialMaterialState
    ) -> UniaxialMaterialResponse: ...


def _positive_finite(name: str, value: float, *, allow_zero: bool = False) -> float:
    value = float(value)
    valid = value >= 0.0 if allow_zero else value > 0.0
    if not math.isfinite(value) or not valid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


@dataclass(frozen=True)
class ElasticPerfectlyPlastic:
    """One-dimensional associative elastic-perfectly-plastic material."""

    E: float
    yield_stress: float
    yield_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        object.__setattr__(self, "E", _positive_finite("E", self.E))
        object.__setattr__(
            self, "yield_stress", _positive_finite("yield_stress", self.yield_stress)
        )
        object.__setattr__(
            self, "yield_tolerance",
            _positive_finite("yield_tolerance", self.yield_tolerance, allow_zero=True),
        )

    def initial_state(self) -> UniaxialMaterialState:
        return UniaxialMaterialState()

    def update(
        self, strain: float, committed: UniaxialMaterialState
    ) -> UniaxialMaterialResponse:
        return _return_mapping(
            float(strain), committed, self.E, self.yield_stress, 0.0,
            self.yield_tolerance,
        )


@dataclass(frozen=True)
class BilinearIsotropicHardening:
    """Bilinear uniaxial plasticity with linear isotropic hardening.

    ``tangent_modulus`` is the observable post-yield stress-strain slope.  The
    internal hardening modulus used by return mapping is
    ``H = E*Et/(E-Et)``, so the consistent tangent is exactly ``Et``.
    """

    E: float
    yield_stress: float
    tangent_modulus: float
    yield_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        elastic = _positive_finite("E", self.E)
        tangent = _positive_finite(
            "tangent_modulus", self.tangent_modulus, allow_zero=True
        )
        if tangent >= elastic:
            raise ValueError("tangent_modulus must be smaller than E")
        object.__setattr__(self, "E", elastic)
        object.__setattr__(
            self, "yield_stress", _positive_finite("yield_stress", self.yield_stress)
        )
        object.__setattr__(self, "tangent_modulus", tangent)
        object.__setattr__(
            self, "yield_tolerance",
            _positive_finite("yield_tolerance", self.yield_tolerance, allow_zero=True),
        )

    @property
    def hardening_modulus(self) -> float:
        if self.tangent_modulus == 0.0:
            return 0.0
        return self.E * self.tangent_modulus / (self.E - self.tangent_modulus)

    def initial_state(self) -> UniaxialMaterialState:
        return UniaxialMaterialState()

    def update(
        self, strain: float, committed: UniaxialMaterialState
    ) -> UniaxialMaterialResponse:
        return _return_mapping(
            float(strain), committed, self.E, self.yield_stress,
            self.hardening_modulus, self.yield_tolerance,
        )


def _return_mapping(
    strain: float,
    committed: UniaxialMaterialState,
    elastic_modulus: float,
    yield_stress: float,
    hardening_modulus: float,
    tolerance: float,
) -> UniaxialMaterialResponse:
    """Closed-form backward-Euler return to the one-dimensional yield surface."""

    values = (
        strain, committed.strain, committed.stress, committed.plastic_strain,
        committed.equivalent_plastic_strain,
        committed.dissipated_energy_density,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("strain and committed material history must be finite")
    if committed.equivalent_plastic_strain < 0.0:
        raise ValueError("equivalent plastic strain cannot be negative")
    if committed.dissipated_energy_density < 0.0:
        raise ValueError("dissipated energy cannot be negative")

    trial_stress = elastic_modulus * (strain - committed.plastic_strain)
    current_yield = yield_stress + hardening_modulus * committed.equivalent_plastic_strain
    yield_function = abs(trial_stress) - current_yield
    scale = max(yield_stress, abs(trial_stress), 1.0)
    if yield_function <= tolerance * scale:
        state = UniaxialMaterialState(
            strain=strain,
            stress=trial_stress,
            plastic_strain=committed.plastic_strain,
            equivalent_plastic_strain=committed.equivalent_plastic_strain,
            dissipated_energy_density=committed.dissipated_energy_density,
        )
        return UniaxialMaterialResponse(trial_stress, elastic_modulus, state, False, 0.0)

    direction = 1.0 if trial_stress >= 0.0 else -1.0
    plastic_increment = yield_function / (elastic_modulus + hardening_modulus)
    plastic_strain = committed.plastic_strain + direction * plastic_increment
    equivalent_plastic_strain = committed.equivalent_plastic_strain + plastic_increment
    stress = trial_stress - elastic_modulus * direction * plastic_increment
    tangent = (
        elastic_modulus * hardening_modulus / (elastic_modulus + hardening_modulus)
        if hardening_modulus > 0.0 else 0.0
    )
    dissipated = (
        committed.dissipated_energy_density + yield_stress * plastic_increment
    )
    state = UniaxialMaterialState(
        strain=strain,
        stress=stress,
        plastic_strain=plastic_strain,
        equivalent_plastic_strain=equivalent_plastic_strain,
        dissipated_energy_density=dissipated,
    )
    return UniaxialMaterialResponse(stress, tangent, state, True, plastic_increment)

