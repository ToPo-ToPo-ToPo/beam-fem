import numpy as np
import pytest

from beamfem import (
    BilinearIsotropicHardening,
    ElasticPerfectlyPlastic,
    UniaxialMaterialState,
)


E = 200.0e9
YIELD = 250.0e6


def test_elastic_perfectly_plastic_return_mapping_and_unloading_history():
    material = ElasticPerfectlyPlastic(E, YIELD)
    elastic = material.update(1.0e-3, material.initial_state())
    assert elastic.stress == pytest.approx(200.0e6)
    assert elastic.tangent == E
    assert not elastic.yielded

    plastic = material.update(2.0e-3, elastic.state)
    assert plastic.stress == pytest.approx(YIELD)
    assert plastic.tangent == 0.0
    assert plastic.state.plastic_strain == pytest.approx(0.75e-3)
    assert plastic.state.equivalent_plastic_strain == pytest.approx(0.75e-3)
    assert plastic.state.dissipated_energy_density == pytest.approx(YIELD * 0.75e-3)

    unloaded = material.update(1.0e-3, plastic.state)
    assert unloaded.stress == pytest.approx(50.0e6)
    assert unloaded.tangent == E
    assert not unloaded.yielded
    assert unloaded.state.plastic_strain == plastic.state.plastic_strain
    assert unloaded.state.dissipated_energy_density == plastic.state.dissipated_energy_density


def test_bilinear_isotropic_hardening_has_requested_consistent_tangent():
    post_yield = 10.0e9
    material = BilinearIsotropicHardening(E, YIELD, post_yield)
    strain = 2.0e-3
    response = material.update(strain, material.initial_state())
    expected = YIELD + post_yield * (strain - YIELD / E)
    assert response.stress == pytest.approx(expected)
    assert response.tangent == pytest.approx(post_yield)
    perturbation = 1.0e-8
    perturbed = material.update(strain + perturbation, material.initial_state())
    numerical_tangent = (perturbed.stress - response.stress) / perturbation
    assert numerical_tangent == pytest.approx(post_yield, rel=1.0e-8)

    unloaded = material.update(1.5e-3, response.state)
    assert unloaded.tangent == E
    assert unloaded.stress == pytest.approx(response.stress - E * 0.5e-3)


def test_material_models_reject_invalid_parameters_and_history():
    with pytest.raises(ValueError, match="yield_stress"):
        ElasticPerfectlyPlastic(E, 0.0)
    with pytest.raises(ValueError, match="smaller than E"):
        BilinearIsotropicHardening(E, YIELD, E)
    material = ElasticPerfectlyPlastic(E, YIELD)
    with pytest.raises(ValueError, match="cannot be negative"):
        material.update(0.0, UniaxialMaterialState(equivalent_plastic_strain=-1.0))
    with pytest.raises(ValueError, match="finite"):
        material.update(np.nan, material.initial_state())

