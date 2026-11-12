import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import GuideOptics, units
from pyrayleigh.constants import SPEED_OF_LIGHT

REFERENCE = 193414489032258.06
GVD = -2.1682619391414893e-26


@pytest.fixture
def jax_numpy():
    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    try:
        yield jnp
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture(params=["numpy", "strict", "jax"])
def xp(request):
    if request.param == "numpy":
        return numpy_api
    if request.param == "strict":
        return strict
    return request.getfixturevalue("jax_numpy")


def test_reference_phase_and_group_delay(xp):
    values = [xp.asarray(value, dtype=xp.float64) for value in (REFERENCE, 1.45, 1.47, GVD)]
    guide = GuideOptics(*values)
    np.testing.assert_allclose(np.asarray(guide.reference_propagation_constant),
                               5877818.513168, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(np.asarray(guide.group_delay_per_length(values[0])),
                               4.903392199412835e-9, rtol=1e-14, atol=0.)
    np.testing.assert_array_equal(np.asarray(guide.propagation_constant_offset(values[0])), 0.)


def test_zero_dispersion_constant_index_matches_pyofdr_convention(xp):
    guide = GuideOptics(REFERENCE, 1.4682, 1.4682)
    frequency = xp.asarray([[REFERENCE - 1e10, REFERENCE, REFERENCE + 1e10]], dtype=xp.float64)
    beta = guide.propagation_constant(frequency)
    expected = 2.*np.pi*1.4682*np.asarray(frequency)/SPEED_OF_LIGHT
    assert beta.shape == (1, 3)
    assert beta.dtype == xp.float64
    np.testing.assert_allclose(np.asarray(beta), expected, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(guide.group_delay_per_length(frequency)),
                               np.full((1, 3), 1.4682/SPEED_OF_LIGHT), rtol=1e-14, atol=0.)


def test_small_frequency_offset_keeps_precision(xp):
    guide = GuideOptics(REFERENCE, 1.45, 1.47, GVD)
    frequency = xp.asarray(REFERENCE + 1., dtype=xp.float64)
    actual = guide.propagation_constant_offset(frequency)
    np.testing.assert_allclose(np.asarray(actual), 3.080892182268972e-8, rtol=1e-13, atol=0.)


def test_second_order_dispersion_offset(xp):
    guide = GuideOptics(REFERENCE, 1.45, 1.47, GVD)
    frequency = xp.asarray(REFERENCE + 1e10, dtype=xp.float64)
    actual = guide.propagation_constant_offset(frequency)
    np.testing.assert_allclose(np.asarray(actual), 308.0891754271221, rtol=1e-13, atol=0.)


def test_jax_frequency_derivatives_and_group_delay(jax_numpy):
    import jax
    guide = GuideOptics(REFERENCE, 1.45, 1.47, GVD)
    def beta_offset(frequency_thz):
        return guide.propagation_constant_offset(frequency_thz*1e12)
    frequency = jax_numpy.asarray(REFERENCE/1e12 + .01, dtype=jax_numpy.float64)
    first = jax.jit(jax.grad(beta_offset))(frequency)
    second = jax.jit(jax.grad(jax.grad(beta_offset)))(frequency)
    expected_first = 2.*np.pi*1e12*guide.group_delay_per_length(frequency*1e12)
    np.testing.assert_allclose(first, expected_first, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(second, 4.*np.pi**2*GVD*1e24, rtol=1e-13, atol=0.)


def test_jax_guide_parameters_remain_differentiable(jax_numpy):
    import jax
    def beta(parameters):
        guide = GuideOptics(REFERENCE, parameters[0], parameters[1], parameters[2]*1e-27)
        return guide.propagation_constant(REFERENCE + 1e10)
    parameters = jax_numpy.asarray([1.45, 1.47, GVD*1e27], dtype=jax_numpy.float64)
    value = jax.jit(beta)(parameters)
    gradient = jax.jit(jax.grad(beta))(parameters)
    delta_omega = 2.*np.pi*1e10
    expected = [2.*np.pi*REFERENCE/SPEED_OF_LIGHT,
                delta_omega/SPEED_OF_LIGHT, .5*delta_omega**2*1e-27]
    np.testing.assert_allclose(gradient, expected, rtol=1e-13, atol=0.)
    reference = GuideOptics(REFERENCE, 1.45, 1.47, GVD).propagation_constant(REFERENCE + 1e10)
    np.testing.assert_allclose(value, reference, rtol=1e-14)


def test_numpy_guide_constants_need_an_explicit_jax_namespace(jax_numpy):
    import jax
    reference = units.wavelength_to_frequency(1550e-9)
    guide = GuideOptics(reference, 1.4682, 1.4682)
    frequency = jax_numpy.asarray(REFERENCE/1e12 + .01, dtype=jax_numpy.float64)
    with pytest.raises(TypeError):
        guide.propagation_constant(frequency*1e12)
    def beta(frequency_thz):
        return guide.propagation_constant(frequency_thz*1e12, xp=jax_numpy)
    derivative = jax.jit(jax.grad(beta))(frequency)
    np.testing.assert_allclose(derivative, 2.*np.pi*1e12*1.4682/SPEED_OF_LIGHT,
                               rtol=1e-13, atol=0.)


def test_guide_parameters_are_scalars_and_real():
    with pytest.raises(ValueError, match="scalars"):
        GuideOptics(REFERENCE, np.array([1.45, 1.46]), 1.47).propagation_constant(REFERENCE)
    with pytest.raises(TypeError, match="real values"):
        GuideOptics(REFERENCE, 1.45 + .1j, 1.47).propagation_constant(REFERENCE)
