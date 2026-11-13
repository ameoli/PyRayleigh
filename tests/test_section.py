import math

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import GuideOptics, Section, units

OPTICS = GuideOptics(193414489032258.06, 1.4682, 1.4682)


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


def test_material_length_and_independent_coefficients():
    section = Section(2., 7., OPTICS, attenuation=0., rayleigh_coefficient=1e-8)
    assert section.length == 5.
    assert section.optics is OPTICS
    assert section.rayleigh_coefficient == 1e-8
    np.testing.assert_array_equal(section.attenuation_at([1e14, 2e14]), [0., 0.])


@pytest.mark.parametrize("start,end,error", [
    (-1., 1., ValueError), (1., 1., ValueError), (2., 1., ValueError),
    (float("nan"), 1., ValueError), (0., float("inf"), ValueError),
    (True, 1., TypeError), (np.array(0.), 1., TypeError), (0., "1", TypeError),
])
def test_fixed_geometry_is_checked(start, end, error):
    with pytest.raises(error):
        Section(start, end, OPTICS, 0., 0.)


def test_section_requires_guide_optics():
    with pytest.raises(TypeError, match="GuideOptics"):
        Section(0., 1., None, 0., 0.)


@pytest.mark.parametrize("attenuation,rayleigh,error", [
    (np.array([.2]), 1e-8, ValueError), (.2, np.array([1e-8]), ValueError),
    (.2j, 1e-8, TypeError), (.2, 1e-8j, TypeError),
])
def test_coefficients_are_real_scalars(attenuation, rayleigh, error):
    with pytest.raises(error):
        Section(0., 1., OPTICS, attenuation, rayleigh)


@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
def test_constant_attenuation_follows_frequency_shape(xp, shape):
    alpha = xp.asarray(4.605170185988092e-5, dtype=xp.float64)
    rayleigh = xp.asarray(1e-8, dtype=xp.float64)
    section = Section(0., 1000., OPTICS, alpha, rayleigh)
    frequency = xp.full(shape, 193414489032258.06, dtype=xp.float64)
    actual = section.attenuation_at(frequency)
    assert actual.shape == shape
    assert actual.dtype == xp.float64
    assert section.attenuation is alpha
    assert section.rayleigh_coefficient is rayleigh
    np.testing.assert_array_equal(np.asarray(actual), np.full(shape, 4.605170185988092e-5))


def test_jax_power_loss_derivatives(jax_numpy):
    import jax
    def power(attenuation_db_per_km):
        alpha = units.attenuation_from_db_per_km(attenuation_db_per_km)
        section = Section(0., 1000., OPTICS, alpha, 0.)
        return jax_numpy.exp(-section.attenuation_at(193414489032258.06)*section.length)
    attenuation = jax_numpy.asarray(.2, dtype=jax_numpy.float64)
    value = jax.jit(power)(attenuation)
    first = jax.jit(jax.grad(power))(attenuation)
    second = jax.jit(jax.grad(jax.grad(power)))(attenuation)
    expected = .954992586021436
    scale = math.log(10.)/10.
    np.testing.assert_allclose(value, expected, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(first, -scale*expected, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(second, scale**2*expected, rtol=1e-13, atol=0.)


def test_jax_rayleigh_parameter_with_numpy_attenuation(jax_numpy):
    import jax
    alpha = units.attenuation_from_db_per_km(.2)
    def level(strength):
        section = Section(0., 5., OPTICS, alpha, strength*1e-8)
        return units.linear_to_db(section.rayleigh_coefficient)
    strength = jax_numpy.asarray(2., dtype=jax_numpy.float64)
    value = jax.jit(level)(strength)
    first = jax.jit(jax.grad(level))(strength)
    second = jax.jit(jax.grad(jax.grad(level)))(strength)
    np.testing.assert_allclose(value, -76.98970004336019, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(first, 10./(math.log(10.)*2.), rtol=1e-13, atol=0.)
    np.testing.assert_allclose(second, -10./(math.log(10.)*4.), rtol=1e-13, atol=0.)
