import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import units


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


def test_power_ratios_and_decibels(xp):
    decibels = xp.asarray([-20., 0., 20.], dtype=xp.float64)
    ratios = units.db_to_linear(decibels)
    np.testing.assert_allclose(np.asarray(ratios), [.01, 1., 100.], rtol=1e-14)
    np.testing.assert_allclose(np.asarray(units.linear_to_db(ratios)), [-20., 0., 20.], atol=1e-14)
    assert ratios.dtype == xp.float64


def test_one_kilometre_power_and_field_loss(xp):
    alpha = units.attenuation_from_db_per_km(.2, xp=xp)
    np.testing.assert_allclose(np.asarray(alpha), 4.605170185988092e-5, rtol=1e-14, atol=0.)
    power = xp.exp(-alpha * 1000.)
    one_way_field = xp.exp(-alpha * 1000. / 2.)
    round_trip_field = one_way_field**2
    expected = 0.954992586021436
    np.testing.assert_allclose(np.asarray(power), expected, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(round_trip_field), expected, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(units.attenuation_to_db_per_km(alpha)), .2, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(units.attenuation_from_db_per_km(0., xp=xp)), 0.)


def test_vacuum_wavelength_and_absolute_frequency(xp):
    wavelength = xp.asarray([1310e-9, 1550e-9], dtype=xp.float64)
    frequency = units.wavelength_to_frequency(wavelength)
    np.testing.assert_allclose(np.asarray(frequency)[1], 193414489032258.06, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(units.frequency_to_wavelength(frequency)),
                               [1310e-9, 1550e-9], rtol=1e-14, atol=0.)


def test_dispersion_units_sign_and_broadcasting(xp):
    dispersion = xp.asarray([[-17.], [0.], [17.]], dtype=xp.float64) * units.PS_PER_NM_KM
    wavelength = xp.asarray([1550e-9, 3100e-9], dtype=xp.float64)
    beta2 = units.dispersion_to_gvd_at_wavelength(dispersion, wavelength)
    reference = -2.1682619391414893e-26
    expected = reference * np.array([[-1., -4.], [0., 0.], [1., 4.]])
    assert beta2.shape == (3, 2)
    np.testing.assert_allclose(np.asarray(beta2), expected, rtol=1e-14, atol=0.)


def test_complex_units_are_rejected(xp):
    with pytest.raises(TypeError, match="real values"):
        units.db_to_linear(xp.asarray([1. + 2j], dtype=xp.complex128))


def test_jax_dispersion_derivatives_are_scaled(jax_numpy):
    import jax
    def beta2_in_ps2_per_km(wavelength_um):
        return units.dispersion_to_gvd_at_wavelength(17e-6, wavelength_um * 1e-6) * 1e27
    wavelength = jax_numpy.asarray(1.55, dtype=jax_numpy.float64)
    expected = -21.682619391414893
    value = jax.jit(beta2_in_ps2_per_km)(wavelength)
    first = jax.jit(jax.grad(beta2_in_ps2_per_km))(wavelength)
    second = jax.jit(jax.grad(jax.grad(beta2_in_ps2_per_km)))(wavelength)
    np.testing.assert_allclose(value, expected, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(first, 2.*expected/1.55, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(second, 2.*expected/1.55**2, rtol=1e-13, atol=0.)


def test_jax_frequency_derivatives_are_scaled(jax_numpy):
    import jax
    def wavelength_um(frequency_thz):
        return units.frequency_to_wavelength(frequency_thz * 1e12) * 1e6
    frequency = jax_numpy.asarray(193.41448903225806, dtype=jax_numpy.float64)
    value = jax.jit(wavelength_um)(frequency)
    first = jax.jit(jax.grad(wavelength_um))(frequency)
    second = jax.jit(jax.grad(jax.grad(wavelength_um)))(frequency)
    np.testing.assert_allclose(value, 1.55, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(first, -1.55/193.41448903225806, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(second, 2.*1.55/193.41448903225806**2, rtol=1e-13, atol=0.)
