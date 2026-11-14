import math

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest
from scipy import special

from pyrayleigh import weak_fbg_reflection
from pyrayleigh.constants import SPEED_OF_LIGHT
from pyrayleigh.scattering.fbg import _sinc

WAVELENGTH, LENGTH, INDEX, REFLECTIVITY = 1550e-9, .005, 1.4682, .04
FREQUENCY = SPEED_OF_LIGHT/WAVELENGTH
BANDWIDTH = SPEED_OF_LIGHT/(2.*INDEX*LENGTH)


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


def test_native_sinc_matches_numpy_in_the_centre_and_sidelobes(xp):
    values = np.r_[np.linspace(-20., 20., 101), 0., 1e-14, -1e-10,
                    .1/math.pi*(1. - 1e-9), .1/math.pi*(1. + 1e-9)]
    native = xp.asarray(values.tolist())
    result = _sinc(native, xp)
    assert result.dtype == xp.float64
    assert type(result) is type(native)
    np.testing.assert_allclose(np.asarray(result), np.sinc(values), rtol=1e-13, atol=1e-16)


def test_field_spectrum_has_the_pyofdr_peak_nulls_and_signed_sidelobes(xp):
    frequency = xp.asarray((FREQUENCY + BANDWIDTH*np.array([0., 1., 1.5, 2., 2.5])).tolist())
    result = weak_fbg_reflection(frequency, WAVELENGTH, LENGTH, REFLECTIVITY, INDEX)
    expected = math.sqrt(REFLECTIVITY)*np.sinc(2.*INDEX*LENGTH*(np.asarray(frequency) - FREQUENCY)/SPEED_OF_LIGHT)
    assert result.dtype == xp.complex128
    assert type(result) is type(frequency)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(np.asarray(result)[0], math.sqrt(REFLECTIVITY), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(result)[[1, 3]], 0., atol=2e-12)
    assert np.asarray(result)[2].real < 0. and np.asarray(result)[4].real > 0.
    np.testing.assert_array_equal(np.asarray(result).imag, np.zeros(5))


def test_peak_reflectivity_sets_power_and_amplitude_scaling(xp):
    frequency = xp.asarray((FREQUENCY + BANDWIDTH*np.array([-.4, 0., .4])).tolist())
    first = weak_fbg_reflection(frequency, WAVELENGTH, LENGTH, .01, INDEX)
    second = weak_fbg_reflection(frequency, WAVELENGTH, LENGTH, .04, INDEX)
    np.testing.assert_allclose(np.asarray(second), 2.*np.asarray(first), rtol=1e-14)
    np.testing.assert_allclose(abs(np.asarray(second)[1])**2, .04, rtol=1e-14)
    zero = weak_fbg_reflection(frequency, WAVELENGTH, LENGTH, 0., INDEX)
    np.testing.assert_array_equal(np.asarray(zero), np.zeros(3, dtype=np.complex128))


def test_grating_length_sets_the_frequency_width(xp):
    frequency = FREQUENCY + xp.asarray([4e9, 8e9, 12e9])
    first = weak_fbg_reflection(frequency, WAVELENGTH, LENGTH, REFLECTIVITY, INDEX)
    doubled = weak_fbg_reflection(FREQUENCY + .5*(frequency - FREQUENCY), WAVELENGTH,
                                   2.*LENGTH, REFLECTIVITY, INDEX)
    np.testing.assert_allclose(np.asarray(doubled), np.asarray(first), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
def test_frequency_shapes_and_complex_dtype_are_kept(xp, shape):
    frequency = xp.full(shape, FREQUENCY, dtype=xp.float64)
    result = weak_fbg_reflection(frequency, WAVELENGTH, LENGTH, REFLECTIVITY, INDEX)
    assert result.shape == shape
    assert result.dtype == xp.complex128
    np.testing.assert_allclose(np.asarray(result), np.full(shape, math.sqrt(REFLECTIVITY)), rtol=1e-14)


def test_all_physical_inputs_broadcast_normally(xp):
    wavelength = xp.asarray([[WAVELENGTH], [WAVELENGTH*1.0001]])
    length = xp.asarray([[LENGTH], [2.*LENGTH]], dtype=xp.float32)
    reflectivity = xp.asarray([[.01], [.04]])
    frequency = xp.asarray((FREQUENCY + BANDWIDTH*np.array([-.5, 0., .5])).tolist())
    result = weak_fbg_reflection(frequency, wavelength, length, reflectivity, INDEX)
    detuning = np.asarray(frequency) - SPEED_OF_LIGHT/np.asarray(wavelength)
    expected = np.sqrt(np.asarray(reflectivity))*np.sinc(
        2.*INDEX*np.asarray(length, dtype=np.float64)*detuning/SPEED_OF_LIGHT)
    assert result.shape == (2, 3)
    assert result.dtype == xp.complex128
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("name", ["frequency", "bragg_wavelength", "length", "peak_reflectivity", "effective_index"])
def test_physical_parameters_are_real(name):
    inputs = dict(frequency=FREQUENCY, bragg_wavelength=WAVELENGTH, length=LENGTH,
                   peak_reflectivity=REFLECTIVITY, effective_index=INDEX)
    inputs[name] = inputs[name] + 1j
    with pytest.raises(TypeError):
        weak_fbg_reflection(**inputs)


def test_mixed_namespaces_need_explicit_selection(jax_numpy):
    frequency, length = np.asarray([FREQUENCY]), jax_numpy.asarray(LENGTH)
    with pytest.raises(TypeError):
        weak_fbg_reflection(frequency, WAVELENGTH, length, REFLECTIVITY, INDEX)
    result = weak_fbg_reflection(frequency, WAVELENGTH, length, REFLECTIVITY, INDEX, xp=jax_numpy)
    assert type(result) is type(length)


def test_jax_sinc_derivatives_match_spherical_bessel_identities_near_zero(jax_numpy):
    import jax
    def sinc(value):
        return _sinc(value, jax_numpy)
    points = np.array([0., 1e-12, 1e-5, .099999999/math.pi, .100000001/math.pi, .5, 1., 1.5])
    angle = math.pi*points
    expected = (special.spherical_jn(0, angle), -math.pi*special.spherical_jn(1, angle),
                 -math.pi**2*special.spherical_jn(1, angle, derivative=True))
    for function, reference in zip((sinc, jax.grad(sinc), jax.hessian(sinc)), expected):
        actual = jax.jit(jax.vmap(function))(jax_numpy.asarray(points.tolist()))
        np.testing.assert_allclose(np.asarray(actual), reference, rtol=2e-11, atol=1e-24)


def test_jax_field_peak_has_finite_curvature_at_exact_bragg_frequency(jax_numpy):
    import jax
    def reflection(offset):
        return jax_numpy.real(weak_fbg_reflection(FREQUENCY + BANDWIDTH*offset, WAVELENGTH,
                                                  LENGTH, REFLECTIVITY, INDEX))
    offset = jax_numpy.asarray(0.)
    np.testing.assert_allclose(jax.jit(reflection)(offset), math.sqrt(REFLECTIVITY), rtol=1e-14)
    np.testing.assert_allclose(jax.jit(jax.grad(reflection))(offset), 0., atol=2e-12)
    np.testing.assert_allclose(jax.jit(jax.hessian(reflection))(offset),
                               -math.sqrt(REFLECTIVITY)*math.pi**2/3., rtol=1e-12)


def test_jax_wavelength_length_index_and_reflectivity_gradients_and_hessian(jax_numpy):
    import jax
    def reflection(parameters):
        offset, wavelength_scale, length_scale, index_scale, reflectivity = parameters
        return jax_numpy.real(weak_fbg_reflection(FREQUENCY + BANDWIDTH*offset,
            WAVELENGTH*wavelength_scale, LENGTH*length_scale, reflectivity, INDEX*index_scale))
    x, reflectivity = .7, REFLECTIVITY
    ratio = FREQUENCY/BANDWIDTH
    slope_x = np.array([1., ratio, x, x, 0.])
    second_x = np.zeros((5, 5))
    second_x[1, 1] = -2.*ratio
    for index in (2, 3):
        second_x[0, index] = second_x[index, 0] = 1.
        second_x[1, index] = second_x[index, 1] = ratio
    second_x[2, 3] = second_x[3, 2] = x
    s = np.sinc(x)
    ds = -math.pi*special.spherical_jn(1, math.pi*x)
    d2s = -math.pi**2*special.spherical_jn(1, math.pi*x, derivative=True)
    amplitude = math.sqrt(reflectivity)
    dr = np.array([0., 0., 0., 0., 1.])
    first = amplitude*ds*slope_x + s/(2.*amplitude)*dr
    second = (amplitude*(d2s*np.outer(slope_x, slope_x) + ds*second_x)
              + ds/(2.*amplitude)*(np.outer(slope_x, dr) + np.outer(dr, slope_x))
              - s/(4.*reflectivity**1.5)*np.outer(dr, dr))
    parameters = jax_numpy.asarray([x, 1., 1., 1., reflectivity])
    np.testing.assert_allclose(jax.jit(reflection)(parameters), amplitude*s, rtol=1e-11)
    np.testing.assert_allclose(jax.jit(jax.grad(reflection))(parameters), first, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(jax.jit(jax.hessian(reflection))(parameters), second, rtol=2e-9, atol=1e-9)
