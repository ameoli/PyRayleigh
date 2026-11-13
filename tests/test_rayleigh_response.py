from dataclasses import replace

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (Fiber, GuideOptics, PointReflector, RayleighRealization,
                        Section, lossless_reflector, optical_path,
                        rayleigh_from_normals, rayleigh_response)
from pyrayleigh.constants import SPEED_OF_LIGHT

REFERENCE = 193414489032258.06
GUIDE = GuideOptics(REFERENCE, 1.4682, 1.4682)


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


def make_fiber(sample=None, elements=(), attenuation=.2, length=2., rate=1e-8):
    return Fiber("spool-1", length, 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                  [Section(0., length, GUIDE, attenuation, rate)], elements, sample)


def test_stored_coefficients_are_summed_without_new_cell_scaling(xp):
    coefficients = xp.asarray([.01 + .02j, -.03j, .02], dtype=xp.complex128)
    sample = RayleighRealization(coefficients, [0., .75, 1.75], [0., .5, 1.25, 2.])
    fiber = make_fiber(sample)
    frequency = xp.asarray([REFERENCE - 1e9, REFERENCE, REFERENCE + 1e9])
    path = optical_path(fiber)
    expected = (.01 + .02j - .03j*np.asarray(path.round_trip(0., .75, frequency))
                + .02*np.asarray(path.round_trip(0., 1.75, frequency)))
    actual = rayleigh_response(fiber, frequency)
    assert actual.dtype == xp.complex128
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-13, atol=0.)
    np.testing.assert_array_equal(np.asarray(sample.coefficients), np.asarray(coefficients))


def test_points_at_or_before_a_sample_supply_complex_through_factors(xp):
    sample = RayleighRealization(xp.asarray([.01, .02, .03]), [.25, 1., 1.75], [0., .5, 1.5, 2.])
    values = [xp.asarray(value, dtype=xp.complex128) for value in (.1, 0., .5j, .4j)]
    point = PointReflector(1., *values)
    fiber = make_fiber(sample, [point])
    frequency = xp.asarray([REFERENCE, REFERENCE + 1e9])
    path = optical_path(fiber)
    expected = (.01*np.asarray(path.round_trip(0., .25, frequency))
                - .004*np.asarray(path.round_trip(0., 1., frequency))
                - .006*np.asarray(path.round_trip(0., 1.75, frequency)))
    np.testing.assert_allclose(np.asarray(rayleigh_response(fiber, frequency)), expected,
                               rtol=1e-13, atol=0.)


def test_repeated_rayleigh_positions_cross_a_point_only_once(xp):
    sample = RayleighRealization(xp.asarray([.01, .02]), [1., 1.], [0., 1., 2.])
    values = [xp.asarray(value, dtype=xp.complex128) for value in (0., 0., .5j, .4j)]
    fiber = make_fiber(sample, [PointReflector(1., *values)])
    frequency = xp.asarray(REFERENCE, dtype=xp.float64)
    expected = -.006*np.asarray(optical_path(fiber).round_trip(0., 1., frequency))
    np.testing.assert_allclose(np.asarray(rayleigh_response(fiber, frequency)), expected,
                               rtol=1e-13, atol=0.)


def test_opaque_input_point_blocks_coincident_and_later_rayleigh(xp):
    sample = RayleighRealization(xp.asarray([.01j, .02j]), [0., 2.], [0., 1., 2.])
    fiber = make_fiber(sample, [lossless_reflector(0., 1., xp=xp)])
    frequency = xp.asarray([REFERENCE, REFERENCE + 1e9])
    np.testing.assert_array_equal(np.asarray(rayleigh_response(fiber, frequency)), [0j, 0j])


@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
@pytest.mark.parametrize("has_sample", [False, True])
def test_frequency_shape_and_missing_realization(xp, shape, has_sample):
    sample = RayleighRealization(xp.asarray([.01j]), [1.], [0., 2.]) if has_sample else None
    fiber = make_fiber(sample)
    frequency = xp.full(shape, REFERENCE, dtype=xp.float64)
    actual = rayleigh_response(fiber, frequency)
    assert actual.shape == shape
    assert actual.dtype == xp.complex128
    expected_magnitude = .01*np.exp(-.2) if has_sample else 0.
    np.testing.assert_allclose(np.abs(np.asarray(actual)), np.full(shape, expected_magnitude),
                               rtol=1e-14, atol=0.)


def test_numpy_sample_and_jax_frequency_require_explicit_selection(jax_numpy):
    sample = RayleighRealization([.01j], [1.], [0., 2.])
    fiber = make_fiber(sample)
    frequency = jax_numpy.asarray(REFERENCE, dtype=jax_numpy.float64)
    with pytest.raises(TypeError):
        rayleigh_response(fiber, frequency)
    actual = rayleigh_response(fiber, frequency, xp=jax_numpy)
    assert actual.dtype == jax_numpy.complex128
    np.testing.assert_allclose(np.abs(np.asarray(actual)), .01*np.exp(-.2), rtol=1e-14, atol=0.)


def test_frequency_is_real_without_a_stored_sample():
    with pytest.raises(TypeError):
        rayleigh_response(make_fiber(), [REFERENCE + 1j])


def test_jax_fixed_normals_keep_rate_and_loss_derivatives(jax_numpy):
    import jax
    spacing = 1_000_000.25*SPEED_OF_LIGHT/(GUIDE.effective_index*REFERENCE)
    normals = np.array([[1., 0.], [2., 0.]])
    def power(parameters):
        rate_scale, optical_depth = parameters
        fiber = make_fiber(attenuation=optical_depth/spacing,
                           length=2.*spacing, rate=rate_scale*1e-8)
        sample = rayleigh_from_normals(fiber, normals, cell_length=spacing, xp=jax_numpy)
        fiber = replace(fiber, rayleigh=sample)
        field = rayleigh_response(fiber, REFERENCE)
        return jax_numpy.real(jax_numpy.conj(field)*field)
    parameters = jax_numpy.asarray([2., .2], dtype=jax_numpy.float64)
    value = jax.jit(power)(parameters)
    gradient = jax.jit(jax.grad(power))(parameters)
    hessian = jax.jit(jax.hessian(power))(parameters)
    rate, depth = np.asarray(parameters)
    scale = 1e-8*spacing/2.
    b = 2.*np.exp(-depth)
    f = 1. - b
    expected_gradient = scale*np.array([f*f, 2.*rate*f*b])
    expected_hessian = scale*np.array([[0., 2.*f*b], [2.*f*b, 2.*rate*(b*b - f*b)]])
    np.testing.assert_allclose(value, scale*rate*f*f, rtol=1e-12, atol=0.)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-21)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-12, atol=1e-21)
